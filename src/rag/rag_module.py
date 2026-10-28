"""
RAG Module: FAISS + E5 embeddings + cross-encoder re-ranking

Pipeline (everything runs locally once the models are cached):
1. Extraction    : PDF (page by page) and TXT files
2. Segmentation  : sentence-aware chunks with overlap, never crossing a page
3. Indexing      : E5 embeddings (Sentence-Transformers) in an exact FAISS index
                   (inner product on L2-normalised vectors = cosine similarity)
4. Retrieval     : dense candidates (+ optional BM25 fused with RRF)
5. Re-ranking    : cross-encoder (sentence_transformers.CrossEncoder) rescoring
                   of the candidates against the query
6. Sources       : every result carries its document, page and chunk id
"""

import glob
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# config must be imported first: it may set HF_HUB_OFFLINE before the Hub client loads
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAG_INDEX_DIR,
    RAG_EMBEDDING_MODEL,
    RAG_RERANKER_MODEL,
    RAG_USE_RERANKER,
    RAG_RERANK_CANDIDATES,
)

import faiss
import numpy as np

logger = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)  # noisy on malformed PDFs

# PDF support
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False
        logger.warning("pypdf not installed. PDF support disabled.")

# BM25 support for hybrid search
try:
    from rank_bm25 import BM25Okapi
    BM25_SUPPORT = True
except ImportError:
    BM25_SUPPORT = False
    logger.warning("rank_bm25 not installed. Hybrid search disabled.")

INDEX_FORMAT_VERSION = 1
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class RAGModule:
    """
    RAG module: FAISS dense retrieval (E5), optional BM25 fusion, cross-encoder re-ranking.

    `embedder` / `reranker` can be injected (any object exposing `encode` /
    `predict`), which is how the unit tests run without downloading models.
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persistence_path: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        hybrid_weight: float = 0.3,
        embedding_model: str = None,
        reranker_model: str = None,
        use_reranker: bool = None,
        rerank_candidates: int = None,
        embedder=None,
        reranker=None,
    ):
        self.persistence_path = Path(persistence_path or RAG_INDEX_DIR)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hybrid_weight = hybrid_weight
        self.embedding_model = embedding_model or RAG_EMBEDDING_MODEL
        self.reranker_model = reranker_model or RAG_RERANKER_MODEL
        self.use_reranker = RAG_USE_RERANKER if use_reranker is None else use_reranker
        self.rerank_candidates = rerank_candidates or RAG_RERANK_CANDIDATES

        logger.info(f"Initializing RAG Module ({collection_name}, {self.embedding_model})...")
        self.embedder = embedder
        self.reranker = reranker
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model)

        # E5 models were trained with "query: " / "passage: " prefixes
        self._e5 = "e5" in self.embedding_model.lower()

        self.index: Optional[faiss.Index] = None
        self.chunks: List[str] = []
        self.metadatas: List[dict] = []
        self.bm25_index = None
        self._fingerprint: Optional[str] = None

        self._load()
        logger.info(
            f"RAG initialized. PDF: {PDF_SUPPORT}, BM25: {BM25_SUPPORT}, "
            f"Reranker: {self.use_reranker}, Chunks: {len(self.chunks)}"
        )

    # ------------------------------------------------------------------ #
    # Embeddings / reranker
    # ------------------------------------------------------------------ #

    def _embed(self, texts: List[str], is_query: bool) -> np.ndarray:
        """Encode texts into L2-normalised float32 vectors (cosine == inner product)."""
        if self._e5:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        vectors = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.ascontiguousarray(np.asarray(vectors, dtype="float32"))

    def _get_reranker(self):
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder {self.reranker_model}...")
            self.reranker = CrossEncoder(self.reranker_model, max_length=512)
        return self.reranker

    # ------------------------------------------------------------------ #
    # 1. Extraction
    # ------------------------------------------------------------------ #

    def _read_pdf(self, file_path: str) -> List[Tuple[Optional[int], str]]:
        """Extract text from a PDF as (page_number, text) pairs (1-based pages)."""
        if not PDF_SUPPORT:
            return []
        try:
            reader = PdfReader(file_path)
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
            logger.debug(f"Extracted {sum(len(t) for _, t in pages)} chars from {file_path}")
            return pages
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return []

    def _read_txt(self, file_path: str) -> List[Tuple[Optional[int], str]]:
        """Read a text file (no page notion)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [(None, f.read())]
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # 2. Segmentation
    # ------------------------------------------------------------------ #

    def _overlap_tail(self, chunk: str) -> str:
        """Last `chunk_overlap` characters of a chunk, cut on a word boundary."""
        if self.chunk_overlap <= 0 or len(chunk) <= self.chunk_overlap:
            return "" if self.chunk_overlap <= 0 else chunk
        tail = chunk[-self.chunk_overlap:]
        space = tail.find(" ")
        return tail[space + 1:] if space != -1 else ""

    def _split_long(self, sentence: str) -> List[str]:
        """Hard-split a sentence longer than chunk_size on word boundaries."""
        parts, current = [], ""
        for word in sentence.split(" "):
            while len(word) > self.chunk_size:  # e.g. a long URL or formula
                if current:
                    parts.append(current)
                    current = ""
                parts.append(word[: self.chunk_size])
                word = word[self.chunk_size:]
            if current and len(current) + 1 + len(word) > self.chunk_size:
                parts.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            parts.append(current)
        return parts

    def _chunk_text(self, text: str, source: str, page: Optional[int] = None) -> List[Tuple[str, dict]]:
        """Split text into sentence-aware chunks of <= chunk_size chars, with overlap."""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        def meta(idx: int) -> dict:
            return {"source": source, "chunk": idx, "page": page}

        if len(text) <= self.chunk_size:
            return [(text, meta(0))]

        pieces: List[str] = []
        for sentence in _SENTENCE_RE.split(text):
            if len(sentence) > self.chunk_size:
                pieces.extend(self._split_long(sentence))
            elif sentence:
                pieces.append(sentence)

        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > self.chunk_size:
                chunks.append(current)
                tail = self._overlap_tail(current)
                # the overlap must never push the chunk past the size limit
                current = f"{tail} {piece}" if tail and len(tail) + 1 + len(piece) <= self.chunk_size else piece
            else:
                current = f"{current} {piece}" if current else piece
        if current:
            chunks.append(current)

        return [(c, meta(i)) for i, c in enumerate(chunks)]

    # ------------------------------------------------------------------ #
    # 3. Indexing + persistence
    # ------------------------------------------------------------------ #

    def _list_files(self, directory_path: str, recursive: bool = True) -> List[str]:
        # "**/" matches zero or more directories, so files at the top level are found too
        prefix = "**" if recursive else ""
        files = glob.glob(os.path.join(directory_path, prefix, "*.txt"), recursive=recursive)
        if PDF_SUPPORT:
            files += glob.glob(os.path.join(directory_path, prefix, "*.pdf"), recursive=recursive)
        return sorted(set(files))

    def _compute_fingerprint(self, directory_path: str, files: List[str]) -> str:
        """Hash of everything the index depends on: files, chunking and embedding model."""
        h = hashlib.sha256()
        h.update(json.dumps([
            INDEX_FORMAT_VERSION, self.embedding_model, self.chunk_size, self.chunk_overlap
        ]).encode())
        for fp in files:
            h.update(os.path.relpath(fp, directory_path).encode())
            with open(fp, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        return h.hexdigest()

    def _store_dir(self) -> Path:
        return self.persistence_path / self.collection_name

    def _save(self):
        store = self._store_dir()
        store.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(store / "index.faiss"))
        with open(store / "chunks.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "metadatas": self.metadatas}, f, ensure_ascii=False)
        with open(store / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "format": INDEX_FORMAT_VERSION,
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "fingerprint": self._fingerprint,
                "num_chunks": len(self.chunks),
            }, f, indent=2)
        logger.info(f"Saved FAISS index to {store}")

    def _load(self):
        """Load a persisted index, if any (BM25 is cheap to rebuild from the chunks)."""
        store = self._store_dir()
        try:
            manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("format") != INDEX_FORMAT_VERSION or manifest.get("embedding_model") != self.embedding_model:
                return
            index = faiss.read_index(str(store / "index.faiss"))
            data = json.loads((store / "chunks.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, RuntimeError):
            return
        self.index = index
        self.chunks = data["chunks"]
        self.metadatas = data["metadatas"]
        self._fingerprint = manifest.get("fingerprint")
        self._build_bm25_index()
        logger.info(f"Loaded persisted index ({len(self.chunks)} chunks)")

    def ingest(self, directory_path: str, recursive: bool = True, force: bool = False):
        """
        Extract, segment and index all documents of a directory.

        The index is rebuilt only when the files, chunking parameters or embedding
        model changed since the persisted one (or when `force` is set).
        """
        files = self._list_files(directory_path, recursive)
        logger.info(f"Found {len(files)} documents")

        fingerprint = self._compute_fingerprint(directory_path, files)
        if not force and self.index is not None and fingerprint == self._fingerprint:
            logger.info("Index up to date, skipping ingestion")
            return

        all_chunks, all_metadatas = [], []
        for file_path in files:
            filename = os.path.basename(file_path)
            relative_path = os.path.relpath(file_path, directory_path)
            path_parts = relative_path.split(os.sep)
            level = path_parts[0] if len(path_parts) > 1 else "general"
            source = f"{level}/{filename}"

            pages = self._read_pdf(file_path) if file_path.lower().endswith(".pdf") else self._read_txt(file_path)
            chunk_idx = 0
            for page, text in pages:
                for chunk_text, metadata in self._chunk_text(text, source, page):
                    metadata.update({"chunk": chunk_idx, "level": level, "filename": filename})
                    chunk_idx += 1
                    all_chunks.append(chunk_text)
                    all_metadatas.append(metadata)

        if not all_chunks:
            logger.warning("No content found to ingest")
            return

        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        vectors = self._embed(all_chunks, is_query=False)
        index = faiss.IndexFlatIP(vectors.shape[1])  # exact search: the corpus is only a few thousand chunks
        index.add(vectors)

        self.index = index
        self.chunks = all_chunks
        self.metadatas = all_metadatas
        self._fingerprint = fingerprint
        self._build_bm25_index()
        self._save()
        logger.info(f"Ingested {len(all_chunks)} chunks from {len(files)} files")

    def _build_bm25_index(self):
        if not BM25_SUPPORT or not self.chunks:
            self.bm25_index = None
            return
        self.bm25_index = BM25Okapi([_tokenize(c) for c in self.chunks])

    # ------------------------------------------------------------------ #
    # 4-6. Retrieval, re-ranking, sources
    # ------------------------------------------------------------------ #

    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(self._embed([query], is_query=True), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def _bm25_search(self, query: str, k: int) -> List[int]:
        scores = self.bm25_index.get_scores(_tokenize(query))
        top = np.argsort(-scores)[:k]
        return [int(i) for i in top if scores[i] > 0]

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        use_hybrid: bool = True,
        use_reranker: Optional[bool] = None,
    ) -> Tuple[List[str], List[dict]]:
        """
        Retrieve the passages most relevant to `query`.

        Returns (documents, metadatas). Each metadata dict is a sourced reference:
        `source` ("level/filename"), `page` (PDF page, None for TXT), `chunk`
        (chunk id inside the document), `level`, `filename`, plus the scores
        (`dense_score`, and `rerank_score` when the cross-encoder was used).
        """
        if self.index is None or not self.chunks:
            return [], []
        rerank = self.use_reranker if use_reranker is None else use_reranker
        hybrid = use_hybrid and BM25_SUPPORT and self.bm25_index is not None

        n_candidates = max(self.rerank_candidates, n_results) if rerank else n_results * 2
        n_candidates = max(n_candidates, n_results)

        dense = self._dense_search(query, n_candidates)
        dense_scores = dict(dense)

        if hybrid:
            # Reciprocal Rank Fusion of the dense and BM25 rankings
            k_rrf = 60
            fused = {}
            for rank, (idx, _) in enumerate(dense):
                fused[idx] = 1 / (k_rrf + rank + 1)
            for rank, idx in enumerate(self._bm25_search(query, n_candidates)):
                fused[idx] = fused.get(idx, 0) + self.hybrid_weight / (k_rrf + rank + 1)
            candidates = sorted(fused, key=fused.get, reverse=True)[:n_candidates]
        else:
            candidates = [idx for idx, _ in dense]

        rerank_scores = {}
        if rerank and candidates:
            pairs = [(query, self.chunks[i]) for i in candidates]
            scores = self._get_reranker().predict(pairs, show_progress_bar=False)
            rerank_scores = {idx: float(s) for idx, s in zip(candidates, scores)}
            candidates = sorted(candidates, key=rerank_scores.get, reverse=True)

        documents, metadatas = [], []
        for idx in candidates[:n_results]:
            meta = dict(self.metadatas[idx])
            meta["dense_score"] = dense_scores.get(idx)
            if idx in rerank_scores:
                meta["rerank_score"] = rerank_scores[idx]
            documents.append(self.chunks[idx])
            metadatas.append(meta)
        return documents, metadatas

    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.chunks),
            "chunk_size": self.chunk_size,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model if self.use_reranker else None,
            "bm25_enabled": self.bm25_index is not None,
        }
