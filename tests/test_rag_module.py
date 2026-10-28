"""
Unit tests for the RAG pipeline logic (extraction, chunking, FAISS, re-ranking, sources).

They use deterministic stand-ins for the neural models (a hashed bag-of-words embedder and
a token-overlap "cross-encoder"), so they run offline and fast. They check the plumbing, NOT
retrieval quality: quality numbers come from `python -m src.rag.rag_benchmark --run`.
"""

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rag.rag_module import RAGModule  # noqa: E402


class StubEmbedder:
    DIM = 256

    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.DIM), dtype="float32")
        for row, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                if tok not in ("query", "passage"):
                    out[row, zlib.crc32(tok.encode()) % self.DIM] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class StubReranker:
    """Scores by number of distinct shared tokens: a different signal from the embedder."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs, show_progress_bar=False):
        self.pairs.extend(pairs)
        return [len(set(re.findall(r"\w+", q.lower())) & set(re.findall(r"\w+", d.lower()))) for q, d in pairs]


@pytest.fixture
def kb(tmp_path):
    root = tmp_path / "kb"
    (root / "math").mkdir(parents=True)
    (root / "physics").mkdir()
    (root / "math" / "fractions.txt").write_text(
        "Une fraction est un quotient de deux nombres entiers. Le numérateur est au dessus. "
        "Le dénominateur est en dessous et ne peut pas être nul.")
    (root / "math" / "geometrie.txt").write_text(
        "Le théorème de Pythagore relie les côtés d'un triangle rectangle. "
        "L'hypoténuse est le côté opposé à l'angle droit.")
    (root / "physics" / "ohm.txt").write_text(
        "La loi d'Ohm relie la tension, la résistance et l'intensité du courant dans un circuit électrique.")
    return root


def make_rag(tmp_path, **kw):
    kw.setdefault("embedder", StubEmbedder())
    kw.setdefault("embedding_model", "intfloat/multilingual-e5-small")
    return RAGModule(collection_name="t", persistence_path=str(tmp_path / "idx"), **kw)


def test_no_chromadb_dependency():
    import rag.rag_module as m
    assert "chromadb" not in sys.modules
    assert hasattr(m, "faiss")


def test_chunking_respects_size_overlap_and_words(tmp_path):
    rag = make_rag(tmp_path, chunk_size=120, chunk_overlap=30)
    text = " ".join(f"Phrase numéro {i} sur les fractions décimales." for i in range(40))
    chunks = rag._chunk_text(text, "x.txt", page=3)
    assert len(chunks) > 3
    assert all(len(c) <= 120 for c, _ in chunks)
    assert [m["chunk"] for _, m in chunks] == list(range(len(chunks)))
    assert all(m["page"] == 3 and m["source"] == "x.txt" for _, m in chunks)
    # overlap: each chunk starts with the end of the previous one, on a word boundary
    for (prev, _), (nxt, _) in zip(chunks, chunks[1:]):
        first_words = nxt.split(" ")[:2]
        assert first_words[0] in prev.split(" ")


def test_long_sentence_without_period_is_split(tmp_path):
    rag = make_rag(tmp_path, chunk_size=100, chunk_overlap=10)
    chunks = rag._chunk_text("mot " * 200, "x.txt")
    assert len(chunks) > 1 and all(len(c) <= 100 for c, _ in chunks)


def test_ingest_and_dense_retrieval_with_sources(tmp_path, kb):
    rag = make_rag(tmp_path, use_reranker=False)
    rag.ingest(str(kb))
    assert rag.index.ntotal == len(rag.chunks) == 3
    docs, metas = rag.retrieve("Qu'est-ce que la loi d'Ohm ?", n_results=2, use_hybrid=False)
    assert "Ohm" in docs[0]
    assert metas[0]["source"] == "physics/ohm.txt"
    assert metas[0]["level"] == "physics" and metas[0]["filename"] == "ohm.txt"
    assert "chunk" in metas[0] and "page" in metas[0] and "dense_score" in metas[0]
    assert "rerank_score" not in metas[0]


def test_e5_prefixes_are_applied(tmp_path, kb):
    emb = StubEmbedder()
    rag = make_rag(tmp_path, embedder=emb, use_reranker=False)
    rag.ingest(str(kb))
    rag.retrieve("fraction", n_results=1)
    assert all(t.startswith("passage: ") for t in emb.calls[0])
    assert emb.calls[-1] == ["query: fraction"]

    emb2 = StubEmbedder()
    rag2 = make_rag(tmp_path / "other", embedder=emb2, embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
    rag2.ingest(str(kb))
    assert not any(t.startswith("passage: ") for t in emb2.calls[0])


def test_reranker_reorders_candidates_and_exposes_scores(tmp_path, kb):
    rer = StubReranker()
    rag = make_rag(tmp_path, reranker=rer, use_reranker=True, rerank_candidates=3)
    rag.ingest(str(kb))
    docs, metas = rag.retrieve("théorème Pythagore triangle rectangle hypoténuse", n_results=3, use_hybrid=False)
    assert len(rer.pairs) == 3  # all candidates were rescored against the query
    scores = [m["rerank_score"] for m in metas]
    assert scores == sorted(scores, reverse=True)
    assert metas[0]["source"] == "math/geometrie.txt"
    # explicit opt-out
    rer.pairs.clear()
    _, metas = rag.retrieve("fraction", n_results=2, use_reranker=False)
    assert not rer.pairs and "rerank_score" not in metas[0]


def test_hybrid_uses_bm25_and_returns_n_results(tmp_path, kb):
    rag = make_rag(tmp_path, use_reranker=False)
    rag.ingest(str(kb))
    assert rag.bm25_index is not None
    docs, metas = rag.retrieve("dénominateur", n_results=2, use_hybrid=True)
    assert len(docs) == 2 and metas[0]["source"] == "math/fractions.txt"


def test_index_is_persisted_and_not_rebuilt_when_unchanged(tmp_path, kb):
    emb = StubEmbedder()
    rag = make_rag(tmp_path, embedder=emb, use_reranker=False)
    rag.ingest(str(kb))
    n_calls = len(emb.calls)

    emb2 = StubEmbedder()
    rag2 = make_rag(tmp_path, embedder=emb2, use_reranker=False)
    assert len(rag2.chunks) == 3          # loaded from disk
    rag2.ingest(str(kb))
    assert emb2.calls == []               # nothing re-embedded
    docs, _ = rag2.retrieve("loi d'Ohm", n_results=1)
    assert "Ohm" in docs[0]

    (kb / "math" / "nouveau.txt").write_text("Un polyèdre est un solide délimité par des polygones.")
    rag2.ingest(str(kb))                  # content changed -> rebuilt
    assert len(rag2.chunks) == 4 and emb2.calls
    assert n_calls >= 1


def test_index_not_reused_for_another_embedding_model(tmp_path, kb):
    rag = make_rag(tmp_path, use_reranker=False)
    rag.ingest(str(kb))
    other = make_rag(tmp_path, embedding_model="intfloat/multilingual-e5-base", use_reranker=False)
    assert other.index is None


def test_empty_index_returns_nothing(tmp_path):
    rag = make_rag(tmp_path)
    assert rag.retrieve("anything") == ([], [])


def test_pdf_pages_are_tracked(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter
    # a blank PDF has no text: extraction must degrade gracefully (no chunk, no crash)
    root = tmp_path / "kb" / "math"
    root.mkdir(parents=True)
    w = PdfWriter()
    w.add_blank_page(200, 200)
    with open(root / "blank.pdf", "wb") as f:
        w.write(f)
    rag = make_rag(tmp_path, use_reranker=False)
    assert rag._read_pdf(str(root / "blank.pdf")) == [(1, "")]
    rag.ingest(str(tmp_path / "kb"))
    assert rag.index is None
