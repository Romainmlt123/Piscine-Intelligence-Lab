"""
RAG Configuration Definitions

Defines the configurations compared by the benchmark. Each one is a point in the
design space: embedding model x chunking x (BM25 fusion) x (cross-encoder re-ranking).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """Configuration for a RAG experiment"""
    name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    use_hybrid: bool = False      # fuse BM25 with the dense ranking (RRF)
    use_reranker: bool = False    # re-rank candidates with the cross-encoder

    def __str__(self):
        return (f"{self.name} (chunk={self.chunk_size}, k={self.top_k}, "
                f"hybrid={self.use_hybrid}, rerank={self.use_reranker})")


# Embedding models available
EMBEDDING_MODELS = {
    "minilm_legacy": "paraphrase-multilingual-MiniLM-L12-v2",  # previous model, kept as baseline
    "e5_small": "intfloat/multilingual-e5-small",  # default
    "e5_base": "intfloat/multilingual-e5-base",    # larger, optional
}

_E5S = EMBEDDING_MODELS["e5_small"]
_E5B = EMBEDDING_MODELS["e5_base"]

# Configurations to benchmark. Configs sharing (embedding, chunking) share one FAISS index.
BENCHMARK_CONFIGS = [
    # Baseline: previous embedding model, now served by FAISS (dense only)
    RAGConfig("minilm_dense", EMBEDDING_MODELS["minilm_legacy"], 500, 50, 5),
    # E5-small ablation: each component added in turn
    RAGConfig("e5_small_dense", _E5S, 500, 50, 5),
    RAGConfig("e5_small_hybrid", _E5S, 500, 50, 5, use_hybrid=True),
    RAGConfig("e5_small_rerank", _E5S, 500, 50, 5, use_reranker=True),
    RAGConfig("e5_small_hybrid_rerank", _E5S, 500, 50, 5, use_hybrid=True, use_reranker=True),
    # Chunk size sensitivity (full pipeline)
    RAGConfig("e5_small_hybrid_rerank_c250", _E5S, 250, 25, 5, use_hybrid=True, use_reranker=True),
    RAGConfig("e5_small_hybrid_rerank_c1000", _E5S, 1000, 100, 5, use_hybrid=True, use_reranker=True),
    # E5-base option
    RAGConfig("e5_base_dense", _E5B, 500, 50, 5),
    RAGConfig("e5_base_rerank", _E5B, 500, 50, 5, use_reranker=True),
]


def get_config_by_name(name: str) -> Optional[RAGConfig]:
    """Get a configuration by its name"""
    for config in BENCHMARK_CONFIGS:
        if config.name == name:
            return config
    return None


def list_configs() -> list:
    """List all available configuration names"""
    return [c.name for c in BENCHMARK_CONFIGS]
