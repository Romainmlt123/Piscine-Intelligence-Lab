"""
RAG Configuration Definitions

Defines configurations for RAG benchmarking and production use.
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
    
    def __str__(self):
        return f"{self.name} (chunk={self.chunk_size}, k={self.top_k})"


# Embedding models available
EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",  # Fast, English-focused
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # Multi-language (recommended)
    "camembert": "dangvantuan/sentence-camembert-base",  # French-specific (slower)
}

# Configurations to benchmark
BENCHMARK_CONFIGS = [
    RAGConfig(
        name="baseline",
        embedding_model=EMBEDDING_MODELS["minilm"],
        chunk_size=500,
        chunk_overlap=50,
        top_k=5
    ),
    RAGConfig(
        name="multilingual",
        embedding_model=EMBEDDING_MODELS["multilingual"],
        chunk_size=500,
        chunk_overlap=50,
        top_k=5
    ),
    RAGConfig(
        name="small_chunks",
        embedding_model=EMBEDDING_MODELS["minilm"],
        chunk_size=250,
        chunk_overlap=25,
        top_k=5
    ),
    RAGConfig(
        name="large_chunks",
        embedding_model=EMBEDDING_MODELS["minilm"],
        chunk_size=1000,
        chunk_overlap=100,
        top_k=5
    ),
    RAGConfig(
        name="more_context",
        embedding_model=EMBEDDING_MODELS["minilm"],
        chunk_size=500,
        chunk_overlap=50,
        top_k=10
    ),
    RAGConfig(
        name="less_context",
        embedding_model=EMBEDDING_MODELS["minilm"],
        chunk_size=500,
        chunk_overlap=50,
        top_k=3
    ),
    RAGConfig(
        name="multilingual_small",
        embedding_model=EMBEDDING_MODELS["multilingual"],
        chunk_size=250,
        chunk_overlap=25,
        top_k=5
    ),
    RAGConfig(
        name="multilingual_more_k",
        embedding_model=EMBEDDING_MODELS["multilingual"],
        chunk_size=500,
        chunk_overlap=50,
        top_k=10
    ),
]

# CamemBERT config (French-specific, slower)
CAMEMBERT_CONFIG = RAGConfig(
    name="camembert_fr",
    embedding_model=EMBEDDING_MODELS["camembert"],
    chunk_size=500,
    chunk_overlap=50,
    top_k=5
)


def get_config_by_name(name: str) -> Optional[RAGConfig]:
    """Get a configuration by its name"""
    for config in BENCHMARK_CONFIGS:
        if config.name == name:
            return config
    if name == "camembert_fr":
        return CAMEMBERT_CONFIG
    return None


def list_configs() -> list:
    """List all available configuration names"""
    return [c.name for c in BENCHMARK_CONFIGS] + ["camembert_fr"]
