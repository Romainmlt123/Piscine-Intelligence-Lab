"""
Download the RAG models once, so the application can then run fully offline.

Usage:
    python -m src.rag.download_models              # default models (E5-small + cross-encoder)
    python -m src.rag.download_models --e5-base    # also fetch multilingual-e5-base

Afterwards, set RAG_OFFLINE=1 to forbid any contact with the Hugging Face Hub.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAG_EMBEDDING_MODEL, RAG_RERANKER_MODEL


def main():
    parser = argparse.ArgumentParser(description="Pre-download RAG models")
    parser.add_argument("--e5-base", action="store_true", help="also download intfloat/multilingual-e5-base")
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer

    embedders = [RAG_EMBEDDING_MODEL] + (["intfloat/multilingual-e5-base"] if args.e5_base else [])
    for name in embedders:
        print(f"Downloading embedder {name}...")
        SentenceTransformer(name)
    print(f"Downloading cross-encoder {RAG_RERANKER_MODEL}...")
    CrossEncoder(RAG_RERANKER_MODEL)
    print("Done. Set RAG_OFFLINE=1 to run without network access.")


if __name__ == "__main__":
    main()
