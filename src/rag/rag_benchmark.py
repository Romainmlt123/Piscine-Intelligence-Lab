"""
RAG Benchmark

Evaluates RAG configurations using the actual RAG module (FAISS + E5 + cross-encoder).
Configurations that share the same embedding model and chunking share one index, so
the dense / hybrid / re-ranked variants are compared on exactly the same chunks.

Two relevance criteria are reported for each query:
- keyword hit : >= 40% of `expected_keywords` appear in a retrieved chunk (lenient)
- source hit  : a retrieved chunk comes from a document matching
                `expected_source_pattern` (stricter on provenance, blind to the passage)

Usage:
    python -m src.rag.rag_benchmark --run
    python -m src.rag.rag_benchmark --run --config e5_small_hybrid_rerank
    python -m src.rag.rag_benchmark --report
"""

import json
import logging
import os
import platform
import re
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_module import RAGModule
from rag.rag_configs import BENCHMARK_CONFIGS, RAGConfig, get_config_by_name
from config import KNOWLEDGE_BASE_DIR, DATA_DIR, RAG_RERANKER_MODEL, RAG_RERANK_CANDIDATES

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Aggregated results for a configuration"""
    config_name: str
    num_queries: int
    keyword_hits: int
    source_hits: int
    keyword_mrr: float
    source_mrr: float
    avg_retrieval_time_ms: float
    p95_retrieval_time_ms: float
    index_build_time_s: float  # time to extract + chunk + embed + index (shared by configs of a group)
    num_chunks: int
    details: List[dict]

    @property
    def keyword_hit_rate(self) -> float:
        return self.keyword_hits / self.num_queries

    @property
    def source_hit_rate(self) -> float:
        return self.source_hits / self.num_queries


def _environment() -> dict:
    import faiss
    import sentence_transformers
    import torch
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": torch.cuda.is_available(),
        "faiss": faiss.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "reranker_model": RAG_RERANKER_MODEL,
        "rerank_candidates": RAG_RERANK_CANDIDATES,
    }


class RAGBenchmark:
    """Benchmark driven by the real RAG module (single source of truth for RAG logic)."""

    def __init__(self, knowledge_base_dir: str = None):
        self.knowledge_base_dir = knowledge_base_dir or str(KNOWLEDGE_BASE_DIR)
        self.results_dir = Path(DATA_DIR) / "benchmark_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        queries_path = Path(__file__).parent / "test_queries.json"
        with open(queries_path, 'r', encoding='utf-8') as f:
            self.test_queries = json.load(f)['queries']
        logger.info(f"Loaded {len(self.test_queries)} test queries")

    # ------------------------------------------------------------------ #

    def run_group(self, configs: List[RAGConfig], reranker=None) -> List[BenchmarkResult]:
        """Build one index for configs sharing (embedding, chunking), then evaluate each."""
        first = configs[0]
        logger.info(f"\n{'=' * 60}\nIndex: {first.embedding_model} (chunk={first.chunk_size}, overlap={first.chunk_overlap})\n{'=' * 60}")

        tmp_path = Path(DATA_DIR) / "benchmark_tmp" / re.sub(r"\W+", "_", f"{first.embedding_model}_{first.chunk_size}")
        shutil.rmtree(tmp_path, ignore_errors=True)

        start = time.perf_counter()
        rag = RAGModule(
            collection_name="benchmark",
            persistence_path=str(tmp_path),
            chunk_size=first.chunk_size,
            chunk_overlap=first.chunk_overlap,
            embedding_model=first.embedding_model,
            reranker=reranker,
        )
        rag.ingest(self.knowledge_base_dir, force=True)
        build_time = time.perf_counter() - start
        logger.info(f"Indexed {len(rag.chunks)} chunks in {build_time:.1f}s")

        results = [self._evaluate(rag, c, build_time) for c in configs]
        shutil.rmtree(tmp_path, ignore_errors=True)
        return results

    def _evaluate(self, rag: RAGModule, config: RAGConfig, build_time: float) -> BenchmarkResult:
        logger.info(f"\n--- {config} ---")
        # Warm-up: first call pays lazy model loading / allocator costs, not part of the latency
        rag.retrieve(self.test_queries[0]['question'], n_results=config.top_k,
                     use_hybrid=config.use_hybrid, use_reranker=config.use_reranker)

        details = []
        for q in self.test_queries:
            start = time.perf_counter()
            docs, metas = rag.retrieve(q['question'], n_results=config.top_k,
                                       use_hybrid=config.use_hybrid, use_reranker=config.use_reranker)
            elapsed_ms = (time.perf_counter() - start) * 1000

            kw_rank = self._keyword_rank(docs, [k.lower() for k in q.get('expected_keywords', [])])
            src_rank = self._source_rank(metas, q.get('expected_source_pattern'))
            logger.info(f"  {q['id']}: keyword_rank={kw_rank} source_rank={src_rank} time={elapsed_ms:.0f}ms")
            details.append({
                'query_id': q['id'], 'keyword_rank': kw_rank, 'source_rank': src_rank,
                'time_ms': elapsed_ms,
                'sources': [f"{m['source']}#p{m.get('page')}" for m in metas],
            })

        n = len(details)
        times = sorted(d['time_ms'] for d in details)
        result = BenchmarkResult(
            config_name=config.name,
            num_queries=n,
            keyword_hits=sum(1 for d in details if d['keyword_rank'] > 0),
            source_hits=sum(1 for d in details if d['source_rank'] > 0),
            keyword_mrr=sum(1 / d['keyword_rank'] for d in details if d['keyword_rank'] > 0) / n,
            source_mrr=sum(1 / d['source_rank'] for d in details if d['source_rank'] > 0) / n,
            avg_retrieval_time_ms=sum(times) / n,
            p95_retrieval_time_ms=times[min(n - 1, int(round(0.95 * n)) - 1)],
            index_build_time_s=build_time,
            num_chunks=len(rag.chunks),
            details=details,
        )
        logger.info(f"=> keyword {result.keyword_hits}/{n} (MRR {result.keyword_mrr:.3f}), "
                    f"source {result.source_hits}/{n} (MRR {result.source_mrr:.3f}), "
                    f"{result.avg_retrieval_time_ms:.1f}ms avg")
        return result

    @staticmethod
    def _keyword_rank(docs: List[str], keywords: List[str]) -> int:
        """1-based rank of the first chunk containing >= 40% of the keywords (0 if none)."""
        for i, doc in enumerate(docs):
            text = doc.lower()
            if sum(1 for kw in keywords if kw in text) >= len(keywords) * 0.4:
                return i + 1
        return 0

    @staticmethod
    def _source_rank(metas: List[dict], pattern: Optional[str]) -> int:
        """1-based rank of the first chunk whose document matches the expected pattern (0 if none)."""
        if not pattern:
            return 0
        for i, meta in enumerate(metas):
            if re.search(pattern, meta['source']):
                return i + 1
        return 0

    # ------------------------------------------------------------------ #

    def run_all(self, configs: List[RAGConfig] = None) -> List[BenchmarkResult]:
        """Run the benchmark for all configurations (grouped by shared index)."""
        configs = configs or BENCHMARK_CONFIGS

        groups = OrderedDict()
        for c in configs:
            groups.setdefault((c.embedding_model, c.chunk_size, c.chunk_overlap), []).append(c)

        reranker = None
        if any(c.use_reranker for c in configs):
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(RAG_RERANKER_MODEL, max_length=512)

        results = []
        for group in groups.values():
            try:
                results.extend(self.run_group(group, reranker=reranker))
                self._save_results(results)
            except Exception as e:
                logger.error(f"Error testing group {group[0].embedding_model}: {e}")
        return results

    def _save_results(self, results: List[BenchmarkResult]):
        data = {
            "timestamp": datetime.now().isoformat(),
            "environment": _environment(),
            "results": [asdict(r) for r in results],
        }
        output_path = self.results_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        for path in (output_path, self.results_dir / "latest_results.json"):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    def generate_report(self) -> str:
        """Generate a markdown report from the latest results."""
        latest_path = self.results_dir / "latest_results.json"
        if not latest_path.exists():
            return "No results found. Run benchmark first."
        with open(latest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        env = data.get('environment', {})
        lines = ["# RAG Benchmark Results\n", f"**Date**: {data['timestamp']}  ",
                 f"**Environment**: {json.dumps(env)}\n",
                 "| Config | Keyword hit | Keyword MRR | Source hit | Source MRR | Latency avg / p95 (ms) | Chunks | Index build (s) |",
                 "|--------|-------------|-------------|------------|------------|------------------------|--------|-----------------|"]
        for r in data['results']:
            n = r['num_queries']
            lines.append(
                f"| {r['config_name']} | {r['keyword_hits']}/{n} | {r['keyword_mrr']:.3f} | "
                f"{r['source_hits']}/{n} | {r['source_mrr']:.3f} | "
                f"{r['avg_retrieval_time_ms']:.1f} / {r['p95_retrieval_time_ms']:.1f} | "
                f"{r['num_chunks']} | {r['index_build_time_s']:.1f} |")
        lines.append(f"\n_{data['results'][0]['num_queries']} queries: differences of one or two queries are within noise._")
        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG Benchmark")
    parser.add_argument("--run", action="store_true", help="Run benchmark")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--config", type=str, help="Specific config name")
    args = parser.parse_args()
    benchmark = RAGBenchmark()

    if args.run:
        if args.config:
            config = get_config_by_name(args.config)
            if not config:
                logger.error(f"Config '{args.config}' not found")
                sys.exit(1)
            benchmark.run_all([config])
        else:
            benchmark.run_all()

    if args.report:
        print(benchmark.generate_report())


if __name__ == "__main__":
    main()
