"""
RAG Benchmark Script (Simplified)

This script evaluates RAG configurations using the actual RAG module.
No code duplication - uses rag_module.py directly.

Usage:
    python -m src.rag.rag_benchmark --run
    python -m src.rag.rag_benchmark --run --config multilingual
    python -m src.rag.rag_benchmark --report
"""

import json
import time
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_module import RAGModule
from rag.rag_configs import BENCHMARK_CONFIGS, RAGConfig, get_config_by_name
from config import KNOWLEDGE_BASE_DIR, DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Aggregated results for a configuration"""
    config_name: str
    hit_rate: float
    mrr: float
    avg_retrieval_time_ms: float
    total_ingestion_time_s: float
    num_chunks: int
    details: List[dict]


class RAGBenchmark:
    """
    Simplified benchmark using the actual RAG module.
    No code duplication - single source of truth for RAG logic.
    """
    
    def __init__(self, knowledge_base_dir: str = None):
        self.knowledge_base_dir = knowledge_base_dir or str(KNOWLEDGE_BASE_DIR)
        self.results_dir = Path(DATA_DIR) / "benchmark_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test queries
        queries_path = Path(__file__).parent / "test_queries.json"
        with open(queries_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.test_queries = data['queries']
        
        logger.info(f"Loaded {len(self.test_queries)} test queries")
    
    def run_config(self, config: RAGConfig) -> BenchmarkResult:
        """Run benchmark for a single configuration using the real RAG module"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing config: {config.name}")
        logger.info(f"  Embedding: {config.embedding_model}")
        logger.info(f"  Chunk size: {config.chunk_size}, Top-k: {config.top_k}")
        logger.info(f"{'='*60}")
        
        # Create temporary directory for this benchmark
        temp_db_path = Path(DATA_DIR) / f"benchmark_temp_{config.name}"
        shutil.rmtree(temp_db_path, ignore_errors=True)
        
        # Initialize RAG module with this config's settings
        start_ingest = time.time()
        rag = RAGModule(
            collection_name=f"benchmark_{config.name}",
            persistence_path=str(temp_db_path),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Temporarily override embedding model if different
        if config.embedding_model != rag.embedder.get_sentence_embedding_dimension():
            from sentence_transformers import SentenceTransformer
            rag.embedder = SentenceTransformer(config.embedding_model)
        
        # Ingest using the real RAG module
        rag.ingest(self.knowledge_base_dir)
        ingestion_time = time.time() - start_ingest
        
        num_chunks = rag.collection.count()
        logger.info(f"Ingested {num_chunks} chunks in {ingestion_time:.1f}s")
        
        # Run test queries
        logger.info("Running test queries...")
        results = []
        
        for q in self.test_queries:
            start = time.time()
            docs, metas = rag.retrieve(q['question'], n_results=config.top_k)
            elapsed = (time.time() - start) * 1000
            
            # Check hit based on keywords
            expected_keywords = [kw.lower() for kw in q.get('expected_keywords', [])]
            hit, rank = self._check_hit(docs, expected_keywords)
            
            status = "✅" if hit else "❌"
            logger.info(f"  {status} {q['id']}: rank={rank}, time={elapsed:.0f}ms")
            
            results.append({
                'query_id': q['id'],
                'hit': hit,
                'rank': rank,
                'time_ms': elapsed
            })
        
        # Calculate metrics
        hits = sum(1 for r in results if r['hit'])
        hit_rate = hits / len(results)
        mrr = sum(1/r['rank'] if r['rank'] > 0 else 0 for r in results) / len(results)
        avg_time = sum(r['time_ms'] for r in results) / len(results)
        
        # Cleanup
        shutil.rmtree(temp_db_path, ignore_errors=True)
        
        logger.info(f"\nResults: Hit Rate={hit_rate*100:.1f}%, MRR={mrr:.3f}, Avg={avg_time:.1f}ms")
        
        return BenchmarkResult(
            config_name=config.name,
            hit_rate=hit_rate,
            mrr=mrr,
            avg_retrieval_time_ms=avg_time,
            total_ingestion_time_s=ingestion_time,
            num_chunks=num_chunks,
            details=results
        )
    
    def _check_hit(self, docs: List[str], keywords: List[str]) -> tuple:
        """Check if keywords are found in retrieved docs"""
        for i, doc in enumerate(docs):
            text = doc.lower()
            matches = sum(1 for kw in keywords if kw in text)
            if matches >= len(keywords) * 0.4:
                return True, i + 1
        return False, 0
    
    def run_all(self, configs: List[RAGConfig] = None) -> List[BenchmarkResult]:
        """Run benchmark for all configurations"""
        configs = configs or BENCHMARK_CONFIGS
        results = []
        
        for config in configs:
            try:
                result = self.run_config(config)
                results.append(result)
                self._save_results(results)
            except Exception as e:
                logger.error(f"Error testing {config.name}: {e}")
        
        return results
    
    def _save_results(self, results: List[BenchmarkResult]):
        """Save results to JSON"""
        output_path = self.results_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(r) for r in results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save as latest
        latest_path = self.results_dir / "latest_results.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(self) -> str:
        """Generate markdown report"""
        latest_path = self.results_dir / "latest_results.json"
        if not latest_path.exists():
            return "No results found. Run benchmark first."
        
        with open(latest_path, 'r') as f:
            data = json.load(f)
        
        results = sorted(data['results'], key=lambda x: x['hit_rate'], reverse=True)
        
        lines = ["# RAG Benchmark Results\n"]
        lines.append(f"**Date**: {data['timestamp']}\n")
        lines.append("\n| Config | Hit Rate | MRR | Time (ms) |")
        lines.append("|--------|----------|-----|-----------|")
        
        for r in results:
            lines.append(f"| {r['config_name']} | {r['hit_rate']*100:.1f}% | {r['mrr']:.3f} | {r['avg_retrieval_time_ms']:.1f} |")
        
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
            if config:
                benchmark.run_config(config)
            else:
                logger.error(f"Config '{args.config}' not found")
        else:
            benchmark.run_all()
    
    if args.report:
        print(benchmark.generate_report())


if __name__ == "__main__":
    main()
