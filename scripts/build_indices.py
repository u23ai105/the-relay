"""
Phase 4 (Part 2): Offline Chunking, Index Building & Retrieval Configuration.
Builds 250-token and 500-token chunks, constructs offline Hybrid/Dense indices,
and executes a controlled offline RRF-vs-Reranker ablation check.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import build_and_save_all_chunks
from src.config import load_config
from src.logger import get_logger
from src.retrieval import RetrievalEngineRegistry
from src.evaluation_questions import EVALUATION_QUESTIONS


def main() -> None:
    logger = get_logger("build_indices")
    config = load_config()

    indices_dir = PROJECT_ROOT / "data" / "indices"
    pages_jsonl = indices_dir / "extracted_pages.jsonl"

    if not pages_jsonl.exists():
        logger.error(f"Extracted pages not found at {pages_jsonl}. Run extract_corpus.py first.")
        sys.exit(1)

    logger.info("Building 250-token and 500-token contiguous chunks...")
    manifest = build_and_save_all_chunks(
        extracted_pages_jsonl=pages_jsonl,
        output_dir=indices_dir,
        chunk_sizes=(250, 500),
        boundary_context_tokens=45
    )

    logger.info("Chunks built successfully.")
    for sz, details in manifest["configurations"].items():
        logger.info(f"Size {sz}: {details['total_chunks']} chunks ({details['oversized_split_chunks']} oversized splits).")

    logger.info("Initializing Dense / Hybrid RRF / Reranker offline indices...")
    # Initializing the registry builds the in-memory LSA SVD models and saves .npy caches
    registry = RetrievalEngineRegistry(indices_dir, config)

    # Warm up sizes 250 and 500
    engine_250 = registry.get_engine(250)
    engine_500 = registry.get_engine(500)
    logger.info(f"Initialized retrieval engines for 250 (chunks={len(engine_250.chunks)}) and 500 (chunks={len(engine_500.chunks)}).")

    logger.info("Running offline RRF-vs-Reranker comparison on Eval Question 1...")
    q1 = EVALUATION_QUESTIONS[0]
    comparison = engine_500.compare_rrf_vs_reranker_on_same_pool(
        query=q1.question,
        pool_size=12,
        top_k=4,
        paper_boost_ids=[q1.expected_paper_id] if q1.expected_paper_id else None
    )

    logger.info(f"RRF vs Reranker Top-4 Comparison (12 candidates, {comparison['rerank_ms']} ms):")
    logger.info(f"RRF Top 4 IDs:      {comparison['rrf_top_k_ids']}")
    logger.info(f"Reranked Top 4 IDs: {comparison['reranked_top_k_ids']}")
    logger.info(f"Set changed: {comparison['set_changed']} | Order changed: {comparison['order_changed']}")

    comp_file = indices_dir / "rrf_vs_rerank_q1_ablation.json"

    serializable_comp = {
        k: v for k, v in comparison.items()
        if k not in ("rrf_top_pieces", "reranked_top_pieces")
    }

    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(serializable_comp, f, indent=2)

    logger.info("Indices built and cached successfully.")


if __name__ == "__main__":
    main()