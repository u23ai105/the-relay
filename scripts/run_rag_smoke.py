"""
Live RAG Smoke Test (Phase 6 Integration).
Validates end-to-end Generation, Semantic Verification, and Streaming delivery
via actual Groq free-tier API calls using one safe question.
"""

import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, has_groq_api_key
from src.evaluation_questions import EVALUATION_QUESTIONS
from src.groq_client import GroqCompatibleClient
from src.logger import get_logger
from src.pipeline import RAGPipeline
from src.retrieval import RetrievalEngineRegistry
from src.query_and_evidence import VerifiedAnswerCache


async def async_main():
    logger = get_logger("run_rag_smoke")
    config = load_config()

    if not has_groq_api_key():
        logger.error("GROQ_API_KEY is missing from .env. Cannot run live RAG test.")
        sys.exit(1)

    indices_dir = PROJECT_ROOT / "data" / "indices"
    cache_file = PROJECT_ROOT / "data" / "cache" / "verified_answers.json"

    registry = RetrievalEngineRegistry(indices_dir, config)
    client = GroqCompatibleClient(config, source="real")
    cache = VerifiedAnswerCache(cache_file)
    pipeline = RAGPipeline(config, registry, client, cache)

    # Use q01 (Attention mechanism limits) as a fast, safe query
    q1 = EVALUATION_QUESTIONS[0]

    logger.info(f"Running End-to-End Live RAG on Query: {q1.question}")

    res = await pipeline.run_request(
        question=q1.question,
        request_id="smoke_req_001",
        scenario="smoke_test",
        retrieval_setting="500tok_4to4_rerank_no",
        eval_question=q1,
        enable_cache=False,  # Force live Groq call
    )

    logger.info(f"Pipeline Result Status: {res.status}")
    logger.info(f"Verified Gate Passed: {res.verified}")
    logger.info(f"Abstained: {res.abstained}")
    logger.info(f"Grade: {res.telemetry_row['answer_grade']}")

    print("\n--- Verified Streamed Answer ---")
    print("".join(res.streamed_chunks))
    print("--------------------------------\n")

    logger.info(f"Total Latency: {res.telemetry_row['total_ms']} ms")
    logger.info(f"User TTFT: {res.telemetry_row['first_token_ms']} ms")
    logger.info(f"Provider Calls Made: {res.telemetry_row['total_provider_calls']}")
    logger.info(f"Tokens In: {res.telemetry_row['tokens_in']}, Out: {res.telemetry_row['tokens_out']}")


if __name__ == "__main__":
    asyncio.run(async_main())