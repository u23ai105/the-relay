"""
Live RAG Smoke Test (Phase 6 Integration - Baseline Correctness).
Validates end-to-end Generation, Deterministic Citation Validation, Semantic Verification,
and Answer Grading via the stateful benchmark provider.
Exits non-zero if q01 does not cite exact locator:
  chunk_id: 1706.03762v7_s500_c0007
  paper_id: 1706.03762
  version: v7
  page: 5
  section: 3.2.2 Multi-Head Attention
  paragraph: 12
"""

import asyncio
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation_questions import EVALUATION_QUESTIONS
from src.groq_client import GroqCompatibleClient, end_request_scope, start_request_scope
from src.logger import get_logger
from src.pipeline import RAGPipeline
from src.retrieval import RetrievalEngineRegistry
from scripts.run_load_experiments import start_benchmark_server

EXPECTED_CHUNK_ID = "1706.03762v7_s500_c0007"
EXPECTED_PAPER_ID = "1706.03762"
EXPECTED_VERSION = "v7"
EXPECTED_PAGE = 5
EXPECTED_SECTION = "3.2.2 Multi-Head Attention"
EXPECTED_LOCATOR = "Paragraph 12"


async def async_main() -> int:
    logger = get_logger("run_rag_smoke")
    server, thread = start_benchmark_server(port=8015)

    try:
        config = load_config()
        config["provider"]["standin_base_url"] = "http://127.0.0.1:8015/openai/v1"
        config["provider"]["max_transport_retries"] = 1
        config["pipeline"]["retry_strategy"] = "naive"

        indices_dir = PROJECT_ROOT / "data" / "indices"
        registry = RetrievalEngineRegistry(indices_dir, config)
        client = GroqCompatibleClient(config, source="stand-in")
        pipeline = RAGPipeline(config, registry, client, None)

        q1 = EVALUATION_QUESTIONS[0]
        req_id = "smoke_q01"
        deadline_ms = 45000.0
        deadline_mono = time.monotonic() + (deadline_ms / 1000.0)
        client.register_request_deadline(req_id, deadline_mono)
        scope_token = start_request_scope(req_id, deadline_mono)

        logger.info("Running Q01 Smoke Test on Benchmark Provider")
        try:
            res = await pipeline.run_request(
                question=q1.question,
                request_id=req_id,
                scenario="smoke_test",
                retrieval_setting="500tok_12to4_rerank_yes",
                eval_question=q1,
                enable_cache=False,
                fault_inject="none;req_id=smoke_q01",
                request_deadline_ms=deadline_ms,
            )
        finally:
            end_request_scope(scope_token)
            client.clear_request_deadline(req_id)

        logger.info(f"Pipeline Result Status: {res.status}")
        logger.info(f"Verified Gate Passed: {res.verified}")
        logger.info(f"Abstained: {res.abstained}")
        logger.info(f"Answer Grade: {res.telemetry_row['answer_grade']}")

        final_answer = "".join(res.streamed_chunks)
        print("\n--- Final Verified Streamed Answer ---")
        print(final_answer)
        print("--------------------------------\n")

        trace_data = res.telemetry_row.get("call_trace_json", "[]")
        if isinstance(trace_data, str):
            try:
                call_trace = json.loads(trace_data)
            except json.JSONDecodeError:
                call_trace = []
                logger.error("Failed to decode call_trace_json string.")
        else:
            call_trace = trace_data

        stages = [c.get("stage", "unknown") for c in call_trace if isinstance(c, dict)]
        print(f"Provider Stages Executed: {stages}")

        if (
            res.status != "ok"
            or not res.verified
            or res.abstained
            or res.telemetry_row["answer_grade"] != "correct"
        ):
            logger.error("Q01 Smoke Test failed pipeline status/verification/grade criteria.")
            return 1

        exact_citation_found = False
        observed_citations = []
        if res.generation_output and getattr(res.generation_output, "claims", None):
            for claim in res.generation_output.claims:
                for cit in getattr(claim, "citations", []):
                    loc_str = str(getattr(cit, "locator_detail", "") or "").strip()
                    p_idx = getattr(cit, "paragraph_index", None)
                    observed_citations.append(
                        {
                            "chunk_id": cit.chunk_id,
                            "paper_id": cit.paper_id,
                            "arxiv_version": cit.arxiv_version,
                            "page_number": cit.page_number,
                            "section_title": cit.section_title,
                            "locator_detail": loc_str,
                            "paragraph_index": p_idx,
                        }
                    )
                    if (
                        cit.chunk_id == EXPECTED_CHUNK_ID
                        and cit.paper_id == EXPECTED_PAPER_ID
                        and cit.arxiv_version == EXPECTED_VERSION
                        and cit.page_number == EXPECTED_PAGE
                        and str(cit.section_title).strip() == EXPECTED_SECTION
                        and loc_str == EXPECTED_LOCATOR
                        and (p_idx is None or p_idx == 12)
                    ):
                        exact_citation_found = True

        if not exact_citation_found:
            logger.error(
                f"Q01 Smoke Test exact locator mismatch! Expected "
                f"(chunk_id={EXPECTED_CHUNK_ID}, paper_id={EXPECTED_PAPER_ID}, version={EXPECTED_VERSION}, "
                f"page={EXPECTED_PAGE}, section={EXPECTED_SECTION}, locator={EXPECTED_LOCATOR}), "
                f"got: {observed_citations}"
            )
            return 1

        return 0
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))