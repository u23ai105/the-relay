import pytest
from src.query_and_evidence import process_query_deterministically
from src.evaluation_questions import EVALUATION_QUESTIONS, grade_pipeline_response
from src.prompts_and_schemas import (
    GenerationOutput, 
    ClaimItem, 
    CitationLocator
)
from src.groq_client import GroqCompatibleClient
from src.pipeline import RAGPipeline


def test_deterministic_query_processing():
    pq = process_query_deterministically("What is the d_model size in 1706.03762?")
    assert "1706.03762" in pq.detected_paper_ids
    assert pq.has_domain_signal is True


def test_rubric_grading_correct():
    q = EVALUATION_QUESTIONS[0]
    
    locator = CitationLocator(
        chunk_id="c001", paper_id="1706.03762", arxiv_version="v7",
        page_number=5, section_title="Attention", locator_detail=""
    )
    claim = ClaimItem(claim_id="c1", claim_text="h=8, d_k=64, d_model=512", cited_chunk_ids=["c001"], citations=[locator])
    gen_out = GenerationOutput(abstained=False, abstention_reason="", answer_text="h=8, d_k=64, d_model=512", claims=[claim])
    
    grade = grade_pipeline_response(q, "ok", gen_out, True)
    assert grade == "correct"


def test_pipeline_source_initialization():
    config = {"provider": {"real_base_url": "dummy", "standin_base_url": "dummy"}, "retrieval": {}, "pipeline": {}}
    client_standin = GroqCompatibleClient(config, source="stand-in")
    pipe_standin = RAGPipeline(config, None, client_standin, None)
    assert pipe_standin.source == "stand-in"


@pytest.mark.asyncio
async def test_pipeline_execution_result_question():
    """Verifies that PipelineExecutionResult.question uses the actual query, not the request_id."""
    from src.retrieval import RetrievalResult
    class MockRegistry:
        def retrieve_for_setting(self, *args, **kwargs):
            return RetrievalResult("d", "d", 500, 4, 4, False, "None", 0, 0, [], [], 1.0, 1.0)
    class MockClient:
        source = "stand-in"
    
    config = {
        "provider": {"default_model": "dummy", "max_transport_retries": 0},
        "retrieval": {"default_setting": "dummy", "settings": {"dummy": {"chunk_size_tokens": 500, "candidates_fetched": 4, "pieces_passed": 4, "rerank_enabled": False}}},
        "pipeline": {"max_local_queue_depth": 0} # Force explicit admission rejection
    }
    pipe = RAGPipeline(config, MockRegistry(), MockClient(), None)
    res = await pipe.run_request(question="What is the dimension?", request_id="r1", enable_cache=False)
    
    assert res.request_id == "r1"
    assert res.question == "What is the dimension?"
    assert res.telemetry_row["question_id"] == "unknown"
