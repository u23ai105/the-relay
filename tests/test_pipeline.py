import pytest
import asyncio
from pathlib import Path
from src.query_and_evidence import process_query_deterministically
from src.evaluation_questions import EVALUATION_QUESTIONS, grade_pipeline_response
from src.prompts_and_schemas import (
    GenerationOutput, 
    VerificationOutput, 
    ClaimItem, 
    CitationLocator
)
from src.groq_client import GroqCompatibleClient
from src.pipeline import RAGPipeline
from src.verification import validate_citations_deterministically


def test_deterministic_query_processing():
    pq = process_query_deterministically("What is the d_model size in 1706.03762?")
    assert "1706.03762" in pq.detected_paper_ids
    assert pq.has_domain_signal is True


def test_rubric_grading_correct():
    q = EVALUATION_QUESTIONS[0]
    
    locator = CitationLocator(
        chunk_id="c001",
        paper_id="1706.03762",
        arxiv_version="v7",
        page_number=5,
        section_title="Attention",
        locator_detail=""
    )
    
    claim = ClaimItem(
        claim_id="c1",
        claim_text="h=8, d_k=64, d_model=512",
        cited_chunk_ids=["c001"],
        citations=[locator]
    )
    
    gen_out = GenerationOutput(
        abstained=False,
        abstention_reason="",
        answer_text="h=8, d_k=64, d_model=512",
        claims=[claim]
    )
    
    grade = grade_pipeline_response(q, "ok", gen_out, True)
    assert grade == "correct"


def test_rubric_grading_refused_correctly():
    q = EVALUATION_QUESTIONS[12]
    gen_out = GenerationOutput(
        abstained=True,
        abstention_reason="Language not mentioned.",
        answer_text="Language not mentioned.",
        claims=[]
    )
    grade = grade_pipeline_response(q, "ok", gen_out, True)
    assert grade == "refused_correctly"


def test_pipeline_source_initialization():
    config = {
        "provider": {"real_base_url": "dummy", "standin_base_url": "dummy"},
        "retrieval": {},
        "pipeline": {}
    }
    client_standin = GroqCompatibleClient(config, source="stand-in")
    pipe_standin = RAGPipeline(config, None, client_standin, None)
    assert pipe_standin.source == "stand-in"
    
    client_real = GroqCompatibleClient(config, source="real")
    pipe_real = RAGPipeline(config, None, client_real, None)
    assert pipe_real.source == "real"


@pytest.mark.asyncio
async def test_refusal_expansion_guard_failure_abort_ratelimit():
    from src.groq_client import ProviderCallResult
    class MockRegistry:
        def __init__(self):
            self.call_count = 0
            
        def retrieve_for_setting(self, query: str, *args, **kwargs):
            self.call_count += 1
            from src.retrieval import RetrievalResult
            from src.chunking import ChunkRecord
            from src.retrieval import RetrievedPiece
            
            chunk_id = "c002" if "mimi neural audio codec" in query.lower() else "c001"
            dummy = RetrievedPiece(
                chunk=ChunkRecord(
                    chunk_id=chunk_id, paper_id="1706.03762", arxiv_version="v7", title="", filename="", sha256="",
                    chunk_size_target=500, chunk_index_in_paper=1, global_chunk_index=1,
                    primary_page=1, page_start=1, page_end=1, page_numbers=[1],
                    section_title="A", section_path="A", all_sections=["A"], paragraph_refs=[],
                    table_caption_ids=[], table_references=[], figure_caption_ids=[], figure_references=[],
                    equation_numbers=[], has_math=False, is_oversized_split=False, token_count=10, word_count=10,
                    char_count=50, text="text"
                ),
                rank=1, score=1.0
            )
            return RetrievalResult(
                setting_name="dummy", retrieval_method="hybrid_rrf", chunk_size_tokens=500,
                candidates_fetched=4, pieces_passed=4, rerank_enabled=False, reranker_backend_used="None",
                retrieval_ms=0, rerank_ms=0, candidates_before_rerank=[dummy], final_pieces=[dummy],
                max_lexical_overlap=1.0, top_score=1.0
            )
            
    class MockClient:
        source = "stand-in"
        def __init__(self):
            self.calls_made = []
            
        async def execute_single_call(self, stage, *args, **kwargs):
            self.calls_made.append(stage)
            if stage == "generate":
                return ProviderCallResult(
                    success=True, stage="generate", model_id="dummy", http_status=200,
                    content='{"abstained": true, "abstention_reason": "test", "answer_text": "no", "claims": []}',
                    provider_first_token_ms=10.0, work_ms=50.0, tokens_in=10, tokens_out=10, accounting_mode="exact_reported"
                )
            elif stage == "regenerate":
                return ProviderCallResult(
                    success=False, stage="regenerate", model_id="dummy", http_status=429,
                    content="", provider_first_token_ms=None, work_ms=20.0, tokens_in=0, tokens_out=0,
                    accounting_mode="none", failure_reason="rpm_limit", is_retriable=False
                )
            raise AssertionError(f"Unexpected stage called: {stage}")
            
    config = {
        "provider": {"default_model": "dummy", "max_transport_retries": 0},
        "retrieval": {
            "default_setting": "dummy",
            "settings": {"dummy": {"chunk_size_tokens": 500, "candidates_fetched": 4, "pieces_passed": 4, "rerank_enabled": False}}
        },
        "pipeline": {}
    }
    client = MockClient()
    registry = MockRegistry()
    pipe = RAGPipeline(config, registry, client, None)
    res = await pipe.run_request("Trigger expansion alias: audio codec", enable_cache=False)
    
    assert registry.call_count == 2
    assert client.calls_made == ["generate", "regenerate"]
    assert res.status == "rate_limited"
    assert res.telemetry_row["failure_reason"] == "rpm_limit"
    assert res.telemetry_row["correction_attempts"] == 1
    assert res.telemetry_row["failed_attempt_work_ms"] == 20.0
    assert not res.streamed_chunks


@pytest.mark.asyncio
async def test_refusal_expansion_guard_failure_abort_timeout():
    from src.groq_client import ProviderCallResult
    class MockRegistry:
        def __init__(self):
            self.call_count = 0
        def retrieve_for_setting(self, query: str, *args, **kwargs):
            self.call_count += 1
            from src.retrieval import RetrievalResult
            from src.chunking import ChunkRecord
            from src.retrieval import RetrievedPiece
            chunk_id = "c002" if "mimi neural audio codec" in query.lower() else "c001"
            dummy = RetrievedPiece(
                chunk=ChunkRecord(
                    chunk_id=chunk_id, paper_id="1706.03762", arxiv_version="v7", title="", filename="", sha256="",
                    chunk_size_target=500, chunk_index_in_paper=1, global_chunk_index=1,
                    primary_page=1, page_start=1, page_end=1, page_numbers=[1],
                    section_title="A", section_path="A", all_sections=["A"], paragraph_refs=[],
                    table_caption_ids=[], table_references=[], figure_caption_ids=[], figure_references=[],
                    equation_numbers=[], has_math=False, is_oversized_split=False, token_count=10, word_count=10,
                    char_count=50, text="text"
                ),
                rank=1, score=1.0
            )
            return RetrievalResult(
                setting_name="dummy", retrieval_method="hybrid_rrf", chunk_size_tokens=500,
                candidates_fetched=4, pieces_passed=4, rerank_enabled=False, reranker_backend_used="None",
                retrieval_ms=0, rerank_ms=0, candidates_before_rerank=[dummy], final_pieces=[dummy],
                max_lexical_overlap=1.0, top_score=1.0
            )
            
    class MockClient:
        source = "stand-in"
        def __init__(self):
            self.calls_made = []
        async def execute_single_call(self, stage, *args, **kwargs):
            self.calls_made.append(stage)
            if stage == "generate":
                return ProviderCallResult(
                    success=True, stage="generate", model_id="dummy", http_status=200,
                    content='{"abstained": true, "abstention_reason": "test", "answer_text": "no", "claims": []}',
                    provider_first_token_ms=10.0, work_ms=50.0, tokens_in=10, tokens_out=10, accounting_mode="exact_reported"
                )
            elif stage == "regenerate":
                return ProviderCallResult(
                    success=False, stage="regenerate", model_id="dummy", http_status=0,
                    content="", provider_first_token_ms=None, work_ms=20.0, tokens_in=0, tokens_out=0,
                    accounting_mode="none", failure_reason="ttft_timeout", is_retriable=False
                )
            raise AssertionError(f"Unexpected stage called: {stage}")
            
    config = {
        "provider": {"default_model": "dummy", "max_transport_retries": 0},
        "retrieval": {
            "default_setting": "dummy",
            "settings": {"dummy": {"chunk_size_tokens": 500, "candidates_fetched": 4, "pieces_passed": 4, "rerank_enabled": False}}
        },
        "pipeline": {}
    }
    client = MockClient()
    registry = MockRegistry()
    pipe = RAGPipeline(config, registry, client, None)
    res = await pipe.run_request("Trigger expansion alias: audio codec", enable_cache=False)
    
    assert registry.call_count == 2
    assert client.calls_made == ["generate", "regenerate"]
    assert res.status == "timeout"
    assert res.telemetry_row["client_abandoned"] is True
    assert not res.streamed_chunks


@pytest.mark.asyncio
async def test_json_truncation_triggers_regeneration_and_recovers():
    from src.groq_client import ProviderCallResult
    import json
    
    class MockRegistry:
        def retrieve_for_setting(self, *args, **kwargs):
            from src.retrieval import RetrievalResult
            from src.chunking import ChunkRecord
            from src.retrieval import RetrievedPiece
            dummy = RetrievedPiece(
                chunk=ChunkRecord(
                    chunk_id="c001", paper_id="1706.03762", arxiv_version="v7", title="", filename="", sha256="",
                    chunk_size_target=500, chunk_index_in_paper=1, global_chunk_index=1,
                    primary_page=1, page_start=1, page_end=1, page_numbers=[1],
                    section_title="A", section_path="A", all_sections=["A"], paragraph_refs=[],
                    table_caption_ids=[], table_references=[], figure_caption_ids=[], figure_references=[],
                    equation_numbers=[], has_math=False, is_oversized_split=False, token_count=10, word_count=10,
                    char_count=50, text="text"
                ),
                rank=1, score=1.0
            )
            return RetrievalResult(
                setting_name="dummy", retrieval_method="hybrid_rrf", chunk_size_tokens=500,
                candidates_fetched=4, pieces_passed=4, rerank_enabled=False, reranker_backend_used="None",
                retrieval_ms=0, rerank_ms=0, candidates_before_rerank=[dummy], final_pieces=[dummy],
                max_lexical_overlap=1.0, top_score=1.0
            )
            
    class MockClient:
        source = "stand-in"
        def __init__(self):
            self.calls = []
            
        async def execute_single_call(self, stage, *args, **kwargs):
            self.calls.append(stage)
            if stage == "generate":
                # Call 1: Truncated, invalid JSON
                return ProviderCallResult(
                    success=True, stage="generate", model_id="dummy", http_status=200,
                    content='{"abstained": false, "answer_text": "truncated',
                    provider_first_token_ms=10.0, work_ms=50.0, tokens_in=10, tokens_out=700, accounting_mode="exact_reported"
                )
            elif stage == "regenerate":
                # Call 2: Recovery into valid, structured, non-abstained answer
                valid_resp = {
                    "abstained": False,
                    "abstention_reason": "",
                    "answer_text": "Recovered valid answer",
                    "claims": [{
                        "claim_id": "c1",
                        "claim_text": "This is a recovered claim.",
                        "cited_chunk_ids": ["c001"],
                        "citations": [{
                            "chunk_id": "c001",
                            "paper_id": "1706.03762",
                            "arxiv_version": "v7",
                            "page_number": 1,
                            "section_title": "A",
                            "locator_detail": ""
                        }]
                    }]
                }
                return ProviderCallResult(
                    success=True, stage="regenerate", model_id="dummy", http_status=200,
                    content=json.dumps(valid_resp),
                    provider_first_token_ms=10.0, work_ms=50.0, tokens_in=10, tokens_out=250, accounting_mode="exact_reported"
                )
            elif stage == "reverify":
                # Call 3: Semantic Verification on the regenerated output
                valid_ver = {
                    "verified": True, 
                    "supported_claim_ids": ["c1"], 
                    "unsupported_claim_ids": [], 
                    "refusal_valid": None, 
                    "overlooked_chunk_ids": [], 
                    "feedback": ""
                }
                return ProviderCallResult(
                    success=True, stage="reverify", model_id="dummy", http_status=200,
                    content=json.dumps(valid_ver),
                    provider_first_token_ms=10.0, work_ms=50.0, tokens_in=10, tokens_out=50, accounting_mode="exact_reported"
                )
            raise AssertionError(f"Unexpected stage: {stage}")
            
    config = {
        "provider": {"default_model": "dummy", "max_transport_retries": 0},
        "retrieval": {
            "default_setting": "dummy",
            "settings": {"dummy": {"chunk_size_tokens": 500, "candidates_fetched": 4, "pieces_passed": 4, "rerank_enabled": False}}
        },
        "pipeline": {"max_correction_attempts": 1}
    }
    
    client = MockClient()
    pipe = RAGPipeline(config, MockRegistry(), client, None)
    res = await pipe.run_request("Test truncation recovery", enable_cache=False)
    
    assert client.calls == ["generate", "regenerate", "reverify"]
    assert res.status == "ok"
    assert res.verified is True
    assert res.abstained is False
    assert res.generation_output is not None
    assert res.generation_output.answer_text == "Recovered valid answer"
    
    # Verify deterministic citation validation passes on the recovered output
    det_check = validate_citations_deterministically(res.generation_output, res.retrieved_pieces)
    assert det_check.valid is True
    assert det_check.errors == []
    assert det_check.cited_chunk_ids == ["c001"]
    
    assert "This is a recovered claim." in res.user_response_text
    assert "c001" in res.user_response_text
    assert "1706.03762v7" in res.user_response_text
    assert res.verification_output is not None
    assert res.verification_output.verified is True
    assert res.telemetry_row["correction_attempts"] == 1
    assert res.telemetry_row["total_provider_calls"] == 3
    assert res.telemetry_row["tokens_in"] == 30
    assert res.telemetry_row["tokens_out"] == 1000  # 700 + 250 + 50
