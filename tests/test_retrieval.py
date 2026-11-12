import pytest
from pathlib import Path
from src.chunking import ChunkRecord
from src.retrieval import BM25Retriever, DenseRetriever, HybridRRFRetriever, CrossEncoderReranker, RetrievedPiece

@pytest.fixture
def sample_chunks():
    return [
        ChunkRecord(
            chunk_id="c001", paper_id="1706.03762", arxiv_version="v7",
            title="Attention", filename="1706.pdf", sha256="hash",
            chunk_size_target=500, chunk_index_in_paper=1, global_chunk_index=1,
            primary_page=1, page_start=1, page_end=1, page_numbers=[1],
            section_title="Intro", section_path="Intro", all_sections=["Intro"],
            paragraph_refs=[], table_caption_ids=[], table_references=[],
            figure_caption_ids=[], figure_references=[], equation_numbers=["1"],
            has_math=True, is_oversized_split=False, token_count=10, word_count=10, char_count=50,
            text="The Transformer relies on multi-head attention."
        ),
        ChunkRecord(
            chunk_id="c002", paper_id="2410.00037", arxiv_version="v1",
            title="Moshi", filename="2410.pdf", sha256="hash",
            chunk_size_target=500, chunk_index_in_paper=1, global_chunk_index=2,
            primary_page=1, page_start=1, page_end=1, page_numbers=[1],
            section_title="Intro", section_path="Intro", all_sections=["Intro"],
            paragraph_refs=[], table_caption_ids=[], table_references=[],
            figure_caption_ids=[], figure_references=[], equation_numbers=[],
            has_math=False, is_oversized_split=False, token_count=10, word_count=10, char_count=50,
            text="Moshi uses a Helium language model."
        )
    ]

def test_bm25_retrieval(sample_chunks):
    ret = BM25Retriever(sample_chunks)
    res = ret.search("multi-head attention", top_k=1)
    assert len(res) == 1
    assert res[0].chunk.chunk_id == "c001"

def test_dense_lightweight_retrieval(sample_chunks):
    ret = DenseRetriever(sample_chunks, backend="lightweight")
    res = ret.search("helium model", top_k=1)
    assert len(res) == 1
    assert res[0].chunk.chunk_id == "c002"

def test_hybrid_rrf_combines_ranks(sample_chunks):
    ret = HybridRRFRetriever(sample_chunks, rrf_k=60, dense_backend="lightweight")
    res = ret.retrieve_candidates("attention", candidates_k=2, method="hybrid_rrf")
    assert len(res) == 2
    assert res[0].rrf_score is not None

def test_reranker_backend_reporting_and_fallback(sample_chunks):
    candidates = [
        RetrievedPiece(chunk=sample_chunks[0], rank=1, score=1.0, rrf_score=1.0),
        RetrievedPiece(chunk=sample_chunks[1], rank=2, score=0.8, rrf_score=0.8)
    ]
    
    # 1. Fallback / Lightweight
    reranker_light = CrossEncoderReranker(backend="lightweight")
    _, _, backend_light = reranker_light.rerank("helium", candidates, top_k=2)
    assert backend_light == "lightweight"
    
    # 2. Missing model fallback
    reranker_st = CrossEncoderReranker(backend="sentence_transformers", model_name="invalid-fake-model-xyz123")
    _, _, backend_st = reranker_st.rerank("helium", candidates, top_k=2)
    assert backend_st == "lightweight"  # Caught exception and fell back
