import pytest
from src.chunking import (
    _split_oversized_text_lossless,
    count_tokens,
    _expand_pages_to_units,
)

def test_token_counting():
    assert count_tokens("Hello world!") == 3
    assert count_tokens("") == 0
    # "Model", "uses", "d_model", "=", "512", "." -> 6 tokens
    assert count_tokens("Model uses d_model=512.") == 6

def test_oversized_text_splitting():
    text = "This is sentence one. This is sentence two. " * 20
    assert count_tokens(text) > 100
    
    pieces = _split_oversized_text_lossless(text, max_tokens=50)
    assert len(pieces) > 2
    for p in pieces:
        assert count_tokens(p) <= 50
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")

def test_expand_pages_to_units_preserves_metadata():
    pages = [{
        "paper_id": "1706.03762",
        "arxiv_version": "v7",
        "title": "Attention",
        "filename": "1706.pdf",
        "sha256": "fakehash",
        "page_number": 1,
        "structured_paragraphs": [
            {
                "paragraph_index": 0,
                "text": "Regular paragraph text here.",
                "active_section": "1 Intro",
                "section_path": "1 Intro",
                "table_caption_ids": ["1"],
                "has_math": True
            }
        ]
    }]
    
    units = _expand_pages_to_units(pages, chunk_size_tokens=250)
    assert len(units) == 1
    u = units[0]
    assert u.paper_id == "1706.03762"
    assert u.table_caption_ids == ["1"]
    assert u.has_math is True
