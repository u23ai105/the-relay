import json
from pathlib import Path
import pytest
import pymupdf

from src.extraction import (
    EXPECTED_CORPUS,
    _build_page_lines,
    compute_file_sha256,
    extract_corpus,
    load_and_verify_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
MANIFEST_PATH = PAPERS_DIR / "provenance_manifest.json"


def _create_synthetic_pdf(
    pdf_path: Path,
    paper_id: str,
    version: str,
    empty_page: bool = False,
    include_stamp: bool = True,
) -> str:
    doc = pymupdf.open()
    page1 = doc.new_page()
    # Title (large font before Abstract -> must be rejected as section heading)
    page1.insert_text((72, 60), "Synthetic Research Paper Title", fontsize=16, fontname="helv")
    if include_stamp:
        page1.insert_text((72, 85), f"arXiv:{paper_id}{version} [cs.LG] 1 Jan 2026", fontsize=9)
    page1.insert_text((72, 110), "Abstract", fontsize=12, fontname="hebo")
    page1.insert_text(
        (72, 128),
        "This paper investigates sequence modeling and attention complexity in transformers.",
        fontsize=10,
    )
    # Split number and title across two horizontal insertions on the same baseline (y=155)
    page1.insert_text((72, 155), "2.1", fontsize=12, fontname="hebo")
    page1.insert_text((102, 155), "Architectural Overview", fontsize=12, fontname="hebo")
    page1.insert_text(
        (72, 173),
        "As shown in Table 1 and Figure 1, our method optimizes Equation (1) below.",
        fontsize=10,
    )
    # Bounded 4-level numbered headings + generic leaf sub-header + trailing punctuation on numbered heading
    page1.insert_text((72, 195), "2.1.1", fontsize=11, fontname="hebo")
    page1.insert_text((108, 195), "Routing Module:", fontsize=11, fontname="hebo")
    page1.insert_text((72, 212), "2.1.1.1", fontsize=10.5, fontname="hebo")
    page1.insert_text((118, 212), "Boundary Probability.", fontsize=10.5, fontname="hebo")
    page1.insert_text((72, 229), "Results", fontsize=10.5, fontname="hebo")
    page1.insert_text((72, 245), "Attention(Q, K, V) = softmax(QK^T)V   (1)", fontsize=10)

    # Unnumbered run-in paragraph lead-in ending with period (must be rejected)
    page1.insert_text((72, 265), "Deduplication.", fontsize=10, fontname="hebo")

    # Inline bold run-in with raised superscript footnote (y=281 vs baseline y=285) and body prose
    page1.insert_text((72, 285), "3. Token Generation:", fontsize=10, fontname="hebo")
    page1.insert_text((178, 281), "10", fontsize=7, fontname="helv")
    page1.insert_text(
        (192, 285),
        "To perform a model step, we do the following for a 1-stage hierarchy:",
        fontsize=10,
        fontname="helv",
    )

    # Ordinary small-font digit on the SAME baseline y=302 (NOT raised -> must NOT become [^2])
    page1.insert_text((72, 302), "Baseline version", fontsize=10, fontname="helv")
    page1.insert_text((152, 302), "2", fontsize=7, fontname="helv")
    page1.insert_text((162, 302), "is evaluated here.", fontsize=10, fontname="helv")

    # Mathematical exponent x^2 (raised digit '2' at y=314 vs baseline y=318 after variable 'x' -> must NOT become [^2])
    page1.insert_text((72, 318), "Complexity scales as x", fontsize=10, fontname="helv")
    page1.insert_text((177, 314), "2", fontsize=7, fontname="helv")
    page1.insert_text((186, 318), "in sequence length.", fontsize=10, fontname="helv")

    # Captions: Table 1 & Figure 1 are also cited in prose above; Table 2 & Figure 2 appear ONLY in captions
    page1.insert_text((72, 338), "Table 1: Model latency comparison across settings.", fontsize=10)
    page1.insert_text((72, 355), "Table 2: Ablation study across chunking thresholds.", fontsize=10)
    page1.insert_text((72, 372), "Figure 1: System architecture diagram.", fontsize=10)
    page1.insert_text((72, 389), "Figure 2: Token boundary routing visualization.", fontsize=10)

    if empty_page:
        doc.new_page()

    doc.save(pdf_path)
    doc.close()
    return compute_file_sha256(pdf_path)


def test_hash_mismatch_rejected(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    manifest_entries = []
    for pid, ver in EXPECTED_CORPUS:
        p = pdf_dir / f"{pid}{ver}.pdf"
        _ = _create_synthetic_pdf(p, pid, ver)
        manifest_entries.append(
            {
                "paper_id": pid,
                "version": ver,
                "filename": p.name,
                "sha256": "0" * 64,
            }
        )
    manifest_file = pdf_dir / "provenance_manifest.json"
    manifest_file.write_text(json.dumps({"papers": manifest_entries}), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_and_verify_manifest(pdf_dir, manifest_file)


def test_unexpected_paper_version_rejected(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    manifest_entries = []
    for pid, ver in EXPECTED_CORPUS:
        wrong_ver = "v6" if pid == "1706.03762" else ver
        p = pdf_dir / f"{pid}{wrong_ver}.pdf"
        sha = _create_synthetic_pdf(p, pid, wrong_ver)
        manifest_entries.append(
            {"paper_id": pid, "version": wrong_ver, "filename": p.name, "sha256": sha}
        )
    manifest_file = pdf_dir / "provenance_manifest.json"
    manifest_file.write_text(json.dumps({"papers": manifest_entries}), encoding="utf-8")

    with pytest.raises(ValueError, match="Version mismatch"):
        load_and_verify_manifest(pdf_dir, manifest_file)


def test_missing_in_text_arxiv_stamp_rejected(tmp_path):
    pdf_dir = tmp_path / "papers"
    out_dir = tmp_path / "indices"
    pdf_dir.mkdir()
    manifest_entries = []
    for idx, (pid, ver) in enumerate(EXPECTED_CORPUS):
        p = pdf_dir / f"{pid}{ver}.pdf"
        sha = _create_synthetic_pdf(p, pid, ver, include_stamp=(idx != 0))
        manifest_entries.append(
            {"paper_id": pid, "version": ver, "filename": p.name, "sha256": sha}
        )
    manifest_file = pdf_dir / "provenance_manifest.json"
    manifest_file.write_text(json.dumps({"papers": manifest_entries}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected arXiv version marker"):
        extract_corpus(pdf_dir, manifest_file, out_dir)


def test_empty_page_triggers_extraction_error(tmp_path):
    pdf_dir = tmp_path / "papers"
    out_dir = tmp_path / "indices"
    pdf_dir.mkdir()
    manifest_entries = []
    for idx, (pid, ver) in enumerate(EXPECTED_CORPUS):
        p = pdf_dir / f"{pid}{ver}.pdf"
        sha = _create_synthetic_pdf(p, pid, ver, empty_page=(idx == 0))
        manifest_entries.append(
            {"paper_id": pid, "version": ver, "filename": p.name, "sha256": sha}
        )
    manifest_file = pdf_dir / "provenance_manifest.json"
    manifest_file.write_text(json.dumps({"papers": manifest_entries}), encoding="utf-8")

    with pytest.raises(ValueError, match="empty pages"):
        extract_corpus(pdf_dir, manifest_file, out_dir)


def test_superscript_geometry_nested_hierarchy_and_table_metadata(tmp_path):
    pdf_dir = tmp_path / "papers"
    out_dir = tmp_path / "indices"
    pdf_dir.mkdir()
    manifest_entries = []
    for pid, ver in EXPECTED_CORPUS:
        p = pdf_dir / f"{pid}{ver}.pdf"
        sha = _create_synthetic_pdf(p, pid, ver, empty_page=False)
        manifest_entries.append(
            {"paper_id": pid, "version": ver, "filename": p.name, "sha256": sha}
        )
    manifest_file = pdf_dir / "provenance_manifest.json"
    manifest_file.write_text(json.dumps({"papers": manifest_entries}), encoding="utf-8")

    summary = extract_corpus(pdf_dir, manifest_file, out_dir)
    first_paper = summary["papers"][0]
    sections = first_paper["unique_sections_detected"]

    # 1. Split headings & numbered headings with trailing colon/period are confirmed (with trailing punct stripped)
    assert "2.1 Architectural Overview" in sections
    assert "2.1.1 Routing Module" in sections
    assert "2.1.1.1 Boundary Probability" in sections

    # 2. Unnumbered run-in "Deduplication." and inline "3. Token Generation:" are rejected
    assert "Deduplication." not in sections
    assert "Deduplication" not in sections
    assert not any("Token Generation" in s for s in sections)

    pages_file = out_dir / "extracted_pages.jsonl"
    first_page = json.loads(pages_file.read_text(encoding="utf-8").splitlines()[0])

    # 3. Verify bounded 4-level numbered hierarchy + generic sub-header "Results" never overwrites 2.1.1.1
    paths = [sp["section_path"] for sp in first_page["structured_paragraphs"]]
    assert any(
        "2.1 Architectural Overview > 2.1.1 Routing Module > 2.1.1.1 Boundary Probability > Results"
        in p
        for p in paths
    )
    assert first_page["active_section_end"] == "2.1.1.1 Boundary Probability"

    # 4. Verify raised superscript footnote [^10] is preserved in extracted paragraph text
    assert "[^10]" in first_page["full_text"]
    assert any(
        "3. Token Generation: [^10] To perform a model step" in sp["text"]
        for sp in first_page["structured_paragraphs"]
    )

    # 5. Verify ordinary small-font digit on same baseline AND raised mathematical exponent x^2 are NOT converted into [^2]
    assert "[^2]" not in first_page["full_text"]
    assert "Baseline version 2 is evaluated here." in first_page["full_text"]
    assert "Complexity scales as x2 in sequence length." in first_page["full_text"]

    # 6. Verify prose mention of "Table 1" / "Figure 1" goes into references, while captions are NOT counted as references
    prose_mention_paras = [
        sp
        for sp in first_page["structured_paragraphs"]
        if "As shown in Table 1" in sp["text"]
    ]
    assert len(prose_mention_paras) == 1
    assert prose_mention_paras[0]["table_references"] == ["1"]
    assert prose_mention_paras[0]["figure_references"] == ["1"]
    assert prose_mention_paras[0]["table_caption_ids"] == []
    assert prose_mention_paras[0]["figure_caption_ids"] == []
    assert prose_mention_paras[0]["is_table_caption"] is False

    caption_paras = [
        sp
        for sp in first_page["structured_paragraphs"]
        if "Table 1: Model latency comparison" in sp["text"]
    ]
    assert len(caption_paras) == 1
    assert caption_paras[0]["is_table_caption"] is True
    assert caption_paras[0]["table_caption_ids"] == ["1"]
    assert caption_paras[0]["table_references"] == []

    table2_caption_paras = [
        sp
        for sp in first_page["structured_paragraphs"]
        if "Table 2: Ablation study" in sp["text"]
    ]
    assert len(table2_caption_paras) == 1
    assert table2_caption_paras[0]["is_table_caption"] is True
    assert table2_caption_paras[0]["table_caption_ids"] == ["2"]
    assert table2_caption_paras[0]["table_references"] == []

    fig2_caption_paras = [
        sp
        for sp in first_page["structured_paragraphs"]
        if "Figure 2: Token boundary" in sp["text"]
    ]
    assert len(fig2_caption_paras) == 1
    assert fig2_caption_paras[0]["figure_caption_ids"] == ["2"]
    assert fig2_caption_paras[0]["figure_references"] == []

    assert first_page["table_caption_ids"] == ["1", "2"]
    assert first_page["table_references"] == ["1"]
    assert first_page["figure_caption_ids"] == ["1", "2"]
    assert first_page["figure_references"] == ["1"]


def test_two_column_reading_order_and_gutter_isolation(tmp_path):
    pdf_dir = tmp_path / "papers"
    out_dir = tmp_path / "indices"
    pdf_dir.mkdir()
    manifest_entries = []

    for idx, (pid, ver) in enumerate(EXPECTED_CORPUS):
        p = pdf_dir / f"{pid}{ver}.pdf"
        if idx == 0:
            doc = pymupdf.open()
            page = doc.new_page(width=600, height=800)
            page.insert_text((72, 50), f"arXiv:{pid}{ver} [cs.CL] 1 Jan 2026", fontsize=9)
            page.insert_text(
                (120, 80),
                "Full Width Two-Column Layout Paper Title Across Center",
                fontsize=14,
                fontname="hebo",
            )
            page.insert_text((72, 110), "Abstract", fontsize=12, fontname="hebo")
            page.insert_text((72, 160), "1 Left Section", fontsize=12, fontname="hebo")
            page.insert_text(
                (72, 185),
                "Left column first paragraph line one text.",
                fontsize=10,
            )
            page.insert_text(
                (72, 210),
                "Left column second paragraph line two text.",
                fontsize=10,
            )

            page.insert_text((315, 160), "2 Right Section", fontsize=12, fontname="hebo")
            page.insert_text(
                (315, 185),
                "Right column first paragraph line one text.",
                fontsize=10,
            )
            page.insert_text(
                (315, 210),
                "Right column second paragraph line two text.",
                fontsize=10,
            )
            doc.save(p)
            doc.close()
            sha = compute_file_sha256(p)
        else:
            sha = _create_synthetic_pdf(p, pid, ver)

        manifest_entries.append(
            {"paper_id": pid, "version": ver, "filename": p.name, "sha256": sha}
        )

    manifest_file = pdf_dir / "provenance_manifest.json"
    manifest_file.write_text(json.dumps({"papers": manifest_entries}), encoding="utf-8")

    extract_corpus(pdf_dir, manifest_file, out_dir)
    pages_file = out_dir / "extracted_pages.jsonl"
    first_page = json.loads(pages_file.read_text(encoding="utf-8").splitlines()[0])
    full_text = first_page["full_text"]

    assert full_text.index("Left column second paragraph") < full_text.index(
        "2 Right Section"
    )
    assert full_text.index("2 Right Section") < full_text.index(
        "Right column first paragraph"
    )


def test_math_font_superscript_safeguard_direct():
    """
    Tests the current-span math-font safeguard (`not sp_math`) AND deferred whitespace
    handling when explicit whitespace-only spans appear around a raised exponent and prose:
    - The preceding word 'prose' uses a non-math font ('CMR10').
    - An explicit whitespace-only span {'text': ' '} precedes the raised digit '2' ('CMMI10').
    - Another explicit whitespace-only span {'text': ' '} precedes 'in the bound.'.
    Verifies that pending whitespace before the raised exponent is discarded ('prose2'),
    while pending whitespace before ordinary prose is preserved ('prose2 in the bound.').
    """
    blocks = [
        {
            "type": 0,
            "lines": [
                {
                    "bbox": [72.0, 186.0, 260.0, 200.0],
                    "spans": [
                        {
                            "text": "Given prose",
                            "size": 10.0,
                            "flags": 0,
                            "font": "CMR10",
                            "bbox": [72.0, 190.0, 128.0, 200.0],
                            "origin": [72.0, 198.0],
                        },
                        {
                            "text": " ",
                            "size": 10.0,
                            "flags": 0,
                            "font": "CMR10",
                            "bbox": [128.0, 190.0, 130.5, 200.0],
                            "origin": [128.0, 198.0],
                        },
                        {
                            "text": "2",
                            "size": 7.0,
                            "flags": 0,
                            "font": "CMMI10",
                            "bbox": [130.5, 186.0, 135.0, 193.0],
                            "origin": [130.5, 194.0],
                        },
                        {
                            "text": " ",
                            "size": 10.0,
                            "flags": 0,
                            "font": "CMR10",
                            "bbox": [135.0, 190.0, 138.0, 200.0],
                            "origin": [135.0, 198.0],
                        },
                        {
                            "text": "in the bound.",
                            "size": 10.0,
                            "flags": 0,
                            "font": "CMR10",
                            "bbox": [138.0, 190.0, 210.0, 200.0],
                            "origin": [138.0, 198.0],
                        },
                    ],
                }
            ],
        }
    ]

    lines, stamp, page_has_math, rejected = _build_page_lines(
        blocks=blocks,
        body_font_size=10.0,
        page_number=2,
        page_width=612.0,
        page_height=792.0,
    )

    assert len(lines) == 1
    assert page_has_math is True
    assert lines[0].has_math_font is True
    assert "[^2]" not in lines[0].text
    assert lines[0].text == "Given prose2 in the bound."


def test_preceding_math_span_superscript_safeguard_direct():
    """
    Independently tests the preceding-span math-font safeguard (`not prev_nonempty_span_math`):
    - Base span is a multi-letter mathematical identifier ('theta') with font 'CMMI10'
      (so `follows_prose_word` is True and `prev_nonempty_span_math` is True).
    - Raised numeric span '2' uses a non-math font ('helv', so `not sp_math` is True).
    Verifies that `not prev_nonempty_span_math` prevents conversion into `[^2]` and
    that deferred whitespace before the raised exponent is suppressed.
    """
    blocks = [
        {
            "type": 0,
            "lines": [
                {
                    "bbox": [72.0, 186.0, 275.0, 200.0],
                    "spans": [
                        {
                            "text": "Given ",
                            "size": 10.0,
                            "flags": 0,
                            "font": "helv",
                            "bbox": [72.0, 190.0, 102.0, 200.0],
                            "origin": [72.0, 198.0],
                        },
                        {
                            "text": "theta",
                            "size": 10.0,
                            "flags": 0,
                            "font": "CMMI10",
                            "bbox": [102.0, 190.0, 130.0, 200.0],
                            "origin": [102.0, 198.0],
                        },
                        {
                            "text": " ",
                            "size": 10.0,
                            "flags": 0,
                            "font": "helv",
                            "bbox": [130.0, 190.0, 132.0, 200.0],
                            "origin": [130.0, 198.0],
                        },
                        {
                            "text": "2",
                            "size": 7.0,
                            "flags": 0,
                            "font": "helv",
                            "bbox": [132.0, 186.0, 136.5, 193.0],
                            "origin": [132.0, 194.0],
                        },
                        {
                            "text": " in the bound.",
                            "size": 10.0,
                            "flags": 0,
                            "font": "helv",
                            "bbox": [138.0, 190.0, 215.0, 200.0],
                            "origin": [138.0, 198.0],
                        },
                    ],
                }
            ],
        }
    ]

    lines, stamp, page_has_math, rejected = _build_page_lines(
        blocks=blocks,
        body_font_size=10.0,
        page_number=2,
        page_width=612.0,
        page_height=792.0,
    )

    assert len(lines) == 1
    assert page_has_math is True
    assert lines[0].has_math_font is True
    assert "[^2]" not in lines[0].text
    assert lines[0].text == "Given theta2 in the bound."


@pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason="Real provenance_manifest.json not present in data/papers/",
)
def test_real_corpus_regression_checks(tmp_path):
    isolated_out_dir = tmp_path / "indices"
    summary = extract_corpus(PAPERS_DIR, MANIFEST_PATH, isolated_out_dir)
    papers_by_id = {p["paper_id"]: p for p in summary["papers"]}

    # 1. Attention Is All You Need (1706.03762v7) regressions
    s_attn = papers_by_id["1706.03762"]["unique_sections_detected"]
    assert "1 Introduction" in s_attn
    assert "2 Background" in s_attn
    assert "3 Model Architecture" in s_attn
    assert "3.1 Encoder and Decoder Stacks" in s_attn
    assert "3.2 Attention" in s_attn
    assert "3.2.1 Scaled Dot-Product Attention" in s_attn
    assert "3.2.2 Multi-Head Attention" in s_attn
    assert "4 Why Self-Attention" in s_attn
    assert "5 Training" in s_attn
    assert "6 Results" in s_attn
    assert "7 Conclusion" in s_attn
    assert "Attention Visualizations" in s_attn

    # 2. Moshi (2410.00037v1) regressions
    s_moshi = papers_by_id["2410.00037"]["unique_sections_detected"]
    assert "1 Introduction" in s_moshi
    assert "3.2 The Helium Text Language Model" in s_moshi
    assert "5.1 Text Language Modeling" in s_moshi
    assert "5.2 Audio Tokenization" in s_moshi
    assert not any("Deduplication" in s for s in s_moshi)
    assert not any("Audio Language Modeling." in s for s in s_moshi)

    # 3. Dynamic Chunking (2507.07955v2) regressions
    s_dc = papers_by_id["2507.07955"]["unique_sections_detected"]
    assert "1 Introduction" in s_dc
    assert "2 H-Net Architecture" in s_dc
    assert "2.1 Architectural Overview" in s_dc
    assert "2.2 Dynamic Chunking (DC)" in s_dc
    assert not any("Token Generation" in s for s in s_dc)
    assert not any(s.endswith(".") for s in s_dc)

    with open(isolated_out_dir / "extracted_pages.jsonl", "r", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]

    # Verify Moshi p.21 preserves active_section_end == "5.1 Text Language Modeling"
    moshi_p21 = [
        p for p in pages if p["paper_id"] == "2410.00037" and p["page_number"] == 21
    ][0]
    assert moshi_p21["active_section_end"] == "5.1 Text Language Modeling"

    # Verify Dynamic Chunking p.10 preserves [^10] in paragraph text while rejecting "3. Token Generation:" as heading
    dc_p10 = [
        p for p in pages if p["paper_id"] == "2507.07955" and p["page_number"] == 10
    ][0]
    assert "[^10]" in dc_p10["full_text"]
