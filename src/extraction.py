"""
Phase 4 (Part 1): Geometry-aware, column-aware PDF text extraction and
cryptographic/version provenance verification using `import pymupdf`.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

EXPECTED_CORPUS: List[Tuple[str, str]] = [
    ("1706.03762", "v7"),
    ("2410.00037", "v1"),
    ("2507.07955", "v2"),
]

ARXIV_STAMP_RE = re.compile(
    r"arXiv\s*:\s*(\d{4}\.\d{4,5})(v\d+)\b", re.IGNORECASE
)

STANDALONE_SEC_NUM_RE = re.compile(
    r"^(?:Section\s+)?((?:\d+(?:\.\d+){0,3})|(?:[A-Z](?:\.\d+){1,3}))\.?$"
)

NUMBERED_SECTION_RE = re.compile(
    r"^(?:Section\s+)?((?:\d+(?:\.\d+){0,3})|(?:[A-Z](?:\.\d+){0,3}))(\.)?\s+([A-Z][A-Za-z0-9\s\-\–\—,/()']{1,85}?)([.:])?$"
)

CANONICAL_TOP_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "model architecture",
    "method",
    "methods",
    "methodology",
    "experiments",
    "conclusion",
    "conclusions",
    "acknowledgements",
    "acknowledgments",
    "references",
    "appendix",
    "broader impact",
}

CANONICAL_SUB_HEADERS = {
    "results",
    "discussion",
    "experimental setup",
}

CAPTION_START_RE = re.compile(
    r"^(Figure|Fig\.|Table|Algorithm)\s+(\d+|[A-Z]\.\d+)(?:\s*[:\.\-\–]|\s+)",
    re.IGNORECASE,
)

TABLE_REF_RE = re.compile(r"\bTable\s+(\d+|[A-Z]\.\d+)\b", re.IGNORECASE)
FIGURE_REF_RE = re.compile(r"\b(?:Figure|Fig\.)\s+(\d+|[A-Z]\.\d+)\b", re.IGNORECASE)
EQUATION_REF_RE = re.compile(
    r"\b(?:Equation|Eq\.|Eqs\.)\s*\(?(\d+|[A-Z]\.\d+)\)?", re.IGNORECASE
)
TRAILING_EQ_NUM_RE = re.compile(r"(?:^|\s)\((\d{1,2}|[A-Z]\.\d{1,2})\)\s*$")

MATH_FONT_PREFIXES = ("cmmi", "cmsy", "cmex", "msbm", "msam", "math", "stixmath")

MATH_FUNC_WORDS = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "log", "ln", "exp",
    "lim", "max", "min", "sup", "inf", "det",
    "dim", "ker", "arg", "argmin", "argmax", "softmax",
    "mod", "gcd", "lcm", "tr", "diag", "rank",
}

PROSE_FOOTNOTE_ANCHOR_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_\\^])([A-Za-z]{2,})([.,:;!?'\"\u2019\u201d]?)$"
)


@dataclass
class HeadingItem:
    text: str
    section_number: str
    section_title: str
    hierarchy_level: int  # Bounded 4-level numbered hierarchy (1..4); 5 = generic leaf sub-header
    font_size: float
    is_bold: bool
    confidence: str  # "confirmed" or "candidate_rejected"
    reason: str


@dataclass
class _RawSpanGroup:
    x0: float
    y0: float
    x1: float
    y1: float
    spans: List[Dict[str, Any]]
    block_idx: int
    raw_text: str

    @property
    def y_mid(self) -> float:
        return 0.5 * (self.y0 + self.y1)


@dataclass
class _ExtractedLine:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    max_font_size: float
    bold_chars: int
    alnum_chars: int
    has_math_font: bool
    block_idx: int
    col_id: int = 0  # 0 = full-width/single-col, 1 = left col, 2 = right col

    @property
    def bold_ratio(self) -> float:
        return self.bold_chars / max(1, self.alnum_chars)

    @property
    def is_bold(self) -> bool:
        return self.bold_ratio >= 0.80 and self.alnum_chars > 0

    @property
    def y_mid(self) -> float:
        return 0.5 * (self.y0 + self.y1)


@dataclass
class PageExtractionRecord:
    paper_id: str
    arxiv_version: str
    title: str
    filename: str
    sha256: str
    page_number: int
    total_pages: int
    char_count: int
    word_count: int
    paragraph_count: int
    active_section_start: str
    active_section_end: str
    active_section_path_start: str
    active_section_path_end: str
    confirmed_headings: List[Dict[str, Any]]
    heading_candidates: List[Dict[str, Any]]
    equation_numbers: List[str]
    equation_references: List[str]
    has_math_spans: bool
    table_captions: List[str]
    table_caption_ids: List[str]
    table_references: List[str]
    table_Ids: List[str]
    pymupdf_tables_detected: int
    extracted_tables: List[Dict[str, Any]]
    figure_captions: List[str]
    figure_caption_ids: List[str]
    figure_references: List[str]
    figure_ids: List[str]
    image_count: int
    drawing_count: int
    arxiv_stamp_detected: str
    extraction_warnings: List[str]
    structured_paragraphs: List[Dict[str, Any]]
    paragraphs: List[str]
    full_text: str


def compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_version(ver: str) -> str:
    v = str(ver).strip().lower()
    return v if v.startswith("v") else f"v{v}"


def load_and_verify_manifest(
    papers_dir: Path, manifest_path: Path
) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Provenance manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        raw_manifest = json.load(f)

    entries_raw: List[Dict[str, Any]] = []
    if isinstance(raw_manifest, list):
        entries_raw = raw_manifest
    elif isinstance(raw_manifest, dict):
        if "papers" in raw_manifest and isinstance(raw_manifest["papers"], list):
            entries_raw = raw_manifest["papers"]
        elif "papers" in raw_manifest and isinstance(raw_manifest["papers"], dict):
            for k, v in raw_manifest["papers"].items():
                item = dict(v)
                item.setdefault("paper_id", k)
                entries_raw.append(item)
        else:
            for k, v in raw_manifest.items():
                if isinstance(v, dict):
                    item = dict(v)
                    item.setdefault("paper_id", k)
                    entries_raw.append(item)

    normalized_by_id: Dict[str, Dict[str, str]] = {}
    for item in entries_raw:
        raw_id = str(
            item.get("paper_id") or item.get("id") or item.get("arxiv_id") or ""
        ).strip()
        raw_ver = str(
            item.get("arxiv_version") or item.get("version") or ""
        ).strip()

        m = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", raw_id)
        if m:
            pid = m.group(1)
            ver = _normalize_version(raw_ver if raw_ver else (m.group(2) or ""))
        else:
            pid = raw_id
            ver = _normalize_version(raw_ver) if raw_ver else ""

        sha256_expected = str(
            item.get("sha256")
            or item.get("sha256_hash")
            or item.get("hash")
            or ""
        ).strip().lower()

        filename = str(
            item.get("filename") or item.get("file") or item.get("path") or ""
        ).strip()
        if filename:
            pdf_path = papers_dir / Path(filename).name
        else:
            pdf_path = papers_dir / f"{pid}{ver}.pdf"
            if not pdf_path.exists():
                candidates = sorted(papers_dir.glob(f"*{pid}*.pdf"))
                if candidates:
                    pdf_path = candidates[0]

        title = str(item.get("title") or "").strip()
        normalized_by_id[pid] = {
            "paper_id": pid,
            "arxiv_version": ver,
            "title": title,
            "filename": pdf_path.name,
            "pdf_path": str(pdf_path),
            "sha256": sha256_expected,
        }

    verified_list: List[Dict[str, str]] = []
    for expected_id, expected_ver in EXPECTED_CORPUS:
        if expected_id not in normalized_by_id:
            raise ValueError(
                f"Required paper {expected_id} missing from {manifest_path}"
            )
        entry = normalized_by_id[expected_id]
        if entry["arxiv_version"] != expected_ver:
            raise ValueError(
                f"Version mismatch for {expected_id}: expected {expected_ver}, got {entry['arxiv_version']}"
            )
        pdf_path = Path(entry["pdf_path"])
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"PDF file missing or empty for {expected_id}: {pdf_path}"
            )
        if not entry["sha256"]:
            raise ValueError(f"Missing SHA-256 hash in manifest for {expected_id}")

        actual_sha256 = compute_file_sha256(pdf_path)
        if actual_sha256 != entry["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {pdf_path.name}: expected {entry['sha256']}, computed {actual_sha256}"
            )
        verified_list.append(entry)

    if len(normalized_by_id) != len(EXPECTED_CORPUS):
        unexpected = set(normalized_by_id.keys()) - {p[0] for p in EXPECTED_CORPUS}
        if unexpected:
            raise ValueError(f"Unexpected papers in manifest: {sorted(unexpected)}")

    return verified_list


def _estimate_document_body_font_size(doc: pymupdf.Document) -> float:
    sizes: List[float] = []
    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = float(span.get("size", 0.0))
                    if len(text) >= 15 and 7.0 <= size <= 14.0:
                        sizes.extend([round(size, 1)] * min(len(text), 60))
    if not sizes:
        return 10.0
    return float(median(sizes))


def _is_bold_span(span: Dict[str, Any]) -> bool:
    flags = int(span.get("flags", 0))
    font_name = str(span.get("font", "")).lower()
    return (
        bool(flags & 16)
        or ("bold" in font_name)
        or ("medi" in font_name)
        or ("romand" in font_name)
        or ("demi" in font_name)
    )


def _is_math_span(span: Dict[str, Any]) -> bool:
    font_name = str(span.get("font", "")).lower()
    return any(prefix in font_name for prefix in MATH_FONT_PREFIXES)


def _span_origin_y(span: Dict[str, Any]) -> float:
    orig = span.get("origin")
    if isinstance(orig, (list, tuple)) and len(orig) >= 2:
        return float(orig[1])
    bbox = span.get("bbox", [0.0, 0.0, 0.0, 0.0])
    return float(bbox[3])


def _span_y_mid(span: Dict[str, Any]) -> float:
    bbox = span.get("bbox", [0.0, 0.0, 0.0, 0.0])
    return 0.5 * (float(bbox[1]) + float(bbox[3]))


def _has_two_column_overlap(
    groups: List[_RawSpanGroup], x_center: float
) -> bool:
    left_y_intervals: List[Tuple[float, float]] = []
    right_y_intervals: List[Tuple[float, float]] = []

    for g in groups:
        txt = g.raw_text.strip()
        if len(txt) < 12 or TRAILING_EQ_NUM_RE.match(f" {txt}"):
            continue
        if g.x1 <= x_center + 12.0:
            left_y_intervals.append((g.y0, g.y1))
        elif g.x0 >= x_center - 12.0:
            right_y_intervals.append((g.y0, g.y1))

    if len(left_y_intervals) < 2 or len(right_y_intervals) < 2:
        return False

    overlapping_right = 0
    for ry0, ry1 in right_y_intervals:
        if any(max(ly0, ry0) <= min(ly1, ry1) + 15.0 for ly0, ly1 in left_y_intervals):
            overlapping_right += 1
    return overlapping_right >= 2


def _order_lines_column_aware(
    lines: List[_ExtractedLine], x_center: float, is_two_col: bool
) -> List[_ExtractedLine]:
    if not is_two_col:
        lines.sort(key=lambda ln: (round(ln.y0 / 2.0), ln.x0))
        for ln in lines:
            ln.col_id = 0
        return lines

    temp_sorted = sorted(lines, key=lambda ln: (round(ln.y0 / 2.0), ln.x0))
    ordered: List[_ExtractedLine] = []
    left_band: List[_ExtractedLine] = []
    right_band: List[_ExtractedLine] = []

    def _flush_band() -> None:
        if left_band:
            left_band.sort(key=lambda ln: (round(ln.y0 / 2.0), ln.x0))
            ordered.extend(left_band)
            left_band.clear()
        if right_band:
            right_band.sort(key=lambda ln: (round(ln.y0 / 2.0), ln.x0))
            ordered.extend(right_band)
            right_band.clear()

    for ln in temp_sorted:
        if ln.x1 <= x_center + 12.0:
            ln.col_id = 1
            left_band.append(ln)
        elif ln.x0 >= x_center - 12.0:
            ln.col_id = 2
            right_band.append(ln)
        else:
            _flush_band()
            ln.col_id = 0
            ordered.append(ln)

    _flush_band()
    return ordered


def _cluster_and_merge_visual_rows(
    raw_groups: List[_RawSpanGroup], x_center: float, is_two_col: bool
) -> List[_RawSpanGroup]:
    """
    Clusters span groups that share a visual row (vertical overlap or baseline alignment,
    including raised superscripts, with disjoint horizontal intervals in the same column),
    sorts each row strictly left-to-right by x0, and merges adjacent horizontal groups.
    """
    sorted_by_y = sorted(raw_groups, key=lambda g: (g.y_mid, g.x0))
    rows: List[List[_RawSpanGroup]] = []

    for g in sorted_by_y:
        placed = False
        g_height = max(1.0, g.y1 - g.y0)

        for row in reversed(rows[-6:]):
            row_y0 = min(item.y0 for item in row)
            row_y1 = max(item.y1 for item in row)
            row_height = max(1.0, row_y1 - row_y0)
            row_y_mid = 0.5 * (row_y0 + row_y1)

            if abs(g.y_mid - row_y_mid) > 8.0 and (g.y0 > row_y1 + 2.0):
                continue

            if is_two_col:
                crosses = any(
                    (item.x1 <= x_center + 12.0 and g.x0 >= x_center - 12.0)
                    or (g.x1 <= x_center + 12.0 and item.x0 >= x_center - 12.0)
                    for item in row
                )
                if crosses:
                    continue

            horiz_disjoint = all(
                g.x1 <= item.x0 + 3.0 or g.x0 >= item.x1 - 3.0 for item in row
            )
            if not horiz_disjoint:
                continue

            v_overlap = min(row_y1, g.y1) - max(row_y0, g.y0)
            min_h = min(row_height, g_height)
            vertically_aligned = (
                v_overlap >= 0.35 * min_h
                or abs(g.y_mid - row_y_mid) <= 4.5
                or abs(g.y1 - row_y1) <= 4.5
                or (g.y0 >= row_y0 - 3.5 and g.y1 <= row_y1 + 3.5)
            )

            if vertically_aligned:
                row.append(g)
                placed = True
                break

        if not placed:
            rows.append([g])

    merged_groups: List[_RawSpanGroup] = []
    for row in rows:
        row.sort(key=lambda item: item.x0)
        row_merged: List[_RawSpanGroup] = []
        for g in row:
            if not row_merged:
                row_merged.append(g)
                continue

            prev = row_merged[-1]
            horiz_gap = g.x0 - prev.x1
            prev_is_sec_num = bool(STANDALONE_SEC_NUM_RE.match(prev.raw_text.strip()))
            max_gap = 55.0 if prev_is_sec_num else 25.0

            if -4.0 <= horiz_gap <= max_gap:
                combined_spans = prev.spans + g.spans
                row_merged[-1] = _RawSpanGroup(
                    x0=min(prev.x0, g.x0),
                    y0=min(prev.y0, g.y0),
                    x1=max(prev.x1, g.x1),
                    y1=max(prev.y1, g.y1),
                    spans=combined_spans,
                    block_idx=prev.block_idx,
                    raw_text=f"{prev.raw_text.rstrip()} {g.raw_text.lstrip()}",
                )
            else:
                row_merged.append(g)

        merged_groups.extend(row_merged)

    return merged_groups


def _build_page_lines(
    blocks: List[Dict[str, Any]],
    body_font_size: float,
    page_number: int,
    page_width: float,
    page_height: float,
) -> Tuple[List[_ExtractedLine], str, bool, List[Dict[str, Any]]]:
    """
    1. Collects raw PyMuPDF lines and filters arXiv stamps and standalone margin page numbers.
    2. Detects whether the page has a two-column layout.
    3. Clusters horizontally adjacent span groups on the same visual row within the same
       column and orders them strictly left-to-right by x0 before merging, so split section
       numbers ("2" + "Background", "2.1" + "Title") and raised superscript spans ("10")
       are never split or reordered across discretization buckets.
    4. Uses deferred whitespace handling and evaluates superscript footnotes using vertical
       bbox/origin geometry and math-span safeguards, suppressing whitespace before raised
       mathematical exponents while preserving footnotes and ordinary prose spacing.
    5. Orders lines using column-aware reading order.
    """
    raw_groups: List[_RawSpanGroup] = []
    arxiv_stamp_str = ""
    page_has_math = False
    inline_rejected_candidates: List[Dict[str, Any]] = []
    x_center = 0.5 * page_width

    for b_idx, block in enumerate(blocks):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "")]
            if not spans:
                continue
            joined = re.sub(
                r"\s+", " ", "".join(s.get("text", "") for s in spans)
            ).strip()
            if not joined:
                continue

            m_stamp = ARXIV_STAMP_RE.search(joined)
            if m_stamp:
                arxiv_stamp_str = joined
                continue

            bbox = line.get("bbox", [0.0, 0.0, 0.0, 0.0])
            lx0, ly0, lx1, ly1 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )

            is_in_page_margin = ly0 >= (page_height - 75.0) or ly1 <= 60.0
            has_bold = any(_is_bold_span(s) for s in spans)
            if joined.isdigit() and is_in_page_margin and not has_bold:
                continue

            raw_groups.append(
                _RawSpanGroup(
                    x0=lx0,
                    y0=ly0,
                    x1=lx1,
                    y1=ly1,
                    spans=list(spans),
                    block_idx=b_idx,
                    raw_text=joined,
                )
            )

    is_two_col = _has_two_column_overlap(raw_groups, x_center)
    merged_groups = _cluster_and_merge_visual_rows(raw_groups, x_center, is_two_col)

    extracted_lines: List[_ExtractedLine] = []

    for grp in merged_groups:
        spans = grp.spans
        ref_spans = [
            s
            for s in spans
            if float(s.get("size", 0.0)) >= (body_font_size - 1.0)
            and sum(1 for c in s.get("text", "") if c.isalnum()) > 0
        ]
        if not ref_spans:
            ref_spans = spans

        ref_font_size = max(float(s.get("size", 0.0)) for s in ref_spans)
        ref_origin_y = median(_span_origin_y(s) for s in ref_spans)
        ref_y_mid = median(_span_y_mid(s) for s in ref_spans)

        reconstructed_parts: List[str] = []
        prefix_bold_parts: List[str] = []
        body_parts_after_bold: List[str] = []
        in_bold_prefix = True

        max_size = 0.0
        bold_chars = 0
        alnum_chars = 0
        line_math = False
        prev_span_x1: Optional[float] = None
        prev_nonempty_span_math: bool = False
        pending_whitespace: bool = False

        for sp in spans:
            raw_t = sp.get("text", "")
            if not raw_t:
                continue

            sp_bbox = sp.get("bbox", [0.0, 0.0, 0.0, 0.0])
            sp_x0, sp_x1 = float(sp_bbox[0]), float(sp_bbox[2])
            horiz_gap_from_prev = (
                (sp_x0 - prev_span_x1) if prev_span_x1 is not None else 0.0
            )

            stripped_t = raw_t.strip()
            if not stripped_t:
                pending_whitespace = True
                prev_span_x1 = sp_x1
                continue

            sz = float(sp.get("size", 0.0))
            sp_bold = _is_bold_span(sp)
            sp_math = _is_math_span(sp)
            if sp_math:
                line_math = True
                page_has_math = True

            sp_orig_y = _span_origin_y(sp)
            sp_mid_y = _span_y_mid(sp)

            is_raised_geometry = (
                (ref_origin_y - sp_orig_y) >= 1.0
                and (ref_y_mid - sp_mid_y) >= 0.8
            )

            prev_text_stripped = "".join(reconstructed_parts).rstrip()
            m_anchor = PROSE_FOOTNOTE_ANCHOR_RE.search(prev_text_stripped)
            anchor_word = m_anchor.group(1).lower() if m_anchor else ""
            follows_prose_word = bool(
                m_anchor and anchor_word not in MATH_FUNC_WORDS
            )

            is_superscript_footnote = (
                stripped_t.isdigit()
                and 1 <= len(stripped_t) <= 3
                and len(reconstructed_parts) > 0
                and sz <= (ref_font_size - 1.5)
                and is_raised_geometry
                and not sp_math
                and not prev_nonempty_span_math
                and follows_prose_word
                and horiz_gap_from_prev <= 12.0
            )

            is_raised_math_exponent = (
                stripped_t.isdigit()
                and 1 <= len(stripped_t) <= 3
                and sz <= (ref_font_size - 1.5)
                and is_raised_geometry
                and not is_superscript_footnote
            )

            if is_superscript_footnote:
                pending_whitespace = False
                prev_span_x1 = sp_x1
                prev_nonempty_span_math = sp_math
                footnote_token = f" [^{stripped_t}] "
                reconstructed_parts.append(footnote_token)
                in_bold_prefix = False
                body_parts_after_bold.append(footnote_token)
                continue

            if is_raised_math_exponent:
                pending_whitespace = False
                if reconstructed_parts and reconstructed_parts[-1].endswith(" "):
                    reconstructed_parts[-1] = reconstructed_parts[-1].rstrip(" ")
                    if in_bold_prefix and prefix_bold_parts and prefix_bold_parts[-1].endswith(" "):
                        prefix_bold_parts[-1] = prefix_bold_parts[-1].rstrip(" ")
                    elif not in_bold_prefix and body_parts_after_bold and body_parts_after_bold[-1].endswith(" "):
                        body_parts_after_bold[-1] = body_parts_after_bold[-1].rstrip(" ")
            else:
                should_add_space = pending_whitespace or (
                    prev_span_x1 is not None and horiz_gap_from_prev >= 2.5
                )
                if (
                    should_add_space
                    and reconstructed_parts
                    and not reconstructed_parts[-1].endswith(" ")
                    and not raw_t.startswith(" ")
                ):
                    reconstructed_parts.append(" ")
                    if in_bold_prefix:
                        prefix_bold_parts.append(" ")
                    else:
                        body_parts_after_bold.append(" ")
                pending_whitespace = False

            prev_span_x1 = sp_x1
            prev_nonempty_span_math = sp_math

            if sz > max_size:
                max_size = sz

            c_alnum = sum(1 for c in raw_t if c.isalnum())
            alnum_chars += c_alnum
            if sp_bold:
                bold_chars += c_alnum

            if in_bold_prefix and sp_bold:
                prefix_bold_parts.append(raw_t)
            else:
                in_bold_prefix = False
                body_parts_after_bold.append(raw_t)

            reconstructed_parts.append(raw_t)

        line_str = re.sub(r"\s+", " ", "".join(reconstructed_parts)).strip()
        if not line_str:
            continue

        prefix_str = re.sub(r"\s+", " ", "".join(prefix_bold_parts)).strip()
        rest_str = re.sub(r"\s+", " ", "".join(body_parts_after_bold)).strip()
        if (
            prefix_str
            and rest_str
            and len(prefix_str) >= 3
            and len(rest_str) >= 12
            and not CAPTION_START_RE.match(line_str)
        ):
            inline_rejected_candidates.append(
                asdict(
                    HeadingItem(
                        text=prefix_str,
                        section_number="",
                        section_title="",
                        hierarchy_level=0,
                        font_size=round(max_size, 2),
                        is_bold=True,
                        confidence="candidate_rejected",
                        reason="inline_bold_lead_in_split_from_body",
                    )
                )
            )
            clean_combined = f"{prefix_str} {rest_str}"
            extracted_lines.append(
                _ExtractedLine(
                    x0=grp.x0,
                    y0=grp.y0,
                    x1=grp.x1,
                    y1=grp.y1,
                    text=clean_combined,
                    max_font_size=max_size,
                    bold_chars=bold_chars,
                    alnum_chars=alnum_chars,
                    has_math_font=line_math,
                    block_idx=grp.block_idx,
                )
            )
            continue

        extracted_lines.append(
            _ExtractedLine(
                x0=grp.x0,
                y0=grp.y0,
                x1=grp.x1,
                y1=grp.y1,
                text=line_str,
                max_font_size=max_size,
                bold_chars=bold_chars,
                alnum_chars=alnum_chars,
                has_math_font=line_math,
                block_idx=grp.block_idx,
            )
        )

    ordered_lines = _order_lines_column_aware(extracted_lines, x_center, is_two_col)
    return ordered_lines, arxiv_stamp_str, page_has_math, inline_rejected_candidates


def _section_level_from_number(sec_num: str) -> int:
    """Returns bounded hierarchy depth (1..4) from a dotted section number."""
    if not sec_num:
        return 1
    return min(4, len(sec_num.split(".")))


def _classify_heading_line(
    ln: _ExtractedLine,
    body_font_size: float,
    page_number: int,
    seen_abstract: bool,
    seen_references: bool,
) -> Optional[HeadingItem]:
    clean = re.sub(r"\s+", " ", ln.text).strip()
    if not clean or len(clean) < 2:
        return None

    max_font_size = ln.max_font_size
    is_bold = ln.is_bold
    is_larger = max_font_size >= (body_font_size + 0.8)
    is_top_size = max_font_size >= (body_font_size + 1.5)

    if not (is_larger or is_bold):
        return None

    # Guard 1: Reject Figure/Table/Algorithm captions
    if CAPTION_START_RE.match(clean):
        return HeadingItem(
            text=clean,
            section_number="",
            section_title="",
            hierarchy_level=0,
            font_size=round(max_font_size, 2),
            is_bold=is_bold,
            confidence="candidate_rejected",
            reason="caption_line",
        )

    # Guard 2: Reject lines dominated by math fonts or equation expressions
    if ln.has_math_font and (
        "=" in clean or "∑" in clean or "∫" in clean or len(clean) <= 12
    ):
        return HeadingItem(
            text=clean,
            section_number="",
            section_title="",
            hierarchy_level=0,
            font_size=round(max_font_size, 2),
            is_bold=is_bold,
            confidence="candidate_rejected",
            reason="math_or_operator_span",
        )

    lower_clean_no_punct = clean.lower().rstrip(".:")

    # Guard 3: Before Abstract on page 1, reject paper title, authors, and affiliations
    if page_number == 1 and not seen_abstract and lower_clean_no_punct != "abstract":
        return HeadingItem(
            text=clean,
            section_number="",
            section_title="",
            hierarchy_level=0,
            font_size=round(max_font_size, 2),
            is_bold=is_bold,
            confidence="candidate_rejected",
            reason="pre_abstract_title_or_author",
        )

    # Guard 4: Evaluate Numbered Section Headings (bounded 4-level support) BEFORE rejecting for terminal punctuation
    m_num = NUMBERED_SECTION_RE.match(clean)
    if m_num:
        sec_num = m_num.group(1)
        had_trailing_dot_on_num = bool(m_num.group(2))
        sec_title = m_num.group(3).strip()

        alpha_count = sum(c.isalpha() for c in sec_title)
        digit_count = sum(c.isdigit() for c in sec_title)
        has_internal_colon = ":" in sec_title

        is_single_digit_with_dot = ("." not in sec_num) and had_trailing_dot_on_num
        valid_single_digit = (not is_single_digit_with_dot) or is_larger

        if (
            is_bold
            and valid_single_digit
            and not has_internal_colon
            and alpha_count >= 3
            and digit_count <= alpha_count
            and len(sec_title.split()) <= 12
        ):
            lvl = _section_level_from_number(sec_num)
            return HeadingItem(
                text=clean,
                section_number=sec_num,
                section_title=f"{sec_num} {sec_title}",
                hierarchy_level=lvl,
                font_size=round(max_font_size, 2),
                is_bold=is_bold,
                confidence="confirmed",
                reason="numbered_section_header",
            )

    # Guard 5: Reject unnumbered lines ending with a period or colon (run-in paragraph lead-ins)
    if clean.endswith(".") or clean.endswith(":"):
        return HeadingItem(
            text=clean,
            section_number="",
            section_title="",
            hierarchy_level=0,
            font_size=round(max_font_size, 2),
            is_bold=is_bold,
            confidence="candidate_rejected",
            reason="paragraph_lead_in_with_terminal_punctuation",
        )

    lower_clean = clean.lower()

    # Guard 6: Canonical unnumbered top-level sections
    if lower_clean in CANONICAL_TOP_SECTIONS and (
        is_bold or is_larger or lower_clean == "abstract"
    ):
        return HeadingItem(
            text=clean,
            section_number="",
            section_title=clean,
            hierarchy_level=1,
            font_size=round(max_font_size, 2),
            is_bold=is_bold,
            confidence="confirmed",
            reason="canonical_section_header",
        )

    # Guard 7: Canonical sub-headers inside sections (stored at leaf level 5 so they never overwrite numbered levels 1..4)
    if lower_clean in CANONICAL_SUB_HEADERS and is_bold:
        return HeadingItem(
            text=clean,
            section_number="",
            section_title=clean,
            hierarchy_level=5,
            font_size=round(max_font_size, 2),
            is_bold=is_bold,
            confidence="confirmed",
            reason="canonical_sub_header",
        )

    # Guard 8: Unnumbered appendix section header after References at top-level section font size
    if (
        seen_references
        and is_bold
        and is_top_size
        and ln.x0 <= 135.0
        and 4 <= len(clean) <= 60
        and sum(c.isalpha() for c in clean) >= 4
    ):
        return HeadingItem(
            text=clean,
            section_number="",
            section_title=clean,
            hierarchy_level=1,
            font_size=round(max_font_size, 2),
            is_bold=is_bold,
            confidence="confirmed",
            reason="unnumbered_appendix_section_header",
        )

    return HeadingItem(
        text=clean,
        section_number="",
        section_title="",
        hierarchy_level=0,
        font_size=round(max_font_size, 2),
        is_bold=is_bold,
        confidence="candidate_rejected",
        reason="unnumbered_styled_line_or_diagram_label",
    )


def _update_section_hierarchy(
    hierarchy: Dict[int, str], heading: HeadingItem
) -> Tuple[str, str]:
    """
    Maintains a bounded four-level numbered section hierarchy (levels 1..4, matching
    NUMBERED_SECTION_RE) and places generic unnumbered sub-headers at leaf level 5 so
    they never overwrite any numbered heading up to depth 4 (e.g., 3.2.1 or 2.1.1.1).
    """
    lvl = heading.hierarchy_level
    title = heading.section_title

    if lvl == 1:
        hierarchy.clear()
        hierarchy[1] = title
    elif 2 <= lvl <= 4:
        if 1 not in hierarchy or hierarchy[1] == "Front Matter":
            hierarchy[1] = title
        else:
            hierarchy[lvl] = title
        for deeper in [k for k in list(hierarchy.keys()) if k > lvl]:
            hierarchy.pop(deeper, None)
    elif lvl >= 5:
        hierarchy[5] = title

    path_parts = [hierarchy[k] for k in sorted(hierarchy.keys()) if hierarchy.get(k)]
    full_path = " > ".join(path_parts) if path_parts else "Front Matter"

    numbered_levels = [k for k in sorted(hierarchy.keys()) if 1 <= k <= 4 and hierarchy.get(k)]
    if numbered_levels:
        active_primary = hierarchy[numbered_levels[-1]]
    else:
        active_primary = full_path
    return active_primary, full_path


def _boxes_overlap(b1: List[float], b2: List[float], tol: float = 2.0) -> bool:
    return not (
        b1[2] < b2[0] - tol
        or b1[0] > b2[2] + tol
        or b1[3] < b2[1] - tol
        or b1[1] > b2[3] + tol
    )


def extract_single_pdf(
    paper_meta: Dict[str, str],
) -> Tuple[List[PageExtractionRecord], Dict[str, Any]]:
    pdf_path = Path(paper_meta["pdf_path"])
    paper_id = paper_meta["paper_id"]
    arxiv_version = paper_meta["arxiv_version"]
    title = paper_meta.get("title") or paper_id
    sha256 = paper_meta["sha256"]

    doc = pymupdf.open(pdf_path)
    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError(f"PDF has 0 pages: {pdf_path}")

        body_font_size = _estimate_document_body_font_size(doc)
        hierarchy: Dict[int, str] = {1: "Front Matter"}
        active_section = "Front Matter"
        active_section_path = "Front Matter"
        seen_abstract = False
        seen_references = False
        page_records: List[PageExtractionRecord] = []
        version_verified_in_text = False

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_number = page_idx + 1
            section_at_start = active_section
            section_path_at_start = active_section_path

            page_dict = page.get_text("dict", sort=True)
            blocks = page_dict.get("blocks", [])
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)

            extracted_tables: List[Dict[str, Any]] = []
            try:
                tabs = page.find_tables()
                if tabs and hasattr(tabs, "tables"):
                    for t_idx, tab in enumerate(tabs.tables):
                        t_bbox = [round(float(c), 2) for c in tab.bbox]
                        raw_rows = tab.extract() if hasattr(tab, "extract") else []
                        extracted_tables.append(
                            {
                                "table_index": t_idx,
                                "bbox": t_bbox,
                                "row_count": len(raw_rows),
                                "col_count": max((len(r) for r in raw_rows), default=0),
                                "rows": raw_rows,
                            }
                        )
            except Exception:
                extracted_tables = []

            pymupdf_tables_count = len(extracted_tables)

            merged_lines, arxiv_stamp_str, page_has_math, inline_rejected = _build_page_lines(
                blocks=blocks,
                body_font_size=body_font_size,
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
            )

            if arxiv_stamp_str:
                m_stamp = ARXIV_STAMP_RE.search(arxiv_stamp_str)
                if m_stamp:
                    stamp_id, stamp_ver = m_stamp.group(1), m_stamp.group(2).lower()
                    if stamp_id == paper_id and stamp_ver == arxiv_version:
                        version_verified_in_text = True
                    elif stamp_id == paper_id and stamp_ver != arxiv_version:
                        raise ValueError(
                            f"arXiv stamp version mismatch on {pdf_path.name} p.{page_number}: "
                            f"found {stamp_id}{stamp_ver}, expected {paper_id}{arxiv_version}"
                        )

            paragraphs: List[str] = []
            structured_paragraphs: List[Dict[str, Any]] = []
            confirmed_headings: List[Dict[str, Any]] = []
            heading_candidates: List[Dict[str, Any]] = list(inline_rejected)
            equation_numbers: List[str] = []
            equation_refs: List[str] = []
            table_captions: List[str] = []
            table_caption_ids: List[str] = []
            table_references: List[str] = []
            figure_captions: List[str] = []
            figure_caption_ids: List[str] = []
            figure_references: List[str] = []
            warnings: List[str] = []

            current_para_lines: List[_ExtractedLine] = []
            current_para_section = active_section
            current_para_path = active_section_path

            def _flush_paragraph() -> None:
                if not current_para_lines:
                    return
                para_text = "\n".join(l.text for l in current_para_lines).strip()
                if not para_text:
                    current_para_lines.clear()
                    return

                px0 = round(min(l.x0 for l in current_para_lines), 2)
                py0 = round(min(l.y0 for l in current_para_lines), 2)
                px1 = round(max(l.x1 for l in current_para_lines), 2)
                py1 = round(max(l.y1 for l in current_para_lines), 2)
                p_bbox = [px0, py0, px1, py1]

                para_table_caption_ids: List[str] = []
                para_table_refs: List[str] = []
                para_fig_caption_ids: List[str] = []
                para_fig_refs: List[str] = []
                para_eq_nums: List[str] = []
                para_has_math = any(l.has_math_font for l in current_para_lines)
                non_caption_lines: List[str] = []

                for l in current_para_lines:
                    m_cap = CAPTION_START_RE.match(l.text)
                    if m_cap:
                        kind = m_cap.group(1).lower()
                        ident = m_cap.group(2)
                        if kind.startswith("tab") and ident not in para_table_caption_ids:
                            para_table_caption_ids.append(ident)
                        elif kind.startswith("fig") and ident not in para_fig_caption_ids:
                            para_fig_caption_ids.append(ident)
                    else:
                        non_caption_lines.append(l.text)

                    m_eq = TRAILING_EQ_NUM_RE.search(l.text)
                    if m_eq and (l.has_math_font or "=" in l.text or len(l.text) < 90):
                        eq_id = m_eq.group(1)
                        if eq_id not in equation_numbers:
                            equation_numbers.append(eq_id)
                        if eq_id not in para_eq_nums:
                            para_eq_nums.append(eq_id)

                non_caption_text = "\n".join(non_caption_lines)
                for m_t in TABLE_REF_RE.finditer(non_caption_text):
                    tid = m_t.group(1)
                    if tid not in table_references:
                        table_references.append(tid)
                    if tid not in para_table_refs:
                        para_table_refs.append(tid)
                for m_f in FIGURE_REF_RE.finditer(non_caption_text):
                    fid = m_f.group(1)
                    if fid not in figure_references:
                        figure_references.append(fid)
                    if fid not in para_fig_refs:
                        para_fig_refs.append(fid)
                for m_e in EQUATION_REF_RE.finditer(non_caption_text):
                    eid = m_e.group(1)
                    if eid not in equation_refs:
                        equation_refs.append(eid)

                overlaps_grid_table = any(
                    _boxes_overlap(p_bbox, t["bbox"]) for t in extracted_tables
                )

                paragraphs.append(para_text)
                structured_paragraphs.append(
                    {
                        "paragraph_index": len(structured_paragraphs),
                        "text": para_text,
                        "active_section": current_para_section,
                        "section_path": current_para_path,
                        "column_id": current_para_lines[0].col_id,
                        "bbox": p_bbox,
                        "is_table_caption": bool(para_table_caption_ids),
                        "overlaps_extracted_table": overlaps_grid_table,
                        "table_caption_ids": para_table_caption_ids,
                        "table_references": para_table_refs,
                        "figure_caption_ids": para_fig_caption_ids,
                        "figure_references": para_fig_refs,
                        "equation_numbers": para_eq_nums,
                        "has_math": para_has_math,
                    }
                )
                current_para_lines.clear()

            prev_block_idx: Optional[int] = None
            prev_col_id: Optional[int] = None

            for ln in merged_lines:
                line_str = ln.text

                m_cap = CAPTION_START_RE.match(line_str)
                if m_cap:
                    kind = m_cap.group(1).lower()
                    ident = m_cap.group(2)
                    if kind.startswith("tab"):
                        table_captions.append(line_str)
                        if ident not in table_caption_ids:
                            table_caption_ids.append(ident)
                    elif kind.startswith("fig"):
                        figure_captions.append(line_str)
                        if ident not in figure_caption_ids:
                            figure_caption_ids.append(ident)

                h_item = _classify_heading_line(
                    ln=ln,
                    body_font_size=body_font_size,
                    page_number=page_number,
                    seen_abstract=seen_abstract,
                    seen_references=seen_references,
                )

                if h_item is not None:
                    heading_candidates.append(asdict(h_item))
                    if h_item.confidence == "confirmed":
                        _flush_paragraph()
                        confirmed_headings.append(asdict(h_item))
                        active_section, active_section_path = _update_section_hierarchy(
                            hierarchy, h_item
                        )
                        current_para_section = active_section
                        current_para_path = active_section_path
                        if h_item.section_title.lower() == "abstract":
                            seen_abstract = True
                        if h_item.section_title.lower() == "references":
                            seen_references = True
                        current_para_lines.append(ln)
                        prev_block_idx = ln.block_idx
                        prev_col_id = ln.col_id
                        continue

                if "abstract" in line_str.lower() and page_number == 1:
                    seen_abstract = True

                if prev_block_idx is not None and (
                    ln.block_idx != prev_block_idx or ln.col_id != prev_col_id
                ):
                    _flush_paragraph()
                    current_para_section = active_section
                    current_para_path = active_section_path

                current_para_lines.append(ln)
                prev_block_idx = ln.block_idx
                prev_col_id = ln.col_id

            _flush_paragraph()

            full_text = "\n\n".join(paragraphs)
            char_count = len(full_text)
            word_count = len(full_text.split())

            image_count = len(page.get_images(full=True))
            drawing_count = len(page.get_drawings())

            if char_count == 0:
                warnings.append("empty_page_text")
            elif char_count < 120 and (image_count > 0 or drawing_count > 10):
                warnings.append("figure_or_diagram_heavy_page_sparse_text")
            elif char_count < 120:
                warnings.append("sparse_page_text")

            if page_has_math or equation_numbers:
                warnings.append("contains_linearized_pdf_math")
            if table_captions and pymupdf_tables_count == 0:
                warnings.append("borderless_table_preserved_as_text_lines")

            page_records.append(
                PageExtractionRecord(
                    paper_id=paper_id,
                    arxiv_version=arxiv_version,
                    title=title,
                    filename=pdf_path.name,
                    sha256=sha256,
                    page_number=page_number,
                    total_pages=total_pages,
                    char_count=char_count,
                    word_count=word_count,
                    paragraph_count=len(paragraphs),
                    active_section_start=section_at_start,
                    active_section_end=active_section,
                    active_section_path_start=section_path_at_start,
                    active_section_path_end=active_section_path,
                    confirmed_headings=confirmed_headings,
                    heading_candidates=heading_candidates,
                    equation_numbers=equation_numbers,
                    equation_references=equation_refs,
                    has_math_spans=page_has_math,
                    table_captions=table_captions,
                    table_caption_ids=table_caption_ids,
                    table_references=table_references,
                    table_Ids=table_caption_ids,
                    pymupdf_tables_detected=pymupdf_tables_count,
                    extracted_tables=extracted_tables,
                    figure_captions=figure_captions,
                    figure_caption_ids=figure_caption_ids,
                    figure_references=figure_references,
                    figure_ids=figure_caption_ids,
                    image_count=image_count,
                    drawing_count=drawing_count,
                    arxiv_stamp_detected=arxiv_stamp_str,
                    extraction_warnings=warnings,
                    structured_paragraphs=structured_paragraphs,
                    paragraphs=paragraphs,
                    full_text=full_text,
                )
            )

        extracted_page_nums = [r.page_number for r in page_records]
        expected_page_nums = list(range(1, total_pages + 1))
        if extracted_page_nums != expected_page_nums:
            raise ValueError(
                f"Missing or out-of-order pages for {paper_id}: {extracted_page_nums}"
            )

        empty_pages = [r.page_number for r in page_records if r.char_count == 0]
        if empty_pages:
            raise ValueError(
                f"Extraction produced completely empty pages for {paper_id}: {empty_pages}"
            )

        if not version_verified_in_text:
            raise ValueError(
                f"Expected arXiv version marker {paper_id}{arxiv_version} not found in {pdf_path.name}"
            )

        paper_summary = {
            "paper_id": paper_id,
            "arxiv_version": arxiv_version,
            "title": title,
            "filename": pdf_path.name,
            "sha256": sha256,
            "version_stamp_found_in_pdf": version_verified_in_text,
            "total_pages": total_pages,
            "body_font_size_pt": round(body_font_size, 2),
            "total_characters": sum(r.char_count for r in page_records),
            "total_words": sum(r.word_count for r in page_records),
            "total_paragraphs": sum(r.paragraph_count for r in page_records),
            "confirmed_headings_count": sum(
                len(r.confirmed_headings) for r in page_records
            ),
            "rejected_heading_candidates_count": sum(
                len(r.heading_candidates) - len(r.confirmed_headings)
                for r in page_records
            ),
            "unique_sections_detected": list(
                dict.fromkeys(
                    h["section_title"]
                    for r in page_records
                    for h in r.confirmed_headings
                )
            ),
            "pages_with_math": sum(1 for r in page_records if r.has_math_spans),
            "unique_equation_numbers": sorted(
                {eq for r in page_records for eq in r.equation_numbers}
            ),
            "unique_table_caption_ids": sorted(
                {tid for r in page_records for tid in r.table_caption_ids}
            ),
            "unique_table_reference_ids": sorted(
                {tid for r in page_records for tid in r.table_references}
            ),
            "unique_table_ids": sorted(
                {tid for r in page_records for tid in r.table_caption_ids}
            ),
            "table_captions_found": sum(len(r.table_captions) for r in page_records),
            "pymupdf_grid_tables_found": sum(
                r.pymupdf_tables_detected for r in page_records
            ),
            "unique_figure_caption_ids": sorted(
                {fid for r in page_records for fid in r.figure_caption_ids}
            ),
            "unique_figure_reference_ids": sorted(
                {fid for r in page_records for fid in r.figure_references}
            ),
            "unique_figure_ids": sorted(
                {fid for r in page_records for fid in r.figure_caption_ids}
            ),
            "figure_captions_found": sum(len(r.figure_captions) for r in page_records),
            "sparse_or_figure_heavy_pages": [
                r.page_number
                for r in page_records
                if "figure_or_diagram_heavy_page_sparse_text" in r.extraction_warnings
                or "sparse_page_text" in r.extraction_warnings
            ],
        }
        return page_records, paper_summary
    finally:
        doc.close()


def extract_corpus(
    papers_dir: Path,
    manifest_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    verified_papers = load_and_verify_manifest(papers_dir, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pages: List[PageExtractionRecord] = []
    paper_summaries: List[Dict[str, Any]] = []

    for paper_meta in verified_papers:
        pages, summary = extract_single_pdf(paper_meta)
        all_pages.extend(pages)
        paper_summaries.append(summary)

    pages_jsonl_path = output_dir / "extracted_pages.jsonl"
    with open(pages_jsonl_path, "w", encoding="utf-8") as f:
        for page_rec in all_pages:
            f.write(json.dumps(asdict(page_rec), ensure_ascii=False) + "\n")

    corpus_summary = {
        "corpus_size_papers": len(paper_summaries),
        "total_pages_extracted": len(all_pages),
        "total_words_extracted": sum(p["total_words"] for p in paper_summaries),
        "output_pages_jsonl": str(pages_jsonl_path),
        "papers": paper_summaries,
    }

    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(corpus_summary, f, indent=2, ensure_ascii=False)

    return corpus_summary
