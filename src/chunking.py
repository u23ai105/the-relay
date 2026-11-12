"""
Phase 4 (Part 2): Deterministic paragraph-aware chunking for 250-token and 500-token
configurations. Preserves full paper, version, page, section, paragraph, and table/math
provenance, handles oversized paragraphs without dropping or duplicating content, and
attaches adjacent boundary context snippets for downstream semantic verification.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"])")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(TOKEN_RE.findall(text))


def _tail_tokens_text(text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return ""
    words = text.split()
    selected: List[str] = []
    acc = 0
    for w in reversed(words):
        t_cnt = max(1, count_tokens(w))
        if acc + t_cnt > max_tokens and selected:
            break
        selected.append(w)
        acc += t_cnt
    return " ".join(reversed(selected))


def _head_tokens_text(text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return ""
    words = text.split()
    selected: List[str] = []
    acc = 0
    for w in words:
        t_cnt = max(1, count_tokens(w))
        if acc + t_cnt > max_tokens and selected:
            break
        selected.append(w)
        acc += t_cnt
    return " ".join(selected)


@dataclass
class ChunkRecord:
    chunk_id: str
    paper_id: str
    arxiv_version: str
    title: str
    filename: str
    sha256: str
    chunk_size_target: int
    chunk_index_in_paper: int
    global_chunk_index: int
    primary_page: int
    page_start: int
    page_end: int
    page_numbers: List[int]
    section_title: str
    section_path: str
    all_sections: List[str]
    paragraph_refs: List[Dict[str, int]]
    table_caption_ids: List[str]
    table_references: List[str]
    figure_caption_ids: List[str]
    figure_references: List[str]
    equation_numbers: List[str]
    has_math: bool
    is_oversized_split: bool
    token_count: int
    word_count: int
    char_count: int
    text: str
    prev_boundary_context: str = ""
    next_boundary_context: str = ""


@dataclass
class _ParagraphUnit:
    paper_id: str
    arxiv_version: str
    title: str
    filename: str
    sha256: str
    page_number: int
    paragraph_index: int
    sub_part_index: int
    is_oversized_split: bool
    active_section: str
    section_path: str
    table_caption_ids: List[str]
    table_references: List[str]
    figure_caption_ids: List[str]
    figure_references: List[str]
    equation_numbers: List[str]
    has_math: bool
    text: str
    token_count: int


def _split_oversized_text_lossless(text: str, max_tokens: int) -> List[str]:
    """
    Splits an oversized paragraph into contiguous non-overlapping pieces each <= max_tokens
    without dropping or duplicating any words. Prefers sentence boundaries first, and falls
    back to exact word boundaries when a single sentence exceeds max_tokens.
    """
    clean = text.strip()
    if not clean:
        return []
    if count_tokens(clean) <= max_tokens:
        return [clean]

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(clean) if s.strip()]
    pieces: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    def _flush_current() -> None:
        nonlocal current_tokens
        if current_sentences:
            pieces.append(" ".join(current_sentences))
            current_sentences.clear()
            current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if sent_tokens <= max_tokens:
            if current_tokens + sent_tokens <= max_tokens:
                current_sentences.append(sent)
                current_tokens += sent_tokens
            else:
                _flush_current()
                current_sentences.append(sent)
                current_tokens = sent_tokens
        else:
            # Single sentence exceeds max_tokens: split deterministically by whitespace words
            _flush_current()
            words = sent.split()
            word_buf: List[str] = []
            word_toks = 0
            for w in words:
                w_tok = max(1, count_tokens(w))
                if word_toks + w_tok > max_tokens and word_buf:
                    pieces.append(" ".join(word_buf))
                    word_buf = [w]
                    word_toks = w_tok
                else:
                    word_buf.append(w)
                    word_toks += w_tok
            if word_buf:
                pieces.append(" ".join(word_buf))

    _flush_current()
    return pieces


def _expand_pages_to_units(
    pages: List[Dict[str, Any]], chunk_size_tokens: int
) -> List[_ParagraphUnit]:
    units: List[_ParagraphUnit] = []
    for p in pages:
        paper_id = p["paper_id"]
        arxiv_version = p["arxiv_version"]
        title = p["title"]
        filename = p["filename"]
        sha256 = p["sha256"]
        page_num = int(p["page_number"])

        struct_paras = p.get("structured_paragraphs", [])
        if not struct_paras and p.get("paragraphs"):
            struct_paras = [
                {
                    "paragraph_index": idx,
                    "text": txt,
                    "active_section": p.get("active_section_end", "Front Matter"),
                    "section_path": p.get("active_section_path_end", "Front Matter"),
                    "table_caption_ids": [],
                    "table_references": [],
                    "figure_caption_ids": [],
                    "figure_references": [],
                    "equation_numbers": [],
                    "has_math": bool(p.get("has_math_spans", False)),
                }
                for idx, txt in enumerate(p["paragraphs"])
            ]

        for sp in struct_paras:
            raw_text = str(sp.get("text", "")).strip()
            if not raw_text:
                continue
            p_idx = int(sp.get("paragraph_index", 0))
            tok_cnt = count_tokens(raw_text)

            if tok_cnt <= chunk_size_tokens:
                units.append(
                    _ParagraphUnit(
                        paper_id=paper_id,
                        arxiv_version=arxiv_version,
                        title=title,
                        filename=filename,
                        sha256=sha256,
                        page_number=page_num,
                        paragraph_index=p_idx,
                        sub_part_index=0,
                        is_oversized_split=False,
                        active_section=str(sp.get("active_section", "Front Matter")),
                        section_path=str(sp.get("section_path", "Front Matter")),
                        table_caption_ids=list(sp.get("table_caption_ids", [])),
                        table_references=list(sp.get("table_references", [])),
                        figure_caption_ids=list(sp.get("figure_caption_ids", [])),
                        figure_references=list(sp.get("figure_references", [])),
                        equation_numbers=list(sp.get("equation_numbers", [])),
                        has_math=bool(sp.get("has_math", False)),
                        text=raw_text,
                        token_count=tok_cnt,
                    )
                )
            else:
                sub_texts = _split_oversized_text_lossless(raw_text, chunk_size_tokens)
                for sub_i, stext in enumerate(sub_texts):
                    units.append(
                        _ParagraphUnit(
                            paper_id=paper_id,
                            arxiv_version=arxiv_version,
                            title=title,
                            filename=filename,
                            sha256=sha256,
                            page_number=page_num,
                            paragraph_index=p_idx,
                            sub_part_index=sub_i,
                            is_oversized_split=True,
                            active_section=str(sp.get("active_section", "Front Matter")),
                            section_path=str(sp.get("section_path", "Front Matter")),
                            table_caption_ids=list(sp.get("table_caption_ids", [])),
                            table_references=list(sp.get("table_references", [])),
                            figure_caption_ids=list(sp.get("figure_caption_ids", [])),
                            figure_references=list(sp.get("figure_references", [])),
                            equation_numbers=list(sp.get("equation_numbers", [])),
                            has_math=bool(sp.get("has_math", False)),
                            text=stext,
                            token_count=count_tokens(stext),
                        )
                    )
    return units


def _top_section_key(section_path: str) -> str:
    return section_path.split(" > ")[0].strip() if section_path else "Front Matter"


def build_chunks_for_size(
    extracted_pages: List[Dict[str, Any]],
    chunk_size_tokens: int,
    boundary_context_tokens: int = 45,
    min_section_break_tokens: int = 40,
) -> List[ChunkRecord]:
    """
    Deterministically chunks all extracted pages for a target token budget (`250` or `500`),
    grouped by paper. Never mixes pages from different papers in a single chunk.
    """
    if chunk_size_tokens <= 0:
        raise ValueError(f"chunk_size_tokens must be positive, got {chunk_size_tokens}")

    papers_order: List[str] = []
    pages_by_paper: Dict[str, List[Dict[str, Any]]] = {}
    for p in extracted_pages:
        pid = p["paper_id"]
        if pid not in pages_by_paper:
            papers_order.append(pid)
            pages_by_paper[pid] = []
        pages_by_paper[pid].append(p)

    all_chunks: List[ChunkRecord] = []
    global_idx = 0

    for pid in papers_order:
        p_pages = sorted(pages_by_paper[pid], key=lambda x: int(x["page_number"]))
        units = _expand_pages_to_units(p_pages, chunk_size_tokens)
        paper_chunks: List[ChunkRecord] = []
        buf: List[_ParagraphUnit] = []
        buf_tokens = 0

        def _flush_buffer() -> None:
            nonlocal buf_tokens, global_idx
            if not buf:
                return
            first_u = buf[0]
            paper_idx = len(paper_chunks) + 1
            global_idx += 1

            ver = first_u.arxiv_version
            chunk_id = f"{pid}{ver}_s{chunk_size_tokens}_c{paper_idx:04d}"

            joined_text = "\n\n".join(u.text for u in buf)
            actual_tokens = count_tokens(joined_text)
            page_nums = list(dict.fromkeys(u.page_number for u in buf))
            all_secs = list(dict.fromkeys(u.active_section for u in buf if u.active_section))

            # Choose the section owning the most tokens in this chunk as primary section_title
            sec_weights: Dict[str, int] = {}
            path_by_sec: Dict[str, str] = {}
            for u in buf:
                sec_weights[u.active_section] = sec_weights.get(u.active_section, 0) + u.token_count
                path_by_sec[u.active_section] = u.section_path
            primary_sec = max(sec_weights.items(), key=lambda kv: kv[1])[0]
            primary_path = path_by_sec.get(primary_sec, first_u.section_path)

            para_refs: List[Dict[str, int]] = []
            seen_refs = set()
            for u in buf:
                key = (u.page_number, u.paragraph_index)
                if key not in seen_refs:
                    seen_refs.add(key)
                    para_refs.append(
                        {"page_number": u.page_number, "paragraph_index": u.paragraph_index}
                    )

            t_caps = list(dict.fromkeys(cid for u in buf for cid in u.table_caption_ids))
            t_refs = list(dict.fromkeys(rid for u in buf for rid in u.table_references))
            f_caps = list(dict.fromkeys(cid for u in buf for cid in u.figure_caption_ids))
            f_refs = list(dict.fromkeys(rid for u in buf for rid in u.figure_references))
            eq_nums = list(dict.fromkeys(eid for u in buf for eid in u.equation_numbers))

            paper_chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    paper_id=pid,
                    arxiv_version=ver,
                    title=first_u.title,
                    filename=first_u.filename,
                    sha256=first_u.sha256,
                    chunk_size_target=chunk_size_tokens,
                    chunk_index_in_paper=paper_idx,
                    global_chunk_index=global_idx,
                    primary_page=page_nums[0],
                    page_start=min(page_nums),
                    page_end=max(page_nums),
                    page_numbers=page_nums,
                    section_title=primary_sec,
                    section_path=primary_path,
                    all_sections=all_secs,
                    paragraph_refs=para_refs,
                    table_caption_ids=t_caps,
                    table_references=t_refs,
                    figure_caption_ids=f_caps,
                    figure_references=f_refs,
                    equation_numbers=eq_nums,
                    has_math=any(u.has_math for u in buf),
                    is_oversized_split=any(u.is_oversized_split for u in buf),
                    token_count=actual_tokens,
                    word_count=len(joined_text.split()),
                    char_count=len(joined_text),
                    text=joined_text,
                )
            )
            buf.clear()
            buf_tokens = 0

        for u in units:
            if not buf:
                buf.append(u)
                buf_tokens = u.token_count
                continue

            top_sec_changed = _top_section_key(u.section_path) != _top_section_key(
                buf[-1].section_path
            )
            would_exceed = (buf_tokens + u.token_count) > chunk_size_tokens

            if would_exceed or (top_sec_changed and buf_tokens >= min_section_break_tokens):
                _flush_buffer()
                buf.append(u)
                buf_tokens = u.token_count
            else:
                buf.append(u)
                buf_tokens += u.token_count

        _flush_buffer()

        # Populate prev_boundary_context and next_boundary_context within the same paper
        for i, ch in enumerate(paper_chunks):
            if i > 0:
                ch.prev_boundary_context = _tail_tokens_text(
                    paper_chunks[i - 1].text, boundary_context_tokens
                )
            if i + 1 < len(paper_chunks):
                ch.next_boundary_context = _head_tokens_text(
                    paper_chunks[i + 1].text, boundary_context_tokens
                )

        all_chunks.extend(paper_chunks)

    return all_chunks


def build_and_save_all_chunks(
    extracted_pages_jsonl: Path,
    output_dir: Path,
    chunk_sizes: Tuple[int, ...] = (250, 500),
    boundary_context_tokens: int = 45,
) -> Dict[str, Any]:
    if not extracted_pages_jsonl.exists():
        raise FileNotFoundError(
            f"Extracted pages JSONL not found: {extracted_pages_jsonl}"
        )

    with open(extracted_pages_jsonl, "r", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f if line.strip()]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_configs: Dict[str, Any] = {}

    for sz in chunk_sizes:
        chunks = build_chunks_for_size(
            extracted_pages=pages,
            chunk_size_tokens=sz,
            boundary_context_tokens=boundary_context_tokens,
        )
        out_file = output_dir / f"chunks_{sz}.jsonl"
        h = hashlib.sha256()
        with open(out_file, "w", encoding="utf-8") as f:
            for ch in chunks:
                line = json.dumps(asdict(ch), ensure_ascii=False) + "\n"
                f.write(line)
                h.update(line.encode("utf-8"))

        per_paper_counts: Dict[str, int] = {}
        for ch in chunks:
            key = f"{ch.paper_id}{ch.arxiv_version}"
            per_paper_counts[key] = per_paper_counts.get(key, 0) + 1

        tok_counts = [ch.token_count for ch in chunks]
        manifest_configs[str(sz)] = {
            "chunk_size_target": sz,
            "output_file": str(out_file),
            "sha256": h.hexdigest(),
            "total_chunks": len(chunks),
            "chunks_per_paper": per_paper_counts,
            "min_tokens": min(tok_counts) if tok_counts else 0,
            "max_tokens": max(tok_counts) if tok_counts else 0,
            "mean_tokens": round(sum(tok_counts) / max(1, len(tok_counts)), 2),
            "oversized_split_chunks": sum(1 for ch in chunks if ch.is_oversized_split),
        }

    manifest = {
        "source_pages_jsonl": str(extracted_pages_jsonl),
        "total_source_pages": len(pages),
        "boundary_context_tokens": boundary_context_tokens,
        "configurations": manifest_configs,
    }
    manifest_path = output_dir / "chunks_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def load_chunks_jsonl(chunks_path: Path) -> List[ChunkRecord]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks JSONL file not found: {chunks_path}")
    records: List[ChunkRecord] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(ChunkRecord(**json.loads(line)))
    return records