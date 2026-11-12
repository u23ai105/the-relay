import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction import extract_corpus
from src.logger import get_logger


def main() -> None:
    logger = get_logger("extract_corpus")
    papers_dir = PROJECT_ROOT / "data" / "papers"
    manifest_path = papers_dir / "provenance_manifest.json"
    output_dir = PROJECT_ROOT / "data" / "indices"

    logger.info(f"Verifying manifest and extracting PDFs from {papers_dir}...")
    summary = extract_corpus(
        papers_dir=papers_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
    )

    for p in summary["papers"]:
        logger.info(
            f"Extracted {p['paper_id']}{p['arxiv_version']} ({p['filename']}): "
            f"pages={p['total_pages']}, words={p['total_words']}, "
            f"confirmed_headings={p['confirmed_headings_count']}, "
            f"rejected_heading_candidates={p['rejected_heading_candidates_count']}, "
            f"table_captions={p['table_captions_found']}, "
            f"figure_captions={p['figure_captions_found']}, "
            f"math_pages={p['pages_with_math']}"
        )

    logger.info(
        f"Corpus extraction complete: {summary['total_pages_extracted']} total pages written to {summary['output_pages_jsonl']}"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
