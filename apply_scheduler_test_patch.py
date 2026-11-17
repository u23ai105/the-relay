"""Apply the safe mid-stream-disconnect assertion update to tests/test_scheduler.py.

Run this script from the The Relay project root. It preserves a timestamped backup
and changes only the stale assertion that expects incomplete stream text to be returned.
"""
from datetime import datetime
from pathlib import Path
import sys

path = Path("tests/test_scheduler.py")
if not path.is_file():
    raise SystemExit(f"Could not find {path}. Run this script from the project root.")

source = path.read_text(encoding="utf-8")
old = '    assert "truncated" in res.content'
new = '''    # Incomplete provider output is deliberately discarded after a mid-stream disconnect.
    assert res.content == ""
    assert res.failure_reason == "disconnect_mid_stream"
    assert res.accounting_mode == "partial_estimated"
    assert res.tokens_out > 0'''

if old not in source:
    already_patched = all(
        marker in source
        for marker in (
            'assert res.content == ""',
            'assert res.failure_reason == "disconnect_mid_stream"',
            'assert res.accounting_mode == "partial_estimated"',
        )
    )
    if already_patched:
        print(f"{path} already contains the safe disconnect assertions; no change made.")
        raise SystemExit(0)
    raise SystemExit(
        f"Expected stale assertion was not found in {path}. No file was changed. "
        "Inspect test_midstream_disconnect_retry manually rather than applying a broad replacement."
    )

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = path.with_name(f"{path.name}.{stamp}.bak")
backup.write_text(source, encoding="utf-8")
path.write_text(source.replace(old, new, 1), encoding="utf-8")
print(f"Updated {path}")
print(f"Original preserved at {backup}")
