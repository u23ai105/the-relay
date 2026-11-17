"""Surgically classify post-TTFT HTTPX read failures as mid-stream disconnects.

Run from the The Relay project root. Creates a timestamped backup before editing.
"""
from datetime import datetime
from pathlib import Path

path = Path("src/groq_client.py")
if not path.is_file():
    raise SystemExit(f"Could not find {path}. Run this script from the project root.")

source = path.read_text(encoding="utf-8")
old = '''            elif isinstance(exc, httpx.RemoteProtocolError):
                f_reason = "disconnect_mid_stream"
                retriable = True
            else:
                f_reason = "network_error"
                retriable = True
'''
new = '''            elif first_token_ms is not None and isinstance(exc, (httpx.ReadError, httpx.RemoteProtocolError)):
                # A transport read failure after SSE content has started is a
                # mid-stream disconnect, even when HTTPX surfaces it as ReadError.
                f_reason = "disconnect_mid_stream"
                retriable = True
            elif isinstance(exc, httpx.RemoteProtocolError):
                # Preserve protocol-disconnect classification even before a token
                # has been decoded from the stream.
                f_reason = "disconnect_mid_stream"
                retriable = True
            else:
                f_reason = "network_error"
                retriable = True
'''

if new in source:
    print(f"{path} already contains the read-error disconnect classification; no change made.")
    raise SystemExit(0)
if source.count(old) != 1:
    raise SystemExit(
        f"Expected exactly one target exception-classification block in {path}; "
        f"found {source.count(old)}. No file was changed. Inspect the current handler manually."
    )

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = path.with_name(f"{path.name}.{stamp}.bak")
backup.write_text(source, encoding="utf-8")
path.write_text(source.replace(old, new, 1), encoding="utf-8")
print(f"Updated {path}")
print(f"Original preserved at {backup}")
