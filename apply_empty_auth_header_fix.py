"""Omit the Authorization header when the client has no API key.

Run from the The Relay project root. This is required for unauthenticated local
stand-in/test servers: `Authorization: Bearer ` is an invalid HTTP header value.
The real Groq client still rejects a missing key before making a request.
"""
from datetime import datetime
from pathlib import Path

path = Path("src/groq_client.py")
if not path.is_file():
    raise SystemExit(f"Could not find {path}. Run this script from the project root.")

source = path.read_text(encoding="utf-8")
old = '''        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Relay-Stage": stage,
        }
        if fault_inject:
            headers["X-Relay-Fault"] = fault_inject'''
new = '''        headers = {
            "Content-Type": "application/json",
            "X-Relay-Stage": stage,
        }
        # The local stand-in does not require authentication. Never emit an
        # empty `Bearer ` value: HTTPX rejects it before the request is sent.
        if self.api_key and self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"
        if fault_inject:
            headers["X-Relay-Fault"] = fault_inject'''

if old not in source:
    already_fixed = (
        '"Authorization": f"Bearer {self.api_key}"' not in source
        and 'if self.api_key and self.api_key.strip():' in source
    )
    if already_fixed:
        print(f"{path} already omits the Authorization header when the API key is empty; no change made.")
        raise SystemExit(0)
    raise SystemExit(
        f"Expected Authorization header block was not found in {path}. No file was changed. "
        "Inspect the current headers block before applying a manual edit."
    )

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = path.with_name(f"{path.name}.{stamp}.bak")
backup.write_text(source, encoding="utf-8")
path.write_text(source.replace(old, new, 1), encoding="utf-8")
print(f"Updated {path}")
print(f"Original preserved at {backup}")
