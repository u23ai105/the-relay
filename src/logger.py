from datetime import datetime, timezone
import json
import logging
import re
import sys
from typing import Any, Dict

SENSITIVE_KEYS = {"authorization", "api_key", "x-api-key", "groq_api_key"}
KEY_PATTERN = re.compile(r"(gsk_[A-Za-z0-9]{10,}|Bearer\s+[A-Za-z0-9_\-\.]{10,})", re.IGNORECASE)


def redact_value(val: Any) -> Any:
    if isinstance(val, str):
        return KEY_PATTERN.sub("[REDACTED_CREDENTIAL]", val)
    if isinstance(val, dict):
        return redact_dict(val)
    if isinstance(val, (list, tuple)):
        return [redact_value(item) for item in val]
    return val


def redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for k, v in data.items():
        if str(k).lower() in SENSITIVE_KEYS:
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = redact_value(v)
    return sanitized


class RedactingJSONFormatter(logging.Formatter):
    """Formats log records as single-line UTC JSON with automatic credential redaction."""

    def format(self, record: logging.LogRecord) -> str:
        utc_dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        timestamp_utc = utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        raw_message = record.getMessage()
        safe_message = redact_value(raw_message)

        payload: Dict[str, Any] = {
            "timestamp": timestamp_utc,
            "level": record.levelname,
            "logger": record.name,
            "message": safe_message,
        }

        extra_data = getattr(record, "structured_data", None)
        if isinstance(extra_data, dict):
            payload["data"] = redact_dict(extra_data)
        elif extra_data is not None:
            payload["data"] = redact_value(extra_data)

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            payload["exception"] = redact_value(exc_text)

        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(RedactingJSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger