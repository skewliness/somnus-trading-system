from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None, *, json_logs: bool | None = None) -> None:
    resolved_level = (level or os.getenv("TS_LOG_LEVEL", "INFO")).upper()
    use_json = json_logs if json_logs is not None else os.getenv("TS_LOG_FORMAT", "json").lower() == "json"
    handler = logging.StreamHandler(sys.stderr)
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=getattr(logging, resolved_level, logging.INFO), handlers=[handler], force=True)
