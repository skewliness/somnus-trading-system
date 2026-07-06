from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class GateEventRecorder:
    """Append-only recorder for strategy gate decisions.

    It is disabled by default so production behavior and IO remain unchanged
    unless an operator explicitly enables the recorder.
    """

    def __init__(self, path: Path, *, enabled: bool = False) -> None:
        self.path = path
        self.enabled = enabled

    @classmethod
    def from_env(cls, root: Path | None = None) -> "GateEventRecorder":
        enabled = _env_bool("TS_V6_GATE_EVENT_RECORDER_ENABLED", False)
        configured = os.getenv("TS_V6_GATE_EVENT_FILE", "data/trading_system/v6_execution/gate_events.jsonl")
        path = Path(configured)
        if root is not None and not path.is_absolute():
            path = root / path
        return cls(path, enabled=enabled)

    def record_entry_block(self, signal: Any, block: dict[str, Any]) -> None:
        if not self.enabled:
            return
        event = {
            "event_type": "entry_structure_block",
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "symbol": str(getattr(signal, "symbol", "") or ""),
            "side_bias": str(getattr(signal, "side_bias", "") or ""),
            "price": getattr(signal, "price", None),
            "long_probability": getattr(signal, "long_probability", None),
            "short_probability": getattr(signal, "short_probability", None),
            "wait_probability": getattr(signal, "wait_probability", None),
            "computed_at": getattr(signal, "computed_at", None),
            "block": dict(block or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
