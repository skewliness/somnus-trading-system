from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from trading_system.events.models import EventEnvelope, ExitEvent, FillEvent, OrderEvent, PositionEvent, SignalEvent
from trading_system.persistence import file_lock


LOGGER = logging.getLogger(__name__)


class EventWriter:
    """Optional append-only V6 event stream.

    The writer is a sidecar. It never raises into the execution path through
    ``safe_emit_run_summary`` and it does not replace the existing WAL or state.
    """

    def __init__(
        self,
        path: Path,
        *,
        enabled: bool = False,
        account_id: str = "default",
        runtime_mode: str = "dry_run",
        source: str = "v6_execution",
        strategy_version: str = "6.0-baseline",
        max_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self.path = path
        self.enabled = bool(enabled)
        self.account_id = str(account_id or "default")
        self.runtime_mode = str(runtime_mode or "dry_run")
        self.source = str(source or "v6_execution")
        self.strategy_version = str(strategy_version or "6.0-baseline")
        self.max_bytes = max(1024, int(max_bytes))

    def _envelope(self, event_type: str, payload: Mapping[str, Any], *, occurred_at: str = "") -> EventEnvelope:
        return EventEnvelope.create(
            event_type,
            payload,
            account_id=self.account_id,
            runtime_mode=self.runtime_mode,
            source=self.source,
            strategy_version=self.strategy_version,
            occurred_at=occurred_at,
        )

    def _append(self, events: Iterable[EventEnvelope]) -> int:
        rows = [event.to_dict() for event in events]
        if not self.enabled or not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path):
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                rotated = self.path.with_suffix(self.path.suffix + ".1")
                rotated.unlink(missing_ok=True)
                self.path.replace(rotated)
            with self.path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return len(rows)

    def emit_run_summary(self, summary: Mapping[str, Any]) -> int:
        if not self.enabled:
            return 0
        run_at = str(summary.get("generated_at") or "")
        events: list[EventEnvelope] = []
        for row in summary.get("signals") or []:
            if isinstance(row, Mapping):
                item = SignalEvent.from_summary(row)
                events.append(self._envelope("signal", item.to_payload(), occurred_at=item.computed_at or run_at))
        for row in summary.get("intents") or []:
            if not isinstance(row, Mapping):
                continue
            item = OrderEvent.from_summary(row)
            events.append(self._envelope("order", item.to_payload(), occurred_at=item.generated_at or run_at))
            if item.action in {"close_long", "close_short"}:
                exit_item = ExitEvent.from_intent(row)
                events.append(self._envelope("exit", exit_item.to_payload(), occurred_at=item.generated_at or run_at))
        for row in summary.get("results") or []:
            if not isinstance(row, Mapping):
                continue
            item = FillEvent.from_summary(row)
            if not (item.action or item.order_id or item.client_order_id):
                continue
            events.append(self._envelope("fill", item.to_payload(), occurred_at=run_at))
            if item.action in {"close_long", "close_short"}:
                exit_item = ExitEvent.from_result(row)
                events.append(self._envelope("exit", exit_item.to_payload(), occurred_at=run_at))
        for row in summary.get("positions") or []:
            if isinstance(row, Mapping):
                item = PositionEvent.from_summary(row)
                events.append(self._envelope("position", item.to_payload(), occurred_at=run_at))
        return self._append(events)

    def safe_emit_run_summary(self, summary: Mapping[str, Any]) -> int:
        try:
            return self.emit_run_summary(summary)
        except Exception:
            LOGGER.exception("V6 sidecar event logging failed")
            return 0
