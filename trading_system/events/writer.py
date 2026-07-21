from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from trading_system.errors import log_classified_exception
from trading_system.events.identity import derive_trade_id, extract_trade_id, trade_side
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
        config_version: str = "",
        model_version: str = "",
        config_hash: str = "",
        max_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self.path = path
        self.enabled = bool(enabled)
        self.account_id = str(account_id or "default")
        self.runtime_mode = str(runtime_mode or "dry_run")
        self.source = str(source or "v6_execution")
        self.strategy_version = str(strategy_version or "6.0-baseline")
        self.config_version = str(config_version or "")
        self.model_version = str(model_version or "")
        self.config_hash = str(config_hash or "")
        self.max_bytes = max(1024, int(max_bytes))

    def _envelope(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        occurred_at: str = "",
        trade_id: str = "",
        parent_event_id: str = "",
    ) -> EventEnvelope:
        return EventEnvelope.create(
            event_type,
            payload,
            account_id=self.account_id,
            runtime_mode=self.runtime_mode,
            source=self.source,
            strategy_version=self.strategy_version,
            config_version=self.config_version,
            model_version=self.model_version,
            config_hash=self.config_hash,
            occurred_at=occurred_at,
            trade_id=trade_id,
            parent_event_id=parent_event_id,
        )

    def _derived_trade_id(self, row: Mapping[str, Any], *, seed: str) -> str:
        return derive_trade_id(
            account_id=self.account_id,
            symbol=str(row.get("symbol") or "UNKNOWN"),
            side=trade_side(row),
            seed=seed,
        )

    @staticmethod
    def _intent_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(row.get("symbol") or "").upper(),
            str(row.get("action") or "").lower(),
            str(row.get("generated_at") or ""),
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
        position_trade_ids: dict[tuple[str, str], str] = {}
        position_rows = [row for row in summary.get("positions") or [] if isinstance(row, Mapping)]
        for row in position_rows:
            key = (str(row.get("symbol") or "").upper(), trade_side(row))
            seed = str(row.get("opened_at") or row.get("entry_price") or run_at)
            position_trade_ids[key] = extract_trade_id(row) or self._derived_trade_id(row, seed=seed)

        intent_rows = [row for row in summary.get("intents") or [] if isinstance(row, Mapping)]
        intent_trade_ids: dict[tuple[str, str, str], str] = {}
        intent_trade_ids_by_action: dict[tuple[str, str], str] = {}
        symbol_trade_ids: dict[str, list[str]] = {}
        for row in intent_rows:
            key = self._intent_key(row)
            position_key = (key[0], trade_side(row))
            seed = str(row.get("generated_at") or run_at)
            trade_id = extract_trade_id(row) or position_trade_ids.get(position_key) or self._derived_trade_id(row, seed=seed)
            intent_trade_ids[key] = trade_id
            intent_trade_ids_by_action[(key[0], key[1])] = trade_id
            symbol_trade_ids.setdefault(key[0], []).append(trade_id)

        signal_parent_ids: dict[str, str] = {}
        for row in summary.get("signals") or []:
            if isinstance(row, Mapping):
                item = SignalEvent.from_summary(row)
                linked_ids = list(dict.fromkeys(symbol_trade_ids.get(item.symbol.upper()) or []))
                if not linked_ids:
                    matching_position = position_trade_ids.get((item.symbol.upper(), str(item.side_bias or "").lower()))
                    linked_ids = [
                        matching_position
                        or derive_trade_id(
                            account_id=self.account_id,
                            symbol=item.symbol,
                            side=item.side_bias or "unknown",
                            seed=item.computed_at or run_at,
                        )
                    ]
                for trade_id in linked_ids:
                    event = self._envelope("signal", item.to_payload(), occurred_at=item.computed_at or run_at, trade_id=trade_id)
                    events.append(event)
                    signal_parent_ids[trade_id] = event.event_id

        order_parent_ids: dict[str, str] = {}
        for row in intent_rows:
            item = OrderEvent.from_summary(row)
            trade_id = intent_trade_ids[self._intent_key(row)]
            order_event = self._envelope(
                "order",
                item.to_payload(),
                occurred_at=item.generated_at or run_at,
                trade_id=trade_id,
                parent_event_id=signal_parent_ids.get(trade_id, ""),
            )
            events.append(order_event)
            order_parent_ids[trade_id] = order_event.event_id
            if item.action in {"close_long", "close_short"}:
                exit_item = ExitEvent.from_intent(row)
                events.append(
                    self._envelope(
                        "exit",
                        exit_item.to_payload(),
                        occurred_at=item.generated_at or run_at,
                        trade_id=trade_id,
                        parent_event_id=order_event.event_id,
                    )
                )

        fill_parent_ids: dict[str, str] = {}
        for row in summary.get("results") or []:
            if not isinstance(row, Mapping):
                continue
            item = FillEvent.from_summary(row)
            if not (item.action or item.order_id or item.client_order_id):
                continue
            intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else row
            intent_key = self._intent_key(intent)
            position_key = (str(item.symbol or "").upper(), trade_side(intent))
            trade_id = (
                extract_trade_id(row)
                or intent_trade_ids.get(intent_key)
                or intent_trade_ids_by_action.get((intent_key[0], intent_key[1]))
                or position_trade_ids.get(position_key)
                or self._derived_trade_id(intent, seed=str(intent.get("generated_at") or run_at))
            )
            fill_event = self._envelope(
                "fill",
                item.to_payload(),
                occurred_at=run_at,
                trade_id=trade_id,
                parent_event_id=order_parent_ids.get(trade_id, ""),
            )
            events.append(fill_event)
            fill_parent_ids[trade_id] = fill_event.event_id
            if item.action in {"close_long", "close_short"}:
                exit_item = ExitEvent.from_result(row)
                events.append(
                    self._envelope(
                        "exit",
                        exit_item.to_payload(),
                        occurred_at=run_at,
                        trade_id=trade_id,
                        parent_event_id=fill_event.event_id,
                    )
                )
        for row in position_rows:
            item = PositionEvent.from_summary(row)
            trade_id = position_trade_ids[(item.symbol.upper(), str(item.side or "unknown").lower())]
            events.append(
                self._envelope(
                    "position",
                    item.to_payload(),
                    occurred_at=run_at,
                    trade_id=trade_id,
                    parent_event_id=(
                        fill_parent_ids.get(trade_id)
                        or order_parent_ids.get(trade_id)
                        or signal_parent_ids.get(trade_id, "")
                    ),
                )
            )
        return self._append(events)

    def safe_emit_run_summary(self, summary: Mapping[str, Any]) -> int:
        try:
            return self.emit_run_summary(summary)
        except Exception as exc:
            log_classified_exception(LOGGER, exc, operation="event_writer.emit_run_summary", level=logging.ERROR)
            return 0
