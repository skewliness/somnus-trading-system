from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping

from trading_system.redaction import redact_sensitive_text


EVENT_SCHEMA_VERSION = 1
VALID_EVENT_TYPES = {"signal", "order", "fill", "position", "exit"}
SENSITIVE_KEY_PARTS = ("api_key", "secret", "token", "cookie", "authorization", "password", "signature")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(part in lowered for part in SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(item_key): _sanitize(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)[:2000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_sensitive_text(value)[:2000]


def _compact_identifier(value: Any) -> str:
    return str(value or "").strip()[:160]


@dataclass(frozen=True, slots=True)
class SignalEvent:
    symbol: str
    computed_at: str
    side_bias: str
    price: float | None
    long_probability: float | None
    short_probability: float | None
    wait_probability: float | None
    model_quality_score: float | None

    @classmethod
    def from_summary(cls, row: Mapping[str, Any]) -> "SignalEvent":
        return cls(
            symbol=_compact_identifier(row.get("symbol")),
            computed_at=_compact_identifier(row.get("computed_at")),
            side_bias=_compact_identifier(row.get("side_bias")),
            price=row.get("price"),
            long_probability=row.get("long_probability"),
            short_probability=row.get("short_probability"),
            wait_probability=row.get("wait_probability"),
            model_quality_score=row.get("model_quality_score"),
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    symbol: str
    action: str
    side: str
    generated_at: str
    quantity_hint_usdt: float | None
    leverage: int | None
    reduce_only: bool
    entry_price: float | None
    stop_loss_price: float | None
    take_profit_price: float | None
    close_fraction: float | None
    target_name: str
    rationale: str

    @classmethod
    def from_summary(cls, row: Mapping[str, Any]) -> "OrderEvent":
        return cls(
            symbol=_compact_identifier(row.get("symbol")),
            action=_compact_identifier(row.get("action")),
            side=_compact_identifier(row.get("side")),
            generated_at=_compact_identifier(row.get("generated_at")),
            quantity_hint_usdt=row.get("quantity_hint_usdt"),
            leverage=row.get("leverage"),
            reduce_only=bool(row.get("reduce_only")),
            entry_price=row.get("entry_price"),
            stop_loss_price=row.get("stop_loss_price"),
            take_profit_price=row.get("take_profit_price"),
            close_fraction=row.get("close_fraction"),
            target_name=_compact_identifier(row.get("target_name")),
            rationale=redact_sensitive_text(row.get("rationale") or row.get("reason") or "")[:1200],
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FillEvent:
    symbol: str
    action: str
    status: str
    order_id: str
    client_order_id: str
    executed_quantity: float | None
    average_price: float | None
    error: str

    @classmethod
    def from_summary(cls, row: Mapping[str, Any]) -> "FillEvent":
        intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else {}
        return cls(
            symbol=_compact_identifier(row.get("symbol") or intent.get("symbol")),
            action=_compact_identifier(row.get("action") or intent.get("action")),
            status=_compact_identifier(row.get("status")),
            order_id=_compact_identifier(row.get("order_id")),
            client_order_id=_compact_identifier(row.get("client_order_id")),
            executed_quantity=row.get("executed_quantity") or row.get("executed_qty") or row.get("quantity"),
            average_price=row.get("average_price") or row.get("avg_price") or row.get("price"),
            error=redact_sensitive_text(row.get("error") or "")[:1200],
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PositionEvent:
    symbol: str
    side: str
    quantity: float | None
    entry_price: float | None
    current_price: float | None
    horizon: str
    entry_timeframe: str

    @classmethod
    def from_summary(cls, row: Mapping[str, Any]) -> "PositionEvent":
        return cls(
            symbol=_compact_identifier(row.get("symbol")),
            side=_compact_identifier(row.get("side")),
            quantity=row.get("position_qty") or row.get("remaining_quantity") or row.get("quantity"),
            entry_price=row.get("entry_price"),
            current_price=row.get("current_price") or row.get("mark_price"),
            horizon=_compact_identifier(row.get("horizon")),
            entry_timeframe=_compact_identifier(row.get("entry_timeframe")),
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExitEvent:
    symbol: str
    action: str
    stage: str
    status: str
    order_id: str
    close_fraction: float | None
    target_name: str
    reason: str

    @classmethod
    def from_intent(cls, row: Mapping[str, Any]) -> "ExitEvent":
        return cls(
            symbol=_compact_identifier(row.get("symbol")),
            action=_compact_identifier(row.get("action")),
            stage="intent_created",
            status="created",
            order_id="",
            close_fraction=row.get("close_fraction"),
            target_name=_compact_identifier(row.get("target_name")),
            reason=redact_sensitive_text(row.get("rationale") or row.get("reason") or "")[:1200],
        )

    @classmethod
    def from_result(cls, row: Mapping[str, Any]) -> "ExitEvent":
        intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else {}
        return cls(
            symbol=_compact_identifier(row.get("symbol") or intent.get("symbol")),
            action=_compact_identifier(row.get("action") or intent.get("action")),
            stage="execution_result",
            status=_compact_identifier(row.get("status")),
            order_id=_compact_identifier(row.get("order_id") or row.get("client_order_id")),
            close_fraction=intent.get("close_fraction") or row.get("close_fraction"),
            target_name=_compact_identifier(intent.get("target_name") or row.get("target_name")),
            reason=redact_sensitive_text(intent.get("rationale") or row.get("reason") or row.get("error") or "")[:1200],
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    schema_version: int
    event_id: str
    event_type: str
    occurred_at: str
    emitted_at: str
    account_id: str
    runtime_mode: str
    source: str
    strategy_version: str
    payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        account_id: str,
        runtime_mode: str,
        source: str,
        strategy_version: str,
        occurred_at: str = "",
    ) -> "EventEnvelope":
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"unsupported event type: {event_type}")
        safe_payload = _sanitize(dict(payload))
        timestamp = _compact_identifier(occurred_at) or _utc_now_iso()
        identity = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_type": event_type,
            "occurred_at": timestamp,
            "account_id": _compact_identifier(account_id),
            "runtime_mode": _compact_identifier(runtime_mode),
            "source": _compact_identifier(source),
            "strategy_version": _compact_identifier(strategy_version),
            "payload": safe_payload,
        }
        encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return cls(
            event_id=hashlib.sha256(encoded).hexdigest(),
            emitted_at=_utc_now_iso(),
            **identity,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
