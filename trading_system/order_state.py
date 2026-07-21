from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from trading_system.errors import StateError, classify_exception
from trading_system.persistence import atomic_write_json, file_lock, load_json


class OrderState(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    ACKED = "acked"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    UNKNOWN_PENDING_RECONCILE = "unknown_pending_reconcile"
    RECONCILED_FILLED = "reconciled_filled"
    RECONCILED_NOT_FILLED = "reconciled_not_filled"
    EXPIRED = "expired"


TERMINAL_STATES = {
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.REJECTED,
    OrderState.FAILED,
    OrderState.RECONCILED_FILLED,
    OrderState.RECONCILED_NOT_FILLED,
    OrderState.EXPIRED,
}

ALLOWED_TRANSITIONS = {
    OrderState.CREATED: {
        OrderState.SUBMITTED,
        OrderState.REJECTED,
        OrderState.FAILED,
        OrderState.EXPIRED,
        OrderState.RECONCILED_FILLED,
        OrderState.RECONCILED_NOT_FILLED,
    },
    OrderState.SUBMITTED: {
        OrderState.ACKED,
        OrderState.PARTIAL_FILLED,
        OrderState.FILLED,
        OrderState.REJECTED,
        OrderState.FAILED,
        OrderState.UNKNOWN_PENDING_RECONCILE,
        OrderState.CANCEL_PENDING,
    },
    OrderState.ACKED: {
        OrderState.PARTIAL_FILLED,
        OrderState.FILLED,
        OrderState.REJECTED,
        OrderState.FAILED,
        OrderState.UNKNOWN_PENDING_RECONCILE,
        OrderState.CANCEL_PENDING,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIAL_FILLED: {
        OrderState.FILLED,
        OrderState.CANCEL_PENDING,
        OrderState.CANCELLED,
        OrderState.UNKNOWN_PENDING_RECONCILE,
        OrderState.FAILED,
        OrderState.EXPIRED,
    },
    OrderState.CANCEL_PENDING: {
        OrderState.CANCELLED,
        OrderState.PARTIAL_FILLED,
        OrderState.FILLED,
        OrderState.REJECTED,
        OrderState.UNKNOWN_PENDING_RECONCILE,
        OrderState.FAILED,
    },
    OrderState.UNKNOWN_PENDING_RECONCILE: {
        OrderState.RECONCILED_FILLED,
        OrderState.RECONCILED_NOT_FILLED,
        OrderState.FAILED,
        OrderState.EXPIRED,
    },
}


@dataclass(frozen=True, slots=True)
class OrderTransition:
    from_state: str
    to_state: str
    event: str
    occurred_at: str
    idempotent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OrderStateProjection:
    order_key: str
    current_state: str
    terminal: bool
    transitions: tuple[OrderTransition, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_key": self.order_key,
            "current_state": self.current_state,
            "terminal": self.terminal,
            "transitions": [item.to_dict() for item in self.transitions],
        }


class OrderStateMachine:
    def __init__(self, *, initial_state: OrderState = OrderState.CREATED) -> None:
        self.state = initial_state
        self.transitions: list[OrderTransition] = []

    def transition(self, target: OrderState, event: str, *, occurred_at: str = "") -> OrderTransition:
        timestamp = str(occurred_at or datetime.now(timezone.utc).isoformat())
        if target == self.state:
            transition = OrderTransition(self.state.value, target.value, str(event), timestamp, idempotent=True)
            self.transitions.append(transition)
            return transition
        allowed = ALLOWED_TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise StateError(
                f"invalid order state transition: {self.state.value} -> {target.value}",
                code="invalid_order_state_transition",
            )
        transition = OrderTransition(self.state.value, target.value, str(event), timestamp)
        self.state = target
        self.transitions.append(transition)
        return transition


ORDER_STATE_SCHEMA_VERSION = 1
PENDING_RECOVERY_STATES = {
    OrderState.SUBMITTED,
    OrderState.PARTIAL_FILLED,
    OrderState.CANCEL_PENDING,
    OrderState.UNKNOWN_PENDING_RECONCILE,
}


class OrderStateStore:
    """Durable sidecar for order lifecycles.

    The sidecar never decides whether an order should be placed. It records the
    existing execution path and keeps ambiguous outcomes available for exchange
    reconciliation after a process restart.
    """

    def __init__(self, path: Path, *, max_records: int = 5000) -> None:
        self.path = Path(path)
        self.max_records = max(100, int(max_records))

    @staticmethod
    def _empty_payload() -> dict[str, Any]:
        return {
            "schema_version": ORDER_STATE_SCHEMA_VERSION,
            "updated_at": "",
            "orders": {},
        }

    def _load_unlocked(self) -> dict[str, Any]:
        payload = load_json(self.path, self._empty_payload(), lock=False)
        if not isinstance(payload, dict):
            payload = self._empty_payload()
        orders = payload.get("orders")
        if not isinstance(orders, dict):
            orders = {}
        payload["schema_version"] = ORDER_STATE_SCHEMA_VERSION
        payload["orders"] = orders
        return payload

    @staticmethod
    def _copy(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _prune(self, payload: dict[str, Any]) -> None:
        orders = payload.get("orders") or {}
        if len(orders) <= self.max_records:
            return
        unresolved = {
            key: value
            for key, value in orders.items()
            if str(value.get("current_state") or "") in {state.value for state in PENDING_RECOVERY_STATES}
        }
        terminal = [
            (key, value)
            for key, value in orders.items()
            if key not in unresolved
        ]
        terminal.sort(key=lambda item: str(item[1].get("updated_at") or ""), reverse=True)
        remaining = max(0, self.max_records - len(unresolved))
        payload["orders"] = {**unresolved, **dict(terminal[:remaining])}

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = self._now()
        self._prune(payload)
        atomic_write_json(self.path, payload, lock=False)

    @staticmethod
    def _transition_record(record: dict[str, Any], target: OrderState, event: str, *, occurred_at: str = "") -> None:
        current = OrderState(str(record.get("current_state") or OrderState.CREATED.value))
        machine = OrderStateMachine(initial_state=current)
        transition = machine.transition(target, event, occurred_at=occurred_at)
        transitions = list(record.get("transitions") or [])
        transitions.append(transition.to_dict())
        record["current_state"] = machine.state.value
        record["terminal"] = machine.state in TERMINAL_STATES
        record["transitions"] = transitions
        record["updated_at"] = transition.occurred_at

    def begin_order(
        self,
        *,
        symbol: str,
        action: str,
        request: Mapping[str, Any] | None = None,
        order_key: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = str(order_key or f"attempt:{uuid4().hex}")[:160]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path):
            payload = self._load_unlocked()
            orders = payload["orders"]
            existing = orders.get(key)
            if isinstance(existing, dict):
                result = self._copy(existing)
                result["duplicate"] = True
                return result
            now = self._now()
            record = {
                "order_key": key,
                "exchange_order_id": "",
                "client_order_id": "",
                "symbol": str(symbol or "").upper()[:40],
                "action": str(action or "")[:80],
                "current_state": OrderState.CREATED.value,
                "terminal": False,
                "created_at": now,
                "updated_at": now,
                "request": self._copy(dict(request or {})),
                "response": {},
                "last_error": {},
                "metadata": self._copy(dict(metadata or {})),
                "transitions": [],
            }
            orders[key] = record
            self._write_unlocked(payload)
            return self._copy(record)

    def get(self, order_key: str) -> dict[str, Any] | None:
        with file_lock(self.path):
            payload = self._load_unlocked()
            record = payload["orders"].get(str(order_key))
            return self._copy(record) if isinstance(record, dict) else None

    def transition(self, order_key: str, target: OrderState, event: str, *, occurred_at: str = "") -> dict[str, Any]:
        with file_lock(self.path):
            payload = self._load_unlocked()
            record = payload["orders"].get(str(order_key))
            if not isinstance(record, dict):
                raise StateError(f"order state record not found: {order_key}", code="order_state_record_missing")
            self._transition_record(record, target, event, occurred_at=occurred_at)
            self._write_unlocked(payload)
            return self._copy(record)

    def mark_submitted(self, order_key: str, *, occurred_at: str = "") -> dict[str, Any]:
        return self.transition(order_key, OrderState.SUBMITTED, "submission_attempted", occurred_at=occurred_at)

    def apply_exchange_response(
        self,
        order_key: str,
        response: Mapping[str, Any] | None,
        *,
        occurred_at: str = "",
        force_state: OrderState | None = None,
    ) -> dict[str, Any]:
        response_dict = dict(response or {})
        target = force_state or _fill_state({"response": response_dict})
        timestamp = str(occurred_at or self._now())
        with file_lock(self.path):
            payload = self._load_unlocked()
            record = payload["orders"].get(str(order_key))
            if not isinstance(record, dict):
                raise StateError(f"order state record not found: {order_key}", code="order_state_record_missing")
            current = OrderState(str(record.get("current_state") or OrderState.CREATED.value))
            if current == OrderState.CREATED:
                self._transition_record(record, OrderState.SUBMITTED, "submission_attempted", occurred_at=timestamp)
                current = OrderState.SUBMITTED
            if target in {OrderState.PARTIAL_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED}:
                if current == OrderState.SUBMITTED:
                    self._transition_record(record, OrderState.ACKED, "exchange_acknowledged", occurred_at=timestamp)
                    current = OrderState.ACKED
                if target == OrderState.CANCELLED and current != OrderState.CANCEL_PENDING:
                    self._transition_record(record, OrderState.CANCEL_PENDING, "exchange_cancel_pending", occurred_at=timestamp)
            self._transition_record(record, target, f"exchange_status:{_exchange_status({'response': response_dict}) or target.value}", occurred_at=timestamp)
            record["response"] = self._copy(response_dict)
            exchange_id = response_dict.get("orderId") or response_dict.get("algoId")
            client_id = response_dict.get("clientOrderId")
            if exchange_id not in (None, ""):
                record["exchange_order_id"] = str(exchange_id)[:160]
            if client_id not in (None, ""):
                record["client_order_id"] = str(client_id)[:160]
            self._write_unlocked(payload)
            return self._copy(record)

    def mark_submission_exception(self, order_key: str, exc: BaseException, *, operation: str) -> dict[str, Any]:
        error = classify_exception(exc, operation=operation)
        message = error.message.lower()
        explicit_rejection = error.code in {
            "invalid_symbol",
            "reduce_only_rejected",
        } or (" 400:" in f" {message}" and not error.ambiguous_order_status)
        transport_ambiguous = error.ambiguous_order_status or error.code in {
            "network_timeout",
            "dns_failure",
            "tls_failure",
        } or any(token in message for token in ("502 bad gateway", "503", "504", "connection aborted"))
        target = (
            OrderState.REJECTED
            if explicit_rejection
            else OrderState.UNKNOWN_PENDING_RECONCILE
            if transport_ambiguous
            else OrderState.FAILED
        )
        with file_lock(self.path):
            payload = self._load_unlocked()
            record = payload["orders"].get(str(order_key))
            if not isinstance(record, dict):
                raise StateError(f"order state record not found: {order_key}", code="order_state_record_missing")
            if OrderState(str(record.get("current_state") or OrderState.CREATED.value)) == OrderState.CREATED:
                self._transition_record(record, OrderState.SUBMITTED, "submission_attempted")
            self._transition_record(record, target, f"submission_exception:{error.code}")
            record["last_error"] = error.to_dict()
            self._write_unlocked(payload)
            return self._copy(record)

    def ensure_external_order(self, *, symbol: str, order_id: int | str, action: str = "cancel_order") -> dict[str, Any]:
        key = f"exchange:{str(symbol).upper()}:{order_id}"[:160]
        record = self.begin_order(symbol=symbol, action=action, order_key=key, metadata={"external_order": True})
        state = OrderState(str(record.get("current_state") or OrderState.CREATED.value))
        if state == OrderState.CREATED:
            record = self.mark_submitted(key)
            record = self.transition(key, OrderState.ACKED, "external_order_observed")
        return record

    def mark_cancel_pending(self, order_key: str) -> dict[str, Any]:
        return self.transition(order_key, OrderState.CANCEL_PENDING, "cancel_requested")

    def reconcile(self, order_key: str, *, filled: bool, event: str = "exchange_reconciled") -> dict[str, Any]:
        target = OrderState.RECONCILED_FILLED if filled else OrderState.RECONCILED_NOT_FILLED
        return self.transition(order_key, target, event)

    def pending_reconciliation(self) -> list[dict[str, Any]]:
        with file_lock(self.path):
            payload = self._load_unlocked()
            pending = [
                self._copy(record)
                for record in payload["orders"].values()
                if isinstance(record, dict)
                and str(record.get("current_state") or "") in {state.value for state in PENDING_RECOVERY_STATES}
            ]
        pending.sort(key=lambda item: str(item.get("updated_at") or ""))
        return pending


def _order_key(row: Mapping[str, Any]) -> str:
    response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
    intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else {}
    explicit = row.get("order_id") or row.get("client_order_id") or response.get("orderId") or response.get("clientOrderId")
    if explicit:
        return str(explicit)[:160]
    identity = {
        "symbol": row.get("symbol") or intent.get("symbol"),
        "action": row.get("action") or intent.get("action"),
        "submitted_at": row.get("submitted_at"),
        "generated_at": intent.get("generated_at"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"derived:{hashlib.sha256(encoded).hexdigest()[:24]}"


def _exchange_status(row: Mapping[str, Any]) -> str:
    response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
    return str(response.get("status") or row.get("exchange_status") or "").upper()


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fill_state(row: Mapping[str, Any]) -> OrderState:
    response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
    exchange_status = _exchange_status(row)
    if exchange_status == "FILLED":
        return OrderState.FILLED
    if exchange_status in {"PARTIALLY_FILLED", "PARTIAL_FILLED"}:
        return OrderState.PARTIAL_FILLED
    if exchange_status in {"CANCELED", "CANCELLED"}:
        return OrderState.CANCELLED
    if exchange_status == "EXPIRED":
        return OrderState.EXPIRED
    if exchange_status == "REJECTED":
        return OrderState.REJECTED
    executed = _float(response.get("executedQty") or row.get("executed_quantity") or row.get("executed_qty"))
    original = _float(response.get("origQty") or row.get("original_quantity") or row.get("quantity"))
    if executed > 0 and original > executed:
        return OrderState.PARTIAL_FILLED
    if executed > 0:
        return OrderState.FILLED
    return OrderState.ACKED


def _reconcile_found_expected_position(row: Mapping[str, Any]) -> bool:
    reconcile = row.get("reconcile") if isinstance(row.get("reconcile"), Mapping) else {}
    intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else {}
    symbol = str(row.get("symbol") or intent.get("symbol") or "").upper()
    action = str(row.get("action") or intent.get("action") or "")
    side = "short" if action == "open_short" else "long"
    expected = f"{symbol}:{side}".upper()
    return expected in {str(item).upper() for item in reconcile.get("actual_positions") or []}


def project_order_result(row: Mapping[str, Any]) -> OrderStateProjection:
    """Project a legacy V6 result into an auditable state path.

    This function is intentionally read-only. The exchange and existing V6
    reconciliation code remain authoritative for execution and local state.
    """

    machine = OrderStateMachine()
    status = str(row.get("status") or "").lower()
    occurred_at = str(row.get("submitted_at") or "")
    error = str(row.get("original_error") or row.get("error") or "").lower()
    submitted_statuses = {"submitted", "submitted_unprotected", "closed_after_protective_stop_failed"}
    if status in {"planned", "dry_run", "skipped_protective_stop_immediate_trigger", "close_waiting_for_protective_stop_cancel"}:
        return OrderStateProjection(_order_key(row), machine.state.value, False, tuple(machine.transitions))
    if status in submitted_statuses:
        machine.transition(OrderState.SUBMITTED, "submission_attempted", occurred_at=occurred_at)
        target = _fill_state(row)
        if target in {OrderState.PARTIAL_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED}:
            machine.transition(OrderState.ACKED, "exchange_acknowledged", occurred_at=occurred_at)
        if target == OrderState.CANCELLED:
            machine.transition(OrderState.CANCEL_PENDING, "exchange_cancel_pending", occurred_at=occurred_at)
        machine.transition(target, f"exchange_status:{_exchange_status(row) or 'acknowledged'}", occurred_at=occurred_at)
    elif status == "submitted_no_position_delta":
        machine.transition(OrderState.SUBMITTED, "submission_attempted", occurred_at=occurred_at)
        machine.transition(OrderState.UNKNOWN_PENDING_RECONCILE, "position_delta_not_confirmed", occurred_at=occurred_at)
    elif status in {"status_unknown_position_still_open", "status_unknown_closed_reconciled"}:
        machine.transition(OrderState.SUBMITTED, "submission_attempted", occurred_at=occurred_at)
        machine.transition(OrderState.UNKNOWN_PENDING_RECONCILE, "exchange_status_unknown", occurred_at=occurred_at)
        action = str((row.get("intent") or {}).get("action") or "") if isinstance(row.get("intent"), Mapping) else ""
        reconciled_filled = (
            status == "status_unknown_closed_reconciled" and action in {"close_long", "close_short"}
        ) or (
            status == "status_unknown_position_still_open" and action in {"open_long", "open_short"}
        )
        machine.transition(
            OrderState.RECONCILED_FILLED if reconciled_filled else OrderState.RECONCILED_NOT_FILLED,
            "exchange_position_reconciled",
            occurred_at=occurred_at,
        )
    elif status == "status_unknown_no_position":
        machine.transition(OrderState.SUBMITTED, "submission_attempted", occurred_at=occurred_at)
        machine.transition(OrderState.UNKNOWN_PENDING_RECONCILE, "exchange_status_unknown", occurred_at=occurred_at)
        machine.transition(OrderState.RECONCILED_NOT_FILLED, "exchange_position_absent", occurred_at=occurred_at)
    elif status == "failed_open_reconciled":
        machine.transition(OrderState.SUBMITTED, "submission_attempted", occurred_at=occurred_at)
        machine.transition(OrderState.UNKNOWN_PENDING_RECONCILE, "submission_exception", occurred_at=occurred_at)
        reconciled = OrderState.RECONCILED_FILLED if _reconcile_found_expected_position(row) else OrderState.RECONCILED_NOT_FILLED
        machine.transition(reconciled, "exchange_position_reconciled", occurred_at=occurred_at)
    elif status in {"reconciled_missing_exchange_position", "skipped_zero_partial_close_quantity"}:
        machine.transition(OrderState.RECONCILED_NOT_FILLED, "exchange_position_absent", occurred_at=occurred_at)
    elif "-1007" in error or "status unknown" in error or "timeout waiting for response" in error:
        machine.transition(OrderState.SUBMITTED, "submission_attempted", occurred_at=occurred_at)
        machine.transition(OrderState.UNKNOWN_PENDING_RECONCILE, "exchange_status_unknown", occurred_at=occurred_at)
    elif status in {"reduce_only_rejected_reconciled", "skipped_percent_price_retry_failed", "failed"} and (
        "rejected" in error or "-2022" in error or "-4131" in error
    ):
        machine.transition(OrderState.REJECTED, "exchange_rejected", occurred_at=occurred_at)
    elif status:
        machine.transition(OrderState.FAILED, f"legacy_status:{status}", occurred_at=occurred_at)
    return OrderStateProjection(
        order_key=_order_key(row),
        current_state=machine.state.value,
        terminal=machine.state in TERMINAL_STATES,
        transitions=tuple(machine.transitions),
    )
