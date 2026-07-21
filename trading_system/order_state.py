from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from trading_system.errors import StateError


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
        machine.transition(OrderState.RECONCILED_FILLED, "exchange_position_reconciled", occurred_at=occurred_at)
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
