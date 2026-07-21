from __future__ import annotations

import copy

import pytest

from trading_system.errors import StateError, V6ExecutionError
from trading_system.order_state import OrderState, OrderStateMachine, OrderStateStore, project_order_result


def test_filled_exchange_result_projects_full_lifecycle_without_mutation() -> None:
    result = {
        "status": "submitted",
        "submitted_at": "2026-07-21T00:00:00+00:00",
        "intent": {"symbol": "BTCUSDT", "action": "open_long"},
        "response": {"orderId": 123, "status": "FILLED", "executedQty": "0.01", "origQty": "0.01"},
    }
    before = copy.deepcopy(result)

    projection = project_order_result(result)

    assert result == before
    assert projection.order_key == "123"
    assert projection.current_state == "filled"
    assert projection.terminal is True
    assert [item.to_state for item in projection.transitions] == ["submitted", "acked", "filled"]


def test_partial_fill_remains_non_terminal() -> None:
    projection = project_order_result(
        {
            "status": "submitted",
            "intent": {"symbol": "ETHUSDT", "action": "close_long"},
            "response": {"status": "PARTIALLY_FILLED", "executedQty": "0.05", "origQty": "0.10"},
        }
    )

    assert projection.current_state == "partial_filled"
    assert projection.terminal is False


def test_timeout_waits_for_reconcile_instead_of_assuming_failure() -> None:
    projection = project_order_result(
        {
            "status": "failed",
            "intent": {"symbol": "SOLUSDT", "action": "open_short"},
            "error": 'Binance 408: {"code":-1007,"msg":"status unknown"}',
        }
    )

    assert projection.current_state == "unknown_pending_reconcile"
    assert projection.terminal is False


def test_reconcile_is_exchange_authoritative_for_ambiguous_open() -> None:
    filled = project_order_result(
        {
            "status": "failed_open_reconciled",
            "intent": {"symbol": "HYPEUSDT", "action": "open_short"},
            "reconcile": {"actual_positions": ["HYPEUSDT:SHORT"]},
        }
    )
    absent = project_order_result(
        {
            "status": "status_unknown_no_position",
            "intent": {"symbol": "HYPEUSDT", "action": "open_short"},
        }
    )

    assert filled.current_state == "reconciled_filled"
    assert absent.current_state == "reconciled_not_filled"


def test_duplicate_transition_is_idempotent_and_invalid_transition_is_rejected() -> None:
    machine = OrderStateMachine()
    machine.transition(OrderState.SUBMITTED, "submit")
    duplicate = machine.transition(OrderState.SUBMITTED, "duplicate callback")

    assert duplicate.idempotent is True
    with pytest.raises(StateError, match="invalid order state transition"):
        machine.transition(OrderState.RECONCILED_FILLED, "skipped reconcile")


def test_cancel_and_expiry_paths_are_supported() -> None:
    cancelled = project_order_result(
        {
            "status": "submitted",
            "intent": {"symbol": "BNBUSDT", "action": "open_long"},
            "response": {"status": "CANCELED", "orderId": 8},
        }
    )
    expired = project_order_result(
        {
            "status": "submitted",
            "intent": {"symbol": "BNBUSDT", "action": "open_long"},
            "response": {"status": "EXPIRED", "orderId": 9},
        }
    )

    assert cancelled.current_state == "cancelled"
    assert expired.current_state == "expired"


def test_duplicate_begin_is_idempotent_and_does_not_add_transitions(tmp_path) -> None:
    store = OrderStateStore(tmp_path / "orders.json")
    first = store.begin_order(symbol="BTCUSDT", action="open_long", order_key="intent-1")
    duplicate = store.begin_order(symbol="BTCUSDT", action="open_long", order_key="intent-1")

    assert first["current_state"] == "created"
    assert duplicate["duplicate"] is True
    assert duplicate["transitions"] == []
    assert len(store.pending_reconciliation()) == 0


def test_partial_fill_survives_restart_and_can_finish(tmp_path) -> None:
    path = tmp_path / "orders.json"
    first_process = OrderStateStore(path)
    created = first_process.begin_order(symbol="ETHUSDT", action="open_long")
    first_process.mark_submitted(created["order_key"])
    partial = first_process.apply_exchange_response(
        created["order_key"],
        {"orderId": 91, "status": "PARTIALLY_FILLED", "executedQty": "0.5", "origQty": "1"},
    )

    second_process = OrderStateStore(path)
    restored = second_process.get(created["order_key"])
    completed = second_process.apply_exchange_response(
        created["order_key"],
        {"orderId": 91, "status": "FILLED", "executedQty": "1", "origQty": "1"},
    )

    assert partial["current_state"] == "partial_filled"
    assert restored is not None and restored["current_state"] == "partial_filled"
    assert completed["current_state"] == "filled"
    assert completed["terminal"] is True


def test_timeout_survives_restart_until_exchange_reconciliation(tmp_path) -> None:
    path = tmp_path / "orders.json"
    store = OrderStateStore(path)
    created = store.begin_order(symbol="SOLUSDT", action="open_short")
    store.mark_submitted(created["order_key"])
    unknown = store.mark_submission_exception(
        created["order_key"],
        V6ExecutionError('Binance 408: {"code":-1007,"msg":"status unknown"}'),
        operation="POST /fapi/v1/order",
    )

    restarted = OrderStateStore(path)
    pending = restarted.pending_reconciliation()
    reconciled = restarted.reconcile(created["order_key"], filled=False)

    assert unknown["current_state"] == "unknown_pending_reconcile"
    assert [item["order_key"] for item in pending] == [created["order_key"]]
    assert reconciled["current_state"] == "reconciled_not_filled"
    assert restarted.pending_reconciliation() == []


def test_cancel_rejection_is_persisted_without_losing_exchange_identity(tmp_path) -> None:
    store = OrderStateStore(tmp_path / "orders.json")
    observed = store.ensure_external_order(symbol="HYPEUSDT", order_id=712)
    store.mark_cancel_pending(observed["order_key"])
    rejected = store.mark_submission_exception(
        observed["order_key"],
        V6ExecutionError('Binance signed request failed 400: {"code":-2011,"msg":"Unknown order"}'),
        operation="DELETE /fapi/v1/algoOrder",
    )

    assert rejected["current_state"] == "rejected"
    assert rejected["terminal"] is True
    assert rejected["order_key"] == "exchange:HYPEUSDT:712"


def test_status_unknown_close_projection_uses_exchange_position_as_authority() -> None:
    still_open = project_order_result(
        {
            "status": "status_unknown_position_still_open",
            "intent": {"symbol": "BTCUSDT", "action": "close_short"},
        }
    )
    closed = project_order_result(
        {
            "status": "status_unknown_closed_reconciled",
            "intent": {"symbol": "BTCUSDT", "action": "close_short"},
        }
    )

    assert still_open.current_state == "reconciled_not_filled"
    assert closed.current_state == "reconciled_filled"
