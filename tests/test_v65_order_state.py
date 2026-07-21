from __future__ import annotations

import copy

import pytest

from trading_system.errors import StateError
from trading_system.order_state import OrderState, OrderStateMachine, project_order_result


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
