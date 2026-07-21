from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from typing import Any
from unittest.mock import patch

import scripts.telegram_monitor_v5_v6 as telegram
from scripts.run_v5_position_reconcile_loop import guard_transient_absences, reconcile_once
from scripts.run_v6_ledger_reconcile import run_reconcile as run_v6_ledger_reconcile, v6_execution_lock_context
from trading_system.close_resolver import coalesce_close_intents, resolve_close_quantity
from trading_system.v6_execution import (
    V6BinanceClient,
    V6ExecutionConfig,
    V6ExecutionEngine,
    V6OrderIntent,
    V6Signal,
    V6ExecutionError,
)


def make_close_intent(
    *,
    action: str = "close_long",
    close_fraction: float = 1.0,
    generated_at: str = "2026-07-15T00:00:00+00:00",
    rationale: str = "test close",
) -> V6OrderIntent:
    return V6OrderIntent(
        action=action,
        symbol="BTCUSDT",
        side="SELL" if action == "close_long" else "BUY",
        quantity_hint_usdt=100.0,
        leverage=8,
        reduce_only=True,
        entry_price=100.0,
        stop_loss_price=95.0,
        take_profit_price=110.0,
        long_probability=0.5,
        short_probability=0.3,
        wait_probability=0.2,
        rationale=rationale,
        generated_at=generated_at,
        close_fraction=close_fraction,
        target_name="tp_stage_1",
    )


def make_position() -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "side": "long",
        "entry_price": 100.0,
        "current_price": 100.0,
        "position_qty": 1.0,
        "remaining_quantity": 1.0,
        "notional_usdt": 100.0,
        "original_notional_usdt": 100.0,
        "remaining_fraction": 1.0,
        "lots": [
            {
                "quantity": 1.0,
                "remaining_quantity": 1.0,
                "status": "open",
                "entry_price": 100.0,
            }
        ],
    }


def make_close_result(
    *,
    after_qty: float | None,
    delta_qty: float,
    after_error: str = "",
) -> dict[str, Any]:
    reconcile: dict[str, Any] = {
        "delta_qty_abs": delta_qty,
        "after_error": after_error,
    }
    if after_qty is not None:
        reconcile["after"] = {
            "position_amt": after_qty,
            "abs_position_qty": abs(after_qty),
            "entry_price": 100.0,
            "mark_price": 101.0,
            "notional": abs(after_qty) * 101.0,
        }
    return {
        "status": "submitted",
        "submitted_at": "2026-07-15T00:00:01+00:00",
        "response": {"orderId": 123, "executedQty": str(delta_qty), "avgPrice": "101"},
        "position_reconcile": reconcile,
    }


def test_close_resolver_uses_exchange_quantity_and_flattens_at_ninety_percent() -> None:
    config = V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run", close_resolver_enabled=True)
    client = V6BinanceClient(config)
    with (
        patch.object(
            client,
            "actual_position_for_intent",
            return_value={"symbol": "BTCUSDT", "positionAmt": "4.7", "markPrice": "100"},
        ),
        patch.object(client, "quantity_rules", return_value=(0.001, 0.001)),
        patch.object(client, "minimum_notional_rule", return_value=5.0),
    ):
        intent = make_close_intent(close_fraction=0.90)
        quantity = client.resolve_close_quantity(intent)

    assert float(quantity) == 4.7
    assert intent.risk_plan["close_decision"]["mode"] == "full_flatten"


def test_close_resolver_flattens_when_partial_would_leave_dust() -> None:
    config = V6ExecutionConfig(
        symbols=["BTCUSDT"],
        order_mode="dry_run",
        close_resolver_enabled=True,
        strategy_dust_position_notional_usdt=5.0,
    )
    client = V6BinanceClient(config)
    with (
        patch.object(
            client,
            "actual_position_for_intent",
            return_value={"symbol": "BTCUSDT", "positionAmt": "0.08", "markPrice": "100"},
        ),
        patch.object(client, "quantity_rules", return_value=(0.01, 0.01)),
        patch.object(client, "minimum_notional_rule", return_value=1.0),
    ):
        intent = make_close_intent(close_fraction=0.50)
        quantity = client.resolve_close_quantity(intent)

    assert float(quantity) == 0.08
    assert intent.risk_plan["close_decision"]["reason"] == "partial_remainder_below_dust_notional_escalated_to_full_flatten"


def test_any_fractional_close_is_treated_as_partial_for_stop_resync() -> None:
    intent = make_close_intent(close_fraction=0.30)
    intent.target_name = "v6_smart_signal_long_crowded_flush_reduce_30"
    intent.risk_plan = {
        "partial_take_profit": {
            "type": "v6_smart_signal_risk_guard",
            "fraction_of_initial": 0.30,
        }
    }

    assert V6ExecutionEngine.is_partial_take_profit_intent(intent)


def test_zero_partial_close_preflight_happens_before_protective_stop_cancel(tmp_path) -> None:
    config = V6ExecutionConfig(
        symbols=["BTCUSDT"],
        order_mode="testnet_order",
        state_dir=tmp_path,
        state_file=tmp_path / "state.json",
        signals_file=tmp_path / "signals.jsonl",
        orders_file=tmp_path / "orders.jsonl",
        wal_file=tmp_path / "wal.jsonl",
        latest_summary_file=tmp_path / "summary.json",
        closed_trade_review_queue_file=tmp_path / "review.jsonl",
        post_trade_training_samples_file=tmp_path / "samples.jsonl",
    )
    engine = V6ExecutionEngine(config)
    position = make_position() | {
        "protective_stop_order_id": 123,
        "profit_guard_stages": [
            {
                "name": "v6_smart_signal_long_crowded_flush_reduce_30",
                "status": "pending",
            }
        ],
    }
    intent = make_close_intent(close_fraction=0.30)
    intent.target_name = "v6_smart_signal_long_crowded_flush_reduce_30"
    intent.risk_plan = {
        "partial_take_profit": {
            "type": "v6_smart_signal_risk_guard",
            "stages": [{"name": intent.target_name, "status": "pending"}],
        }
    }
    signal = V6Signal(
        symbol="BTCUSDT",
        computed_at="2026-07-16T00:00:00+00:00",
        price=100.0,
        side_bias="long",
        long_probability=0.5,
        short_probability=0.3,
        wait_probability=0.2,
        model_quality_score=0.8,
        reasons=[],
        h4={},
        fifteen_m={},
        one_m={},
        mid_retest={},
        levels={},
    )

    class FakeClient:
        def require_live_trading_confirmation(self, _operation: str) -> None:
            return None

        def resolve_close_quantity(self, candidate: V6OrderIntent) -> str:
            candidate.risk_plan["close_decision"] = {
                "mode": "skip_zero",
                "reason": "partial_notional_below_min_notional",
            }
            raise V6ExecutionError("resolved zero partial close quantity for BTCUSDT close_long")

        def submit_order(self, _intent: V6OrderIntent) -> dict[str, Any]:
            raise AssertionError("zero preflight must not reach order submission")

        def close(self) -> None:
            return None

    engine.client = FakeClient()  # type: ignore[assignment]
    with (
        patch.object(engine, "load_state", return_value={"positions": [position], "journal": []}),
        patch.object(engine, "build_signal", return_value=signal),
        patch.object(engine, "intent_for_signal", return_value=intent),
        patch.object(engine, "cancel_protective_stop_if_present") as cancel_stop,
        patch.object(engine, "reconcile_state_with_exchange", return_value={"status": "ok"}),
        patch.object(engine, "run_ledger_reconcile_after_action", return_value=None),
    ):
        summary = engine._run_once_locked(submit=True)

    cancel_stop.assert_not_called()
    result = summary["results"][0]
    assert result["status"] == "skipped_zero_partial_close_quantity"
    assert "pre_submit_cancelled_protective_stop" not in result
    assert result["retry_wait"]["event"] == "partial_close_retry_wait"
    assert result["retry_wait"]["stage_retry"]["stage_key"] == "profit_guard_stages"
    assert position["profit_guard_stages"][0]["status"] == "retry_wait"


def test_smart_signal_partial_close_updates_profit_guard_stage() -> None:
    engine = V6ExecutionEngine(V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run"))
    position = make_position() | {
        "profit_guard_stages": [
            {
                "name": "v6_smart_signal_long_crowded_flush_reduce_30",
                "status": "pending",
                "fraction_of_initial": 0.30,
            }
        ]
    }
    intent = make_close_intent(close_fraction=0.30)
    intent.target_name = "v6_smart_signal_long_crowded_flush_reduce_30"
    intent.risk_plan = {
        "partial_take_profit": {
            "type": "v6_smart_signal_risk_guard",
            "fraction_of_initial": 0.30,
        },
        "close_decision": {"mode": "partial", "reason": "requested_fraction"},
    }

    updated = engine.apply_partial_take_profit_to_position(
        position,
        intent,
        make_close_result(after_qty=0.70, delta_qty=0.30),
    )

    assert updated is position
    assert position["profit_guard_stages"][0]["status"] == "executed"
    assert position["remaining_fraction"] == 0.7


def test_standalone_close_resolver_uses_exchange_position_and_step_precision() -> None:
    decision = resolve_close_quantity(
        symbol="BTCUSDT",
        side="long",
        requested_fraction=1.0,
        local_position={"position_qty": 9.0},
        exchange_position={"positionAmt": "4.7"},
        mark_price=60_000.0,
        step_size=0.001,
        min_notional_usdt=5.0,
        dust_notional_usdt=20.0,
    )

    assert decision.mode == "full_flatten"
    assert decision.quantity == "4.700"
    assert decision.quantity_float == 4.7


def test_standalone_close_resolver_handles_dust_minimum_and_missing_position() -> None:
    common = {
        "symbol": "BTCUSDT",
        "side": "long",
        "local_position": {"position_qty": 9.0},
        "step_size": 0.001,
    }
    dust = resolve_close_quantity(
        **common,
        requested_fraction=0.85,
        exchange_position={"positionAmt": "0.010"},
        mark_price=60_000.0,
        min_notional_usdt=5.0,
        dust_notional_usdt=150.0,
    )
    below_minimum = resolve_close_quantity(
        **common,
        requested_fraction=0.1,
        exchange_position={"positionAmt": "0.010"},
        mark_price=1_000.0,
        min_notional_usdt=5.0,
        dust_notional_usdt=0.0,
    )
    missing = resolve_close_quantity(
        **common,
        requested_fraction=1.0,
        exchange_position=None,
        mark_price=60_000.0,
        min_notional_usdt=5.0,
        dust_notional_usdt=20.0,
    )

    assert dust.mode == "full_flatten"
    assert "dust" in dust.reason
    assert below_minimum.mode == "skip_zero"
    assert below_minimum.reason == "partial_notional_below_min_notional"
    assert missing.mode == "skip_missing_exchange_position"


def test_standalone_close_resolver_never_emits_float_artifacts() -> None:
    decision = resolve_close_quantity(
        symbol="BTCUSDT",
        side="long",
        requested_fraction=0.1 + 0.2,
        local_position={},
        exchange_position={"positionAmt": "1.0"},
        mark_price=1_000.0,
        step_size=0.1,
        min_notional_usdt=0.0,
        dust_notional_usdt=0.0,
    )

    assert decision.quantity == "0.3"
    assert "e" not in decision.quantity.lower()


def test_close_intents_are_coalesced_once_per_symbol_and_action() -> None:
    first = make_close_intent(close_fraction=0.30, rationale="tp1")
    second = make_close_intent(close_fraction=0.20, rationale="tp2")

    merged = coalesce_close_intents([first, second])

    assert len(merged) == 1
    assert merged[0].close_fraction == 0.50
    assert "coalesced n=2" in merged[0].rationale


def test_reconcile_after_error_is_not_a_confirmed_position_delta() -> None:
    client = V6BinanceClient(V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run"))
    intent = make_close_intent()

    payload = client.position_reconcile_payload(
        intent,
        {"symbol": "BTCUSDT", "positionAmt": "1", "markPrice": "100"},
        None,
        after_error="ReadTimeout: exchange snapshot unavailable",
    )
    result = {"status": "submitted", "position_reconcile": payload}

    assert payload["delta_qty_abs"] == 0.0
    assert payload["confirmation_state"] == "unknown"
    assert V6ExecutionEngine.submitted_order_has_no_confirmed_position_delta(result)


def test_full_close_unknown_snapshot_keeps_local_position_pending_reconcile() -> None:
    engine = V6ExecutionEngine(
        V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run", close_resolver_enabled=True)
    )
    position = make_position()

    residual = engine.residual_position_after_full_close(
        position,
        make_close_intent(),
        make_close_result(after_qty=None, delta_qty=1.0, after_error="timeout"),
    )

    assert residual is position
    assert residual["reconcile_pending"] is True
    assert residual["position_qty"] == 1.0
    assert residual["lots"][0]["status"] == "open"


def test_full_close_partial_fill_keeps_exchange_residual() -> None:
    engine = V6ExecutionEngine(
        V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run", close_resolver_enabled=True)
    )
    position = make_position()

    residual = engine.residual_position_after_full_close(
        position,
        make_close_intent(),
        make_close_result(after_qty=0.4, delta_qty=0.6),
    )

    assert residual is position
    assert residual["position_qty"] == 0.4
    assert sum(float(lot["remaining_quantity"]) for lot in residual["lots"]) == 0.4


def test_apply_intent_to_state_keeps_partial_fill_residual() -> None:
    engine = V6ExecutionEngine(
        V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run", close_resolver_enabled=True)
    )
    state = {"positions": [make_position()]}

    engine.apply_intent_to_state(
        state,
        make_close_intent(),
        make_close_result(after_qty=0.4, delta_qty=0.6),
    )

    assert len(state["positions"]) == 1
    assert state["positions"][0]["position_qty"] == 0.4


def test_full_close_confirmed_flat_removes_local_position() -> None:
    engine = V6ExecutionEngine(
        V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run", close_resolver_enabled=True)
    )
    position = make_position()

    residual = engine.residual_position_after_full_close(
        position,
        make_close_intent(),
        make_close_result(after_qty=0.0, delta_qty=1.0),
    )

    assert residual is None
    assert all(lot["status"] == "closed" for lot in position["lots"])


def test_v6_activity_fingerprint_ignores_generated_and_submitted_timestamps() -> None:
    first_intent = make_close_intent(generated_at="2026-07-15T00:00:00+00:00").to_dict()
    second_intent = make_close_intent(generated_at="2026-07-15T00:05:00+00:00").to_dict()
    first = {
        "intents": [first_intent],
        "results": [
            {
                "status": "failed",
                "submitted_at": "2026-07-15T00:00:01+00:00",
                "error": 'Binance 400: {"code":-2022,"msg":"ReduceOnly rejected"}',
                "intent": first_intent,
            }
        ],
    }
    second = {
        "intents": [second_intent],
        "results": [
            {
                "status": "failed",
                "submitted_at": "2026-07-15T00:05:01+00:00",
                "error": 'Binance 400: {"code":-2022,"msg":"ReduceOnly rejected"}',
                "intent": second_intent,
            }
        ],
    }

    assert telegram.stable_v6_activity_fingerprint(first) == telegram.stable_v6_activity_fingerprint(second)


def test_v6_activity_fingerprint_distinguishes_exchange_error_codes() -> None:
    intent = make_close_intent().to_dict()
    rejected = {
        "intents": [intent],
        "results": [{"status": "failed", "error": "code=-2022 rejected", "intent": intent}],
    }
    timeout = {
        "intents": [intent],
        "results": [{"status": "failed", "error": "code=-1007 timeout", "intent": intent}],
    }

    assert telegram.stable_v6_activity_fingerprint(rejected) != telegram.stable_v6_activity_fingerprint(timeout)


def test_telegram_close_intent_exposes_resolved_close_mode() -> None:
    intent = make_close_intent().to_dict()
    intent["risk_plan"] = {
        "close_decision": {
            "mode": "full_flatten",
            "reason": "partial_remainder_below_dust_notional_escalated_to_full_flatten",
        }
    }

    assert telegram.zh_close_mode({"mode": "partial"}) == "部分平仓"
    assert telegram.zh_close_mode({"mode": "full_flatten", "reason": "requested_fraction"}) == "全平"
    assert telegram.zh_close_mode(intent["risk_plan"]["close_decision"]) == "粉尘全平"
    assert "模式=粉尘全平" in telegram.format_compact_execution_intent(intent)


def test_entry_gate_decision_is_persisted_for_later_outcome_labeling(tmp_path) -> None:
    output = tmp_path / "entry_gate_events.jsonl"
    engine = V6ExecutionEngine(
        V6ExecutionConfig(
            symbols=["BTCUSDT"],
            order_mode="dry_run",
            entry_gate_event_recording_enabled=True,
            entry_gate_events_file=output,
        )
    )
    signal = V6Signal(
        symbol="BTCUSDT",
        computed_at="2026-07-15T00:00:00+00:00",
        price=60000.0,
        side_bias="long",
        long_probability=0.55,
        short_probability=0.25,
        wait_probability=0.20,
        model_quality_score=0.8,
        reasons=[],
        h4={},
        fifteen_m={},
        one_m={},
        mid_retest={},
        levels={},
    )
    decision = {
        "state": "allowed",
        "allowed": True,
        "notional_multiplier": 0.8,
        "blocking_reason": "",
        "soft_reasons": ["macro_mismatch"],
    }

    engine.record_entry_gate_event(
        signal,
        "open_long",
        {"selected_entry_level": {"name": "bb300_mid_15m", "timeframe": "15m"}},
        decision,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["decision"]["allowed"] is True
    assert row["selected_entry_level"]["name"] == "bb300_mid_15m"
    assert row["outcomes"] == {
        "status": "pending",
        "execution": "next_15m_bar_open",
        "horizons": ["1h", "4h", "24h", "48h", "168h"],
    }


def test_entry_gate_event_log_rotates_at_configured_limit(tmp_path) -> None:
    output = tmp_path / "entry_gate_events.jsonl"
    output.write_text("old-event\n", encoding="utf-8")
    engine = V6ExecutionEngine(
        V6ExecutionConfig(
            symbols=["BTCUSDT"],
            order_mode="dry_run",
            entry_gate_event_recording_enabled=True,
            entry_gate_events_file=output,
            entry_gate_events_max_bytes=1,
        )
    )
    signal = V6Signal(
        symbol="BTCUSDT",
        computed_at="2026-07-15T00:00:00+00:00",
        price=60000.0,
        side_bias="long",
        long_probability=0.55,
        short_probability=0.25,
        wait_probability=0.20,
        model_quality_score=0.8,
        reasons=[],
        h4={},
        fifteen_m={},
        one_m={},
        mid_retest={},
        levels={},
    )

    engine.record_entry_gate_event(signal, "open_long", {}, {"state": "allowed", "allowed": True})

    assert output.with_suffix(".jsonl.1").read_text(encoding="utf-8") == "old-event\n"
    assert json.loads(output.read_text(encoding="utf-8"))["symbol"] == "BTCUSDT"


def test_partial_take_profit_full_flatten_keeps_state_when_post_order_snapshot_is_unknown() -> None:
    engine = V6ExecutionEngine(
        V6ExecutionConfig(
            symbols=["BTCUSDT"],
            order_mode="dry_run",
            close_resolver_enabled=True,
        )
    )
    position = make_position()
    position["take_profit_stages"] = [{"name": "tp_stage_1", "status": "pending", "fraction_of_initial": 1.0}]
    intent = make_close_intent(close_fraction=1.0)
    intent.risk_plan = {
        "partial_take_profit": {"type": "bb_take_profit", "fraction_of_initial": 1.0},
        "close_decision": {"mode": "full_flatten"},
    }
    result = {
        "status": "submitted",
        "position_reconcile": {
            "confirmation_state": "unknown",
            "after_state": "unknown",
            "after": {},
            "after_error": "timeout",
        },
    }

    updated = engine.apply_partial_take_profit_to_position(position, intent, result)

    assert updated is position
    assert float(updated["position_qty"]) > 0


def test_partial_take_profit_dry_run_full_close_keeps_legacy_paper_state_behavior() -> None:
    engine = V6ExecutionEngine(
        V6ExecutionConfig(
            symbols=["BTCUSDT"],
            order_mode="dry_run",
            close_resolver_enabled=True,
        )
    )
    position = make_position()
    position["take_profit_stages"] = [{"name": "tp_stage_1", "status": "pending", "fraction_of_initial": 1.0}]
    intent = make_close_intent(close_fraction=1.0)
    intent.risk_plan = {
        "partial_take_profit": {"type": "bb_take_profit", "fraction_of_initial": 1.0},
        "close_decision": {"mode": "full_flatten"},
    }

    assert engine.apply_partial_take_profit_to_position(position, intent, {"status": "dry_run"}) is None


def test_partial_take_profit_escalated_to_full_close_enqueues_review(tmp_path) -> None:
    engine = V6ExecutionEngine(
        V6ExecutionConfig(
            symbols=["BTCUSDT"],
            order_mode="dry_run",
            close_resolver_enabled=True,
            closed_trade_review_queue_file=tmp_path / "closed_trade_review_queue.jsonl",
        )
    )
    position = make_position()
    position["take_profit_stages"] = [{"name": "tp_stage_1", "status": "pending", "fraction_of_initial": 0.9}]
    state = {"positions": [position], "journal": []}
    intent = make_close_intent(close_fraction=0.9, rationale="staged long take profit touched")
    intent.risk_plan = {
        "partial_take_profit": {"type": "bb_take_profit", "fraction_of_initial": 0.9},
        "close_decision": {"mode": "full_flatten"},
    }
    result = {
        "status": "submitted",
        "submitted_at": "2026-07-15T00:00:01+00:00",
        "response": {"orderId": 123, "executedQty": "1", "avgPrice": "101"},
        "position_reconcile": {
            "confirmation_state": "confirmed",
            "after_state": "flat",
            "before": {"position_amt": 1.0, "abs_position_qty": 1.0},
            "after": {},
            "delta_qty_abs": 1.0,
        },
    }

    engine.apply_intent_to_state(state, intent, result)

    assert state["positions"] == []
    review = json.loads(engine.config.closed_trade_review_queue_file.read_text(encoding="utf-8"))
    assert review["symbol"] == "BTCUSDT"
    assert review["target_name"] == "tp_stage_1"


class ReconcileEngineStub:
    @staticmethod
    def position_key_from_state(position: dict[str, Any]) -> tuple[str, str]:
        return str(position["symbol"]).upper(), str(position["side"]).lower()

    @staticmethod
    def position_key_from_risk_row(row: dict[str, Any]) -> tuple[str, str]:
        side = str(row.get("positionSide") or "").lower()
        if side not in {"long", "short"}:
            side = "long" if float(row.get("positionAmt") or 0.0) > 0 else "short"
        return str(row["symbol"]).upper(), side

    @staticmethod
    def is_strategy_dust_position(_position: dict[str, Any]) -> bool:
        return False


def test_v5_reconcile_requires_repeated_missing_snapshots_before_removal() -> None:
    state = {"positions": [make_position()]}
    confirmations: dict[str, Any] = {}

    first_rows, confirmations, first_meta = guard_transient_absences(
        ReconcileEngineStub(),
        state,
        [],
        confirmations,
        now_ts=100.0,
        required_confirmations=2,
        minimum_elapsed_seconds=5.0,
    )
    second_rows, confirmations, second_meta = guard_transient_absences(
        ReconcileEngineStub(),
        state,
        [],
        confirmations,
        now_ts=101.0,
        required_confirmations=2,
        minimum_elapsed_seconds=5.0,
    )
    confirmed_rows, confirmations, confirmed_meta = guard_transient_absences(
        ReconcileEngineStub(),
        state,
        [],
        confirmations,
        now_ts=105.0,
        required_confirmations=2,
        minimum_elapsed_seconds=5.0,
    )

    assert len(first_rows) == 1
    assert len(second_rows) == 1
    assert first_meta["protected"] == ["BTCUSDT:long"]
    assert second_meta["protected"] == ["BTCUSDT:long"]
    assert confirmed_rows == []
    assert confirmed_meta["confirmed_absent"] == ["BTCUSDT:long"]


def test_v5_reconcile_reads_exchange_snapshot_after_execution_lock(tmp_path) -> None:
    events: list[str] = []

    @contextmanager
    def fake_execution_lock(_state_dir: Any, _timeout_seconds: float):
        events.append("lock_enter")
        try:
            yield
        finally:
            events.append("lock_exit")

    class ClientStub:
        def actual_position_rows(self) -> list[dict[str, Any]]:
            events.append("exchange_snapshot")
            return []

    class EngineStub(ReconcileEngineStub):
        client = ClientStub()

        @staticmethod
        def reconcile_state_with_exchange(
            _state: dict[str, Any],
            *,
            actual_rows: list[dict[str, Any]],
            sync_exchange_side_effects: bool,
        ) -> dict[str, Any]:
            assert actual_rows == []
            assert sync_exchange_side_effects is False
            events.append("state_reconciled")
            return {"status": "ok", "removed": [], "recovered": []}

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_file = state_dir / "state.json"
    state_file.write_text('{"positions":[]}', encoding="utf-8")

    with patch("scripts.run_v5_position_reconcile_loop.v5_execution_lock", fake_execution_lock):
        payload = reconcile_once(
            EngineStub(),
            state_file,
            state_dir / "summary.json",
            state_dir / "events.jsonl",
            state_dir=state_dir,
            lock_timeout_seconds=0.2,
        )

    assert payload["status"] == "ok"
    assert events == ["lock_enter", "exchange_snapshot", "state_reconciled", "lock_exit"]


def test_v6_ledger_reconcile_reads_exchange_snapshot_after_execution_lock(tmp_path) -> None:
    events: list[str] = []

    @contextmanager
    def fake_execution_lock(_config: V6ExecutionConfig):
        events.append("lock_enter")
        try:
            yield
        finally:
            events.append("lock_exit")

    class ClientStub:
        def actual_position_rows(self) -> list[dict[str, Any]]:
            events.append("exchange_snapshot")
            return []

    class EngineStub(ReconcileEngineStub):
        client = ClientStub()

        @staticmethod
        def reconcile_state_with_exchange(
            _state: dict[str, Any],
            *,
            actual_rows: list[dict[str, Any]],
            sync_exchange_side_effects: bool,
        ) -> dict[str, Any]:
            assert actual_rows == []
            assert sync_exchange_side_effects is False
            events.append("state_reconciled")
            return {"status": "ok", "removed": [], "recovered": []}

        @staticmethod
        def refresh_position_lot_index(_state: dict[str, Any]) -> None:
            return None

        @staticmethod
        def close() -> None:
            return None

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    config = V6ExecutionConfig(
        order_mode="testnet_order",
        state_dir=state_dir,
        state_file=state_dir / "state.json",
    )
    config.state_file.write_text('{"positions":[]}', encoding="utf-8")

    with (
        patch("scripts.run_v6_ledger_reconcile.V6ExecutionConfig.from_env", return_value=config),
        patch("scripts.run_v6_ledger_reconcile.V6ExecutionEngine", return_value=EngineStub()),
        patch("scripts.run_v6_ledger_reconcile.v6_execution_lock_context", fake_execution_lock),
        patch("scripts.run_v6_ledger_reconcile.summarize_exchange_rows", return_value=[]),
        patch("scripts.run_v6_ledger_reconcile.summarize_local_positions", return_value=[]),
        patch("scripts.run_v6_ledger_reconcile.compare_local_exchange", return_value=[]),
    ):
        payload = run_v6_ledger_reconcile(check_only=False, tolerance=0.0)

    assert payload["status"] == "ok"
    assert events == ["lock_enter", "exchange_snapshot", "state_reconciled", "exchange_snapshot", "lock_exit"]


def test_v6_ledger_reconcile_waits_for_an_active_execution_cycle_by_default(tmp_path) -> None:
    config = V6ExecutionConfig(state_dir=tmp_path)
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("scripts.run_v6_ledger_reconcile.timed_file_lock", return_value=nullcontext()) as lock,
    ):
        with v6_execution_lock_context(config):
            pass

    assert lock.call_args.kwargs["timeout_seconds"] == 45.0
