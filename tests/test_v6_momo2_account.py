from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import telegram_monitor_v5_v6 as telegram_monitor
from trading_system.v6_execution import V6ExecutionConfig, V6ExecutionEngine, V6OrderIntent, V6Signal


ROOT = Path(__file__).resolve().parents[1]


def make_signal(*, price: float) -> V6Signal:
    return V6Signal(
        symbol="BTCUSDT",
        computed_at="2026-07-20T00:00:00+00:00",
        price=price,
        side_bias="long",
        long_probability=0.60,
        short_probability=0.20,
        wait_probability=0.20,
        model_quality_score=0.50,
        reasons=[],
        h4={},
        fifteen_m={},
        one_m={},
        mid_retest={},
        levels={},
        indicators={},
    )


def make_engine(root: Path, *, order_mode: str = "dry_run") -> V6ExecutionEngine:
    config = V6ExecutionConfig(
        order_mode=order_mode,
        instance_name="momo2",
        exit_control_mode="stop_only_manual_profit",
        manual_exchange_action_audit_enabled=True,
        state_dir=root,
        state_file=root / "state.json",
        signals_file=root / "signals.jsonl",
        entry_gate_events_file=root / "entry_gate_events.jsonl",
        orders_file=root / "orders.jsonl",
        wal_file=root / "execution_wal.jsonl",
        latest_summary_file=root / "latest_run_summary.json",
        closed_trade_review_queue_file=root / "closed_trade_review_queue.jsonl",
        post_trade_training_samples_file=root / "post_trade_training_samples.jsonl",
        manual_exchange_action_audit_file=root / "manual_exchange_actions.jsonl",
    )
    return V6ExecutionEngine(config)


def test_momo2_blocks_automatic_profit_exit_but_allows_hard_stop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(Path(tmp))
        position = {
            "symbol": "BTCUSDT",
            "side": "long",
            "entry_price": 100.0,
            "stop_loss_price": 95.0,
            "take_profit_price": 105.0,
            "notional_usdt": 300.0,
            "leverage": 8,
            "risk_plan": {},
        }
        try:
            assert engine.exit_intent_if_needed(make_signal(price=110.0), position) is None
            assert position["manual_profit_management"]["enabled"] is True
            assert position["manual_profit_management"]["stop_touched"] is False

            stop_intent = engine.exit_intent_if_needed(make_signal(price=94.0), position)
            assert stop_intent is not None
            assert stop_intent.action == "close_long"
            assert stop_intent.close_fraction == 1.0
            assert stop_intent.rationale == "long stop loss touched"
        finally:
            engine.close()


def test_momo2_open_intent_records_manual_profit_policy_for_telegram() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(Path(tmp))
        intent = V6OrderIntent(
            action="open_long",
            symbol="BTCUSDT",
            side="BUY",
            quantity_hint_usdt=300.0,
            leverage=8,
            reduce_only=False,
            entry_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            long_probability=0.60,
            short_probability=0.20,
            wait_probability=0.20,
            rationale="test",
            generated_at="2026-07-20T00:00:00+00:00",
            risk_plan={},
        )
        try:
            engine.annotate_intent_execution_risk(intent)
            assert intent.risk_plan["exit_control_mode"] == "stop_only_manual_profit"
            assert intent.risk_plan["execution_account"]["instance_name"] == "momo2"
            assert telegram_monitor.format_take_profit_for_intent(intent.to_dict()) == "\u4eba\u5de5"
        finally:
            engine.close()


def test_momo2_reconcile_records_manual_partial_reduction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = make_engine(root, order_mode="testnet_order")
        position = {
            "symbol": "BTCUSDT",
            "side": "long",
            "opened_at": "2026-07-20T00:00:00+00:00",
            "entry_price": 100.0,
            "current_price": 110.0,
            "stop_loss_price": 95.0,
            "take_profit_price": 120.0,
            "position_qty": 1.0,
            "remaining_quantity": 1.0,
            "notional_usdt": 110.0,
            "leverage": 8,
            "risk_plan": {"exit_control_mode": "stop_only_manual_profit"},
            "lots": [
                {
                    "lot_id": "lot-1",
                    "symbol": "BTCUSDT",
                    "side": "long",
                    "quantity": 1.0,
                    "remaining_quantity": 1.0,
                    "entry_price": 100.0,
                    "status": "open",
                }
            ],
        }
        state = {"positions": [position], "journal": []}
        actual_rows = [
            {
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "positionAmt": "0.6",
                "entryPrice": "100",
                "markPrice": "110",
                "notional": "66",
                "leverage": "8",
                "marginType": "isolated",
            }
        ]
        try:
            result = engine.reconcile_state_with_exchange(
                state,
                actual_rows=actual_rows,
                sync_exchange_side_effects=False,
            )
            assert result is not None
            assert result["status"] == "ok"
            assert len(result["manual_exchange_actions"]) == 1
            event = result["manual_exchange_actions"][0]
            assert event["classification"] == "manual_reduce"
            assert event["previous_quantity"] == 1.0
            assert event["exchange_quantity"] == 0.6
            assert state["positions"][0]["position_qty"] == 0.6
            assert state["positions"][0]["lots"][0]["remaining_quantity"] == 0.6

            rows = [json.loads(line) for line in (root / "manual_exchange_actions.jsonl").read_text(encoding="utf-8").splitlines()]
            assert len(rows) == 1
            assert rows[0]["event_id"] == event["event_id"]
        finally:
            engine.close()


def test_momo2_service_contract_is_fully_isolated() -> None:
    runner = (ROOT / "scripts" / "macos_run_somnus_service.sh").read_text(encoding="utf-8")
    installer = (ROOT / "scripts" / "macos_install_v6_live_momo2.sh").read_text(encoding="utf-8")
    assert '.env.v6_live_momo2' in runner
    assert 'state_dir="data/trading_system/v6_execution_live_momo2"' in runner
    assert 'export TS_V6_INSTANCE_NAME=momo2' in runner
    assert 'export TS_V6_EXIT_CONTROL_MODE=stop_only_manual_profit' in runner
    assert 'export TS_V6_MANUAL_EXCHANGE_ACTION_AUDIT_ENABLED=1' in runner
    assert 'v6-execution-live-momo2)' in runner
    assert 'v6-ledger-reconcile-live-momo2)' in runner
    assert 'telegram-v6-live-momo2)' in runner
    assert installer.index("v6-live-readiness-momo2") < installer.index("sudo install")

    for label in (
        "com.somnus.v6-execution-live-momo2",
        "com.somnus.v6-ledger-reconcile-live-momo2",
        "com.somnus.telegram-v6-live-momo2",
    ):
        assert (ROOT / "launchd" / f"{label}.plist").exists()
        for script_name in (
            "macos_launch_somnus_all.sh",
            "macos_stop_somnus_all.sh",
            "macos_status_somnus_all.sh",
        ):
            assert label in (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
