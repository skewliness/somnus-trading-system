from __future__ import annotations

import copy
import json
from pathlib import Path

from trading_system.events import EventEnvelope, EventWriter
from trading_system.v6_execution import V6ExecutionConfig, V6ExecutionEngine, V6OrderIntent


def sample_summary() -> dict:
    return {
        "generated_at": "2026-07-21T00:00:00+00:00",
        "signals": [
            {
                "symbol": "BTCUSDT",
                "computed_at": "2026-07-21T00:00:00+00:00",
                "side_bias": "long",
                "price": 60000.0,
                "long_probability": 0.60,
                "short_probability": 0.25,
                "wait_probability": 0.15,
                "model_quality_score": 0.80,
                "reasons": ["4H right-side long", "15M MA20 held"],
            }
        ],
        "intents": [
            {
                "symbol": "BTCUSDT",
                "action": "close_short",
                "side": "BUY",
                "generated_at": "2026-07-21T00:00:01+00:00",
                "quantity_hint_usdt": 100.0,
                "leverage": 8,
                "reduce_only": True,
                "entry_price": 60000.0,
                "stop_loss_price": 61000.0,
                "take_profit_price": 59000.0,
                "close_fraction": 0.30,
                "target_name": "bb120_mid_15m",
                "rationale": "structure exit; api_key=must-not-leak",
            }
        ],
        "results": [
            {
                "status": "submitted",
                "order_id": "123",
                "executed_quantity": 0.01,
                "average_price": 60010.0,
                "intent": {"symbol": "BTCUSDT", "action": "close_short", "close_fraction": 0.30},
            }
        ],
        "positions": [
            {
                "symbol": "ETHUSDT",
                "side": "long",
                "remaining_quantity": 0.2,
                "entry_price": 2000.0,
                "current_price": 2010.0,
                "horizon": "4h",
                "entry_timeframe": "4h",
            }
        ],
    }


def test_event_writer_disabled_is_noop_and_does_not_mutate_summary(tmp_path: Path) -> None:
    summary = sample_summary()
    before = copy.deepcopy(summary)
    writer = EventWriter(tmp_path / "events.jsonl", enabled=False)

    assert writer.emit_run_summary(summary) == 0
    assert summary == before
    assert not (tmp_path / "events.jsonl").exists()


def test_event_writer_writes_typed_account_scoped_events_and_redacts(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, enabled=True, account_id="momo2", runtime_mode="live_order")

    assert writer.emit_run_summary(sample_summary()) == 6
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [row["event_type"] for row in rows] == ["signal", "order", "exit", "fill", "exit", "position"]
    assert {row["account_id"] for row in rows} == {"momo2"}
    assert {row["runtime_mode"] for row in rows} == {"live_order"}
    assert {row["schema_version"] for row in rows} == {2}
    assert all(len(row["event_id"]) == 64 for row in rows)
    assert all(row["trade_id"] for row in rows)
    assert all(row["timestamp"] == row["occurred_at"] for row in rows)
    btc_trade_ids = {row["trade_id"] for row in rows[:5]}
    assert len(btc_trade_ids) == 1
    assert rows[5]["trade_id"] not in btc_trade_ids
    assert rows[0]["parent_event_id"] == ""
    assert rows[1]["parent_event_id"] == rows[0]["event_id"]
    assert rows[2]["parent_event_id"] == rows[1]["event_id"]
    assert rows[3]["parent_event_id"] == rows[1]["event_id"]
    assert rows[4]["parent_event_id"] == rows[3]["event_id"]
    assert rows[0]["payload"]["signal_reason"] == "4H right-side long; 15M MA20 held"
    assert rows[1]["payload"]["order_reason"].startswith("structure exit")
    assert rows[2]["payload"]["exit_reason"].startswith("structure exit")
    assert "must-not-leak" not in path.read_text(encoding="utf-8")
    assert "<redacted>" in path.read_text(encoding="utf-8")


def test_event_id_is_deterministic_for_same_business_event() -> None:
    kwargs = {
        "account_id": "demo",
        "runtime_mode": "testnet_order",
        "source": "v6_execution",
        "strategy_version": "6.0-baseline",
        "occurred_at": "2026-07-21T00:00:00+00:00",
    }
    first = EventEnvelope.create("signal", {"symbol": "BTCUSDT", "price": 60000.0}, **kwargs)
    second = EventEnvelope.create("signal", {"price": 60000.0, "symbol": "BTCUSDT"}, **kwargs)

    assert first.event_id == second.event_id


def test_trade_chain_is_stable_across_writer_restarts(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    summary = sample_summary()
    trade_id = "trd_existing_btc_long"
    summary["signals"][0]["side_bias"] = "long"
    summary["intents"][0].update(
        {
            "action": "open_long",
            "side": "BUY",
            "reduce_only": False,
            "risk_plan": {"trade_id": trade_id},
        }
    )
    summary["results"][0]["intent"] = dict(summary["intents"][0])
    summary["positions"] = [
        {
            "trade_id": trade_id,
            "symbol": "BTCUSDT",
            "side": "long",
            "remaining_quantity": 0.01,
            "entry_price": 60000.0,
            "opened_at": "2026-07-21T00:00:01+00:00",
        }
    ]

    EventWriter(path, enabled=True, account_id="demo").emit_run_summary(summary)

    close_summary = {
        "generated_at": "2026-07-21T04:00:00+00:00",
        "signals": [{**summary["signals"][0], "computed_at": "2026-07-21T04:00:00+00:00"}],
        "intents": [
            {
                **summary["intents"][0],
                "action": "close_long",
                "reduce_only": True,
                "generated_at": "2026-07-21T04:00:01+00:00",
                "risk_plan": {"trade_id": trade_id},
            }
        ],
        "results": [],
        "positions": [],
    }
    close_summary["results"] = [
        {
            "status": "submitted",
            "order_id": "124",
            "intent": dict(close_summary["intents"][0]),
        }
    ]
    EventWriter(path, enabled=True, account_id="demo").emit_run_summary(close_summary)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {row["trade_id"] for row in rows} == {trade_id}
    assert len({row["event_id"] for row in rows}) == len(rows)


def test_engine_trade_id_migration_and_intent_link_are_deterministic() -> None:
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = type("Config", (), {"instance_name": "demo"})()
    state = {
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "opened_at": "2026-07-21T00:00:01+00:00",
                "entry_price": 60000.0,
                "position_qty": 0.01,
                "risk_plan": {},
                "lots": [{}],
            }
        ]
    }

    assert engine.ensure_state_trade_ids(state) is True
    position_trade_id = state["positions"][0]["trade_id"]
    assert engine.ensure_state_trade_ids(state) is False
    assert state["positions"][0]["risk_plan"]["trade_id"] == position_trade_id
    assert state["positions"][0]["lots"][0]["trade_id"] == position_trade_id

    close_intent = V6OrderIntent(
        action="close_long",
        symbol="BTCUSDT",
        side="SELL",
        quantity_hint_usdt=100.0,
        leverage=8,
        reduce_only=True,
        entry_price=60000.0,
        stop_loss_price=59000.0,
        take_profit_price=61000.0,
        long_probability=0.6,
        short_probability=0.2,
        wait_probability=0.2,
        rationale="close",
        generated_at="2026-07-21T04:00:00+00:00",
    )
    assert engine.ensure_intent_trade_id(close_intent, state) == position_trade_id

    open_intent = V6OrderIntent(
        action="open_short",
        symbol="ETHUSDT",
        side="SELL",
        quantity_hint_usdt=100.0,
        leverage=8,
        reduce_only=False,
        entry_price=2000.0,
        stop_loss_price=2060.0,
        take_profit_price=1900.0,
        long_probability=0.2,
        short_probability=0.6,
        wait_probability=0.2,
        rationale="open",
        generated_at="2026-07-21T04:00:00+00:00",
    )
    first = engine.ensure_intent_trade_id(open_intent, state)
    second = engine.ensure_intent_trade_id(open_intent, state)
    assert first == second
    assert first.startswith("trd_")


def test_engine_trade_ids_separate_hedge_sides_and_preserve_scale_in_lifecycle() -> None:
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = type("Config", (), {"instance_name": "momo"})()
    state = {
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "opened_at": "2026-07-21T00:00:00+00:00",
                "entry_price": 60000.0,
                "position_qty": 0.01,
                "risk_plan": {},
                "lots": [{}],
            },
            {
                "symbol": "BTCUSDT",
                "side": "short",
                "opened_at": "2026-07-21T01:00:00+00:00",
                "entry_price": 61000.0,
                "position_qty": 0.01,
                "risk_plan": {},
                "lots": [{}],
            },
        ]
    }
    engine.ensure_state_trade_ids(state)
    long_trade_id = state["positions"][0]["trade_id"]
    short_trade_id = state["positions"][1]["trade_id"]
    assert long_trade_id != short_trade_id

    def intent(action: str, side: str) -> V6OrderIntent:
        return V6OrderIntent(
            action=action,
            symbol="BTCUSDT",
            side=side,
            quantity_hint_usdt=100.0,
            leverage=8,
            reduce_only=action.startswith("close_"),
            entry_price=60500.0,
            stop_loss_price=59000.0,
            take_profit_price=62000.0,
            long_probability=0.5,
            short_probability=0.3,
            wait_probability=0.2,
            rationale="trade lifecycle test",
            generated_at="2026-07-21T04:00:00+00:00",
        )

    assert engine.ensure_intent_trade_id(intent("close_long", "SELL"), state) == long_trade_id
    assert engine.ensure_intent_trade_id(intent("close_short", "BUY"), state) == short_trade_id

    scale_in = intent("open_long", "BUY")
    scale_in.risk_plan = {"scale_in": {"stage": 2}}
    assert engine.ensure_intent_trade_id(scale_in, state) == long_trade_id


def test_fill_event_reads_exchange_identity_from_nested_response(tmp_path: Path) -> None:
    summary = sample_summary()
    summary["results"] = [
        {
            "status": "submitted",
            "intent": {"symbol": "BTCUSDT", "action": "open_long"},
            "response": {
                "orderId": 9123,
                "clientOrderId": "client-9123",
                "status": "FILLED",
                "executedQty": "0.02",
                "avgPrice": "60123.4",
            },
        }
    ]
    summary["signals"] = []
    summary["intents"] = []
    summary["positions"] = []
    path = tmp_path / "events.jsonl"

    EventWriter(path, enabled=True).emit_run_summary(summary)

    payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
    assert payload["order_id"] == "9123"
    assert payload["client_order_id"] == "client-9123"
    assert payload["executed_quantity"] == "0.02"
    assert payload["average_price"] == "60123.4"


def test_fill_event_preserves_explicit_zero_values(tmp_path: Path) -> None:
    summary = sample_summary()
    summary["signals"] = []
    summary["intents"] = []
    summary["positions"] = []
    summary["results"] = [
        {
            "status": "submitted",
            "intent": {"symbol": "BTCUSDT", "action": "open_long"},
            "executed_quantity": 0,
            "average_price": 0,
            "response": {"orderId": 9124, "executedQty": "1", "avgPrice": "60123.4"},
        }
    ]
    path = tmp_path / "events.jsonl"

    EventWriter(path, enabled=True).emit_run_summary(summary)

    payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
    assert payload["executed_quantity"] == 0
    assert payload["average_price"] == 0


def test_event_writer_isolates_accounts_by_file(tmp_path: Path) -> None:
    demo_path = tmp_path / "demo" / "events.jsonl"
    live_path = tmp_path / "live" / "events.jsonl"
    EventWriter(demo_path, enabled=True, account_id="demo").emit_run_summary(sample_summary())
    EventWriter(live_path, enabled=True, account_id="momo").emit_run_summary(sample_summary())

    assert '"account_id":"demo"' in demo_path.read_text(encoding="utf-8")
    assert '"account_id":"momo"' in live_path.read_text(encoding="utf-8")
    assert '"account_id":"momo"' not in demo_path.read_text(encoding="utf-8")


def test_event_writer_skips_non_order_failures_and_never_raises_from_safe_path(tmp_path: Path) -> None:
    summary = sample_summary()
    summary["results"].append({"symbol": "SOLUSDT", "status": "signal_failed", "error": "market data unavailable"})
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, enabled=True)

    assert writer.emit_run_summary(summary) == 6
    assert "signal_failed" not in path.read_text(encoding="utf-8")
    assert EventWriter(tmp_path, enabled=True).safe_emit_run_summary(summary) == 0


def test_v6_config_event_stream_defaults_disabled_and_follows_state_dir(tmp_path: Path) -> None:
    config = V6ExecutionConfig(state_dir=tmp_path)

    assert config.trade_event_logging_enabled is False
    assert config.trade_events_file == tmp_path / "trade_events.jsonl"
