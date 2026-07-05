import json
from pathlib import Path

from trading_system.config import StrategyConfig
from trading_system.execution_bridge import BinanceSkillExecutionBridge


def test_execution_bridge_writes_request_file(tmp_path: Path) -> None:
    config = StrategyConfig(execution_request_dir=tmp_path / "requests")
    bridge = BinanceSkillExecutionBridge(config)
    summary = bridge.write_requests(
        {
            "entries": [
                {
                    "symbol": "AIXBTUSDT",
                    "notional_usdt": 200,
                    "stop_loss_price": 0.7,
                    "take_profit_price": 0.8,
                    "rationale": "test",
                }
            ],
            "exits": [
                {
                    "symbol": "EXITUSDT",
                    "notional_usdt": 200,
                    "rationale": "score_faded",
                }
            ],
        },
        [{"symbol": "AIXBTUSDT", "score": 80.0}, {"symbol": "EXITUSDT", "score": 20.0}],
    )
    assert summary["intent_count"] == 2
    path = Path(summary["request_file"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["skill_name"] == "derivatives-trading-usds-futures"
    assert payload["binance_requests"][0]["endpoint"] == "/fapi/v1/order/test"
    assert payload["binance_requests"][0]["params"]["side"] == "SELL"
    assert payload["binance_requests"][0]["params"]["reduceOnly"] == "true"
    assert payload["binance_requests"][1]["params"]["side"] == "BUY"
    assert "reduceOnly" not in payload["binance_requests"][1]["params"]


def test_execution_bridge_live_request_does_not_cap_model_chosen_notional(tmp_path: Path) -> None:
    config = StrategyConfig(
        execution_request_dir=tmp_path / "requests",
        execution_test_order=False,
        live_max_order_notional_usdt=100,
    )
    bridge = BinanceSkillExecutionBridge(config)

    summary = bridge.write_requests(
        {
            "entries": [
                {
                    "symbol": "AIXBTUSDT",
                    "entry_price": 1.0,
                    "notional_usdt": 200,
                    "stop_loss_price": 0.95,
                    "take_profit_price": 1.08,
                    "rationale": "test",
                }
            ],
            "exits": [],
        },
        [{"symbol": "AIXBTUSDT", "score": 80.0}],
    )

    assert summary["intent_count"] == 1
    assert summary["blocked_intent_count"] == 0
    assert summary["risk_controls"]["status"] == "ok"
    payload = json.loads(Path(summary["request_file"]).read_text(encoding="utf-8"))
    assert payload["request_mode"] == "live_order"
    assert payload["binance_requests"][0]["meta"]["quantity_hint_usdt"] == 200
    assert payload["risk_controls"]["live_max_order_notional_usdt"] == 100.0


def test_execution_bridge_adds_position_panel_monitor_for_new_entries(tmp_path: Path) -> None:
    config = StrategyConfig(execution_request_dir=tmp_path / "requests")
    bridge = BinanceSkillExecutionBridge(config)
    summary = bridge.write_requests(
        {
            "entries": [
                {
                    "symbol": "AIXBTUSDT",
                    "notional_usdt": 200,
                    "stop_loss_price": 0.7,
                    "take_profit_price": 0.8,
                    "rationale": "smart_money_single_side_90pct",
                    "smart_money_context": {
                        "oi_composition": "single_side_庄家盘",
                        "monitor_action": "watch_add_on_shallow_pullback",
                        "whale_long_count": 100,
                        "smart_money_long_count": 300,
                    },
                }
            ],
            "exits": [],
        },
        [{"symbol": "AIXBTUSDT", "score": 88.0}],
    )
    payload = json.loads(Path(summary["request_file"]).read_text(encoding="utf-8"))
    meta = payload["binance_requests"][0]["meta"]
    assert meta["smart_money_context"]["oi_composition"] == "single_side_庄家盘"
    assert meta["position_panel_monitor"]["enabled"] is True
    assert meta["position_panel_monitor"]["views"] == ["smart_money", "whales"]
    assert meta["position_panel_monitor"]["chart_subjects"] == ["smart_money_position", "whale_position"]
    assert meta["position_panel_monitor"]["monitor_action"] == "watch_add_on_shallow_pullback"


def test_execution_bridge_can_route_shadow_paper_to_testnet_orders(tmp_path: Path) -> None:
    config = StrategyConfig(
        execution_request_dir=tmp_path / "requests",
        execution_order_mode="testnet_order",
        execution_signal_source="shadow_paper",
    )
    bridge = BinanceSkillExecutionBridge(config)

    summary = bridge.write_requests(
        {
            "entries": [],
            "exits": [],
            "shadow_paper": {
                "entries": [
                    {
                        "action": "open_long",
                        "symbol": "BILLUSDT",
                        "entry_price": 0.1,
                        "notional_usdt": 50,
                        "stop_loss_price": 0.096,
                        "take_profit_price": 0.108,
                        "rationale": "shadow_paper",
                        "shadow_position_key": "BILLUSDT:2026-05-14T00:00:00+00:00",
                    }
                ],
                "exits": [],
            },
        },
        [{"symbol": "BILLUSDT", "score": 80.0}],
    )

    payload = json.loads(Path(summary["request_file"]).read_text(encoding="utf-8"))
    assert summary["request_mode"] == "testnet_order"
    assert summary["signal_source"] == "shadow_paper"
    assert payload["request_mode"] == "testnet_order"
    assert payload["binance_requests"][0]["endpoint"] == "/fapi/v1/order"
    assert payload["binance_requests"][0]["meta"]["request_mode"] == "testnet_order"
    assert payload["binance_requests"][0]["meta"]["shadow_position_key"] == "BILLUSDT:2026-05-14T00:00:00+00:00"


def test_execution_bridge_suppresses_same_symbol_reentry_when_exit_exists(tmp_path: Path) -> None:
    config = StrategyConfig(
        execution_request_dir=tmp_path / "requests",
        execution_order_mode="testnet_order",
        execution_signal_source="shadow_paper",
    )
    bridge = BinanceSkillExecutionBridge(config)

    summary = bridge.write_requests(
        {
            "shadow_paper": {
                "entries": [
                    {
                        "action": "open_long",
                        "symbol": "GTCUSDT",
                        "entry_price": 0.163,
                        "notional_usdt": 15,
                        "stop_loss_price": 0.156,
                        "take_profit_price": 0.176,
                        "rationale": "shadow_reentry",
                    }
                ],
                "exits": [
                    {
                        "symbol": "GTCUSDT",
                        "notional_usdt": 15,
                        "rationale": "shadow_take_profit_hit",
                    }
                ],
            },
        },
        [{"symbol": "GTCUSDT", "score": 23.0}],
    )

    payload = json.loads(Path(summary["request_file"]).read_text(encoding="utf-8"))
    assert summary["intent_count"] == 1
    assert summary["suppressed_entry_count"] == 1
    assert payload["risk_controls"]["same_symbol_reentry_suppressed"] == ["GTCUSDT"]
    assert payload["binance_requests"][0]["meta"]["action"] == "close_long"
    assert payload["binance_requests"][0]["params"]["reduceOnly"] == "true"


def test_execution_bridge_blocks_close_intent_for_submit_failed_mirror(tmp_path: Path) -> None:
    config = StrategyConfig(
        execution_request_dir=tmp_path / "requests",
        execution_order_mode="testnet_order",
        execution_signal_source="shadow_paper",
    )
    bridge = BinanceSkillExecutionBridge(config)

    summary = bridge.write_requests(
        {
            "shadow_paper": {
                "entries": [],
                "exits": [
                    {
                        "symbol": "AAAUSDT",
                        "notional_usdt": 15,
                        "rationale": "shadow_stop_loss_hit",
                        "binance_mirror_status": "submit_failed",
                    },
                    {
                        "symbol": "BBBUSDT",
                        "notional_usdt": 15,
                        "rationale": "shadow_max_hold_reached",
                        "binance_mirror_status": "submitted",
                    },
                ],
            },
        },
        [{"symbol": "AAAUSDT", "score": 10.0}, {"symbol": "BBBUSDT", "score": 8.0}],
    )

    payload = json.loads(Path(summary["request_file"]).read_text(encoding="utf-8"))
    assert summary["intent_count"] == 1
    assert len(payload["binance_requests"]) == 1
    assert payload["binance_requests"][0]["params"]["symbol"] == "BBBUSDT"


def test_execution_bridge_blocks_close_intent_for_pending_submit_mirror(tmp_path: Path) -> None:
    config = StrategyConfig(
        execution_request_dir=tmp_path / "requests",
        execution_order_mode="testnet_order",
        execution_signal_source="shadow_paper",
    )
    bridge = BinanceSkillExecutionBridge(config)

    summary = bridge.write_requests(
        {
            "shadow_paper": {
                "entries": [],
                "exits": [
                    {
                        "symbol": "CCCUSDT",
                        "notional_usdt": 15,
                        "rationale": "shadow_stop_loss_hit",
                        "binance_mirror_status": "pending_submit",
                    },
                ],
            },
        },
        [{"symbol": "CCCUSDT", "score": 10.0}],
    )

    payload = json.loads(Path(summary["request_file"]).read_text(encoding="utf-8"))
    assert summary["intent_count"] == 0
    assert len(payload["binance_requests"]) == 0


def test_execution_bridge_allows_close_for_legacy_exit_without_mirror_status(tmp_path: Path) -> None:
    config = StrategyConfig(
        execution_request_dir=tmp_path / "requests",
        execution_order_mode="testnet_order",
        execution_signal_source="shadow_paper",
    )
    bridge = BinanceSkillExecutionBridge(config)

    summary = bridge.write_requests(
        {
            "shadow_paper": {
                "entries": [],
                "exits": [
                    {
                        "symbol": "DDDUSDT",
                        "notional_usdt": 15,
                        "rationale": "shadow_take_profit_hit",
                    },
                ],
            },
        },
        [{"symbol": "DDDUSDT", "score": 10.0}],
    )

    payload = json.loads(Path(summary["request_file"]).read_text(encoding="utf-8"))
    assert summary["intent_count"] == 1
    assert payload["binance_requests"][0]["params"]["symbol"] == "DDDUSDT"
