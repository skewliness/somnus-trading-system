import json
from pathlib import Path

from trading_system.factor_backtest import (
    build_binance_hf_market_jsonl,
    discover_quant_factors,
    distill_factor_training_samples,
    run_factor_backtest,
    run_high_frequency_backtest,
    _symbols_from_replay_snapshot,
)


def test_factor_backtest_emits_bucket_recommendations(tmp_path: Path) -> None:
    dataset = tmp_path / "closed.jsonl"
    rows = [
        {"symbol": "AAAUSDT", "price_change_pct_5m": 4.0, "taker_buy_sell_ratio": 1.08, "pnl_pct": 0.03, "is_win": True, "reason_family": "take_profit"},
        {"symbol": "BBBUSDT", "price_change_pct_5m": 5.0, "taker_buy_sell_ratio": 1.07, "pnl_pct": 0.02, "is_win": True, "reason_family": "take_profit"},
        {"symbol": "CCCUSDT", "price_change_pct_5m": 6.0, "taker_buy_sell_ratio": 1.09, "pnl_pct": 0.01, "is_win": True, "reason_family": "take_profit"},
        {"symbol": "DDDUSDT", "price_change_pct_5m": -2.0, "taker_buy_sell_ratio": 0.95, "pnl_pct": -0.03, "is_win": False, "reason_family": "stop_loss_hit"},
        {"symbol": "EEEUSDT", "price_change_pct_5m": -1.0, "taker_buy_sell_ratio": 0.96, "pnl_pct": -0.02, "is_win": False, "reason_family": "stop_loss_hit"},
        {"symbol": "中文USDT", "price_change_pct_5m": 99.0, "pnl_pct": 1.0, "is_win": True},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "factor.json"

    report = run_factor_backtest(dataset, output, min_samples=2)

    assert report["status"] == "ok"
    assert report["closed_rows"] == 5
    assert output.exists()
    recs = {(item["factor"], item["bucket"], item["action"]) for item in report["recommendations"]}
    assert ("price_change_pct_5m", "warm", "boost") in recs
    assert ("price_change_pct_5m", "negative", "penalize") in recs


def test_high_frequency_backtest_builds_event_trades_and_risk_metrics(tmp_path: Path) -> None:
    market_file = tmp_path / "hf_market.jsonl"
    rows = [
        {"symbol": "AAAUSDT", "timestamp": i, "close": close, "volume": volume, "funding_rate": 0.0002}
        for i, (close, volume) in enumerate(
            [
                (100.0, 1000),
                (100.2, 1100),
                (100.5, 1200),
                (104.0, 4500),
                (108.5, 4200),
                (107.0, 3900),
                (106.0, 3600),
            ],
            start=1,
        )
    ]
    for idx, row in enumerate(rows):
        row["oi_change_pct"] = 0.8 if idx >= 3 else 0.0
    market_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "hf_report.json"

    report = run_high_frequency_backtest(market_file, output, lookback=3, hold_bars=2, min_momentum_pct=2.0, volume_z_min=1.0, min_oi_change_pct=0.5)

    assert report["status"] == "ok"
    assert report["events"] >= 1
    assert report["baseline"]["count"] == report["events"]
    assert report["baseline"]["fixed_horizon_loss_rate"] >= 0
    assert report["risk"]["max_drawdown_pct"] <= 0
    assert report["factor_rankings"]
    assert any(item["factor"] == "hf_momentum_pct" for item in report["recommendations"])
    assert report["strategy"]["name"] == "hf_oi_momentum_warm_start_overheat_filter_v3"
    assert all(item["reason_family"] != "stop_loss_hit" for item in report["sample_events"])
    assert all(item["hf_oi_change_pct"] >= 0.5 for item in report["sample_events"])
    assert all(item["factor"] not in {"hf_forward_return_pct", "exit_price"} for item in report["factor_rankings"])
    assert output.exists()


def test_high_frequency_backtest_filters_overheated_chase_entries(tmp_path: Path) -> None:
    market_file = tmp_path / "hf_market_overheated.jsonl"
    rows = [
        {"symbol": "AAAUSDT", "timestamp": i, "close": close, "volume": volume, "oi_change_pct": oi}
        for i, (close, volume, oi) in enumerate(
            [
                (100.0, 1000, 0.0),
                (100.1, 1050, 0.0),
                (100.2, 1100, 0.0),
                (101.0, 1700, 0.8),  # warm valid entry
                (104.8, 9000, 2.5),  # overheated momentum/volume/OI, should be skipped
                (105.0, 8500, 2.8),
                (105.2, 8400, 3.0),
            ],
            start=1,
        )
    ]
    market_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    unfiltered = run_high_frequency_backtest(
        market_file,
        lookback=3,
        hold_bars=1,
        min_momentum_pct=0.2,
        volume_z_min=0.3,
        min_oi_change_pct=0.5,
    )
    filtered = run_high_frequency_backtest(
        market_file,
        lookback=3,
        hold_bars=1,
        min_momentum_pct=0.2,
        volume_z_min=0.3,
        min_oi_change_pct=0.5,
        max_momentum_pct=2.0,
        volume_z_max=20.0,
        max_oi_change_pct=1.5,
    )

    assert unfiltered["events"] > filtered["events"] >= 1
    assert filtered["filters"]["skipped_overheated"]["combo"] >= 1
    assert filtered["strategy"]["max_momentum_pct"] == 2.0


def test_quant_factor_discovery_ranks_predictive_factors(tmp_path: Path) -> None:
    dataset = tmp_path / "closed.jsonl"
    rows = []
    for i in range(6):
        rows.append({"symbol": f"WIN{i}USDT", "alpha_pressure": 8 + i, "noise_factor": i % 2, "pnl_pct": 0.03, "is_win": True})
    for i in range(6):
        rows.append({"symbol": f"LOSS{i}USDT", "alpha_pressure": -8 - i, "noise_factor": i % 2, "pnl_pct": -0.025, "is_win": False})
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = discover_quant_factors(dataset, min_samples=4, top_n=5)

    assert report["status"] == "ok"
    assert report["top_factors"][0]["factor"] == "alpha_pressure"
    assert report["top_factors"][0]["direction"] == "higher_is_better"
    assert report["top_factors"][0]["edge_score"] > 0



def test_factor_distill_filters_noise_and_writes_model_samples(tmp_path: Path) -> None:
    dataset = tmp_path / "closed.jsonl"
    rows = [
        {"sample_key": "w1", "symbol": "AAAUSDT", "price_change_pct_5m": 4.0, "pnl_pct": 0.03, "is_win": True, "reason_family": "take_profit_hit"},
        {"sample_key": "w2", "symbol": "BBBUSDT", "price_change_pct_5m": 5.0, "pnl_pct": 0.02, "is_win": True, "reason_family": "take_profit_hit"},
        {"sample_key": "l1", "symbol": "CCCUSDT", "price_change_pct_5m": -2.0, "pnl_pct": -0.03, "is_win": False, "reason_family": "stop_loss_hit"},
        {"sample_key": "bad", "symbol": "中文USDT", "price_change_pct_5m": 99.0, "pnl_pct": 1.0, "is_win": True},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = run_factor_backtest(dataset, min_samples=1)
    output = tmp_path / "factor_distilled.jsonl"

    summary = distill_factor_training_samples(dataset, report, output, min_rows_for_model=2)

    assert summary["status"] == "ok"
    assert summary["source_rows"] == 3
    assert summary["distilled_rows"] == 2
    assert summary["rejected_rows"] == 1
    assert summary["model_training_ready"] is True
    distilled = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {row["sample_key"] for row in distilled} == {"w1", "w2"}
    assert all(row["factor_distilled"] for row in distilled)


def test_replay_snapshot_symbols_and_binance_hf_jsonl_builder(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            [
                {"symbol": "AAAUSDT", "mark_price": 1.0},
                {"symbol": "中文USDT", "mark_price": 2.0},
                {"symbol": "BBBUSDT", "mark_price": 3.0},
            ]
        ),
        encoding="utf-8",
    )
    assert _symbols_from_replay_snapshot(replay, limit=10) == ["AAAUSDT", "BBBUSDT"]

    class FakeClient:
        def get_json(self, path, params=None):
            symbol = params["symbol"]
            if path == "/futures/data/openInterestHist":
                return [
                    {"timestamp": 1, "sumOpenInterestValue": "100000"},
                    {"timestamp": 2, "sumOpenInterestValue": "101000"},
                    {"timestamp": 3, "sumOpenInterestValue": "103000"},
                ] if symbol == "AAAUSDT" else []
            assert path == "/fapi/v1/klines"
            return [
                [1, "100", "101", "99", "100", "1000", 0, "100000", 10],
                [2, "100", "106", "99", "105", "5000", 0, "500000", 20],
                [3, "105", "109", "104", "108", "4500", 0, "486000", 30],
            ] if symbol == "AAAUSDT" else []

    output = tmp_path / "hf.jsonl"
    report = build_binance_hf_market_jsonl(["AAAUSDT", "BAD中文USDT"], output, client=FakeClient(), limit=3)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert report["status"] == "ok"
    assert report["rows"] == 3
    assert rows[0]["symbol"] == "AAAUSDT"
    assert rows[0]["timestamp"] == 1
    assert rows[0]["close"] == 100.0
    assert rows[0]["volume"] == 1000.0
    assert rows[0]["open_interest_usdt"] == 100000.0
    assert rows[1]["oi_change_pct"] == 1.0


def test_binance_hf_jsonl_builder_skips_bad_symbols(tmp_path: Path) -> None:
    class FakeClient:
        def get_json(self, path, params=None):
            if path == "/futures/data/openInterestHist":
                return []
            if params["symbol"] == "BADUSDT":
                raise RuntimeError("bad symbol")
            return [[1, "1", "2", "1", "1.5", "10", 0, "15", 1]]

    output = tmp_path / "hf.jsonl"
    report = build_binance_hf_market_jsonl(["BADUSDT", "GOODUSDT"], output, client=FakeClient(), limit=1)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert report["status"] == "ok_partial"
    assert report["rows"] == 1
    assert report["per_symbol"]["BADUSDT"] == 0
    assert "BADUSDT" in report["errors"]
    assert rows[0]["symbol"] == "GOODUSDT"


def test_high_frequency_backtest_preserves_smart_money_and_whale_panel_context(tmp_path: Path) -> None:
    market_file = tmp_path / "hf_market_with_panel.jsonl"
    panel = {
        "single_side_pct": 92.0,
        "oi_composition": "single_side_庄家盘",
        "symbol_class": "small_mid_or_meme",
        "monitor_action": "watch_add_on_shallow_pullback",
        "whale_long_position_usdt": 9_200_000,
        "whale_short_position_usdt": 800_000,
        "smart_money_long_count": 300,
        "smart_money_short_count": 50,
    }
    rows = [
        {"symbol": "AAAUSDT", "timestamp": i, "close": close, "volume": volume, "funding_rate": 0.0002, "oi_change_pct": 0.8 if i >= 4 else 0.0, "smart_money_context": panel}
        for i, (close, volume) in enumerate(
            [(100.0, 1000), (100.2, 1100), (100.5, 1200), (104.0, 4500), (108.5, 4200), (107.0, 3900), (106.0, 3600)],
            start=1,
        )
    ]
    market_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = run_high_frequency_backtest(market_file, lookback=3, hold_bars=2, min_momentum_pct=2.0, volume_z_min=1.0, min_oi_change_pct=0.5)

    assert report["status"] == "ok"
    event = report["sample_events"][0]
    assert event["smart_money_single_side_pct"] == 92.0
    assert event["smart_money_oi_composition"] == "single_side_庄家盘"
    assert event["smart_money_monitor_action"] == "watch_add_on_shallow_pullback"
    assert event["whale_long_position_usdt"] == 9_200_000
