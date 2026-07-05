import json
from dataclasses import replace
from pathlib import Path

from trading_system.config import StrategyConfig
from trading_system.execution import PaperExecutionEngine
from trading_system.execution_runner import BinanceExecutionRunner
from trading_system.models import MarketSnapshot
from trading_system.parameter_optimizer import run_bayesian_parameter_search
from trading_system.pipeline import TradingPipeline
from trading_system.regime import detect_market_regime
from trading_system.self_train_predictor import train_predictive_model


class StubWalRunner(BinanceExecutionRunner):
    def __post_init__(self) -> None:
        self.config.execution_result_dir.mkdir(parents=True, exist_ok=True)
        self.config.execution_archive_dir.mkdir(parents=True, exist_ok=True)
        self.config.execution_wal_file.parent.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        self.calls.append(("leverage", symbol, leverage))

    def _signed_request(self, method: str, endpoint: str, params: dict[str, str]) -> dict:
        self.calls.append((method, endpoint, dict(params)))
        return {"ok": True, "symbol": params.get("symbol"), "endpoint": endpoint}

    def _get_mark_price(self, symbol: str) -> float:
        return 2.0

    def _get_symbol_quantity_rules(self, symbol: str) -> tuple[float, float]:
        return (0.1, 0.1)

    def _get_symbol_tick_size(self, symbol: str) -> float:
        return 0.01


def test_execution_runner_records_wal_lifecycle(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_1.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-13T00:00:00+00:00",
                "request_mode": "live_order",
                "skill_name": "derivatives-trading-usds-futures",
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "AIXBTUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "leverage": 3, "quantity_hint_usdt": 100, "stop_loss_price": 1.87},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_wal_file=tmp_path / "execution_wal.jsonl",
        binance_api_key="key",
        binance_api_secret="secret",
        live_execution_confirmation="CONFIRM_LIVE_BINANCE_FUTURES",
    )
    runner = StubWalRunner(config)

    summary = runner.execute_request_file(request_file)

    wal_rows = [json.loads(line) for line in config.execution_wal_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert summary["status"] == "submitted"
    assert [row["stage"] for row in wal_rows] == ["prepared", "submitted", "protective_stop_submitted", "committed"]
    assert wal_rows[0]["request_id"].endswith(":1")
    assert wal_rows[-1]["status"] == "submitted"


def test_paper_engine_triggers_flash_crash_protection_and_skips_new_entries(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        market_flash_crash_drop_pct_5m=8.0,
        market_flash_crash_symbol_count=2,
    )
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "AAAUSDT",
                        "opened_at": "2026-04-22T00:00:00+00:00",
                        "entry_price": 1.0,
                        "current_price": 1.0,
                        "stop_loss_price": 0.97,
                        "take_profit_price": 1.08,
                        "notional_usdt": 200.0,
                        "score": 82.0,
                        "rationale": "trend",
                        "setup_type": "trend",
                    }
                ],
                "journal": [],
            }
        ),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config)
    crash_a = MarketSnapshot(symbol="AAAUSDT", mark_price=0.91, price_change_pct_5m=-9.0, price_change_pct_24h=-12.0, quote_volume_usdt_24h=90_000_000, funding_rate=-0.0008, open_interest_usdt=20_000_000, oi_change_pct_4h=-10.0, taker_buy_sell_ratio=0.88, listing_days=100, sentiment_score=20.0, sentiment_mentions=2, sentiment_source="square_feed", score=25.0)
    crash_b = MarketSnapshot(symbol="BBBUSDT", mark_price=0.45, price_change_pct_5m=-11.0, price_change_pct_24h=-18.0, quote_volume_usdt_24h=70_000_000, funding_rate=-0.0007, open_interest_usdt=16_000_000, oi_change_pct_4h=-14.0, taker_buy_sell_ratio=0.86, listing_days=100, sentiment_score=18.0, sentiment_mentions=1, sentiment_source="square_feed", score=20.0)
    candidate = MarketSnapshot(symbol="NEWUSDT", mark_price=1.1, price_change_pct_5m=1.0, price_change_pct_24h=14.0, quote_volume_usdt_24h=120_000_000, funding_rate=0.0002, open_interest_usdt=50_000_000, oi_change_pct_4h=12.0, taker_buy_sell_ratio=1.14, listing_days=30, sentiment_score=80.0, sentiment_mentions=15, sentiment_source="square_feed", setup_type="base_breakout_confirmed", setup_quality=85.0, score=83.0)

    result = engine.rebalance([crash_a, crash_b, candidate])

    assert result["entries"] == []
    assert len(result["exits"]) == 1
    assert result["exits"][0]["rationale"] == "market_flash_crash_protection"


def test_predictive_model_training_returns_useful_weights_and_predictions(tmp_path: Path) -> None:
    decision_path = tmp_path / "decisions.jsonl"
    closed_path = tmp_path / "closed.jsonl"
    decision_rows = [
        {"sample_key": "d1", "symbol": "AAAUSDT", "score": 88.0, "sentiment_score": 84.0, "sentiment_mentions": 20, "taker_buy_sell_ratio": 1.18, "oi_change_pct_4h": 16.0, "execution_status": "executable_now", "decision_label": "entered", "selected_for_entry": True, "setup_type": "base_breakout_confirmed"},
        {"sample_key": "d2", "symbol": "BBBUSDT", "score": 52.0, "sentiment_score": 42.0, "sentiment_mentions": 2, "taker_buy_sell_ratio": 0.97, "oi_change_pct_4h": -3.0, "execution_status": "watch_only", "decision_label": "blocked", "selected_for_entry": False, "setup_type": "trend"},
    ]
    closed_rows = [
        {"sample_key": "c1", "symbol": "AAAUSDT", "is_win": True, "pnl_pct": 0.06, "reason_family": "take_profit_hit", "setup_type": "base_breakout_confirmed"},
        {"sample_key": "c2", "symbol": "BBBUSDT", "is_win": False, "pnl_pct": -0.03, "reason_family": "stop_loss_hit", "setup_type": "trend"},
    ]
    decision_path.write_text("\n".join(json.dumps(row) for row in decision_rows) + "\n", encoding="utf-8")
    closed_path.write_text("\n".join(json.dumps(row) for row in closed_rows) + "\n", encoding="utf-8")

    model = train_predictive_model(decision_path, closed_path)

    assert model["status"] == "ok"
    assert model["feature_weights"]["sentiment_score"] > 0
    assert model["feature_weights"]["taker_buy_sell_ratio"] > 0
    assert model["symbol_stats"]["AAAUSDT"]["win_rate"] == 1.0
    assert model["symbol_stats"]["BBBUSDT"]["win_rate"] == 0.0


def test_pipeline_source_independence_detection_flags_price_loop_sources() -> None:
    pipeline = TradingPipeline(StrategyConfig())
    summary = pipeline._build_source_independence_summary(
        [
            {"symbol": "LOOPUSDT", "source": "binance_top_movers,coingecko_trending", "hotness_score": 80.0, "mention_count": 5},
            {"symbol": "SAFEUSDT", "source": "square_feed,manual_distill", "hotness_score": 78.0, "mention_count": 8},
        ]
    )

    risky = next(item for item in summary["symbols"] if item["symbol"] == "LOOPUSDT")
    safe = next(item for item in summary["symbols"] if item["symbol"] == "SAFEUSDT")
    assert risky["independence_state"] == "price_loop_risk"
    assert safe["independence_state"] == "independent"
    assert risky["independence_score"] < safe["independence_score"]


def test_detect_market_regime_identifies_flash_crash_environment() -> None:
    regime = detect_market_regime(
        [
            MarketSnapshot(symbol="AAAUSDT", mark_price=1.0, price_change_pct_5m=-8.2, price_change_pct_24h=-15.0, quote_volume_usdt_24h=100_000_000, funding_rate=-0.0008, open_interest_usdt=20_000_000, oi_change_pct_4h=-9.0, taker_buy_sell_ratio=0.89, listing_days=80, sentiment_score=20.0, sentiment_mentions=2, sentiment_source="square_feed"),
            MarketSnapshot(symbol="BBBUSDT", mark_price=1.0, price_change_pct_5m=-9.1, price_change_pct_24h=-18.0, quote_volume_usdt_24h=100_000_000, funding_rate=-0.0009, open_interest_usdt=20_000_000, oi_change_pct_4h=-11.0, taker_buy_sell_ratio=0.87, listing_days=80, sentiment_score=22.0, sentiment_mentions=2, sentiment_source="square_feed"),
            MarketSnapshot(symbol="CCCUSDT", mark_price=1.0, price_change_pct_5m=-7.4, price_change_pct_24h=-12.0, quote_volume_usdt_24h=100_000_000, funding_rate=-0.0007, open_interest_usdt=20_000_000, oi_change_pct_4h=-10.0, taker_buy_sell_ratio=0.9, listing_days=80, sentiment_score=18.0, sentiment_mentions=1, sentiment_source="square_feed"),
        ],
        StrategyConfig(),
    )

    assert regime["regime"] == "flash_crash"
    assert regime["entry_threshold_multiplier"] > 1.0
    assert regime["allow_new_entries"] is False


def test_bayesian_parameter_search_returns_best_trial(tmp_path: Path) -> None:
    dataset = tmp_path / "closed.jsonl"
    rows = [
        {"sample_key": "1", "symbol": "AAAUSDT", "setup_type": "base_breakout_confirmed", "is_win": True, "pnl_pct": 0.06, "mentions": 20, "hotness_score": 82.0},
        {"sample_key": "2", "symbol": "AAAUSDT", "setup_type": "base_breakout_confirmed", "is_win": True, "pnl_pct": 0.04, "mentions": 18, "hotness_score": 80.0},
        {"sample_key": "3", "symbol": "BBBUSDT", "setup_type": "trend", "is_win": False, "pnl_pct": -0.03, "mentions": 2, "hotness_score": 41.0},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = run_bayesian_parameter_search(dataset, iterations=6)

    assert result["status"] == "ok"
    assert result["best_trial"]["objective"] == max(item["objective"] for item in result["trials"])
    assert "entry_threshold" in result["best_params"]


def test_pipeline_exposes_regime_predictive_and_independence_sections(tmp_path: Path) -> None:
    config = replace(
        StrategyConfig(),
        sentiment_file=tmp_path / "sentiment.csv",
        replay_file=tmp_path / "replay.json",
        square_feed_report_file=tmp_path / "square_feed.json",
        merged_sentiment_file=tmp_path / "merged.csv",
        state_file=tmp_path / "state.json",
        report_file=tmp_path / "scan.json",
        pipeline_report_file=tmp_path / "pipeline.json",
        pipeline_report_md_file=tmp_path / "pipeline.md",
        issues_json_file=tmp_path / "issues.json",
        issues_md_file=tmp_path / "issues.md",
        skill_bridge_dir=tmp_path / "skill_bridges",
        closed_trade_dataset_file=tmp_path / "paper_closed_trades.jsonl",
        decision_dataset_file=tmp_path / "paper_decision_samples.jsonl",
    )
    config.sentiment_file.write_text("symbol,hotness_score,mention_count,source,updated_at,notes\nAAAUSDT,88,12,square_feed,2026-04-12T00:00:00+00:00,test\n", encoding="utf-8")
    config.replay_file.write_text(json.dumps([{"symbol": "AAAUSDT", "mark_price": 1.0, "price_change_pct_5m": -8.5, "price_change_pct_24h": -14.0, "quote_volume_usdt_24h": 150000000, "funding_rate": -0.0008, "open_interest_usdt": 82000000, "oi_change_pct_4h": -11.0, "taker_buy_sell_ratio": 0.88, "listing_days": 43, "sentiment_score": 88.0, "sentiment_mentions": 12, "sentiment_source": "binance_top_movers,coingecko_trending"}]), encoding="utf-8")
    config.square_feed_report_file.write_text(json.dumps({"posts": []}), encoding="utf-8")
    config.closed_trade_dataset_file.write_text(json.dumps({"sample_key": "1", "symbol": "AAAUSDT", "is_win": True, "pnl_pct": 0.03, "setup_type": "trend"}) + "\n", encoding="utf-8")
    config.decision_dataset_file.write_text(json.dumps({"sample_key": "1", "symbol": "AAAUSDT", "score": 88.0, "sentiment_score": 88.0, "sentiment_mentions": 12, "taker_buy_sell_ratio": 0.88, "oi_change_pct_4h": -11.0, "execution_status": "watch_only", "decision_label": "blocked", "selected_for_entry": False, "setup_type": "trend"}) + "\n", encoding="utf-8")

    report = TradingPipeline(config).run(mode="replay", collect_square_feed=False)

    assert report["market_regime"]["regime"] in {"flash_crash", "risk_off"}
    assert report["predictive_model_summary"]["status"] == "ok"
    assert report["source_independence_summary"]["symbols"]
