from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

from trading_system.bridges import BridgeLoadResult
from trading_system.config import StrategyConfig
from trading_system.engine import TradingSystem
from trading_system.models import MarketSnapshot
from trading_system.pipeline import TradingPipeline
from trading_system.reports import write_markdown_report


def test_pipeline_writes_outputs(tmp_path: Path) -> None:
    sentiment = tmp_path / "sentiment.csv"
    sentiment.write_text(
        "symbol,hotness_score,mention_count,source,updated_at,notes\nAIXBTUSDT,91,12,manual_distill,2026-04-12T00:00:00+00:00,test\n",
        encoding="utf-8",
    )
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            [
                {
                    "symbol": "AIXBTUSDT",
                    "mark_price": 0.742,
                    "price_change_pct_24h": 18.4,
                    "quote_volume_usdt_24h": 135000000,
                    "funding_rate": 0.00042,
                    "open_interest_usdt": 82200000,
                    "oi_change_pct_4h": 16.7,
                    "taker_buy_sell_ratio": 1.18,
                    "listing_days": 43,
                    "price_position_7d": 0.95,
                    "price_position_14d": 0.72,
                    "price_position_20d": 0.54,
                    "breakout_distance_7d": 0.016,
                    "compression_score": 0.74
                }
            ]
        ),
        encoding="utf-8",
    )
    square = tmp_path / "square_feed.json"
    square.write_text(json.dumps({"posts": []}), encoding="utf-8")

    config = replace(
        StrategyConfig(),
        sentiment_file=sentiment,
        replay_file=replay,
        square_feed_report_file=square,
        merged_sentiment_file=tmp_path / "merged.csv",
        state_file=tmp_path / "state.json",
        report_file=tmp_path / "scan.json",
        pipeline_report_file=tmp_path / "pipeline.json",
        pipeline_report_md_file=tmp_path / "pipeline.md",
        options_delivery_report_file=tmp_path / "options_delivery.json",
        meme_radar_archive_dir=tmp_path / "meme_radar_daily",
        issues_json_file=tmp_path / "issues.json",
        issues_md_file=tmp_path / "issues.md",
        skill_bridge_dir=tmp_path / "skill_bridges",
        macro_trend_enabled=False,
    )
    pipeline = TradingPipeline(config)
    ranked_snapshot = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=0.742,
        price_change_pct_24h=18.4,
        quote_volume_usdt_24h=135000000,
        funding_rate=0.00042,
        open_interest_usdt=82200000,
        oi_change_pct_4h=16.7,
        taker_buy_sell_ratio=1.18,
        listing_days=43,
        sentiment_score=91.0,
        sentiment_mentions=12,
        sentiment_source="manual_distill",
        score=84.0,
        reasons=["setup=base_breakout_confirmed:92.0"],
        manipulation_stage="markup",
        manipulation_quality=86.0,
        manipulation_flags=["compressed_base", "oi_expanding"],
        price_position_7d=0.95,
        price_position_14d=0.72,
        price_position_20d=0.54,
        breakout_distance_7d=0.016,
        compression_score=0.74,
    )
    replay_rebalance = {
        "entries": [
            {
                "symbol": "AIXBTUSDT",
                "notional_usdt": 100,
                "stop_loss_price": 0.69,
                "take_profit_price": 0.81,
                "rationale": "replay execution path",
                "score": 84.0,
            }
        ],
        "exits": [],
    }

    def fake_scan(self):
        return [ranked_snapshot]

    def fake_rebalance(self, regime=None, predictive_model=None):
        assert self.mode == "replay"
        return replay_rebalance

    with (
        patch.object(TradingSystem, "scan", fake_scan),
        patch.object(TradingSystem, "rebalance", fake_rebalance),
        patch("trading_system.pipeline.OptionsDeliveryBridge.load", return_value=BridgeLoadResult("options_delivery", [], {"status": "ok", "count": 0})),
    ):
        report = pipeline.run(mode="replay", collect_square_feed=False)

    assert report["mode"] == "replay"
    assert report["analysis_mode"] == "replay"
    assert report["execution_mode"] == "replay"
    assert report["execution_realtime_enforced"] is False
    assert config.pipeline_report_file.exists()
    assert config.issues_json_file.exists()
    assert config.merged_sentiment_file.exists()
    bridge_names = {item["bridge_name"] for item in report["bridge_summaries"]}
    assert bridge_names >= {"manual_sentiment", "square_feed", "skill_bridge_directory", "coingecko_trending", "binance_top_movers", "options_delivery"}
    assert report["options_delivery_report_file"] == str(config.options_delivery_report_file)
    assert report["candidate_confirmation"]["symbols"] == ["AIXBTUSDT"]
    assert "opennews_targeted" in report["candidate_confirmation"]
    assert "twitter_targeted" in report["candidate_confirmation"]
    first = report["candidate_basket"][0]
    assert first["symbol"] == "AIXBTUSDT"
    assert first["risk_grade"] in {"A", "B", "C"}
    assert first["manipulation_stage"] == "markup"
    assert first["manipulation_flags"] == ["compressed_base", "oi_expanding"]
    assert first["price_position_20d"] == 0.54
    assert first["compression_score"] == 0.74
    assert report["alerts"] == []
    assert report["macro_trend"]["source"] == "disabled"
    assert {item["bridge_name"] for item in report["bridge_summaries"]} >= {"coingecko_trending", "binance_top_movers"}
    assert report["hotness_priority"]
    assert report["meme_radar"]["candidates"][0]["symbol"] == "AIXBTUSDT"
    assert "oi_expansion" in report["meme_radar"]["candidates"][0]["tags"]
    assert report["meme_radar_archive_file"].endswith(".md")
    assert Path(report["meme_radar_archive_file"]).exists()
    assert "妖币雷达Top3" in config.pipeline_report_md_file.read_text(encoding="utf-8")
    assert report["execution_learning_summary"]["status"] in {"missing", "ok"}


def test_pipeline_summarizes_execution_learning_from_journal(tmp_path: Path) -> None:
    pipeline = TradingPipeline(
        replace(
            StrategyConfig(),
            state_file=tmp_path / "paper_state.json",
            pipeline_report_file=tmp_path / "pipeline.json",
            pipeline_report_md_file=tmp_path / "pipeline.md",
            issues_json_file=tmp_path / "issues.json",
            issues_md_file=tmp_path / "issues.md",
        )
    )
    pipeline.config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [
                    {"action": "exit", "reason": "oi_divergence_signal_fade", "pnl_pct": 0.032},
                    {"action": "exit", "reason": "score_faded", "pnl_pct": -0.011},
                    {"action": "take_profit_partial", "reason": "r_multiple_scaleout_stage_1", "pnl_pct": 0.07},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = pipeline._build_execution_learning_summary()

    assert summary["status"] == "ok"
    assert summary["closed_trades"] == 3
    assert summary["by_reason"][0]["reason"] == "r_multiple_scaleout_stage_1"
    assert summary["by_reason"][0]["win_rate"] == 100.0


def test_pipeline_reconciles_shadow_mirror_by_position_key(tmp_path: Path) -> None:
    config = replace(
        StrategyConfig(),
        state_file=tmp_path / "paper_state.json",
        execution_result_dir=tmp_path / "execution_results",
    )
    config.execution_result_dir.mkdir(parents=True)
    key = "AAAUSDT:2026-05-14T00:00:00+00:00"
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "shadow_positions": [
                    {
                        "symbol": "AAAUSDT",
                        "opened_at": "2026-05-14T00:00:00+00:00",
                        "shadow_position_key": key,
                        "binance_mirror_status": "pending_submit",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (config.execution_result_dir / "execution_request_20260514T000001Z_result.json").write_text(
        json.dumps(
            {
                "status": "submitted",
                "results": [
                    {
                        "id": "execution_request_20260514T000001Z:1",
                        "symbol": "AAAUSDT",
                        "status": "submitted",
                        "submitted_at": "2026-05-14T00:00:02+00:00",
                        "meta": {"action": "open_long", "shadow_position_key": key},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    TradingPipeline(config)._reconcile_shadow_mirror_status(config)

    state = json.loads(config.state_file.read_text(encoding="utf-8"))
    position = state["shadow_positions"][0]
    assert position["binance_mirror_status"] == "submitted"
    assert position["binance_request_id"] == "execution_request_20260514T000001Z:1"


def test_pipeline_does_not_reconcile_pending_shadow_with_stale_symbol_result(tmp_path: Path) -> None:
    config = replace(
        StrategyConfig(),
        state_file=tmp_path / "paper_state.json",
        execution_result_dir=tmp_path / "execution_results",
    )
    config.execution_result_dir.mkdir(parents=True)
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "shadow_positions": [
                    {
                        "symbol": "AAAUSDT",
                        "opened_at": "2026-05-14T01:00:00+00:00",
                        "binance_mirror_status": "pending_submit",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (config.execution_result_dir / "execution_request_20260514T000001Z_result.json").write_text(
        json.dumps(
            {
                "status": "submitted",
                "results": [
                    {
                        "id": "execution_request_20260514T000001Z:1",
                        "symbol": "AAAUSDT",
                        "status": "submitted",
                        "submitted_at": "2026-05-14T00:00:02+00:00",
                        "meta": {"action": "open_long"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    TradingPipeline(config)._reconcile_shadow_mirror_status(config)

    state = json.loads(config.state_file.read_text(encoding="utf-8"))
    position = state["shadow_positions"][0]
    assert position["binance_mirror_status"] == "pending_submit"
    assert "binance_request_id" not in position


def test_pipeline_does_not_reconcile_legacy_shadow_with_future_symbol_result(tmp_path: Path) -> None:
    config = replace(
        StrategyConfig(),
        state_file=tmp_path / "paper_state.json",
        execution_result_dir=tmp_path / "execution_results",
    )
    config.execution_result_dir.mkdir(parents=True)
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "shadow_positions": [
                    {
                        "symbol": "AAAUSDT",
                        "opened_at": "2026-05-14T00:00:00+00:00",
                        "binance_mirror_status": "pending_submit",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (config.execution_result_dir / "execution_request_20260514T020001Z_result.json").write_text(
        json.dumps(
            {
                "status": "submitted",
                "results": [
                    {
                        "id": "execution_request_20260514T020001Z:1",
                        "symbol": "AAAUSDT",
                        "status": "submitted",
                        "submitted_at": "2026-05-14T02:00:02+00:00",
                        "meta": {"action": "open_long"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    TradingPipeline(config)._reconcile_shadow_mirror_status(config)

    state = json.loads(config.state_file.read_text(encoding="utf-8"))
    position = state["shadow_positions"][0]
    assert position["binance_mirror_status"] == "pending_submit"
    assert "binance_request_id" not in position


def test_pipeline_builds_stone_style_performance_dashboard(tmp_path: Path) -> None:
    pipeline = TradingPipeline(
        replace(
            StrategyConfig(),
            state_file=tmp_path / "paper_state.json",
            pipeline_report_file=tmp_path / "pipeline.json",
            pipeline_report_md_file=tmp_path / "pipeline.md",
            issues_json_file=tmp_path / "issues.json",
            issues_md_file=tmp_path / "issues.md",
        )
    )
    journal = [
        {"action": "exit", "symbol": f"L{i}USDT", "notional_usdt": 100, "size_fraction": 1.0, "pnl_pct": -0.01}
        for i in range(6)
    ]
    journal.extend(
        [
            {"action": "exit", "symbol": "W1USDT", "notional_usdt": 100, "size_fraction": 1.0, "pnl_pct": 0.12},
            {"action": "take_profit_partial", "symbol": "W2USDT", "notional_usdt": 100, "size_fraction": 0.5, "pnl_pct": 0.08},
        ]
    )
    pipeline.config.state_file.write_text(
        json.dumps(
            {
                "positions": [
                    {"symbol": "OPENUSDT", "entry_price": 1.0, "current_price": 1.05, "notional_usdt": 100},
                ],
                "journal": journal,
            }
        ),
        encoding="utf-8",
    )

    dashboard = pipeline._build_performance_dashboard()

    assert dashboard["status"] == "ok"
    assert dashboard["closed_trades"] == 8
    assert dashboard["win_rate"] == 25.0
    assert dashboard["realized_pnl_usdt"] == 14.0
    assert dashboard["unrealized_pnl_usdt"] == 5.0
    assert dashboard["total_pnl_usdt"] == 19.0
    assert "低胜率高收益" in dashboard["diagnosis"]


def test_markdown_report_shows_performance_dashboard(tmp_path: Path) -> None:
    report_file = tmp_path / "pipeline.md"
    write_markdown_report(
        {
            "analysis_mode": "paper",
            "square_feed_used": False,
            "issue_count": 0,
            "alerts": [],
            "issues": [],
            "candidate_basket": [],
            "unified_coverage": {"market_data": {"mode": "binance_only"}, "signal": {"mode": "binance_only"}},
            "unified_signal_summary": {"top_symbols": []},
            "execution_bridge_summary": {"intent_count": 0, "request_mode": "paper"},
            "execution_results_summary": {},
            "performance_dashboard": {
                "status": "ok",
                "open_positions": 2,
                "closed_trades": 28,
                "wins": 9,
                "win_rate": 32.1,
                "realized_pnl_usdt": 10459.89,
                "unrealized_pnl_usdt": -297.79,
                "total_pnl_usdt": 10162.1,
                "expectancy_pct_per_closed_trade": 1.23,
                "diagnosis": "低胜率高收益：允许，核心是让盈利单覆盖多笔小亏，继续强化分批止盈/奔跑仓。",
            },
        },
        report_file,
    )

    body = report_file.read_text(encoding="utf-8")
    assert "绩效看板" in body
    assert "已实现 +10459.9U｜未实现 -297.79U｜累计 +10162.1U" in body
    assert "胜率 32.1% (9/28)" in body
    assert "低胜率高收益" in body


def test_pipeline_keeps_candidate_confirmation_binance_only(tmp_path: Path) -> None:
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
        macro_trend_enabled=False,
    )
    config.sentiment_file.write_text(
        "symbol,hotness_score,mention_count,source,updated_at,notes\nAIXBTUSDT,91,12,manual_distill,2026-04-12T00:00:00+00:00,test\n",
        encoding="utf-8",
    )
    config.replay_file.write_text("[]", encoding="utf-8")
    config.square_feed_report_file.write_text(json.dumps({"posts": []}), encoding="utf-8")
    pipeline = TradingPipeline(config)
    ranked_snapshot = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=0.742,
        price_change_pct_24h=18.4,
        quote_volume_usdt_24h=135000000,
        funding_rate=0.00042,
        open_interest_usdt=82200000,
        oi_change_pct_4h=16.7,
        taker_buy_sell_ratio=1.18,
        listing_days=43,
        sentiment_score=91.0,
        sentiment_mentions=12,
        sentiment_source="manual_distill",
        score=84.0,
        reasons=["setup=base_breakout_confirmed:92.0"],
    )

    def fake_scan(self):
        return [ranked_snapshot]

    def fake_rebalance(self, regime=None, predictive_model=None):
        return {"entries": [], "exits": []}

    with patch.object(TradingSystem, "scan", fake_scan), patch.object(TradingSystem, "rebalance", fake_rebalance):
        report = pipeline.run(mode="replay", collect_square_feed=False)

    assert report["candidate_confirmation"]["symbols"] == ["AIXBTUSDT"]
    assert set(report["candidate_confirmation"].keys()) == {"symbols", "opennews_targeted", "twitter_targeted"}
    assert report["meme_radar"]["summary"]["candidate_count"] == 1
    assert report["top_ranked_symbols"][0]["symbol"] == "AIXBTUSDT"


def test_pipeline_flags_fast_five_minute_anomaly_alerts(tmp_path: Path) -> None:
    pipeline = TradingPipeline(replace(StrategyConfig(), pipeline_report_file=tmp_path / "pipeline.json", pipeline_report_md_file=tmp_path / "pipeline.md", issues_json_file=tmp_path / "issues.json", issues_md_file=tmp_path / "issues.md"))
    snapshot = MarketSnapshot(
        symbol="RAVEUSDT",
        mark_price=1.3,
        price_change_pct_5m=34.2,
        price_change_pct_24h=140.0,
        quote_volume_usdt_24h=280_000_000,
        funding_rate=0.0045,
        open_interest_usdt=120_000_000,
        oi_change_pct_4h=26.0,
        taker_buy_sell_ratio=1.22,
        listing_days=9,
        sentiment_score=88.0,
        sentiment_mentions=45,
        sentiment_source="trading-signal,square_feed",
        score=84.0,
        reasons=["setup=squeeze_anomaly:95.0"],
    )

    alerts = pipeline._build_anomaly_alerts([snapshot])

    assert len(alerts) == 1
    assert alerts[0]["symbol"] == "RAVEUSDT"
    assert alerts[0]["price_change_pct_5m"] == 34.2
    assert set(alerts[0]["trigger_flags"]) == {"oi_spike", "volume_spike", "funding_spike"}



def test_pipeline_anomaly_alert_requires_material_oi_spike(tmp_path: Path) -> None:
    config = replace(
        StrategyConfig(),
        pipeline_report_file=tmp_path / "pipeline.json",
        pipeline_report_md_file=tmp_path / "pipeline.md",
        issues_json_file=tmp_path / "issues.json",
        issues_md_file=tmp_path / "issues.md",
    )
    pipeline = TradingPipeline(config)
    snapshot = MarketSnapshot(
        symbol="RAVEUSDT",
        mark_price=1.3,
        price_change_pct_5m=config.anomaly_price_change_pct_5m + 1.0,
        price_change_pct_24h=140.0,
        quote_volume_usdt_24h=config.anomaly_quote_volume_usdt_24h - 1,
        funding_rate=0.0,
        open_interest_usdt=120_000_000,
        oi_change_pct_4h=config.anomaly_oi_change_pct_4h - 0.1,
        taker_buy_sell_ratio=1.22,
        listing_days=9,
        sentiment_score=88.0,
        sentiment_mentions=45,
        sentiment_source="trading-signal,square_feed",
        score=84.0,
    )

    assert pipeline._build_anomaly_alerts([snapshot]) == []

    snapshot.oi_change_pct_4h = config.anomaly_oi_change_pct_4h
    alerts = pipeline._build_anomaly_alerts([snapshot])
    assert alerts[0]["trigger_flags"] == ["oi_spike"]

def test_pipeline_builds_parallel_okx_and_binance_coverage() -> None:
    pipeline = TradingPipeline(StrategyConfig())
    coverage = pipeline._build_unified_coverage(
        [
            {"bridge_name": "trading-signal"},
            {"bridge_name": "crypto-market-rank"},
            {"bridge_name": "okx-dex-signal"},
            {"bridge_name": "okx-dex-market"},
            {"bridge_name": "okx-dex-token"},
            {"bridge_name": "okx-security"},
        ]
    )

    assert coverage["signal"]["parallel_enabled"] is True
    assert coverage["signal"]["mode"] == "parallel"
    assert coverage["security"]["okx_gap_filled"] is True
    assert coverage["ws_realtime"]["mode"] == "missing"


def test_pipeline_builds_unified_signal_summary_with_parallel_sources() -> None:
    pipeline = TradingPipeline(StrategyConfig())

    class Signal:
        def __init__(self, symbol: str, hotness_score: float, mention_count: int, source: str) -> None:
            self.symbol = symbol
            self.hotness_score = hotness_score
            self.mention_count = mention_count
            self.source = source

    summary = pipeline._build_unified_signal_summary(
        [
            Signal("RAVEUSDT", 88.0, 3, "trading-signal"),
            Signal("RAVEUSDT", 91.0, 2, "okx-dex-signal"),
            Signal("AIXBTUSDT", 79.0, 1, "square_feed"),
        ]
    )

    assert summary["parallel_symbol_count"] == 1
    assert summary["okx_supported_symbol_count"] == 1
    first = summary["top_symbols"][0]
    assert first["symbol"] == "RAVEUSDT"
    assert first["parallel_sources"] == ["binance", "okx"]


def test_pipeline_splits_aggregated_source_names_when_counting_parallel_sources() -> None:
    pipeline = TradingPipeline(StrategyConfig())

    class Signal:
        def __init__(self, symbol: str, hotness_score: float, mention_count: int, source: str) -> None:
            self.symbol = symbol
            self.hotness_score = hotness_score
            self.mention_count = mention_count
            self.source = source

    summary = pipeline._build_unified_signal_summary(
        [
            Signal("BTCUSDT", 95.0, 12, "manual_distill,okx-dex-signal,square_feed"),
        ]
    )

    first = summary["top_symbols"][0]
    assert first["sources"] == ["manual_distill", "okx-dex-signal", "square_feed"]
    assert first["parallel_sources"] == ["binance", "okx"]
    assert summary["parallel_symbol_count"] == 1


def test_candidate_basket_surfaces_blocked_reasons_for_watch_only_symbol() -> None:
    basket = TradingPipeline(StrategyConfig())
    snapshot = MarketSnapshot(
        symbol="TSTUSDT",
        mark_price=0.0142,
        price_change_pct_5m=-0.26,
        price_change_pct_24h=30.8,
        quote_volume_usdt_24h=108_181_466,
        funding_rate=0.00005,
        open_interest_usdt=5_018_888,
        oi_change_pct_4h=-8.351,
        taker_buy_sell_ratio=0.9834,
        listing_days=432,
        sentiment_score=74.0,
        sentiment_mentions=3,
        sentiment_source="trading-signal",
        score=9.72,
        setup_type="trend",
        reasons=["hotness=74", "oi_not_expanding", "taker_weak"],
    )

    from trading_system.candidates import build_candidate_basket
    row = build_candidate_basket([snapshot])[0]
    assert row["execution_status"] == "watch_only"
    assert row["blocked_reasons"] == ["oi_not_expanding", "taker_weak"]


def test_candidate_basket_requires_more_mentions_for_untrusted_breakout() -> None:
    from trading_system.candidates import build_candidate_basket

    snapshot = MarketSnapshot(
        symbol="BRKUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=120_000_000,
        funding_rate=0.0002,
        open_interest_usdt=45_000_000,
        oi_change_pct_4h=11.0,
        taker_buy_sell_ratio=1.14,
        listing_days=30,
        sentiment_score=78.0,
        sentiment_mentions=12,
        sentiment_source="opentwitter_search",
        score=84.0,
        setup_type="base_breakout_confirmed",
        setup_quality=86.0,
    )

    row = build_candidate_basket([snapshot])[0]
    assert row["execution_status"] == "confirm_needed"
    assert row["risk_grade"] == "B"


def test_candidate_basket_requires_confirmation_for_mania_followthrough() -> None:
    from trading_system.candidates import build_candidate_basket

    snapshot = MarketSnapshot(
        symbol="FOMOUSDT",
        mark_price=1.0,
        price_change_pct_24h=45.0,
        price_change_pct_5m=4.0,
        quote_volume_usdt_24h=180_000_000,
        funding_rate=0.0003,
        open_interest_usdt=80_000_000,
        oi_change_pct_4h=6.0,
        taker_buy_sell_ratio=1.08,
        listing_days=20,
        sentiment_score=88.0,
        sentiment_mentions=8,
        sentiment_source="binance_top_movers",
        score=86.0,
        setup_type="mania_followthrough",
        setup_quality=82.0,
    )

    row = build_candidate_basket([snapshot])[0]
    assert row["execution_status"] == "confirm_needed"
    assert row["risk_grade"] == "C"


def test_candidate_basket_keeps_trusted_oi_divergence_in_confirm_needed_when_divergence_is_immature() -> None:
    from trading_system.candidates import build_candidate_basket

    snapshot = MarketSnapshot(
        symbol="OIDUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=100_000_000,
        funding_rate=0.0002,
        open_interest_usdt=50_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.09,
        listing_days=30,
        sentiment_score=80.0,
        sentiment_mentions=12,
        sentiment_source="square_feed",
        score=84.0,
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=92.0,
    )

    row = build_candidate_basket([snapshot])[0]
    assert row["execution_status"] == "confirm_needed"
    assert row["risk_grade"] == "B"


def test_candidate_basket_keeps_only_mature_trusted_oi_divergence_executable() -> None:
    from trading_system.candidates import build_candidate_basket

    snapshot = MarketSnapshot(
        symbol="OIDUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=100_000_000,
        funding_rate=0.0002,
        open_interest_usdt=50_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.09,
        listing_days=30,
        sentiment_score=80.0,
        sentiment_mentions=24,
        sentiment_source="square_feed,manual_distill",
        score=84.0,
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=210.0,
    )

    row = build_candidate_basket([snapshot])[0]
    assert row["execution_status"] == "executable_now"
    assert row["risk_grade"] == "A"


def test_candidate_basket_keeps_trusted_breakout_in_confirm_needed_when_mentions_are_zero() -> None:
    from trading_system.candidates import build_candidate_basket

    snapshot = MarketSnapshot(
        symbol="BRKUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=120_000_000,
        funding_rate=0.0002,
        open_interest_usdt=45_000_000,
        oi_change_pct_4h=11.0,
        taker_buy_sell_ratio=1.14,
        listing_days=30,
        sentiment_score=78.0,
        sentiment_mentions=0,
        sentiment_source="square_feed",
        score=84.0,
        setup_type="base_breakout_confirmed",
        setup_quality=86.0,
    )

    row = build_candidate_basket([snapshot])[0]
    assert row["execution_status"] == "confirm_needed"
    assert row["risk_grade"] == "B"


def test_pipeline_builds_oi_divergence_candidates() -> None:
    pipeline = TradingPipeline(StrategyConfig())
    ranked = [
        MarketSnapshot(
            symbol="PEPEUSDT",
            mark_price=0.1,
            price_change_pct_4h=0.8,
            price_change_pct_24h=6.2,
            quote_volume_usdt_24h=120_000_000,
            funding_rate=0.00015,
            open_interest_usdt=88_000_000,
            oi_change_pct_4h=18.5,
            taker_buy_sell_ratio=1.07,
            listing_days=40,
            sentiment_score=80.0,
            sentiment_mentions=18,
            sentiment_source="square_feed,coingecko_trending",
            score=86.0,
            setup_type="oi_divergence_long_candidate",
            setup_quality=83.0,
            oi_price_divergence_score=90.0,
        ),
        MarketSnapshot(
            symbol="BTCUSDT",
            mark_price=60000,
            price_change_pct_4h=4.5,
            price_change_pct_24h=9.0,
            quote_volume_usdt_24h=500_000_000,
            funding_rate=0.0002,
            open_interest_usdt=400_000_000,
            oi_change_pct_4h=7.0,
            taker_buy_sell_ratio=1.02,
            listing_days=365,
            sentiment_score=65.0,
            sentiment_mentions=4,
            sentiment_source="manual_distill",
            score=60.0,
            setup_type="trend",
            setup_quality=0.0,
            oi_price_divergence_score=12.0,
        ),
    ]

    candidates = pipeline._build_oi_divergence_candidates(ranked)

    assert len(candidates) == 1
    assert candidates[0]["symbol"] == "PEPEUSDT"
    assert candidates[0]["timing_label"] == "逼近启动点"
    assert candidates[0]["oi_price_divergence_score"] == 90.0


def test_markdown_report_shows_trade_content_and_error_details(tmp_path: Path) -> None:
    report_file = tmp_path / "pipeline.md"
    write_markdown_report(
        {
            "analysis_mode": "live",
            "square_feed_used": True,
            "issue_count": 0,
            "alerts": [],
            "issues": [],
            "candidate_basket": [],
            "unified_coverage": {"market_data": {"mode": "binance_only"}, "signal": {"mode": "binance_only"}},
            "unified_signal_summary": {"top_symbols": []},
            "execution_bridge_summary": {"intent_count": 1, "request_mode": "live_order"},
            "execution_results_summary": {
                "latest_requests": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "failed",
                        "request_mode": "live_order",
                        "action": "open_long",
                        "side": "BUY",
                        "quantity_hint_usdt": 250,
                        "leverage": 5,
                        "rationale": "oi_divergence_probe",
                    }
                ],
                "latest_errors": [
                    {
                        "symbol": "BTCUSDT",
                        "action": "open_long",
                        "error": "Binance execution failed 400: Quantity less than zero.",
                        "params": {"side": "BUY"},
                    }
                ],
            },
        },
        report_file,
    )

    body = report_file.read_text(encoding="utf-8")
    assert "最近交易内容" in body
    assert "BTCUSDT｜open_long/BUY｜live_order｜名义 250U｜杠杆 5x｜状态 failed" in body
    assert "最近报错" in body
    assert "报错 Binance execution failed 400: Quantity less than zero." in body
