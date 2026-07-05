from datetime import datetime, timedelta, timezone
from pathlib import Path

import json

from trading_system.config import StrategyConfig
from trading_system.execution import PaperExecutionEngine
from trading_system.execution_bridge import BinanceSkillExecutionBridge
from trading_system.models import MarketSnapshot, Position


def test_paper_engine_opens_and_exits_positions(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        score_faded_structure_guard_enabled=False,
    )
    engine = PaperExecutionEngine(config)

    first = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.0,
        price_change_pct_24h=15.0,
        quote_volume_usdt_24h=50_000_000,
        funding_rate=0.0004,
        open_interest_usdt=10_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=1.2,
        listing_days=30,
        sentiment_score=90,
        sentiment_mentions=100,
        sentiment_source="manual_distill",
        price_position_7d=0.95,
        price_position_14d=0.72,
        price_position_20d=0.54,
        breakout_distance_7d=0.016,
        compression_score=0.74,
        setup_type="base_breakout_confirmed",
        score=80.0,
        reasons=["hotness=90"],
    )

    result = engine.rebalance([first])
    assert len(result["entries"]) == 1
    assert len(result["positions"]) == 1

    second = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.1,
        price_change_pct_24h=-5.0,
        quote_volume_usdt_24h=20_000_000,
        funding_rate=-0.0002,
        open_interest_usdt=5_000_000,
        oi_change_pct_4h=-5.0,
        taker_buy_sell_ratio=0.9,
        listing_days=30,
        sentiment_score=20,
        sentiment_mentions=20,
        sentiment_source="manual_distill",
        score=20.0,
        reasons=["score_faded"],
    )

    result = engine.rebalance([second])
    assert len(result["exits"]) == 1
    assert len(result["positions"]) == 0


def test_paper_engine_skips_setup_entries_without_min_mentions(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", min_entry_sentiment_mentions=3, absolute_min_sentiment_mentions=1)
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=0,
        sentiment_source="none",
        setup_type="oi_divergence_long_candidate",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_untrusted_low_hotness_priority_entries(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        absolute_min_sentiment_mentions=1,
        untrusted_entry_min_sentiment_mentions=3,
        untrusted_entry_high_mentions_override=100,
        untrusted_entry_min_source_count=3,
        low_info_min_hotness_priority_score=5,
        base_breakout_require_trusted_confirmation=False,
    )
    engine = PaperExecutionEngine(config)
    low_priority = MarketSnapshot(
        symbol="LOWUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.12,
        listing_days=40,
        sentiment_score=82,
        sentiment_mentions=60,
        sentiment_source="alpha,beta,gamma",
        setup_type="base_breakout_confirmed",
        price_position_7d=0.96,
        breakout_distance_7d=0.015,
        compression_score=0.72,
        hotness_priority_score=4.9,
        score=88.0,
        reasons=["low priority"],
    )
    high_priority = MarketSnapshot(
        **{**low_priority.to_dict(), "symbol": "HIGHUSDT", "hotness_priority_score": 5.1}
    )

    result = engine.rebalance([low_priority, high_priority])

    assert [entry["symbol"] for entry in result["entries"]] == ["HIGHUSDT"]


def test_infer_setup_type_uses_risk_band_for_legacy_training_rows() -> None:
    assert PaperExecutionEngine._infer_setup_type({"risk_band": "oi_divergence"}) == "oi_divergence_long_candidate"
    assert PaperExecutionEngine._infer_setup_type({"risk_band": "base_breakout"}) == "base_breakout_confirmed"


def test_paper_engine_requires_trusted_confirmation_for_oi_divergence_entries(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        min_entry_sentiment_mentions=3,
        low_info_min_sentiment_mentions=5,
        oi_divergence_require_trusted_confirmation=True,
        oi_divergence_min_sentiment_mentions=20,
        oi_divergence_min_divergence_score=140,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="PEPEUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=80,
        sentiment_mentions=24,
        sentiment_source="twitter_search",
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=150.0,
        score=84.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_weak_oi_divergence_even_with_trusted_confirmation(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        oi_divergence_min_sentiment_mentions=20,
        oi_divergence_min_divergence_score=180,
        oi_divergence_min_source_count=2,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="PEPEUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=80,
        sentiment_mentions=16,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=92.0,
        score=84.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_allows_strong_trend_high_hotness_as_small_probe(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        absolute_min_sentiment_mentions=1,
        untrusted_entry_min_sentiment_mentions=3,
        untrusted_entry_high_mentions_override=100,
        untrusted_entry_min_source_count=1,
        trend_followthrough_position_scale=0.25,
        trend_followthrough_min_score=85,
        trend_followthrough_min_hotness_priority_score=5,
        trend_followthrough_min_sentiment_mentions=3,
        trend_followthrough_min_price_change_pct_24h=12,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="RUNUSDT",
        mark_price=1.0,
        price_change_pct_24h=24.0,
        quote_volume_usdt_24h=120_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=10.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=82,
        sentiment_mentions=8,
        sentiment_source="twitter_search",
        setup_type="trend",
        hotness_priority_score=6.0,
        score=88.0,
        reasons=["trend high hotness"],
    )

    result = engine.rebalance([candidate])

    assert [entry["symbol"] for entry in result["entries"]] == ["RUNUSDT"]
    assert result["entries"][0]["setup_type"] == "mania_followthrough"
    assert result["entries"][0]["notional_usdt"] == 50.0
    assert result["entries"][0]["rationale"].startswith("trend_followthrough")


def test_paper_engine_keeps_ordinary_unconfirmed_trend_blocked(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        absolute_min_sentiment_mentions=1,
        untrusted_entry_min_sentiment_mentions=3,
        untrusted_entry_min_source_count=1,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="WEAKUSDT",
        mark_price=1.0,
        price_change_pct_24h=10.0,
        quote_volume_usdt_24h=120_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=10.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=82,
        sentiment_mentions=8,
        sentiment_source="twitter_search",
        setup_type="trend",
        hotness_priority_score=6.0,
        score=88.0,
        reasons=["trend not strong enough"],
    )

    result = engine.rebalance([candidate])

    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_allows_short_cycle_trend_followthrough_hold_extension(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        absolute_min_sentiment_mentions=1,
        untrusted_entry_min_sentiment_mentions=3,
        untrusted_entry_high_mentions_override=100,
        untrusted_entry_min_source_count=1,
        trend_followthrough_position_scale=0.25,
        trend_followthrough_min_score=85,
        trend_followthrough_min_hotness_priority_score=5,
        trend_followthrough_min_sentiment_mentions=3,
        trend_followthrough_min_price_change_pct_24h=12,
        trend_followthrough_max_hold_hours=72,
    )
    engine = PaperExecutionEngine(config)
    opened_at = (datetime.now(timezone.utc) - timedelta(hours=36)).isoformat()
    position = Position(
        symbol="RUNUSDT",
        opened_at=opened_at,
        entry_price=1.0,
        current_price=1.18,
        stop_loss_price=0.96,
        take_profit_price=1.25,
        notional_usdt=50.0,
        score=86.0,
        rationale="trend_followthrough, test",
        setup_type="trend_followthrough",
        take_profit_stage=0,
        initial_risk_pct=0.04,
    )
    current = MarketSnapshot(
        symbol="RUNUSDT",
        mark_price=1.18,
        price_change_pct_24h=22.0,
        quote_volume_usdt_24h=120_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=1.10,
        listing_days=40,
        sentiment_score=86,
        sentiment_mentions=10,
        sentiment_source="twitter_search",
        setup_type="trend_followthrough",
        hotness_priority_score=6.0,
        score=84.0,
        price_change_pct_5m=2.0,
        reasons=["test"],
    )

    plan, updated = engine._evaluate_position(position, current)

    assert plan is None
    assert updated is not None


def test_strategy_config_defaults_to_reachable_oi_divergence_threshold() -> None:
    config = StrategyConfig()
    assert config.oi_divergence_min_divergence_score == 90.0
    assert config.trend_followthrough_position_scale == 0.25
    assert config.trend_followthrough_min_score == 85.0
    assert config.base_breakout_min_sentiment_mentions == 10
    assert config.base_breakout_min_source_count == 2
    assert config.base_breakout_require_trusted_confirmation is True
    assert config.untrusted_entry_min_sentiment_mentions == 50
    assert config.absolute_min_sentiment_mentions == 10
    assert config.low_info_setup_type_block_min_closed_trades == 3
    assert config.low_info_setup_type_block_max_win_rate == 0.1
    assert config.low_info_symbol_block_min_closed_trades == 2
    assert config.low_info_symbol_block_max_win_rate == 0.0
    assert config.symbol_block_low_win_rate_min_closed_trades == 4
    assert config.symbol_block_low_win_rate_max_win_rate == 0.25
    assert config.symbol_block_zero_win_apply_to_confirmed is True
    assert config.stop_loss_setup_block_min_closed_trades == 1
    assert config.stop_loss_block_apply_to_confirmed is True
    assert config.stop_loss_consecutive_symbol_block == 3
    assert config.stop_loss_consecutive_setup_block == 2
    assert config.pdf_cooldown_after_sl_minutes == 120
    assert config.recent_reentry_cooldown_minutes == 120
    assert config.behavioral_sunk_cost_window_minutes == 120
    assert config.default_leverage == 8
    assert config.dynamic_leverage_enabled is True
    assert config.consecutive_stop_loss_half_size_after == 3
    assert config.consecutive_stop_loss_half_size_scale == 0.5
    assert config.flat_score_fade_max_abs_pnl_pct == 0.05
    assert config.symbol_hard_block_underperform_min_closed_trades == 6
    assert config.symbol_hard_block_underperform_max_win_rate == 0.25
    assert config.max_dynamic_stop_loss_pct == 0.04


def test_paper_engine_uses_learning_journal_beyond_trimmed_visible_journal(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    state = {
        "positions": [],
        "journal": [],
        "learning_journal": [
            {
                "timestamp": "2026-04-26T00:00:00+00:00",
                "action": "exit",
                "symbol": "OLDLOSSUSDT",
                "reason": "stop_loss_hit",
                "pnl_pct": -0.05,
                "setup_type": "base_breakout_confirmed",
            }
        ],
    }
    config.state_file.write_text(json.dumps(state), encoding="utf-8")
    candidate = MarketSnapshot(
        symbol="OLDLOSSUSDT",
        mark_price=1.0,
        price_change_pct_24h=16.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0003,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=15.0,
        taker_buy_sell_ratio=1.12,
        listing_days=30,
        sentiment_score=90,
        sentiment_mentions=80,
        sentiment_source="manual_distill",
        setup_type="base_breakout_confirmed",
        score=88.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])

    assert result["entries"] == []
    persisted = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert persisted["learning_journal"]


def test_paper_engine_keeps_empty_learning_journal_independent_from_visible_journal(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", entry_threshold=80)
    engine = PaperExecutionEngine(config)
    config.state_file.write_text(json.dumps({"positions": [], "journal": [], "learning_journal": []}), encoding="utf-8")
    candidate = MarketSnapshot(
        symbol="NEWUSDT",
        mark_price=1.0,
        price_change_pct_24h=16.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0003,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=15.0,
        taker_buy_sell_ratio=1.12,
        listing_days=30,
        sentiment_score=90,
        sentiment_mentions=80,
        sentiment_source="manual_distill",
        setup_type="base_breakout_confirmed",
        score=88.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])

    assert len(result["entries"]) == 1
    persisted = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert len(persisted["journal"]) == 1
    assert len(persisted["learning_journal"]) == 1


def test_consecutive_stop_loss_index_counts_only_latest_run_per_symbol_and_setup() -> None:
    engine = PaperExecutionEngine(StrategyConfig())
    journal = [
        {"action": "exit", "symbol": "AAAUSDT", "reason": "stop_loss_hit", "setup_type": "base_breakout_confirmed"},
        {"action": "exit", "symbol": "AAAUSDT", "reason": "take_profit", "setup_type": "base_breakout_confirmed"},
        {"action": "exit", "symbol": "AAAUSDT", "reason": "stop_loss_hit", "setup_type": "base_breakout_confirmed"},
        {"action": "exit", "symbol": "AAAUSDT", "reason": "stop_loss_hit", "setup_type": "base_breakout_confirmed"},
    ]

    symbol_counts, setup_counts = engine._build_consecutive_stop_loss_index(journal)

    assert symbol_counts["AAAUSDT"] == 2
    assert setup_counts[("AAAUSDT", "base_breakout_confirmed")] == 2


def test_fourth_entry_after_three_stop_losses_is_half_size(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"timestamp": "2026-01-01T00:00:00+00:00", "action": "exit", "symbol": "LOSSUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.03, "setup_type": "base_breakout_confirmed"},\n'
        '    {"timestamp": "2026-01-01T01:00:00+00:00", "action": "exit", "symbol": "LOSSUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.03, "setup_type": "base_breakout_confirmed"},\n'
        '    {"timestamp": "2026-01-01T02:00:00+00:00", "action": "exit", "symbol": "LOSSUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.03, "setup_type": "base_breakout_confirmed"}\n'
        '  ],\n'
        '  "learning_journal": [\n'
        '    {"timestamp": "2026-01-01T00:00:00+00:00", "action": "exit", "symbol": "LOSSUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.03, "setup_type": "base_breakout_confirmed"},\n'
        '    {"timestamp": "2026-01-01T01:00:00+00:00", "action": "exit", "symbol": "LOSSUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.03, "setup_type": "base_breakout_confirmed"},\n'
        '    {"timestamp": "2026-01-01T02:00:00+00:00", "action": "exit", "symbol": "LOSSUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.03, "setup_type": "base_breakout_confirmed"}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        entry_threshold=30,
        position_notional_usdt=200,
        min_entry_notional_usdt=100,
        pdf_cooldown_after_sl_minutes=120,
        recent_reentry_cooldown_minutes=120,
        stop_loss_consecutive_symbol_block=0,
        stop_loss_consecutive_setup_block=0,
        recent_symbol_loss_streak_block=4,
        recent_setup_loss_streak_block=4,
        stop_loss_setup_block_min_closed_trades=4,
        symbol_block_zero_win_min_closed_trades=4,
        symbol_block_min_closed_trades=4,
        setup_block_min_closed_trades=4,
        low_info_setup_type_block_min_closed_trades=4,
        base_breakout_setup_block_min_closed_trades=4,
        low_info_symbol_block_min_closed_trades=4,
        consecutive_stop_loss_half_size_after=3,
        consecutive_stop_loss_half_size_scale=0.5,
        base_breakout_require_trusted_confirmation=False,
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    candidate = MarketSnapshot(
        symbol="LOSSUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=1.1,
        sentiment_score=82,
        sentiment_mentions=20,
        sentiment_source="manual_distill",
        price_position_7d=0.9,
        breakout_distance_7d=0.02,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=80.0,
        reasons=["reentry_after_three_stop_losses"],
    )

    result = engine.rebalance([candidate])

    assert len(result["entries"]) == 1
    assert result["entries"][0]["notional_usdt"] == 100


def test_dynamic_entry_leverage_scales_from_score() -> None:
    bridge = BinanceSkillExecutionBridge(StrategyConfig(default_leverage=8))

    low = bridge._entry_intent({"action": "open_long", "symbol": "LOWUSDT", "notional_usdt": 100}, score=40)
    base = bridge._entry_intent({"action": "open_long", "symbol": "BASEUSDT", "notional_usdt": 100}, score=80)
    high = bridge._entry_intent({"action": "open_long", "symbol": "HIGHUSDT", "notional_usdt": 100}, score=100)
    capped = bridge._entry_intent({"action": "open_long", "symbol": "CAPUSDT", "notional_usdt": 100}, score=200)

    assert low.leverage == 4
    assert base.leverage == 8
    assert high.leverage == 10
    assert capped.leverage == 12


def test_legacy_state_migration_does_not_duplicate_visible_journal_into_learning_journal(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    legacy_journal = [
        {
            "timestamp": "2026-04-26T00:00:00+00:00",
            "action": "exit",
            "symbol": "TSTUSDT",
            "reason": "stop_loss_hit",
            "pnl_pct": -0.02,
            "setup_type": "base_breakout_confirmed",
        }
    ]
    config.state_file.write_text(json.dumps({"positions": [], "journal": legacy_journal}), encoding="utf-8")

    result = engine.rebalance([])

    assert result["positions"] == []
    persisted = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert persisted["journal"] == legacy_journal
    assert persisted["learning_journal"] == legacy_journal


def test_paper_engine_handles_naive_opened_at_without_crashing(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", max_hold_hours=1)
    engine = PaperExecutionEngine(config)
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "OLDUSDT",
                        "opened_at": "2020-01-01T00:00:00",
                        "entry_price": 1.0,
                        "current_price": 1.0,
                        "stop_loss_price": 0.9,
                        "take_profit_price": 1.2,
                        "notional_usdt": 100.0,
                        "score": 90.0,
                        "rationale": "test",
                        "setup_type": "trend",
                    }
                ],
                "journal": [],
                "learning_journal": [],
            }
        ),
        encoding="utf-8",
    )
    current = MarketSnapshot(
        symbol="OLDUSDT",
        mark_price=1.0,
        price_change_pct_24h=1.0,
        quote_volume_usdt_24h=20_000_000,
        funding_rate=0.0,
        open_interest_usdt=10_000_000,
        oi_change_pct_4h=1.0,
        taker_buy_sell_ratio=1.0,
        listing_days=30,
        sentiment_score=70,
        sentiment_mentions=30,
        sentiment_source="manual_distill",
        setup_type="trend",
        score=90.0,
        reasons=["hold"],
    )

    result = engine.rebalance([current])

    assert len(result["exits"]) == 1
    assert result["exits"][0]["rationale"] == "max_hold_reached"


def test_paper_engine_prioritizes_stop_loss_over_breakout_scaleout(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "BRKUSDT",
                        "opened_at": "2026-04-26T00:00:00+00:00",
                        "entry_price": 1.0,
                        "current_price": 1.0,
                        "stop_loss_price": 0.95,
                        "take_profit_price": 1.05,
                        "notional_usdt": 100.0,
                        "score": 90.0,
                        "rationale": "r_based_scaleout",
                        "setup_type": "base_breakout_confirmed",
                        "take_profit_stage": 0,
                        "initial_risk_pct": 0.05,
                    }
                ],
                "journal": [],
                "learning_journal": [],
            }
        ),
        encoding="utf-8",
    )
    current = MarketSnapshot(
        symbol="BRKUSDT",
        mark_price=0.94,
        price_change_pct_24h=20.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0003,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=15.0,
        taker_buy_sell_ratio=1.12,
        listing_days=30,
        sentiment_score=90,
        sentiment_mentions=80,
        sentiment_source="manual_distill",
        setup_type="base_breakout_confirmed",
        score=90.0,
        reasons=["breakout"],
    )

    result = engine.rebalance([current])

    assert len(result["exits"]) == 1
    assert result["exits"][0]["rationale"] == "stop_loss_hit"


def test_paper_engine_does_not_scaleout_legacy_zero_risk_stage_one_position(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", max_hold_hours=999999)
    engine = PaperExecutionEngine(config)
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "RUNUSDT",
                        "opened_at": "2026-04-26T00:00:00+00:00",
                        "entry_price": 1.0,
                        "current_price": 1.02,
                        "stop_loss_price": 1.0,
                        "take_profit_price": 1.1,
                        "notional_usdt": 50.0,
                        "score": 90.0,
                        "rationale": "r_based_scaleout",
                        "setup_type": "base_breakout_confirmed",
                        "take_profit_stage": 1,
                        "initial_risk_pct": 0.0,
                    }
                ],
                "journal": [],
                "learning_journal": [],
            }
        ),
        encoding="utf-8",
    )
    current = MarketSnapshot(
        symbol="RUNUSDT",
        mark_price=1.02,
        price_change_pct_24h=20.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0003,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=15.0,
        taker_buy_sell_ratio=1.12,
        listing_days=30,
        sentiment_score=90,
        sentiment_mentions=80,
        sentiment_source="manual_distill",
        setup_type="base_breakout_confirmed",
        score=90.0,
        reasons=["runner"],
    )

    result = engine.rebalance([current])

    assert result["exits"] == []
    assert len(result["positions"]) == 1


def test_paper_engine_hard_blocks_long_term_underperforming_symbol_even_when_confirmed(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    state = {
        "positions": [],
        "journal": [
            {
                "timestamp": "2026-04-26T00:00:00+00:00",
                "action": "exit",
                "symbol": "TSTUSDT",
                "reason": "stop_loss_hit",
                "pnl_pct": -0.02,
                "setup_type": "base_breakout_confirmed",
            }
            for _ in range(6)
        ],
    }
    config.state_file.write_text(__import__("json").dumps(state), encoding="utf-8")
    candidate = MarketSnapshot(
        symbol="TSTUSDT",
        mark_price=1.0,
        price_change_pct_24h=16.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0003,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=15.0,
        taker_buy_sell_ratio=1.12,
        listing_days=30,
        sentiment_score=90,
        sentiment_mentions=80,
        sentiment_source="manual_distill",
        setup_type="base_breakout_confirmed",
        score=88.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])

    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_confirmed_setup_after_stop_loss_sample(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    state = {
        "positions": [],
        "journal": [
            {
                "timestamp": "2026-04-26T00:00:00+00:00",
                "action": "exit",
                "symbol": "LOSSUSDT",
                "reason": "stop_loss_hit",
                "pnl_pct": -0.06,
                "setup_type": "base_breakout_confirmed",
            }
        ],
    }
    config.state_file.write_text(__import__("json").dumps(state), encoding="utf-8")
    candidate = MarketSnapshot(
        symbol="LOSSUSDT",
        mark_price=1.0,
        price_change_pct_24h=16.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0003,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=15.0,
        taker_buy_sell_ratio=1.12,
        listing_days=30,
        sentiment_score=90,
        sentiment_mentions=80,
        sentiment_source="square_feed,manual_distill",
        setup_type="base_breakout_confirmed",
        score=88.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])

    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_repeated_score_fade_bleed_churn(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    state = {
        "positions": [],
        "journal": [
            {
                "timestamp": "2026-04-26T00:00:00+00:00",
                "action": "exit",
                "symbol": "BLEEDUSDT",
                "reason": "score_faded",
                "pnl_pct": -0.02,
                "setup_type": "trend",
            },
            {
                "timestamp": "2026-04-26T01:00:00+00:00",
                "action": "exit",
                "symbol": "BLEEDUSDT",
                "reason": "score_faded",
                "pnl_pct": -0.018,
                "setup_type": "trend",
            },
        ],
    }
    config.state_file.write_text(__import__("json").dumps(state), encoding="utf-8")
    candidate = MarketSnapshot(
        symbol="BLEEDUSDT",
        mark_price=1.0,
        price_change_pct_24h=16.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0003,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=15.0,
        taker_buy_sell_ratio=1.12,
        listing_days=30,
        sentiment_score=90,
        sentiment_mentions=80,
        sentiment_source="manual_distill",
        setup_type="trend",
        score=88.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])

    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_requires_high_conviction_mentions_for_untrusted_entries(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="ALTUSDT",
        mark_price=1.0,
        price_change_pct_24h=14.0,
        quote_volume_usdt_24h=85_000_000,
        funding_rate=0.00025,
        open_interest_usdt=40_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=1.11,
        listing_days=30,
        sentiment_score=76,
        sentiment_mentions=12,
        sentiment_source="opentwitter_search",
        setup_type="trend",
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_square_only_entry_below_high_conviction_mentions(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="ALTUSDT",
        mark_price=1.0,
        price_change_pct_24h=14.0,
        quote_volume_usdt_24h=85_000_000,
        funding_rate=0.00025,
        open_interest_usdt=40_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=1.11,
        listing_days=30,
        sentiment_score=76,
        sentiment_mentions=12,
        sentiment_source="square_feed",
        setup_type="trend",
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_low_confluence_trusted_entry_without_primary_source(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="ALTUSDT",
        mark_price=1.0,
        price_change_pct_24h=14.0,
        quote_volume_usdt_24h=85_000_000,
        funding_rate=0.00025,
        open_interest_usdt=40_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=1.11,
        listing_days=30,
        sentiment_score=76,
        sentiment_mentions=12,
        sentiment_source="crypto-market-rank",
        setup_type="trend",
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_confirmed_symbol_after_two_zero_win_closes(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{"positions": [], "journal": ['
        '{"action":"exit","symbol":"ALTUSDT","pnl_pct":0.0,"setup_type":"trend"},'
        '{"action":"exit","symbol":"ALTUSDT","pnl_pct":-0.02,"setup_type":"trend"}'
        ']}',
        encoding="utf-8",
    )
    config = StrategyConfig(state_file=state_file)
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="ALTUSDT",
        mark_price=1.0,
        price_change_pct_24h=14.0,
        quote_volume_usdt_24h=85_000_000,
        funding_rate=0.00025,
        open_interest_usdt=40_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=1.11,
        listing_days=30,
        sentiment_score=86,
        sentiment_mentions=80,
        sentiment_source="square_feed,manual_distill",
        setup_type="trend",
        score=90.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_unconfirmed_base_breakout_even_with_mentions(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        base_breakout_require_trusted_confirmation=True,
        base_breakout_min_sentiment_mentions=10,
        base_breakout_min_source_count=2,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="FRESHUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=10.0,
        taker_buy_sell_ratio=1.12,
        listing_days=40,
        sentiment_score=75.0,
        sentiment_mentions=18,
        sentiment_source="binance_top_movers,coingecko_trending",
        price_position_7d=0.93,
        price_position_14d=0.7,
        price_position_20d=0.5,
        breakout_distance_7d=0.02,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=78.0,
        reasons=["setup=base_breakout_confirmed"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_allows_confirmed_base_breakout_with_trusted_source(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        base_breakout_require_trusted_confirmation=True,
        base_breakout_min_sentiment_mentions=10,
        base_breakout_min_source_count=2,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="FRESHUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=10.0,
        taker_buy_sell_ratio=1.12,
        listing_days=40,
        sentiment_score=75.0,
        sentiment_mentions=18,
        sentiment_source="square_feed",
        price_position_7d=0.93,
        price_position_14d=0.7,
        price_position_20d=0.5,
        breakout_distance_7d=0.02,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=78.0,
        reasons=["setup=base_breakout_confirmed"],
    )

    result = engine.rebalance([candidate])
    assert len(result["entries"]) == 1
    assert len(result["positions"]) == 1


def test_paper_engine_blocks_low_info_setup_type_after_full_history_zero_win_run(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        symbol_block_lookback=2,
        low_info_setup_type_block_min_closed_trades=3,
        low_info_setup_type_block_max_win_rate=0.1,
    )
    engine = PaperExecutionEngine(config)
    engine.state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "AAAUSDT", "reason": "score_faded", "pnl_pct": 0.0, "setup_type": "base_breakout_confirmed"},\n'
        '    {"action": "exit", "symbol": "BBBUSDT", "reason": "score_faded", "pnl_pct": 0.0, "setup_type": "base_breakout_confirmed"},\n'
        '    {"action": "exit", "symbol": "CCCUSDT", "reason": "max_hold_reached", "pnl_pct": 0.0, "setup_type": "base_breakout_confirmed"}\n'
        '  ]\n'
        '}',
        encoding="utf-8",
    )
    candidate = MarketSnapshot(
        symbol="FRESHUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=10.0,
        taker_buy_sell_ratio=1.12,
        listing_days=40,
        sentiment_score=75.0,
        sentiment_mentions=18,
        sentiment_source="square_feed",
        price_position_7d=0.93,
        price_position_14d=0.7,
        price_position_20d=0.5,
        breakout_distance_7d=0.02,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=78.0,
        reasons=["setup=base_breakout_confirmed"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_trusted_low_info_symbol_after_two_zero_win_closes(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        symbol_block_lookback=10,
        low_info_symbol_block_min_closed_trades=2,
        low_info_symbol_block_max_win_rate=0.0,
    )
    engine = PaperExecutionEngine(config)
    engine.state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "AIXBTUSDT", "reason": "score_faded", "pnl_pct": 0.0, "setup_type": "oi_divergence_long_candidate"},\n'
        '    {"action": "exit", "symbol": "AIXBTUSDT", "reason": "max_hold_reached", "pnl_pct": 0.0, "setup_type": "oi_divergence_long_candidate"}\n'
        '  ]\n'
        '}',
        encoding="utf-8",
    )
    candidate = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=90_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=82,
        sentiment_mentions=28,
        sentiment_source="square_feed,manual_distill",
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=210.0,
        score=86.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_repeat_low_win_symbol_after_four_negative_closes(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        symbol_block_lookback=10,
        symbol_block_low_win_rate_min_closed_trades=4,
        symbol_block_low_win_rate_max_win_rate=0.25,
    )
    engine = PaperExecutionEngine(config)
    engine.state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "CHURNUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.08, "setup_type": "trend"},\n'
        '    {"action": "exit", "symbol": "CHURNUSDT", "reason": "take_profit_hit", "pnl_pct": 0.03, "setup_type": "trend"},\n'
        '    {"action": "exit", "symbol": "CHURNUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.04, "setup_type": "trend"},\n'
        '    {"action": "exit", "symbol": "CHURNUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.02, "setup_type": "trend"}\n'
        '  ]\n'
        '}',
        encoding="utf-8",
    )
    candidate = MarketSnapshot(
        symbol="CHURNUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=20,
        sentiment_source="binance_top_movers",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_keeps_symbol_with_two_wins_out_of_four_closes(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        symbol_block_lookback=10,
        symbol_block_low_win_rate_min_closed_trades=4,
        symbol_block_low_win_rate_max_win_rate=0.25,
    )
    engine = PaperExecutionEngine(config)
    engine.state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "ALIVEUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.03, "setup_type": "trend"},\n'
        '    {"action": "exit", "symbol": "ALIVEUSDT", "reason": "take_profit_hit", "pnl_pct": 0.06, "setup_type": "trend"},\n'
        '    {"action": "exit", "symbol": "ALIVEUSDT", "reason": "score_faded", "pnl_pct": 0.01, "setup_type": "trend"},\n'
        '    {"action": "exit", "symbol": "ALIVEUSDT", "reason": "stop_loss_hit", "pnl_pct": -0.01, "setup_type": "trend"}\n'
        '  ]\n'
        '}',
        encoding="utf-8",
    )
    candidate = MarketSnapshot(
        symbol="ALIVEUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=20,
        sentiment_source="square_feed",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert len(result["entries"]) == 1
    assert len(result["positions"]) == 1


def test_paper_engine_requires_multiple_sources_for_oi_divergence_entries(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        oi_divergence_min_sentiment_mentions=20,
        oi_divergence_min_divergence_score=180,
        oi_divergence_min_source_count=2,
    )
    engine = PaperExecutionEngine(config)
    single_source = MarketSnapshot(
        symbol="PEPEUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=80,
        sentiment_mentions=24,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=150.0,
        score=84.0,
        reasons=["test"],
    )
    multi_source = MarketSnapshot(
        symbol="GWEIUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=80,
        sentiment_mentions=24,
        sentiment_source="square_feed,opentwitter_search",
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=190.0,
        score=84.0,
        reasons=["test"],
    )

    result = engine.rebalance([single_source, multi_source])
    assert len(result["entries"]) == 1
    assert result["entries"][0]["symbol"] == "GWEIUSDT"
    assert len(result["positions"]) == 1


def test_paper_engine_skips_trend_entry_without_min_mentions_or_trusted_confirmation(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", min_entry_sentiment_mentions=3, absolute_min_sentiment_mentions=1)
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="GWEIUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=2,
        sentiment_source="twitter_search",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_allows_trend_entry_with_trusted_confirmation_even_if_mentions_are_low(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", min_entry_sentiment_mentions=3, absolute_min_sentiment_mentions=1)
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="GWEIUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=2,
        sentiment_source="square_feed",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert len(result["entries"]) == 1
    assert len(result["positions"]) == 1


def test_paper_engine_requires_more_mentions_for_untrusted_entries(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        min_entry_sentiment_mentions=3,
        untrusted_entry_min_sentiment_mentions=13,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="CHIPUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=12,
        sentiment_source="twitter_search",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_single_source_untrusted_entry_without_high_mentions(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        min_entry_sentiment_mentions=3,
        untrusted_entry_min_sentiment_mentions=13,
        untrusted_entry_min_source_count=2,
        untrusted_entry_high_mentions_override=25,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="DENTUSDT",
        mark_price=1.0,
        price_change_pct_24h=22.0,
        quote_volume_usdt_24h=85_000_000,
        funding_rate=0.0004,
        open_interest_usdt=18_000_000,
        oi_change_pct_4h=17.0,
        taker_buy_sell_ratio=1.16,
        listing_days=45,
        sentiment_score=80,
        sentiment_mentions=18,
        sentiment_source="binance_top_movers",
        setup_type="trend",
        setup_quality=92.0,
        score=84.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_allows_multi_source_untrusted_entry_before_high_mentions_override(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        min_entry_sentiment_mentions=3,
        untrusted_entry_min_sentiment_mentions=13,
        untrusted_entry_min_source_count=2,
        untrusted_entry_high_mentions_override=25,
        untrusted_entry_min_score=80,
        absolute_min_sentiment_mentions=1,
        stop_loss_setup_block_min_closed_trades=99,
        stop_loss_consecutive_symbol_block=99,
        stop_loss_consecutive_setup_block=99,
        symbol_block_zero_win_min_closed_trades=99,
        symbol_block_low_win_rate_min_closed_trades=99,
        low_info_symbol_block_min_closed_trades=99,
        low_info_setup_type_block_min_closed_trades=99,
        setup_type_block_min_closed_trades=99,
        non_progress_symbol_block_min_closed_trades=99,
        non_progress_setup_block_min_closed_trades=99,
        flat_score_fade_block_min_closed_trades=99,
        oi_divergence_setup_block_min_closed_trades=99,
        base_breakout_setup_block_min_closed_trades=99,
    )
    engine = PaperExecutionEngine(config)
    confirmed = MarketSnapshot(
        symbol="CHIPUSDT",
        mark_price=1.0,
        price_change_pct_24h=22.0,
        quote_volume_usdt_24h=85_000_000,
        funding_rate=0.0004,
        open_interest_usdt=18_000_000,
        oi_change_pct_4h=17.0,
        taker_buy_sell_ratio=1.16,
        listing_days=45,
        sentiment_score=80,
        sentiment_mentions=18,
        sentiment_source="binance_top_movers,opentwitter_search",
        setup_type="trend",
        setup_quality=92.0,
        score=84.0,
        reasons=["test"],
    )
    weak_only = MarketSnapshot(
        symbol="WEAKUSDT",
        mark_price=1.0,
        price_change_pct_24h=22.0,
        quote_volume_usdt_24h=85_000_000,
        funding_rate=0.0004,
        open_interest_usdt=18_000_000,
        oi_change_pct_4h=17.0,
        taker_buy_sell_ratio=1.16,
        listing_days=45,
        sentiment_score=80,
        sentiment_mentions=18,
        sentiment_source="binance_top_movers,coingecko_trending",
        setup_type="trend",
        setup_quality=92.0,
        score=83.0,
        reasons=["test"],
    )

    result = engine.rebalance([weak_only, confirmed])
    assert len(result["entries"]) == 1
    assert result["entries"][0]["symbol"] == "CHIPUSDT"
    assert len(result["positions"]) == 1


def test_paper_engine_blocks_weak_only_untrusted_entry_even_after_mentions_override(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        min_entry_sentiment_mentions=3,
        untrusted_entry_min_sentiment_mentions=13,
        untrusted_entry_min_source_count=2,
        untrusted_entry_high_mentions_override=25,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="WEAKUSDT",
        mark_price=1.0,
        price_change_pct_24h=22.0,
        quote_volume_usdt_24h=85_000_000,
        funding_rate=0.0004,
        open_interest_usdt=18_000_000,
        oi_change_pct_4h=17.0,
        taker_buy_sell_ratio=1.16,
        listing_days=45,
        sentiment_score=80,
        sentiment_mentions=25,
        sentiment_source="binance_top_movers",
        setup_type="trend",
        setup_quality=92.0,
        score=84.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_skips_low_info_breakout_without_trusted_confirmation(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        min_entry_sentiment_mentions=3,
        low_info_min_sentiment_mentions=5,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="VIRTUALUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=10.0,
        taker_buy_sell_ratio=1.12,
        listing_days=40,
        sentiment_score=75.0,
        sentiment_mentions=4,
        sentiment_source="twitter_search",
        price_position_7d=0.93,
        price_position_14d=0.7,
        price_position_20d=0.5,
        breakout_distance_7d=0.02,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=78.0,
        reasons=["setup=base_breakout_confirmed"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_base_breakout_after_three_global_losses(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "AAAUSDT", "setup_type": "base_breakout_confirmed", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "BBBUSDT", "setup_type": "base_breakout_confirmed", "pnl_pct": 0.0},\n'
        '    {"action": "exit", "symbol": "CCCUSDT", "setup_type": "base_breakout_confirmed", "pnl_pct": -0.01}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        base_breakout_setup_block_min_closed_trades=3,
        base_breakout_setup_block_max_win_rate=0.1,
        setup_type_block_min_closed_trades=99,
        recent_setup_loss_streak_block=99,
        recent_symbol_loss_streak_block=99,
        symbol_block_zero_win_min_closed_trades=99,
        symbol_block_min_closed_trades=99,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="FRESHUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=10.0,
        taker_buy_sell_ratio=1.12,
        listing_days=40,
        sentiment_score=75.0,
        sentiment_mentions=20,
        sentiment_source="square_feed",
        price_position_7d=0.93,
        price_position_14d=0.7,
        price_position_20d=0.5,
        breakout_distance_7d=0.02,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=78.0,
        reasons=["setup=base_breakout_confirmed"],
    )

    result = engine.rebalance([candidate])

    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_exits_stale_position_when_symbol_missing_after_max_hold(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [\n'
        '    {"symbol": "AIXBTUSDT", "opened_at": "2026-04-18T00:00:00+00:00", "entry_price": 1.0, "current_price": 1.0, "stop_loss_price": 0.95, "take_profit_price": 1.05, "notional_usdt": 100.0, "score": 80.0, "rationale": "test", "setup_type": "trend"}\n'
        '  ],\n'
        '  "journal": []\n'
        '}'
    )
    config = StrategyConfig(state_file=state_file, max_hold_hours=1)
    engine = PaperExecutionEngine(config)

    result = engine.rebalance([])

    assert len(result["exits"]) == 1
    assert result["exits"][0]["symbol"] == "AIXBTUSDT"
    assert result["exits"][0]["rationale"] == "market_snapshot_missing_timeout"
    assert result["positions"] == []


def test_paper_engine_blocks_symbol_after_two_zero_win_closes(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "AIXBTUSDT", "pnl_pct": 0.0},\n'
        '    {"action": "exit", "symbol": "AIXBTUSDT", "pnl_pct": -0.02}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        symbol_block_zero_win_min_closed_trades=2,
        symbol_block_min_closed_trades=99,
        symbol_block_lookback=12,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=8,
        sentiment_source="square_feed",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_symbol_after_repeated_stop_losses_for_weak_single_source_entry(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "setup_type": "legacy_breakout", "reason": "stop_loss_hit", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "setup_type": "legacy_breakout", "reason": "stop_loss_hit", "pnl_pct": -0.03},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "setup_type": "legacy_breakout", "reason": "stop_loss_hit", "pnl_pct": -0.01},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "setup_type": "legacy_breakout", "reason": "stop_loss_hit", "pnl_pct": -0.04}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        stop_loss_symbol_block_min_closed_trades=4,
        recent_symbol_loss_streak_block=99,
        symbol_block_zero_win_min_closed_trades=99,
        symbol_block_min_closed_trades=99,
        stop_loss_consecutive_symbol_block=99,
        stop_loss_consecutive_setup_block=99,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="GWEIUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=20,
        sentiment_source="binance_top_movers",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_trusted_entry_after_symbol_repeated_stop_losses(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "setup_type": "legacy_breakout", "reason": "stop_loss_hit", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "setup_type": "legacy_breakout", "reason": "stop_loss_hit", "pnl_pct": -0.03},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "setup_type": "legacy_breakout", "reason": "stop_loss_hit", "pnl_pct": -0.01},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "setup_type": "legacy_breakout", "reason": "stop_loss_hit", "pnl_pct": -0.04}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        stop_loss_symbol_block_min_closed_trades=4,
        recent_symbol_loss_streak_block=99,
        symbol_block_zero_win_min_closed_trades=99,
        symbol_block_min_closed_trades=99,
        stop_loss_consecutive_symbol_block=99,
        stop_loss_consecutive_setup_block=99,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="GWEIUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=20,
        sentiment_source="square_feed",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_setup_after_repeated_stop_losses(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "PEPEUSDT", "setup_type": "oi_divergence_long_candidate", "reason": "stop_loss_hit", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "PEPEUSDT", "setup_type": "oi_divergence_long_candidate", "reason": "stop_loss_hit", "pnl_pct": -0.03},\n'
        '    {"action": "exit", "symbol": "PEPEUSDT", "setup_type": "oi_divergence_long_candidate", "reason": "stop_loss_hit", "pnl_pct": -0.01},\n'
        '    {"action": "exit", "symbol": "PEPEUSDT", "setup_type": "oi_divergence_long_candidate", "reason": "stop_loss_hit", "pnl_pct": -0.04}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        stop_loss_setup_block_min_closed_trades=4,
        recent_setup_loss_streak_block=99,
        recent_symbol_loss_streak_block=99,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="PEPEUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=80,
        sentiment_mentions=24,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=150.0,
        score=84.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_symbol_after_repeated_zero_win_history(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "AIXBTUSDT", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "AIXBTUSDT", "pnl_pct": -0.03},\n'
        '    {"action": "exit", "symbol": "AIXBTUSDT", "pnl_pct": 0.0}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(state_file=state_file, symbol_block_min_closed_trades=3, symbol_block_lookback=12)
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=8,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_globally_failing_setup_type(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "A1USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "A2USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.03},\n'
        '    {"action": "exit", "symbol": "A3USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.01},\n'
        '    {"action": "exit", "symbol": "A4USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.04},\n'
        '    {"action": "exit", "symbol": "A5USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "A6USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": 0.0}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        setup_type_block_min_closed_trades=6,
        setup_type_block_max_win_rate=0.15,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="NEWUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=80,
        sentiment_mentions=20,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        score=84.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])

    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_oi_divergence_setup_earlier_when_win_rate_stays_too_low(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "A1USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "A2USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.03},\n'
        '    {"action": "exit", "symbol": "A3USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.01},\n'
        '    {"action": "exit", "symbol": "A4USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": 0.0}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        setup_type_block_min_closed_trades=99,
        oi_divergence_setup_block_min_closed_trades=4,
        oi_divergence_setup_block_max_win_rate=0.25,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="NEXTUSDT",
        mark_price=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=80,
        sentiment_mentions=22,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=150.0,
        score=84.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])

    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_keeps_other_setup_types_when_global_block_targets_only_one_setup(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "A1USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "A2USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.03},\n'
        '    {"action": "exit", "symbol": "A3USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.01},\n'
        '    {"action": "exit", "symbol": "A4USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.04},\n'
        '    {"action": "exit", "symbol": "A5USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "A6USDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": 0.0}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        setup_type_block_min_closed_trades=6,
        setup_type_block_max_win_rate=0.15,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="TRENDUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=13,
        sentiment_source="square_feed",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])

    assert len(result["entries"]) == 1
    assert len(result["positions"]) == 1


def test_paper_engine_blocks_all_low_info_setups_for_repeat_zero_win_symbol(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "AIXBTUSDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": 0.0},\n'
        '    {"action": "exit", "symbol": "AIXBTUSDT", "setup_type": "oi_divergence_long_candidate", "pnl_pct": -0.02}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        setup_block_min_closed_trades=2,
        symbol_block_zero_win_min_closed_trades=99,
        symbol_block_min_closed_trades=3,
        symbol_block_lookback=12,
        recent_symbol_loss_streak_block=99,
        recent_setup_loss_streak_block=99,
    )
    engine = PaperExecutionEngine(config)
    oi_candidate = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=8,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )
    breakout_candidate = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=10.0,
        taker_buy_sell_ratio=1.12,
        listing_days=40,
        sentiment_score=75.0,
        sentiment_mentions=20,
        sentiment_source="manual_distill",
        price_position_7d=0.93,
        price_position_14d=0.7,
        price_position_20d=0.5,
        breakout_distance_7d=0.02,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=78.0,
        reasons=["setup=base_breakout_confirmed"],
    )

    result = engine.rebalance([oi_candidate, breakout_candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_symbol_after_recent_loss_streak(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "pnl_pct": -0.03},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "pnl_pct": 0.0}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        recent_symbol_loss_streak_block=3,
        recent_setup_loss_streak_block=99,
        symbol_block_min_closed_trades=99,
        setup_block_min_closed_trades=99,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="GWEIUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=8,
        sentiment_source="square_feed",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_resets_symbol_loss_streak_after_win(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "pnl_pct": -0.02},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "pnl_pct": -0.03},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "pnl_pct": 0.05},\n'
        '    {"action": "exit", "symbol": "GWEIUSDT", "pnl_pct": -0.01}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        recent_symbol_loss_streak_block=3,
        recent_setup_loss_streak_block=99,
        symbol_block_min_closed_trades=99,
        setup_block_min_closed_trades=99,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="GWEIUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=10,
        sentiment_source="square_feed",
        setup_type="trend",
        setup_quality=90.0,
        score=82.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert len(result["entries"]) == 1
    assert len(result["positions"]) == 1


def test_paper_engine_blocks_fast_reentry_without_new_edge(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"timestamp": "2026-04-20T00:00:00+00:00", "action": "exit", "symbol": "AIXBTUSDT", "entry_price": 1.0, "exit_price": 1.0, "score": 82.0, "pnl_pct": 0.0}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        recent_reentry_cooldown_minutes=10_000,
        recent_reentry_min_price_move_pct=0.05,
        recent_reentry_min_score_delta=8,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.01,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=8,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=90.0,
        score=84.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_same_cycle_reentry_after_stop_loss(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    opened_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    state_file.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "INUSDT",
                        "opened_at": opened_at,
                        "entry_price": 1.0,
                        "current_price": 1.0,
                        "stop_loss_price": 0.96,
                        "take_profit_price": 1.12,
                        "notional_usdt": 200,
                        "score": 80,
                        "rationale": "manual_test_entry",
                        "setup_type": "trend",
                        "initial_risk_pct": 0.04,
                    }
                ],
                "journal": [],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        state_file=state_file,
        entry_threshold=30,
        pdf_cooldown_after_sl_minutes=120,
        recent_reentry_cooldown_minutes=120,
        base_breakout_require_trusted_confirmation=False,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="INUSDT",
        mark_price=0.95,
        price_change_pct_24h=24.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0003,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.2,
        sentiment_score=82,
        sentiment_mentions=50,
        sentiment_source="manual_distill",
        setup_type="base_breakout_confirmed",
        score=88.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])

    assert [exit["rationale"] for exit in result["exits"]] == ["stop_loss_hit"]
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_blocks_non_progress_reentry_without_fresh_confirmation(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"timestamp": "2026-04-20T00:00:00+00:00", "action": "exit", "symbol": "AIXBTUSDT", "reason": "score_faded", "entry_price": 1.0, "exit_price": 1.0, "score": 82.0, "pnl_pct": 0.0}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        recent_reentry_cooldown_minutes=10_000,
        recent_reentry_min_price_move_pct=0.05,
        recent_reentry_min_score_delta=8,
        non_progress_reentry_cooldown_minutes=10_000,
        non_progress_reentry_min_score_delta=15,
        non_progress_reentry_min_sentiment_mentions=20,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.08,
        price_change_pct_4h=0.1,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=20.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=12,
        sentiment_source="binance_top_movers",
        setup_type="oi_divergence_long_candidate",
        setup_quality=90.0,
        oi_price_divergence_score=150.0,
        score=92.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert result["entries"] == []
    assert result["positions"] == []


def test_paper_engine_allows_reentry_after_material_change(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        '{\n'
        '  "positions": [],\n'
        '  "journal": [\n'
        '    {"timestamp": "2026-04-20T00:00:00+00:00", "action": "exit", "symbol": "AIXBTUSDT", "reason": "score_faded", "entry_price": 1.0, "exit_price": 1.0, "score": 82.0, "pnl_pct": 0.0}\n'
        '  ]\n'
        '}'
    )
    config = StrategyConfig(
        state_file=state_file,
        recent_reentry_cooldown_minutes=10_000,
        recent_reentry_min_price_move_pct=0.05,
        recent_reentry_min_score_delta=8,
        non_progress_reentry_cooldown_minutes=10_000,
        non_progress_reentry_min_score_delta=15,
        non_progress_reentry_min_sentiment_mentions=20,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.08,
        price_change_pct_4h=0.1,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=60_000_000,
        funding_rate=0.0003,
        open_interest_usdt=12_000_000,
        oi_change_pct_4h=20.0,
        taker_buy_sell_ratio=1.15,
        listing_days=30,
        sentiment_score=78,
        sentiment_mentions=24,
        sentiment_source="square_feed,opentwitter_search",
        setup_type="oi_divergence_long_candidate",
        setup_quality=90.0,
        oi_price_divergence_score=150.0,
        score=92.0,
        reasons=["test"],
    )

    result = engine.rebalance([candidate])
    assert len(result["entries"]) == 1
    assert len(result["positions"]) == 1


def test_learning_index_ignores_partial_take_profit_until_final_exit(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    journal = [
        {
            "action": "take_profit_partial",
            "symbol": "CHURNUSDT",
            "setup_type": "base_breakout_confirmed",
            "pnl_pct": 0.04,
        }
    ]

    symbol_learning, setup_learning, setup_type_learning = engine._build_learning_index(journal)
    recent_symbol_loss_streaks, recent_setup_loss_streaks = engine._build_recent_loss_streak_index(journal)

    assert symbol_learning == {}
    assert setup_learning == {}
    assert setup_type_learning == {}
    assert recent_symbol_loss_streaks == {}
    assert recent_setup_loss_streaks == {}


def test_paper_engine_ignores_corrupt_persisted_position_instead_of_crashing(tmp_path: Path) -> None:
    state_file = tmp_path / "paper_state.json"
    state_file.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "BROKENUSDT",
                        "opened_at": "2026-01-01T00:00:00+00:00",
                        "entry_price": 1.0,
                    }
                ],
                "journal": [],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(state_file=state_file)
    engine = PaperExecutionEngine(config)

    result = engine.rebalance([])

    assert result["positions"] == []
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted["positions"] == []


def test_paper_engine_market_flash_crash_guard_exits_open_position(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        market_flash_crash_drop_pct_5m=8,
        market_flash_crash_symbol_count=1,
    )
    state = {
        "positions": [
            {
                "symbol": "CRASHUSDT",
                "opened_at": "2026-01-01T00:00:00+00:00",
                "entry_price": 1.0,
                "current_price": 1.0,
                "stop_loss_price": 0.95,
                "take_profit_price": 1.08,
                "notional_usdt": 200,
                "score": 80,
                "rationale": "structure_stop=0.95, target=1.08",
                "setup_type": "base_breakout_confirmed",
            }
        ],
        "journal": [],
    }
    config.state_file.write_text(json.dumps(state), encoding="utf-8")
    engine = PaperExecutionEngine(config)
    current = MarketSnapshot(
        symbol="CRASHUSDT",
        mark_price=0.90,
        price_change_pct_5m=-9.0,
        price_change_pct_24h=-12.0,
        quote_volume_usdt_24h=100_000_000,
        funding_rate=0.0,
        open_interest_usdt=30_000_000,
        oi_change_pct_4h=3.0,
        taker_buy_sell_ratio=0.8,
        listing_days=30,
        sentiment_score=50,
        sentiment_mentions=20,
        sentiment_source="manual_distill",
        setup_type="base_breakout_confirmed",
        score=70.0,
    )

    result = engine.rebalance([current])

    assert len(result["exits"]) == 1
    assert result["exits"][0]["rationale"] == "market_flash_crash_protection"
    assert result["positions"] == []
    persisted = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert persisted["journal"][-1]["reason"] == "market_flash_crash_protection"


def test_untrusted_discovery_source_can_pass_when_high_mentions_override_met(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        base_breakout_require_trusted_confirmation=False,
        untrusted_entry_high_mentions_override=50,
        untrusted_entry_min_sentiment_mentions=3,
        untrusted_entry_min_source_count=1,
        low_info_min_hotness_priority_score=5,
    )
    engine = PaperExecutionEngine(config)
    candidate = MarketSnapshot(
        symbol="DISCOVERYUSDT",
        mark_price=1.0,
        price_change_pct_24h=14.0,
        quote_volume_usdt_24h=85_000_000,
        funding_rate=0.00025,
        open_interest_usdt=40_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=1.11,
        listing_days=30,
        sentiment_score=86,
        sentiment_mentions=60,
        sentiment_source="binance_top_movers,coingecko_trending",
        setup_type="base_breakout_confirmed",
        price_position_7d=0.94,
        breakout_distance_7d=0.015,
        compression_score=0.72,
        hotness_priority_score=8.0,
        score=92.0,
        reasons=["high mentions discovery"],
    )

    result = engine.rebalance([candidate])

    assert len(result["entries"]) == 1
    assert len(result["positions"]) == 1


def test_paper_engine_extends_hold_when_sentiment_and_oi_still_strong(tmp_path: Path) -> None:
    opened_at = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        max_hold_hours=1,
        exit_threshold=50,
        hold_extension_min_sentiment_score=65,
        hold_extension_min_mentions=20,
        hold_extension_min_oi_change_pct=5,
        hold_extension_min_taker_ratio=1.03,
        hold_extension_min_quote_volume_usdt=20_000_000,
    )
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "HOLDUSDT",
                        "opened_at": opened_at,
                        "entry_price": 1.0,
                        "current_price": 1.02,
                        "stop_loss_price": 0.95,
                        "take_profit_price": 1.20,
                        "notional_usdt": 100.0,
                        "score": 80.0,
                        "rationale": "structure_stop,r_based_scaleout,test",
                        "setup_type": "base_breakout_confirmed",
                        "last_sentiment_score": 70,
                        "last_sentiment_mentions": 40,
                        "last_oi_change_pct_4h": 6,
                        "last_taker_buy_sell_ratio": 1.05,
                    }
                ],
                "journal": [],
                "learning_journal": [],
            }
        ),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config)
    current = MarketSnapshot(
        symbol="HOLDUSDT",
        mark_price=1.03,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=50_000_000,
        funding_rate=0.0,
        open_interest_usdt=20_000_000,
        oi_change_pct_4h=8.0,
        taker_buy_sell_ratio=1.08,
        listing_days=30,
        sentiment_score=72,
        sentiment_mentions=45,
        sentiment_source="manual_distill",
        setup_type="base_breakout_confirmed",
        score=75.0,
    )

    result = engine.rebalance([current])

    assert result["exits"] == []
    assert len(result["positions"]) == 1


def test_paper_engine_does_not_extend_hold_when_signal_fades(tmp_path: Path) -> None:
    opened_at = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", max_hold_hours=1, exit_threshold=50)
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "symbol": "FADEUSDT",
                        "opened_at": opened_at,
                        "entry_price": 1.0,
                        "current_price": 1.0,
                        "stop_loss_price": 0.95,
                        "take_profit_price": 1.20,
                        "notional_usdt": 100.0,
                        "score": 80.0,
                        "rationale": "structure_stop,r_based_scaleout,test",
                        "setup_type": "base_breakout_confirmed",
                        "last_sentiment_score": 80,
                        "last_sentiment_mentions": 80,
                        "last_oi_change_pct_4h": 12,
                        "last_taker_buy_sell_ratio": 1.1,
                    }
                ],
                "journal": [],
                "learning_journal": [],
            }
        ),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config)
    current = MarketSnapshot(
        symbol="FADEUSDT",
        mark_price=1.0,
        quote_volume_usdt_24h=50_000_000,
        oi_change_pct_4h=1.0,
        taker_buy_sell_ratio=0.99,
        sentiment_score=40,
        sentiment_mentions=5,
        sentiment_source="manual_distill",
        setup_type="base_breakout_confirmed",
        score=75.0,
    )

    result = engine.rebalance([current])

    assert len(result["exits"]) == 1
    assert result["exits"][0]["rationale"] == "max_hold_reached"


def test_paper_engine_uses_smart_money_panel_before_entry_and_flags_90pct_single_side(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        absolute_min_sentiment_mentions=1,
        min_entry_sentiment_mentions=1,
        base_breakout_require_trusted_confirmation=False,
        position_notional_usdt=200,
    )
    calls: list[str] = []

    def panel_provider(symbol: str) -> dict:
        calls.append(symbol)
        return {
            "summary": {"nominal_long_short_ratio_pct": 925.0},
            "views": {
                "smart_money": {
                    "long_side": {"current_position_usdt": 9_250_000, "count": 925},
                    "short_side": {"current_position_usdt": 750_000, "count": 75},
                }
            },
        }

    engine = PaperExecutionEngine(config, smart_money_panel_provider=panel_provider)
    candidate = MarketSnapshot(
        symbol="庄币USDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0003,
        open_interest_usdt=60_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.12,
        listing_days=30,
        sentiment_score=80,
        sentiment_mentions=5,
        sentiment_source="manual_distill",
        price_position_7d=0.95,
        breakout_distance_7d=0.01,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=88.0,
        reasons=["breakout"],
    )

    result = engine.rebalance([candidate])

    assert calls == ["庄币USDT"]
    entry = result["entries"][0]
    assert entry["smart_money_context"]["single_side_pct"] == 92.5
    assert entry["smart_money_context"]["oi_composition"] == "single_side_庄家盘"
    assert "smart_money_single_side_90pct" in entry["rationale"]
    assert entry["notional_usdt"] > 200


def test_paper_engine_treats_sub_70pct_oi_growth_as_mixed_consensus_slow_grind(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        absolute_min_sentiment_mentions=1,
        min_entry_sentiment_mentions=1,
        base_breakout_require_trusted_confirmation=False,
        position_notional_usdt=200,
    )

    def panel_provider(symbol: str) -> dict:
        return {
            "summary": {"nominal_long_short_ratio_pct": 190.0},
            "views": {
                "smart_money": {
                    "long_side": {"current_position_usdt": 650_000, "count": 65},
                    "short_side": {"current_position_usdt": 350_000, "count": 35},
                }
            },
        }

    engine = PaperExecutionEngine(config, smart_money_panel_provider=panel_provider)
    candidate = MarketSnapshot(
        symbol="SLOWUSDT",
        mark_price=1.0,
        price_change_pct_24h=10.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=60_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.08,
        listing_days=30,
        sentiment_score=76,
        sentiment_mentions=5,
        sentiment_source="manual_distill",
        price_position_7d=0.94,
        breakout_distance_7d=0.01,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=82.0,
        reasons=["breakout"],
    )

    entry = engine.rebalance([candidate])["entries"][0]

    assert entry["smart_money_context"]["single_side_pct"] == 65.0
    assert entry["smart_money_context"]["oi_composition"] == "mixed_consensus_slow_grind"
    assert "smart_money_mixed_consensus" in entry["rationale"]
    assert entry["notional_usdt"] == 200


def test_major_top30_symbol_does_not_use_90_70_meme_thresholds(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        absolute_min_sentiment_mentions=1,
        min_entry_sentiment_mentions=1,
        base_breakout_require_trusted_confirmation=False,
        position_notional_usdt=200,
    )

    def panel_provider(symbol: str) -> dict:
        return {
            "views": {
                "smart_money": {
                    "long_side": {"current_position_usdt": 9_250_000, "count": 925},
                    "short_side": {"current_position_usdt": 750_000, "count": 75},
                }
            },
        }

    engine = PaperExecutionEngine(config, smart_money_panel_provider=panel_provider)
    candidate = MarketSnapshot(
        symbol="BTCUSDT",
        mark_price=80_000.0,
        price_change_pct_24h=5.0,
        quote_volume_usdt_24h=2_000_000_000,
        funding_rate=0.0001,
        open_interest_usdt=5_000_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=1.05,
        listing_days=3650,
        sentiment_score=80,
        sentiment_mentions=10,
        sentiment_source="manual_distill",
        price_position_7d=0.95,
        breakout_distance_7d=0.01,
        compression_score=0.7,
        setup_type="base_breakout_confirmed",
        score=88.0,
        reasons=["breakout"],
    )

    entry = engine.rebalance([candidate])["entries"][0]

    assert entry["smart_money_context"]["symbol_class"] == "major_top30"
    assert entry["smart_money_context"]["single_side_pct"] == 92.5
    assert entry["smart_money_context"]["oi_composition"] == "large_cap_directional_pressure"
    assert "smart_money_single_side_90pct" not in entry["rationale"]
    assert "smart_money_large_cap_structure" in entry["rationale"]


def test_repeated_signal_4h_intact_allows_small_probe_after_prior_loss(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        position_notional_usdt=200,
        repeated_signal_4h_position_scale=0.6,
    )
    config.state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "positions": [],
                "journal": [
                    {
                        "timestamp": "2024-05-01T00:00:00+00:00",
                        "action": "exit",
                        "symbol": "LABUSDT",
                        "reason": "stop_loss_hit",
                        "pnl_pct": -0.08,
                        "setup_type": "trend",
                    }
                ],
                "learning_journal": [],
            }
        ),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    candidate = MarketSnapshot(
        symbol="LABUSDT",
        mark_price=1.0,
        price_change_pct_4h=2.0,
        price_change_pct_24h=30.0,
        quote_volume_usdt_24h=500_000_000,
        funding_rate=0.001,
        open_interest_usdt=120_000_000,
        oi_change_pct_4h=8.0,
        taker_buy_sell_ratio=1.0,
        sentiment_score=65,
        sentiment_mentions=25,
        sentiment_source="binance_top_movers,coingecko_trending",
        price_position_7d=0.9,
        price_position_20d=0.88,
        breakout_distance_7d=0.12,
        setup_type="trend",
        score=66.0,
        hotness_priority_score=6.0,
        reasons=["repeated_signal"],
    )

    result = engine.rebalance([candidate])

    assert len(result["entries"]) == 1
    entry = result["entries"][0]
    assert entry["symbol"] == "LABUSDT"
    assert entry["notional_usdt"] == 120


def test_repeated_signal_4h_intact_defers_stop_until_structure_breaks(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    position = Position(
        symbol="LABUSDT",
        opened_at=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        entry_price=1.0,
        current_price=1.0,
        stop_loss_price=0.96,
        take_profit_price=1.12,
        notional_usdt=100,
        score=70,
        rationale="mania_followthrough,repeated_signal_4h_probe",
        setup_type="mania_followthrough",
        last_sentiment_score=60,
        last_sentiment_mentions=20,
        last_oi_change_pct_4h=6,
    )
    intact = MarketSnapshot(
        symbol="LABUSDT",
        mark_price=0.95,
        price_change_pct_4h=-1.5,
        oi_change_pct_4h=7,
        taker_buy_sell_ratio=1.0,
        sentiment_score=65,
        sentiment_mentions=25,
        quote_volume_usdt_24h=200_000_000,
        price_position_7d=0.86,
        price_position_20d=0.84,
        breakout_distance_7d=0.2,
        score=62,
        sentiment_source="binance_top_movers,coingecko_trending",
        setup_type="trend",
    )
    broken = MarketSnapshot(
        symbol="LABUSDT",
        mark_price=0.95,
        price_change_pct_4h=-6.0,
        oi_change_pct_4h=1,
        taker_buy_sell_ratio=0.9,
        sentiment_score=40,
        sentiment_mentions=8,
        quote_volume_usdt_24h=200_000_000,
        price_position_7d=0.65,
        price_position_20d=0.63,
        breakout_distance_7d=0.6,
        score=40,
    )

    assert engine._should_exit(position, intact) is None
    assert engine._should_exit(position, broken) == "stop_loss_hit"


def test_score_faded_strong_momo_trend_runner_stays_open(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json")
    engine = PaperExecutionEngine(config)
    position = Position(
        symbol="INUSDT",
        opened_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        entry_price=1.0,
        current_price=1.0,
        stop_loss_price=0.94,
        take_profit_price=1.30,
        notional_usdt=100,
        score=72,
        rationale="trend_followthrough,repeated_signal_4h_probe",
        setup_type="trend",
        last_sentiment_score=45,
        last_sentiment_mentions=24,
        last_oi_change_pct_4h=12,
        high_watermark_pnl_pct=0.015,
    )
    strong_runner = MarketSnapshot(
        symbol="INUSDT",
        mark_price=1.01,
        price_change_pct_4h=14.0,
        price_change_pct_24h=42.0,
        quote_volume_usdt_24h=45_000_000,
        open_interest_usdt=90_000_000,
        oi_change_pct_4h=22.0,
        taker_buy_sell_ratio=0.89,
        sentiment_score=50,
        sentiment_mentions=39,
        sentiment_source="binance_top_movers,opennews_market,opentwitter_search",
        price_position_7d=1.0,
        price_position_20d=0.98,
        breakout_distance_7d=0.48,
        h4_structure_state="valid",
        h4_price_location="above_resistance",
        h4_price_vs_ma20_pct=38.0,
        momo_15m_bars=4404,
        bb210_mid_15m=0.98,
        bb210_lower_15m=0.93,
        bb672_mid_15m=0.96,
        bb672_lower_15m=0.90,
        bb_price_position_210=0.55,
        bb_price_position_672_15m=0.52,
        williams_r_15m=-2.0,
        williams_r_15m_zone="overbought",
        williams_r_15m_120=-69.0,
        williams_r_15m_120_zone="lower_mid",
        setup_type="trend",
        score=31.0,
        reasons=["taker_weak", "pred_model_adj=+7.5"],
    )
    weak_runner = MarketSnapshot(
        symbol="INUSDT",
        mark_price=1.01,
        price_change_pct_4h=14.0,
        price_change_pct_24h=42.0,
        quote_volume_usdt_24h=45_000_000,
        open_interest_usdt=90_000_000,
        oi_change_pct_4h=22.0,
        taker_buy_sell_ratio=0.70,
        sentiment_score=50,
        sentiment_mentions=39,
        price_position_7d=1.0,
        price_position_20d=0.98,
        breakout_distance_7d=0.40,
        h4_structure_state="valid",
        h4_price_location="above_resistance",
        momo_15m_bars=4404,
        bb210_mid_15m=0.98,
        bb210_lower_15m=0.93,
        bb672_mid_15m=0.96,
        bb672_lower_15m=0.90,
        bb_price_position_210=0.55,
        bb_price_position_672_15m=0.52,
        williams_r_15m_zone="overbought",
        williams_r_15m_120_zone="lower_mid",
        setup_type="trend",
        score=31.0,
    )

    assert engine._should_exit(position, strong_runner) is None
    assert engine._should_exit(position, weak_runner) is None


def test_score_faded_does_not_exit_intact_pullback_structure(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", exit_threshold=45)
    engine = PaperExecutionEngine(config)
    position = Position(
        symbol="HANAUSDT",
        opened_at=(datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        entry_price=0.0449341,
        current_price=0.0449341,
        stop_loss_price=0.04322661,
        take_profit_price=0.04834909,
        notional_usdt=143,
        score=37.56,
        rationale="hotness=49,smart_money_directional_pressure",
        setup_type="trend",
        initial_risk_pct=0.038,
        last_sentiment_score=48.94,
        last_sentiment_mentions=21,
        last_oi_change_pct_4h=17.399,
    )
    pullback = MarketSnapshot(
        symbol="HANAUSDT",
        mark_price=0.04442608,
        price_change_pct_4h=-1.8,
        price_change_pct_24h=27.32,
        quote_volume_usdt_24h=25_800_000,
        oi_change_pct_4h=0.53,
        taker_buy_sell_ratio=0.9751,
        sentiment_score=48.94,
        sentiment_mentions=21,
        sentiment_source="binance_top_movers,opennews_market",
        h4_structure_state="valid",
        h4_low_break_recent_20bars=False,
        price_position_7d=0.82,
        price_position_20d=0.80,
        breakout_distance_7d=0.24,
        setup_type="trend",
        score=30.95,
    )

    assert engine._should_exit(position, pullback) is None


def test_score_faded_exits_when_structure_breaks(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", exit_threshold=45)
    engine = PaperExecutionEngine(config)
    position = Position(
        symbol="HANAUSDT",
        opened_at=(datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        entry_price=1.0,
        current_price=1.0,
        stop_loss_price=0.90,
        take_profit_price=1.20,
        notional_usdt=100,
        score=50,
        rationale="hotness=49,smart_money_directional_pressure",
        setup_type="trend",
        initial_risk_pct=0.10,
    )
    broken = MarketSnapshot(
        symbol="HANAUSDT",
        mark_price=0.97,
        price_change_pct_4h=-3.0,
        quote_volume_usdt_24h=25_800_000,
        h4_structure_state="broken",
        h4_low_break_recent_20bars=True,
        setup_type="trend",
        score=30,
    )

    assert engine._should_exit(position, broken) == "score_faded"


def test_external_order_mode_does_not_mutate_paper_only_position_management(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        execution_mode="demo",
        execution_order_mode="testnet_order",
    )
    engine = PaperExecutionEngine(config)
    position = Position(
        symbol="INUSDT",
        opened_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        entry_price=1.0,
        current_price=1.0,
        stop_loss_price=0.96,
        take_profit_price=1.12,
        notional_usdt=100,
        score=80,
        rationale="trend_followthrough,r_based_scaleout",
        setup_type="trend",
        initial_risk_pct=0.04,
    )
    current = MarketSnapshot(
        symbol="INUSDT",
        mark_price=1.09,
        score=82,
        kline_direction=1,
        sentiment_score=80,
        sentiment_mentions=40,
        sentiment_source="binance_top_movers,opentwitter_search",
        oi_change_pct_4h=20,
        taker_buy_sell_ratio=1.2,
    )

    scaleout_plan, scaleout_position = engine._maybe_scale_out_breakout(position, current)
    pyramided = engine._maybe_pyramid_add(position, current)
    trailed = engine._maybe_trailing_stop(position, current)

    assert scaleout_plan is None
    assert scaleout_position == position
    assert pyramided == position
    assert trailed == position


def test_testnet_order_mode_allows_staged_take_profit_partial(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        execution_mode="demo",
        execution_order_mode="testnet_order",
        staged_tp_enabled=True,
    )
    engine = PaperExecutionEngine(config)
    position = Position(
        symbol="INUSDT",
        opened_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        entry_price=1.0,
        current_price=1.0,
        stop_loss_price=0.96,
        take_profit_price=1.06,
        notional_usdt=100,
        score=80,
        rationale="trend_followthrough",
        setup_type="trend",
        initial_risk_pct=0.04,
    )
    current = MarketSnapshot(
        symbol="INUSDT",
        mark_price=1.061,
        score=82,
        sentiment_score=80,
        sentiment_mentions=40,
        oi_change_pct_4h=20,
        taker_buy_sell_ratio=1.2,
    )

    staged_plan, staged_position = engine._maybe_scale_out_staged(position, current)

    assert staged_plan is not None
    assert staged_position is not None
    assert staged_plan.action == "take_profit_partial"
    assert staged_plan.notional_usdt == round(100 * config.staged_tp_stage1_fraction, 8)
    assert staged_plan.stop_loss_price == position.entry_price
    assert staged_position.notional_usdt == round(100 - staged_plan.notional_usdt, 8)
    assert staged_position.take_profit_stage == 1
    assert staged_position.stop_loss_price == position.entry_price


def test_existing_positions_use_staged_targets_instead_of_legacy_take_profit(tmp_path: Path) -> None:
    macro_cache = tmp_path / "macro.json"
    macro_cache.write_text('{"btc_trend": "bullish"}', encoding="utf-8")
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        execution_mode="demo",
        execution_order_mode="testnet_order",
        staged_tp_enabled=True,
        macro_trend_cache_file=macro_cache,
    )
    engine = PaperExecutionEngine(config)
    current_above_legacy_below_stage = MarketSnapshot(
        symbol="PLAYUSDT",
        mark_price=1.081,
        score=82,
        sentiment_score=80,
        sentiment_mentions=40,
    )
    old_position_with_low_take_profit = Position(
        symbol="PLAYUSDT",
        opened_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        entry_price=1.0,
        current_price=1.0,
        stop_loss_price=0.94,
        take_profit_price=1.08,
        notional_usdt=100,
        score=80,
        rationale="trend_followthrough",
        setup_type="trend",
        initial_risk_pct=0.06,
    )
    current_at_stage_target = MarketSnapshot(
        symbol="PLAYUSDT",
        mark_price=1.091,
        score=82,
        sentiment_score=80,
        sentiment_mentions=40,
    )
    old_position_with_high_take_profit = Position(
        symbol="BNBUSDT",
        opened_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        entry_price=1.0,
        current_price=1.0,
        stop_loss_price=0.96,
        take_profit_price=1.08,
        notional_usdt=100,
        score=80,
        rationale="smart_money_large_cap_structure",
        setup_type="trend",
        initial_risk_pct=0.04,
    )
    current_at_early_stage_target = MarketSnapshot(
        symbol="BNBUSDT",
        mark_price=1.061,
        score=82,
        sentiment_score=80,
        sentiment_mentions=40,
    )

    assert engine._should_exit(old_position_with_low_take_profit, current_above_legacy_below_stage) is None
    assert engine._should_exit(old_position_with_low_take_profit, current_at_stage_target) == "staged_take_profit"
    assert engine._should_exit(old_position_with_high_take_profit, current_at_early_stage_target) == "staged_take_profit"


def test_external_order_mode_does_not_defer_exchange_stop(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        execution_mode="demo",
        execution_order_mode="testnet_order",
    )
    engine = PaperExecutionEngine(config)
    position = Position(
        symbol="LABUSDT",
        opened_at=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        entry_price=1.0,
        current_price=1.0,
        stop_loss_price=0.96,
        take_profit_price=1.12,
        notional_usdt=100,
        score=70,
        rationale="mania_followthrough,repeated_signal_4h_probe",
        setup_type="mania_followthrough",
        last_sentiment_score=60,
        last_sentiment_mentions=20,
        last_oi_change_pct_4h=6,
    )
    intact_but_stopped = MarketSnapshot(
        symbol="LABUSDT",
        mark_price=0.95,
        price_change_pct_4h=-1.5,
        oi_change_pct_4h=7,
        taker_buy_sell_ratio=1.0,
        sentiment_score=65,
        sentiment_mentions=25,
        quote_volume_usdt_24h=200_000_000,
        price_position_7d=0.86,
        price_position_20d=0.84,
        breakout_distance_7d=0.2,
        score=62,
        sentiment_source="binance_top_movers,coingecko_trending",
        setup_type="trend",
    )

    assert engine._should_exit(position, intact_but_stopped) == "stop_loss_hit"


def test_first_signal_probe_allows_small_trial_before_full_confirmation(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        position_notional_usdt=200,
        first_signal_probe_position_scale=0.25,
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    first_signal = MarketSnapshot(
        symbol="FHEUSDT",
        mark_price=0.026,
        price_change_pct_4h=1.0,
        price_change_pct_24h=9.0,
        quote_volume_usdt_24h=26_000_000,
        funding_rate=0.00005,
        open_interest_usdt=20_000_000,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.29,
        sentiment_score=84,
        sentiment_mentions=12,
        sentiment_source="coingecko_trending,binance_top_movers",
        price_position_7d=0.99,
        price_position_20d=0.99,
        breakout_distance_7d=0.07,
        setup_type="trend",
        score=44.78,
        hotness_priority_score=6.0,
        reasons=["first_signal_probe"],
    )

    result = engine.rebalance([first_signal])

    assert len(result["entries"]) == 1
    assert result["entries"][0]["notional_usdt"] == 50


def test_retest_add_support_uses_larger_but_still_reduced_size(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        position_notional_usdt=200,
        retest_add_position_scale=0.4,
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    retest = MarketSnapshot(
        symbol="FHEUSDT",
        mark_price=0.0215,
        price_change_pct_4h=2.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=60_000_000,
        oi_change_pct_4h=4.0,
        taker_buy_sell_ratio=1.05,
        sentiment_score=76,
        sentiment_mentions=24,
        sentiment_source="binance_top_movers,opentwitter_search",
        price_position_7d=0.91,
        price_position_20d=0.90,
        breakout_distance_7d=0.08,
        setup_type="trend",
        score=66.0,
        hotness_priority_score=8.0,
        reasons=["retest_hold"],
    )

    result = engine.rebalance([retest])

    assert len(result["entries"]) == 1
    assert result["entries"][0]["notional_usdt"] == 80


def test_shadow_paper_opens_virtual_position_from_high_model_score(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        entry_threshold=90,
        shadow_paper_enabled=True,
        shadow_paper_prob_moon_threshold=0.6,
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    engine._score_with_shadow_models = lambda _features: {
        "scores": {"mixed_undersampled": {"model_label": 2, "prob_0": 0.1, "prob_1": 0.2, "prob_2": 0.7, "prob_moon": 0.9}},
        "errors": {},
        "model_meta": {"mixed_undersampled": {"status": "loaded"}},
    }
    candidate = MarketSnapshot(
        symbol="SHADOWUSDT",
        mark_price=1.0,
        price_change_pct_24h=4.0,
        quote_volume_usdt_24h=50_000_000,
        funding_rate=0.0001,
        open_interest_usdt=10_000_000,
        oi_change_pct_4h=3.0,
        taker_buy_sell_ratio=0.95,
        listing_days=30,
        sentiment_score=40,
        sentiment_mentions=1,
        sentiment_source="binance_top_movers",
        setup_type="trend_followthrough",
        score=20.0,
        reasons=["shadow only"],
    )

    result = engine.rebalance([candidate])

    assert result["entries"] == []
    assert len(result["shadow_paper"]["entries"]) == 1
    assert result["shadow_paper"]["entries"][0]["symbol"] == "SHADOWUSDT"
    persisted = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert len(persisted["shadow_positions"]) == 1
    assert persisted["shadow_positions"][0]["shadow_trigger"]["model"] == "mixed_undersampled"
    assert persisted["shadow_positions"][0]["shadow_position_key"].startswith("SHADOWUSDT:")
    assert persisted["shadow_positions"][0]["binance_mirror_status"] == "pending_submit"
    assert persisted["shadow_positions"][0]["binance_mirror_error"] is None


def test_shadow_paper_blocks_no_edge_trend_entry_by_default(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        shadow_paper_enabled=True,
        shadow_paper_prob_moon_threshold=0.6,
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    engine._score_with_shadow_models = lambda _features: {
        "scores": {"mixed_undersampled": {"model_label": 2, "prob_0": 0.1, "prob_1": 0.2, "prob_2": 0.7, "prob_moon": 0.9}},
        "errors": {},
        "model_meta": {"mixed_undersampled": {"status": "loaded"}},
    }
    candidate = MarketSnapshot(
        symbol="NOEDGEUSDT",
        mark_price=1.0,
        quote_volume_usdt_24h=50_000_000,
        setup_type="trend",
        setup_quality=0.0,
        score=80.0,
    )

    result = engine.rebalance([candidate])

    assert result["shadow_paper"]["entries"] == []
    assert result["shadow_paper"]["blocked_entry_count"] == 1
    assert result["shadow_paper"]["blocked_entries"][0]["reason"] == "setup_type_no_edge"
    persisted = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert persisted["shadow_positions"] == []


def test_shadow_paper_can_disable_trend_entry_block(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        shadow_paper_enabled=True,
        shadow_paper_prob_moon_threshold=0.6,
        block_trend_entry=False,
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    engine._score_with_shadow_models = lambda _features: {
        "scores": {"mixed_undersampled": {"model_label": 2, "prob_0": 0.1, "prob_1": 0.2, "prob_2": 0.7, "prob_moon": 0.9}},
        "errors": {},
        "model_meta": {"mixed_undersampled": {"status": "loaded"}},
    }
    candidate = MarketSnapshot(
        symbol="NOEDGEUSDT",
        mark_price=1.0,
        quote_volume_usdt_24h=50_000_000,
        setup_type="trend",
        setup_quality=0.0,
        score=80.0,
    )

    result = engine.rebalance([candidate])

    assert len(result["shadow_paper"]["entries"]) == 1
    assert result["shadow_paper"]["blocked_entry_count"] == 0


def test_shadow_paper_partially_takes_profit_and_keeps_runner(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", shadow_paper_enabled=True, shadow_paper_prob_moon_threshold=0.6)
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "learning_journal": [],
                "shadow_positions": [
                    {
                        "symbol": "SHADOWUSDT",
                        "side": "long",
                        "opened_at": opened_at,
                        "entry_price": 1.0,
                        "current_price": 1.0,
                        "stop_loss_price": 0.96,
                        "take_profit_price": 1.08,
                        "notional_usdt": 100.0,
                        "score": 80.0,
                        "setup_type": "trend",
                        "rationale": "shadow",
                    }
                ],
                "shadow_journal": [],
            }
        ),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    engine._score_with_shadow_models = lambda _features: {"scores": {}, "errors": {}, "model_meta": {}}
    current = MarketSnapshot(
        symbol="SHADOWUSDT",
        mark_price=1.1,
        price_change_pct_24h=10.0,
        quote_volume_usdt_24h=50_000_000,
        funding_rate=0.0001,
        open_interest_usdt=10_000_000,
        oi_change_pct_4h=3.0,
        taker_buy_sell_ratio=1.05,
        listing_days=30,
        sentiment_score=60,
        sentiment_mentions=10,
        sentiment_source="manual_distill",
        setup_type="trend",
        score=70.0,
    )

    result = engine.rebalance([current])

    assert len(result["shadow_paper"]["exits"]) == 1
    assert result["shadow_paper"]["exits"][0]["action"] == "take_profit_partial"
    assert result["shadow_paper"]["exits"][0]["reason"] == "shadow_partial_take_profit"
    assert abs(result["shadow_paper"]["exits"][0]["notional_usdt"] - 33.3) < 0.01
    assert abs(result["shadow_paper"]["exits"][0]["size_fraction"] - 0.333) < 0.001
    assert result["shadow_paper"]["exits"][0]["pnl_pct"] == 0.1
    persisted = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert len(persisted["shadow_positions"]) == 1
    runner = persisted["shadow_positions"][0]
    assert abs(runner["notional_usdt"] - 66.7) < 0.01
    assert runner["original_notional_usdt"] == 100.0
    assert runner["take_profit_stage"] == 1
    assert runner["stop_loss_price"] == 1.0
    assert runner["high_watermark"] == 1.1
    assert persisted["shadow_journal"][-1]["source"] == "shadow_paper"


def test_shadow_paper_runner_exits_on_trailing_stop(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        shadow_paper_enabled=True,
        shadow_paper_prob_moon_threshold=0.6,
        shadow_paper_trailing_pct=0.05,
    )
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "learning_journal": [],
                "shadow_positions": [
                    {
                        "symbol": "SHADOWUSDT",
                        "side": "long",
                        "opened_at": opened_at,
                        "entry_price": 1.0,
                        "current_price": 1.2,
                        "stop_loss_price": 1.03,
                        "take_profit_price": 1.08,
                        "notional_usdt": 50.0,
                        "original_notional_usdt": 100.0,
                        "take_profit_stage": 2,
                        "initial_risk_pct": 0.03,
                        "high_watermark": 1.2,
                        "score": 80.0,
                        "setup_type": "trend",
                        "rationale": "shadow",
                    }
                ],
                "shadow_journal": [],
            }
        ),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    engine._score_with_shadow_models = lambda _features: {"scores": {}, "errors": {}, "model_meta": {}}
    current = MarketSnapshot(
        symbol="SHADOWUSDT",
        mark_price=1.13,
        price_change_pct_24h=10.0,
        quote_volume_usdt_24h=50_000_000,
        funding_rate=0.0001,
        open_interest_usdt=10_000_000,
        oi_change_pct_4h=3.0,
        taker_buy_sell_ratio=1.05,
        listing_days=30,
        sentiment_score=60,
        sentiment_mentions=10,
        sentiment_source="manual_distill",
        setup_type="trend",
        score=70.0,
    )

    result = engine.rebalance([current])

    assert len(result["shadow_paper"]["exits"]) == 1
    assert result["shadow_paper"]["exits"][0]["reason"] == "shadow_trailing_stop"
    assert result["shadow_paper"]["exits"][0]["notional_usdt"] == 50.0
    assert result["shadow_paper"]["exits"][0]["binance_mirror_status"] == "pending_submit"
    assert result["shadow_paper"]["positions"] == []


def test_shadow_paper_runner_exits_on_max_tp_multiplier(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        shadow_paper_enabled=True,
        shadow_paper_prob_moon_threshold=0.6,
    )
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "learning_journal": [],
                "shadow_positions": [
                    {
                        "symbol": "SHADOWUSDT",
                        "side": "long",
                        "opened_at": opened_at,
                        "entry_price": 1.0,
                        "current_price": 1.12,
                        "stop_loss_price": 1.03,
                        "take_profit_price": 1.08,
                        "notional_usdt": 50.0,
                        "original_notional_usdt": 100.0,
                        "take_profit_stage": 2,
                        "initial_risk_pct": 0.03,
                        "high_watermark": 1.12,
                        "score": 80.0,
                        "setup_type": "trend",
                        "rationale": "shadow",
                    }
                ],
                "shadow_journal": [],
            }
        ),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    engine._score_with_shadow_models = lambda _features: {"scores": {}, "errors": {}, "model_meta": {}}
    current = MarketSnapshot(
        symbol="SHADOWUSDT",
        mark_price=1.16,
        price_change_pct_24h=10.0,
        quote_volume_usdt_24h=50_000_000,
        funding_rate=0.0001,
        open_interest_usdt=10_000_000,
        oi_change_pct_4h=3.0,
        taker_buy_sell_ratio=1.05,
        listing_days=30,
        sentiment_score=60,
        sentiment_mentions=10,
        sentiment_source="manual_distill",
        setup_type="trend",
        score=70.0,
    )

    result = engine.rebalance([current])

    assert len(result["shadow_paper"]["exits"]) == 1
    assert result["shadow_paper"]["exits"][0]["reason"] == "shadow_runner_max_tp"
    assert result["shadow_paper"]["exits"][0]["notional_usdt"] == 50.0
    assert result["shadow_paper"]["positions"] == []


def test_shadow_paper_does_not_reenter_symbol_after_same_cycle_exit(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", shadow_paper_enabled=True, shadow_paper_prob_moon_threshold=0.6)
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "learning_journal": [],
                "shadow_positions": [
                    {
                        "symbol": "GTCUSDT",
                        "side": "long",
                        "opened_at": opened_at,
                        "entry_price": 1.0,
                        "current_price": 1.0,
                        "stop_loss_price": 0.96,
                        "take_profit_price": 1.08,
                        "notional_usdt": 100.0,
                        "score": 20.0,
                        "setup_type": "trend",
                        "rationale": "shadow",
                    }
                ],
                "shadow_journal": [],
            }
        ),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    engine._score_with_shadow_models = lambda _features: {
        "scores": {"decision_class_weighted": {"model_label": 2, "prob_0": 0.05, "prob_1": 0.05, "prob_2": 0.9, "prob_moon": 0.95}},
        "errors": {},
        "model_meta": {"decision_class_weighted": {"status": "loaded"}},
    }
    current = MarketSnapshot(
        symbol="GTCUSDT",
        mark_price=0.95,
        price_change_pct_24h=30.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=-0.001,
        open_interest_usdt=20_000_000,
        oi_change_pct_4h=30.0,
        taker_buy_sell_ratio=1.05,
        listing_days=30,
        sentiment_score=55,
        sentiment_mentions=20,
        sentiment_source="binance_top_movers",
        setup_type="trend",
        score=25.0,
    )

    result = engine.rebalance([current])

    assert len(result["shadow_paper"]["exits"]) == 1
    assert result["shadow_paper"]["entries"] == []
    persisted = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert persisted["shadow_positions"] == []


def test_shadow_paper_reentry_cooldown_blocks_chasing_recent_exit(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        shadow_paper_prob_moon_threshold=0.6,
        shadow_paper_reentry_cooldown_minutes=120,
        shadow_paper_reentry_min_price_move_pct=0.03,
    )
    exited_at = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "learning_journal": [],
                "shadow_positions": [],
                "shadow_journal": [
                    {
                        "timestamp": exited_at,
                        "action": "exit",
                        "source": "shadow_paper",
                        "symbol": "GTCUSDT",
                        "side": "long",
                        "reason": "shadow_take_profit_hit",
                        "exit_price": 1.1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _symbol: {"views": {}})
    engine._score_with_shadow_models = lambda _features: {
        "scores": {"decision_class_weighted": {"model_label": 2, "prob_0": 0.05, "prob_1": 0.05, "prob_2": 0.9, "prob_moon": 0.95}},
        "errors": {},
        "model_meta": {"decision_class_weighted": {"status": "loaded"}},
    }
    candidate = MarketSnapshot(
        symbol="GTCUSDT",
        mark_price=1.12,
        price_change_pct_24h=30.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=-0.001,
        open_interest_usdt=20_000_000,
        oi_change_pct_4h=30.0,
        taker_buy_sell_ratio=1.05,
        listing_days=30,
        sentiment_score=55,
        sentiment_mentions=20,
        sentiment_source="binance_top_movers",
        setup_type="trend",
        score=25.0,
    )

    result = engine.rebalance([candidate])

    assert result["shadow_paper"]["entries"] == []
    persisted = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert persisted["shadow_positions"] == []


# --- trend_up regime exit tests ---


def test_trend_up_long_does_not_exit_at_original_tp(tmp_path: Path) -> None:
    """In trend_up, a long position should NOT exit at the original TP price."""
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        take_profit_pct=0.08,
        trend_up_tp_extension_multiplier=1.5,
    )
    config.state_file.write_text(
        json.dumps({
            "positions": [{
                "symbol": "TRENDUSDT",
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "entry_price": 1.0,
                "current_price": 1.08,
                "stop_loss_price": 0.97,
                "take_profit_price": 1.08,
                "notional_usdt": 100.0,
                "score": 60.0,
                "rationale": "trend_followthrough",
                "setup_type": "trend",
            }],
            "journal": [],
            "learning_journal": [],
        }),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config)
    regime = {"regime": "trend_up", "allow_new_entries": True,
              "entry_threshold_multiplier": 0.96, "stop_loss_multiplier": 1.05,
              "take_profit_multiplier": 1.1}
    current = MarketSnapshot(
        symbol="TRENDUSDT", mark_price=1.08, price_change_pct_24h=10.0,
        quote_volume_usdt_24h=50_000_000, funding_rate=0.0,
        open_interest_usdt=20_000_000, oi_change_pct_4h=6.0,
        taker_buy_sell_ratio=1.06, listing_days=30,
        sentiment_score=70, sentiment_mentions=30,
        sentiment_source="manual_distill", setup_type="trend", score=60.0,
    )
    result = engine.rebalance([current], regime=regime)
    assert result["exits"] == []
    assert len(result["positions"]) == 1


def test_trend_up_long_exits_at_extended_tp(tmp_path: Path) -> None:
    """In trend_up, a long position exits when price reaches extended TP (1.5x distance)."""
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        take_profit_pct=0.08,
        trend_up_tp_extension_multiplier=1.5,
    )
    config.state_file.write_text(
        json.dumps({
            "positions": [{
                "symbol": "TRENDUSDT",
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "entry_price": 1.0,
                "current_price": 1.12,
                "stop_loss_price": 0.97,
                "take_profit_price": 1.08,
                "notional_usdt": 100.0,
                "score": 60.0,
                "rationale": "trend_followthrough",
                "setup_type": "trend",
            }],
            "journal": [],
            "learning_journal": [],
        }),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config)
    regime = {"regime": "trend_up", "allow_new_entries": True,
              "entry_threshold_multiplier": 0.96, "stop_loss_multiplier": 1.05,
              "take_profit_multiplier": 1.1}
    current = MarketSnapshot(
        symbol="TRENDUSDT", mark_price=1.12, price_change_pct_24h=10.0,
        quote_volume_usdt_24h=50_000_000, funding_rate=0.0,
        open_interest_usdt=20_000_000, oi_change_pct_4h=6.0,
        taker_buy_sell_ratio=1.06, listing_days=30,
        sentiment_score=70, sentiment_mentions=30,
        sentiment_source="manual_distill", setup_type="trend", score=60.0,
    )
    result = engine.rebalance([current], regime=regime)
    assert len(result["exits"]) == 1
    assert result["exits"][0]["rationale"] == "take_profit_hit"


def test_trend_up_does_not_relax_short_score_faded(tmp_path: Path) -> None:
    """In trend_up, short positions should still exit at original exit_threshold."""
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        exit_threshold=45,
        trend_up_exit_threshold=30,
    )
    config.state_file.write_text(
        json.dumps({
            "positions": [{
                "symbol": "SHORTUSDT",
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "entry_price": 1.0,
                "current_price": 0.99,
                "stop_loss_price": 1.05,
                "take_profit_price": 0.92,
                "notional_usdt": 100.0,
                "score": 35.0,
                "rationale": "short_squeeze_fade",
                "setup_type": "short_squeeze",
            }],
            "journal": [],
            "learning_journal": [],
        }),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config)
    regime = {"regime": "trend_up", "allow_new_entries": True,
              "entry_threshold_multiplier": 0.96, "stop_loss_multiplier": 1.05,
              "take_profit_multiplier": 1.1}
    current = MarketSnapshot(
        symbol="SHORTUSDT", mark_price=0.99, price_change_pct_24h=10.0,
        quote_volume_usdt_24h=50_000_000, funding_rate=0.0,
        open_interest_usdt=20_000_000, oi_change_pct_4h=6.0,
        taker_buy_sell_ratio=1.06, listing_days=30,
        sentiment_score=70, sentiment_mentions=30,
        sentiment_source="manual_distill", setup_type="short_squeeze", score=35.0,
    )
    result = engine.rebalance([current], regime=regime)
    assert len(result["exits"]) == 1
    assert result["exits"][0]["rationale"] == "score_faded"


def test_trend_up_extends_scaleout_target(tmp_path: Path) -> None:
    """In trend_up, r_based_scaleout should NOT trigger at original 1R target."""
    config = StrategyConfig(
        state_file=tmp_path / "paper_state.json",
        breakout_first_take_profit_r=1.0,
        trend_up_tp_extension_multiplier=1.5,
        breakout_retain_min_sentiment_score=0,
        breakout_retain_min_mentions=0,
        breakout_retain_min_oi_change_pct=0,
        breakout_retain_min_taker_ratio=0,
    )
    config.state_file.write_text(
        json.dumps({
            "positions": [{
                "symbol": "SCALEUSDT",
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "entry_price": 1.0,
                "current_price": 1.03,
                "stop_loss_price": 0.97,
                "take_profit_price": 1.08,
                "notional_usdt": 100.0,
                "score": 70.0,
                "rationale": "structure_stop,r_based_scaleout",
                "setup_type": "base_breakout_confirmed",
                "take_profit_stage": 0,
                "initial_risk_pct": 0.03,
            }],
            "journal": [],
            "learning_journal": [],
        }),
        encoding="utf-8",
    )
    engine = PaperExecutionEngine(config)
    regime = {"regime": "trend_up", "allow_new_entries": True,
              "entry_threshold_multiplier": 0.96, "stop_loss_multiplier": 1.05,
              "take_profit_multiplier": 1.1}
    # Price at original 1R target (1.0 + 0.03*1.0 = 1.03) but below extended (1.0 + 0.03*1.5 = 1.045)
    current = MarketSnapshot(
        symbol="SCALEUSDT", mark_price=1.03, price_change_pct_24h=10.0,
        quote_volume_usdt_24h=50_000_000, funding_rate=0.0,
        open_interest_usdt=20_000_000, oi_change_pct_4h=8.0,
        taker_buy_sell_ratio=1.08, listing_days=30,
        sentiment_score=80, sentiment_mentions=30,
        sentiment_source="manual_distill", setup_type="base_breakout_confirmed", score=70.0,
    )
    result = engine.rebalance([current], regime=regime)
    # Should NOT have partial exit since price hasn't reached extended target
    assert result["exits"] == []
    assert len(result["positions"]) == 1


def test_shadow_paper_exits_position_missing_from_universe_via_fetcher(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", shadow_paper_enabled=True, shadow_paper_prob_moon_threshold=0.6)
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "learning_journal": [],
                "shadow_positions": [
                    {
                        "symbol": "GONEUSDT",
                        "side": "long",
                        "opened_at": opened_at,
                        "entry_price": 1.0,
                        "current_price": 1.0,
                        "stop_loss_price": 0.96,
                        "take_profit_price": 1.08,
                        "notional_usdt": 15.0,
                        "original_notional_usdt": 15.0,
                        "score": 60.0,
                        "entry_score": 60.0,
                        "setup_type": "trend",
                        "take_profit_stage": 0,
                        "high_watermark": None,
                        "low_watermark": None,
                        "initial_risk_pct": 0.04,
                        "pnl_pct": 0.0,
                        "unrealized_pnl_usdt": 0.0,
                        "action": "open_long",
                        "source": "shadow_paper",
                    }
                ],
                "shadow_journal": [],
            }
        ),
        encoding="utf-8",
    )

    def mock_fetcher(symbol: str) -> float | None:
        if symbol == "GONEUSDT":
            return 0.90
        return None

    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _s: {"views": {}}, mark_price_fetcher=mock_fetcher)
    engine._score_with_shadow_models = lambda _features: {"scores": {}, "errors": {}, "model_meta": {}}

    result = engine.rebalance([])

    assert len(result["shadow_paper"]["exits"]) == 1
    assert result["shadow_paper"]["exits"][0]["symbol"] == "GONEUSDT"
    assert result["shadow_paper"]["exits"][0]["reason"] == "shadow_stop_loss_hit"
    assert len(result["shadow_paper"]["positions"]) == 0


def test_shadow_paper_keeps_position_when_fetcher_fails(tmp_path: Path) -> None:
    config = StrategyConfig(state_file=tmp_path / "paper_state.json", shadow_paper_prob_moon_threshold=0.6)
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    config.state_file.write_text(
        json.dumps(
            {
                "positions": [],
                "journal": [],
                "learning_journal": [],
                "shadow_positions": [
                    {
                        "symbol": "GONEUSDT",
                        "side": "long",
                        "opened_at": opened_at,
                        "entry_price": 1.0,
                        "current_price": 1.0,
                        "stop_loss_price": 0.96,
                        "take_profit_price": 1.08,
                        "notional_usdt": 15.0,
                        "original_notional_usdt": 15.0,
                        "score": 60.0,
                        "entry_score": 60.0,
                        "setup_type": "trend",
                        "take_profit_stage": 0,
                        "high_watermark": None,
                        "low_watermark": None,
                        "initial_risk_pct": 0.04,
                        "pnl_pct": 0.0,
                        "unrealized_pnl_usdt": 0.0,
                        "action": "open_long",
                        "source": "shadow_paper",
                    }
                ],
                "shadow_journal": [],
            }
        ),
        encoding="utf-8",
    )

    def failing_fetcher(symbol: str) -> float | None:
        raise RuntimeError("network error")

    engine = PaperExecutionEngine(config, smart_money_panel_provider=lambda _s: {"views": {}}, mark_price_fetcher=failing_fetcher)
    engine._score_with_shadow_models = lambda _features: {"scores": {}, "errors": {}, "model_meta": {}}

    result = engine.rebalance([])

    assert len(result["shadow_paper"]["exits"]) == 0
    assert len(result["shadow_paper"]["positions"]) == 1
    assert result["shadow_paper"]["positions"][0]["symbol"] == "GONEUSDT"


def test_execution_bridge_respects_short_exit_side(tmp_path: Path) -> None:
    config = StrategyConfig(
        execution_request_dir=tmp_path / "requests",
        execution_result_dir=tmp_path / "results",
    )
    bridge = BinanceSkillExecutionBridge(config)
    intent = bridge._exit_intent(
        {"action": "exit", "symbol": "ETHUSDT", "side": "short", "notional_usdt": 125},
        77.0,
    )

    assert intent.action == "close_short"
    assert intent.side == "BUY"
    assert intent.reduce_only is True


def test_execution_bridge_respects_short_partial_take_profit_side(tmp_path: Path) -> None:
    config = StrategyConfig(
        execution_request_dir=tmp_path / "requests",
        execution_result_dir=tmp_path / "results",
    )
    bridge = BinanceSkillExecutionBridge(config)
    intent = bridge._exit_intent(
        {"action": "take_profit_partial", "symbol": "ETHUSDT", "side": "short", "notional_usdt": 50},
        77.0,
    )

    assert intent.action == "take_profit_partial"
    assert intent.side == "BUY"
    assert intent.reduce_only is True
