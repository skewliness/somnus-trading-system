from trading_system.binance import classify_setup
from trading_system.config import StrategyConfig
from trading_system.models import MarketSnapshot
from trading_system.scoring import score_snapshot, _factor_backtest_adjustment


def test_scoring_prefers_hot_and_expanding_symbol() -> None:
    config = StrategyConfig()
    strong = MarketSnapshot(
        symbol="AIXBTUSDT",
        mark_price=1.0,
        price_change_pct_24h=14.0,
        quote_volume_usdt_24h=100_000_000,
        funding_rate=0.0004,
        open_interest_usdt=50_000_000,
        oi_change_pct_4h=14.0,
        taker_buy_sell_ratio=1.15,
        listing_days=45,
        sentiment_score=90,
        sentiment_mentions=100,
        sentiment_source="test",
        setup_type="base_breakout",
        setup_quality=82.0,
    )
    weak = MarketSnapshot(
        symbol="DOGEAIUSDT",
        mark_price=1.0,
        price_change_pct_24h=1.0,
        quote_volume_usdt_24h=2_000_000,
        funding_rate=-0.0001,
        open_interest_usdt=2_000_000,
        oi_change_pct_4h=1.0,
        taker_buy_sell_ratio=0.98,
        listing_days=10,
        sentiment_score=20,
        sentiment_mentions=10,
        sentiment_source="test",
    )

    strong = score_snapshot(strong, config)
    weak = score_snapshot(weak, config)

    assert strong.score > weak.score
    assert strong.score >= config.entry_threshold
    assert weak.score < config.exit_threshold


def test_scoring_penalizes_squeeze_anomaly() -> None:
    config = StrategyConfig()
    anomaly = MarketSnapshot(
        symbol="RAVEUSDT",
        mark_price=9.0,
        price_change_pct_24h=180.0,
        quote_volume_usdt_24h=500_000_000,
        funding_rate=-0.006,
        open_interest_usdt=100_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.11,
        listing_days=2,
        sentiment_score=88,
        sentiment_mentions=50,
        sentiment_source="test",
        setup_type="squeeze_anomaly",
        setup_quality=95.0,
    )
    controlled = MarketSnapshot(
        symbol="PEPEUSDT",
        mark_price=1.0,
        price_change_pct_24h=18.0,
        quote_volume_usdt_24h=500_000_000,
        funding_rate=0.0003,
        open_interest_usdt=100_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.11,
        listing_days=20,
        sentiment_score=88,
        sentiment_mentions=50,
        sentiment_source="test",
        setup_type="base_breakout",
        setup_quality=84.0,
    )

    anomaly = score_snapshot(anomaly, config)
    controlled = score_snapshot(controlled, config)

    assert controlled.score > anomaly.score
    assert any("setup=squeeze_anomaly" in reason for reason in anomaly.reasons)


def test_scoring_treats_base_breakdown_as_short_setup() -> None:
    config = StrategyConfig()
    snapshot = MarketSnapshot(
        symbol="BEARUSDT",
        mark_price=1.0,
        price_change_pct_24h=-12.0,
        quote_volume_usdt_24h=160_000_000,
        funding_rate=-0.0003,
        open_interest_usdt=80_000_000,
        oi_change_pct_4h=12.0,
        taker_buy_sell_ratio=0.86,
        listing_days=120,
        sentiment_score=82,
        sentiment_mentions=80,
        sentiment_source="manual_distill,opennews_market",
        setup_type="base_breakdown_confirmed",
        setup_quality=86.0,
    )

    scored = score_snapshot(snapshot, config)

    assert scored.score >= config.entry_threshold
    assert any("setup=base_breakdown_confirmed" in reason for reason in scored.reasons)
    assert any("short_24h=-12.0%" in reason for reason in scored.reasons)
    assert "funding_negative" not in scored.reasons
    assert "momentum_weak" not in scored.reasons
    assert "taker_weak" not in scored.reasons


def test_classify_setup_marks_base_breakdown_watch_without_confirmation() -> None:
    config = StrategyConfig()
    setup_type, quality = classify_setup(
        price_change_pct_24h=-18.0,
        price_change_pct_4h=-1.5,
        funding_rate=0.0003,
        oi_change_pct_4h=2.0,
        taker_buy_sell_ratio=0.88,
        sentiment_score=0.0,
        sentiment_mentions=0,
        listing_days=180,
        config=config,
        quote_volume_usdt_24h=120_000_000,
        price_position_7d=0.12,
        price_position_20d=0.34,
        breakout_distance_7d=0.0,
        compression_score=0.4,
    )

    assert setup_type == "base_breakdown_watch"
    assert 45.0 <= quality < 80.0


def test_scoring_keeps_base_breakdown_watch_below_entry_threshold() -> None:
    config = StrategyConfig()
    snapshot = MarketSnapshot(
        symbol="WATCHUSDT",
        mark_price=1.0,
        price_change_pct_24h=-18.0,
        quote_volume_usdt_24h=200_000_000,
        funding_rate=0.0003,
        open_interest_usdt=90_000_000,
        oi_change_pct_4h=8.0,
        taker_buy_sell_ratio=0.88,
        listing_days=120,
        sentiment_score=0,
        sentiment_mentions=0,
        sentiment_source="none",
        setup_type="base_breakdown_watch",
        setup_quality=76.0,
    )

    scored = score_snapshot(snapshot, config)

    assert scored.score < config.entry_threshold
    assert "watch_only_no_entry" in scored.reasons


def test_classify_setup_recognizes_short_cycle_trend_followthrough() -> None:
    config = StrategyConfig()
    setup_type, quality = classify_setup(
        price_change_pct_24h=18.0,
        price_change_pct_4h=6.0,
        funding_rate=0.0002,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.12,
        sentiment_score=88.0,
        sentiment_mentions=12,
        listing_days=30,
        config=config,
        price_position_7d=0.96,
        price_position_14d=0.91,
        price_position_20d=0.84,
        breakout_distance_7d=0.02,
        compression_score=0.3,
        price_change_pct_5m=3.5,
        quote_volume_usdt_24h=120_000_000,
    )

    assert setup_type == "mania_followthrough"
    assert quality >= 60.0


def test_classify_setup_prefers_multi_period_base_breakout() -> None:
    config = StrategyConfig()
    setup_type, quality = classify_setup(
        price_change_pct_24h=12.0,
        funding_rate=0.00025,
        oi_change_pct_4h=14.0,
        taker_buy_sell_ratio=1.14,
        sentiment_score=82.0,
        listing_days=35,
        config=config,
        price_position_7d=0.94,
        price_position_14d=0.71,
        price_position_20d=0.52,
        breakout_distance_7d=0.015,
        compression_score=0.72,
    )

    assert setup_type == "base_breakout_confirmed"
    assert quality >= 85.0


def test_classify_setup_rejects_extended_trend_without_base_structure() -> None:
    config = StrategyConfig()
    setup_type, quality = classify_setup(
        price_change_pct_24h=12.0,
        funding_rate=0.00025,
        oi_change_pct_4h=14.0,
        taker_buy_sell_ratio=1.14,
        sentiment_score=82.0,
        listing_days=35,
        config=config,
        price_position_7d=0.99,
        price_position_14d=0.96,
        price_position_20d=0.93,
        breakout_distance_7d=0.09,
        compression_score=0.18,
    )

    assert setup_type == "trend"
    assert quality == 0.0


def test_classify_setup_marks_weak_base_breakout() -> None:
    config = StrategyConfig()
    setup_type, quality = classify_setup(
        price_change_pct_24h=7.0,
        funding_rate=0.00015,
        oi_change_pct_4h=7.5,
        taker_buy_sell_ratio=1.05,
        sentiment_score=48.0,
        listing_days=50,
        config=config,
        price_position_7d=0.89,
        price_position_14d=0.79,
        price_position_20d=0.69,
        breakout_distance_7d=0.031,
        compression_score=0.58,
    )

    assert setup_type == "base_breakout_weak"
    assert 55.0 <= quality < 80.0


def test_classify_setup_marks_squeeze_short_candidate() -> None:
    config = StrategyConfig()
    setup_type, quality = classify_setup(
        price_change_pct_24h=130.0,
        funding_rate=-0.004,
        oi_change_pct_4h=16.0,
        taker_buy_sell_ratio=1.08,
        sentiment_score=70.0,
        listing_days=10,
        config=config,
        price_position_7d=1.0,
        price_position_14d=1.0,
        price_position_20d=1.0,
        breakout_distance_7d=0.12,
        compression_score=0.1,
    )

    assert setup_type == "squeeze_short_candidate"
    assert quality >= 90.0


def test_classify_setup_marks_mania_followthrough() -> None:
    config = StrategyConfig()
    setup_type, quality = classify_setup(
        price_change_pct_24h=31.0,
        price_change_pct_5m=4.2,
        funding_rate=0.00008,
        oi_change_pct_4h=1.2,
        taker_buy_sell_ratio=0.995,
        sentiment_score=78.0,
        sentiment_mentions=6,
        quote_volume_usdt_24h=108_000_000,
        listing_days=120,
        config=config,
        price_position_7d=1.0,
        price_position_14d=1.0,
        price_position_20d=1.0,
        breakout_distance_7d=0.002,
        compression_score=0.02,
    )

    assert setup_type == "mania_followthrough"
    assert quality >= 80.0


def test_classify_setup_marks_oi_divergence_candidate() -> None:
    config = StrategyConfig()
    setup_type, quality = classify_setup(
        price_change_pct_24h=6.0,
        price_change_pct_4h=0.1,
        funding_rate=0.00012,
        oi_change_pct_4h=20.0,
        taker_buy_sell_ratio=1.07,
        sentiment_score=76.0,
        listing_days=55,
        config=config,
        quote_volume_usdt_24h=96_000_000,
        sentiment_mentions=24,
    )

    assert setup_type == "oi_divergence_long_candidate"
    assert quality >= 75.0


def test_classify_setup_rejects_weak_oi_divergence_signal_before_it_enters_replay() -> None:
    config = StrategyConfig()
    setup_type, quality = classify_setup(
        price_change_pct_24h=6.0,
        price_change_pct_4h=0.8,
        funding_rate=0.00012,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.07,
        sentiment_score=76.0,
        listing_days=55,
        config=config,
        quote_volume_usdt_24h=96_000_000,
        sentiment_mentions=8,
    )

    assert setup_type == "trend"
    assert quality == 0.0


def test_scoring_rewards_oi_divergence_candidate() -> None:
    config = StrategyConfig()
    candidate = MarketSnapshot(
        symbol="PEPEUSDT",
        mark_price=0.1,
        price_change_pct_4h=0.1,
        price_change_pct_24h=5.0,
        quote_volume_usdt_24h=110_000_000,
        funding_rate=0.00015,
        open_interest_usdt=88_000_000,
        oi_change_pct_4h=19.0,
        taker_buy_sell_ratio=1.08,
        listing_days=32,
        sentiment_score=82,
        sentiment_mentions=24,
        sentiment_source="square_feed,coingecko_trending",
        setup_type="oi_divergence_long_candidate",
        setup_quality=84.0,
        oi_price_divergence_score=190.0,
    )

    candidate = score_snapshot(candidate, config)

    assert candidate.score >= config.entry_threshold
    assert any("setup=oi_divergence_long_candidate" in reason for reason in candidate.reasons)
    assert any("oi_div=190.0" == reason for reason in candidate.reasons)


def test_scoring_penalizes_unconfirmed_oi_divergence_even_with_mentions() -> None:
    config = StrategyConfig(oi_divergence_require_trusted_confirmation=True)
    candidate = MarketSnapshot(
        symbol="PEPEUSDT",
        mark_price=0.1,
        price_change_pct_4h=0.1,
        price_change_pct_24h=5.0,
        quote_volume_usdt_24h=110_000_000,
        funding_rate=0.00015,
        open_interest_usdt=88_000_000,
        oi_change_pct_4h=19.0,
        taker_buy_sell_ratio=1.08,
        listing_days=32,
        sentiment_score=82,
        sentiment_mentions=24,
        sentiment_source="twitter_search",
        setup_type="oi_divergence_long_candidate",
        setup_quality=84.0,
        oi_price_divergence_score=150.0,
    )

    candidate = score_snapshot(candidate, config)

    assert "oi_divergence_unconfirmed" in candidate.reasons
    assert candidate.score < config.entry_threshold


def test_scoring_penalizes_weak_oi_divergence_signal_even_with_trusted_confirmation() -> None:
    config = StrategyConfig(oi_divergence_min_divergence_score=120, oi_divergence_min_source_count=2)
    candidate = MarketSnapshot(
        symbol="PEPEUSDT",
        mark_price=0.1,
        price_change_pct_4h=0.7,
        price_change_pct_24h=5.0,
        quote_volume_usdt_24h=110_000_000,
        funding_rate=0.00015,
        open_interest_usdt=88_000_000,
        oi_change_pct_4h=19.0,
        taker_buy_sell_ratio=1.08,
        listing_days=32,
        sentiment_score=82,
        sentiment_mentions=16,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=84.0,
        oi_price_divergence_score=92.0,
    )

    candidate = score_snapshot(candidate, config)

    assert "oi_divergence_low_mentions" in candidate.reasons
    assert "oi_divergence_signal_weak" in candidate.reasons
    assert "oi_divergence_single_source" in candidate.reasons
    assert candidate.score < config.entry_threshold


def test_scoring_penalizes_single_source_oi_divergence_even_when_other_checks_pass() -> None:
    config = StrategyConfig(oi_divergence_min_source_count=2)
    candidate = MarketSnapshot(
        symbol="PEPEUSDT",
        mark_price=0.1,
        price_change_pct_4h=0.1,
        price_change_pct_24h=5.0,
        quote_volume_usdt_24h=110_000_000,
        funding_rate=0.00015,
        open_interest_usdt=88_000_000,
        oi_change_pct_4h=19.0,
        taker_buy_sell_ratio=1.08,
        listing_days=32,
        sentiment_score=82,
        sentiment_mentions=24,
        sentiment_source="square_feed",
        setup_type="oi_divergence_long_candidate",
        setup_quality=84.0,
        oi_price_divergence_score=150.0,
    )

    candidate = score_snapshot(candidate, config)

    assert "oi_divergence_single_source" in candidate.reasons
    assert candidate.score < config.entry_threshold


def test_scoring_penalizes_unconfirmed_low_mentions_for_trend() -> None:
    config = StrategyConfig(min_entry_sentiment_mentions=3)
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
    )

    candidate = score_snapshot(candidate, config)

    assert "unconfirmed_low_mentions" in candidate.reasons
    assert candidate.score < config.entry_threshold


def test_scoring_keeps_low_mention_trend_when_trusted_confirmation_exists() -> None:
    config = StrategyConfig(min_entry_sentiment_mentions=3)
    trusted = MarketSnapshot(
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
    )
    unconfirmed = MarketSnapshot(
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
    )

    trusted = score_snapshot(trusted, config)
    unconfirmed = score_snapshot(unconfirmed, config)

    assert "unconfirmed_low_mentions" not in trusted.reasons
    assert trusted.score > unconfirmed.score


def test_scoring_penalizes_single_source_untrusted_entry_before_mentions_override() -> None:
    config = StrategyConfig(
        entry_threshold=60,
        untrusted_entry_min_sentiment_mentions=8,
        untrusted_entry_min_source_count=2,
        untrusted_entry_high_mentions_override=18,
    )
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
        sentiment_mentions=17,
        sentiment_source="binance_top_movers",
        setup_type="trend",
        setup_quality=92.0,
    )

    candidate = score_snapshot(candidate, config)

    assert "single_source_unconfirmed" in candidate.reasons
    assert candidate.score >= config.entry_threshold


def test_scoring_allows_multi_source_untrusted_entry_before_mentions_override() -> None:
    config = StrategyConfig(
        entry_threshold=60,
        untrusted_entry_min_sentiment_mentions=8,
        untrusted_entry_min_source_count=2,
        untrusted_entry_high_mentions_override=18,
    )
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
    )
    weak_only = MarketSnapshot(
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
        sentiment_source="binance_top_movers,coingecko_trending",
        setup_type="trend",
        setup_quality=92.0,
    )

    confirmed = score_snapshot(confirmed, config)
    weak_only = score_snapshot(weak_only, config)

    assert "single_source_unconfirmed" not in confirmed.reasons
    assert "single_source_unconfirmed" in weak_only.reasons
    assert confirmed.score >= config.entry_threshold
    assert weak_only.score < confirmed.score


def test_scoring_keeps_blocking_weak_only_candidate_even_after_mentions_override() -> None:
    config = StrategyConfig(
        entry_threshold=60,
        untrusted_entry_min_sentiment_mentions=8,
        untrusted_entry_min_source_count=2,
        untrusted_entry_high_mentions_override=18,
    )
    candidate = MarketSnapshot(
        symbol="SPKUSDT",
        mark_price=1.0,
        price_change_pct_24h=24.0,
        quote_volume_usdt_24h=120_000_000,
        funding_rate=0.00025,
        open_interest_usdt=28_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.12,
        listing_days=35,
        sentiment_score=84,
        sentiment_mentions=18,
        sentiment_source="binance_top_movers",
        setup_type="base_breakout_confirmed",
        setup_quality=92.0,
    )

    candidate = score_snapshot(candidate, config)

    assert "unconfirmed_low_mentions" not in candidate.reasons
    assert "single_source_unconfirmed" in candidate.reasons
    assert candidate.score > 0


def test_scoring_rewards_accumulation_markup_framework() -> None:
    config = StrategyConfig()
    accumulation = MarketSnapshot(
        symbol="SIRENUSDT",
        mark_price=0.72,
        price_change_pct_24h=3.8,
        quote_volume_usdt_24h=48_000_000,
        funding_rate=0.00012,
        open_interest_usdt=34_000_000,
        oi_change_pct_4h=8.2,
        taker_buy_sell_ratio=1.04,
        listing_days=40,
        sentiment_score=74,
        sentiment_mentions=24,
        sentiment_source="trading-signal,square_feed",
        setup_type="trend",
        manipulation_stage="accumulation",
        manipulation_quality=81.0,
    )
    distribution = MarketSnapshot(
        symbol="RAVEUSDT",
        mark_price=2.4,
        price_change_pct_24h=28.0,
        quote_volume_usdt_24h=220_000_000,
        funding_rate=0.0038,
        open_interest_usdt=80_000_000,
        oi_change_pct_4h=-6.0,
        taker_buy_sell_ratio=0.98,
        listing_days=22,
        sentiment_score=85,
        sentiment_mentions=40,
        sentiment_source="trading-signal,square_feed",
        setup_type="trend",
        manipulation_stage="distribution",
        manipulation_quality=88.0,
    )

    accumulation = score_snapshot(accumulation, config)
    distribution = score_snapshot(distribution, config)

    assert accumulation.score > distribution.score
    assert any("stage=accumulation" in reason for reason in accumulation.reasons)
    assert any("stage=distribution" in reason for reason in distribution.reasons)


def test_scoring_penalizes_distribution_flags() -> None:
    config = StrategyConfig()
    snapshot = MarketSnapshot(
        symbol="RAVEUSDT",
        mark_price=2.4,
        price_change_pct_24h=16.0,
        quote_volume_usdt_24h=220_000_000,
        funding_rate=0.0004,
        open_interest_usdt=80_000_000,
        oi_change_pct_4h=9.0,
        taker_buy_sell_ratio=1.08,
        listing_days=22,
        sentiment_score=88,
        sentiment_mentions=40,
        sentiment_source="psyduck_case_library",
        manipulation_stage="distribution",
        manipulation_quality=87.0,
        manipulation_flags=["controlled_supply", "venue_concentration"],
    )

    scored = score_snapshot(snapshot, config)

    assert scored.score < config.entry_threshold
    assert any(reason == "distribution_risk" for reason in scored.reasons)
    assert any(reason == "flag=controlled_supply" for reason in scored.reasons)


def test_scoring_relaxes_oi_and_taker_penalties_for_mania_followthrough() -> None:
    config = StrategyConfig()
    mania = MarketSnapshot(
        symbol="TSTUSDT",
        mark_price=0.01,
        price_change_pct_5m=4.4,
        price_change_pct_24h=31.0,
        quote_volume_usdt_24h=108_000_000,
        funding_rate=0.00005,
        open_interest_usdt=5_000_000,
        oi_change_pct_4h=1.1,
        taker_buy_sell_ratio=0.995,
        listing_days=120,
        sentiment_score=78,
        sentiment_mentions=6,
        sentiment_source="trading-signal",
        setup_type="mania_followthrough",
        setup_quality=84.0,
    )
    plain = MarketSnapshot(
        symbol="TSTUSDT",
        mark_price=0.01,
        price_change_pct_5m=4.4,
        price_change_pct_24h=31.0,
        quote_volume_usdt_24h=108_000_000,
        funding_rate=0.00005,
        open_interest_usdt=5_000_000,
        oi_change_pct_4h=1.1,
        taker_buy_sell_ratio=0.995,
        listing_days=120,
        sentiment_score=78,
        sentiment_mentions=6,
        sentiment_source="trading-signal",
        setup_type="trend",
        setup_quality=0.0,
    )

    mania = score_snapshot(mania, config)
    plain = score_snapshot(plain, config)

    assert mania.score > plain.score
    assert "oi_not_expanding" not in mania.reasons
    assert "taker_weak" not in mania.reasons

def test_oi_divergence_scoring_uses_configured_threshold_not_legacy_floor() -> None:
    config = StrategyConfig(oi_divergence_min_divergence_score=220)
    below = MarketSnapshot(
        symbol="OIWEAKUSDT",
        mark_price=1.0,
        price_change_pct_24h=12.0,
        quote_volume_usdt_24h=100_000_000,
        funding_rate=0.0003,
        open_interest_usdt=50_000_000,
        oi_change_pct_4h=20.0,
        taker_buy_sell_ratio=1.12,
        listing_days=45,
        sentiment_score=82,
        sentiment_mentions=30,
        sentiment_source="square_feed,opentwitter_search",
        setup_type="oi_divergence_long_candidate",
        setup_quality=88.0,
        oi_price_divergence_score=190.0,
    )

    scored = score_snapshot(below, config)

    assert "oi_divergence_signal_weak" in scored.reasons
