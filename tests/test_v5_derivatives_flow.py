from trading_system.candidates import build_candidate_basket
from trading_system.config import StrategyConfig
from trading_system.models import MarketSnapshot
from trading_system.scoring import score_snapshot
from trading_system.v5_derivatives_flow import (
    classify_derivatives_flow_context,
    compute_perp_spot_volume_ratio,
)


def test_computes_perp_spot_volume_ratio():
    assert compute_perp_spot_volume_ratio(24_000_000, 1_000_000) == 24.0
    assert compute_perp_spot_volume_ratio(24_000_000, 0) is None


def test_negative_funding_squeeze_is_short_risk_not_long_entry_action():
    context = classify_derivatives_flow_context(
        symbol="TESTUSDT",
        side="short",
        price_change_pct_24h=28.0,
        price_change_pct_4h=6.0,
        oi_change_pct_4h=16.0,
        funding_rate=-0.003,
        taker_buy_sell_ratio=1.18,
    )

    assert "negative_funding_squeeze" in context.risk_tags
    assert "derivatives_short_risk" in context.risk_tags
    assert context.confidence_adjustment < 0
    assert not hasattr(context, "action")


def test_top_divergence_and_upper_pressure_reduce_long_confidence():
    context = classify_derivatives_flow_context(
        symbol="TESTUSDT",
        side="long",
        price_change_pct_24h=32.0,
        price_change_pct_4h=9.0,
        oi_change_pct_4h=-3.0,
        funding_rate=0.006,
        upper_wick_pct=2.0,
        candle_body_pct=0.4,
        taker_buy_sell_ratio=0.98,
    )

    assert context.flow_regime == "distribution"
    assert "oi_price_divergence_top" in context.risk_tags
    assert "upper_pressure_active" in context.risk_tags
    assert "derivatives_long_risk" in context.risk_tags
    assert context.confidence_adjustment < 0


def test_scoring_attaches_derivatives_flow_tags_to_snapshot_reasons():
    snapshot = MarketSnapshot(
        symbol="TESTUSDT",
        mark_price=1.0,
        price_change_pct_24h=30.0,
        price_change_pct_4h=8.0,
        quote_volume_usdt_24h=100_000_000,
        funding_rate=0.006,
        open_interest_usdt=20_000_000,
        oi_change_pct_4h=-2.0,
        taker_buy_sell_ratio=0.98,
        sentiment_score=80,
        sentiment_mentions=10,
        sentiment_source="manual_distill",
        kline_upper_shadow_pct=2.0,
        kline_body_pct=0.4,
        setup_type="base_breakout_confirmed",
        setup_quality=80,
    )

    scored = score_snapshot(snapshot, StrategyConfig())

    assert scored.derivatives_flow_confidence_adjustment < 0
    assert "oi_price_divergence_top" in scored.derivatives_flow_tags
    assert any(reason == "deriv_flow=oi_price_divergence_top" for reason in scored.reasons)


def test_candidate_basket_exposes_derivatives_flow_for_signal_screening():
    snapshot = MarketSnapshot(
        symbol="TESTUSDT",
        mark_price=1.0,
        score=75.0,
        derivatives_flow_regime="distribution",
        derivatives_flow_tags=["oi_price_divergence_top", "upper_pressure_active"],
        derivatives_flow_confidence_adjustment=-5.0,
        derivatives_flow_long_risk_score=9.5,
        derivatives_flow_short_risk_score=0.0,
        derivatives_flow_spot_confirmation="unknown",
        derivatives_flow_explanation="oi_price_divergence_top",
    )

    basket = build_candidate_basket([snapshot])

    assert basket[0]["derivatives_flow"]["regime"] == "distribution"
    assert "oi_price_divergence_top" in basket[0]["signal_risk_tags"]
    assert all(not reason.startswith("deriv_flow:") for reason in basket[0]["blocked_reasons"])
