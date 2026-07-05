from __future__ import annotations

import os

from trading_system.config import StrategyConfig
from trading_system.models import MarketSnapshot


TRUSTED_SENTIMENT_SOURCES = (
    "manual_distill",
    "manual_sentiment",
    "square_feed",
    "crypto-market-rank",
    "meme-rush",
    "trading-signal",
    "lanaaielsa-distill",
    "lanaai-square-distill",
    "gmgn-wallet-distill",
    "opennews_news",
    "opennews_listing",
    "opennews_onchain",
)
DEFAULT_CONFIG = StrategyConfig()


def _candidate_basket_limit() -> int:
    raw = os.getenv("TS_CANDIDATE_BASKET_LIMIT", str(DEFAULT_CONFIG.candidate_confirm_limit))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, DEFAULT_CONFIG.candidate_confirm_limit)


def build_candidate_basket(rankings: list[MarketSnapshot]) -> list[dict]:
    basket = []
    for snapshot in rankings[: _candidate_basket_limit()]:
        if snapshot.score <= 0:
            continue
        rationale = list(snapshot.reasons[:6])
        for reason in snapshot.reasons:
            if str(reason).startswith("news_confirm=") and reason not in rationale:
                rationale.append(reason)
        has_trusted_confirmation = _has_trusted_sentiment_confirmation(snapshot)
        has_any_mentions = snapshot.sentiment_mentions >= 1
        has_high_conviction_mentions = snapshot.sentiment_mentions >= DEFAULT_CONFIG.untrusted_entry_min_sentiment_mentions
        risk_band = "high"
        execution_status = "watch_only"
        risk_grade = "C"
        stop_reference = "fixed_pct"
        if snapshot.setup_type == "squeeze_short_candidate":
            risk_band = "anomaly_squeeze"
            execution_status = "executable_now" if snapshot.score >= 80 else "confirm_needed"
            risk_grade = "B" if snapshot.score >= 80 else "C"
            stop_reference = "squeeze_high"
        elif snapshot.setup_type == "base_breakdown_confirmed":
            risk_band = "base_breakdown"
            execution_status = "executable_now" if has_any_mentions and (has_trusted_confirmation or has_high_conviction_mentions) else "confirm_needed"
            risk_grade = "A" if execution_status == "executable_now" else "B"
            stop_reference = "structure_high_7d"
        elif snapshot.setup_type == "base_breakout_confirmed":
            risk_band = "base_breakout"
            execution_status = "executable_now" if (has_any_mentions and (has_trusted_confirmation or has_high_conviction_mentions)) else "confirm_needed"
            risk_grade = "A" if execution_status == "executable_now" else "B"
            stop_reference = "structure_low_7d"
        elif snapshot.setup_type == "oi_divergence_long_candidate":
            risk_band = "oi_divergence"
            divergence_is_mature = snapshot.oi_price_divergence_score >= DEFAULT_CONFIG.oi_divergence_min_divergence_score
            has_required_mentions = snapshot.sentiment_mentions >= DEFAULT_CONFIG.oi_divergence_min_sentiment_mentions
            has_required_source_count = _sentiment_source_count(snapshot) >= DEFAULT_CONFIG.oi_divergence_min_source_count
            execution_status = (
                "executable_now"
                if has_any_mentions
                and divergence_is_mature
                and has_required_mentions
                and has_required_source_count
                and snapshot.score >= 75
                and (has_trusted_confirmation or has_high_conviction_mentions)
                else "confirm_needed"
            )
            risk_grade = "A" if divergence_is_mature and has_required_mentions and has_required_source_count else "B"
            stop_reference = "range_low_4h"
        elif snapshot.setup_type == "mania_followthrough":
            risk_band = "mania_continuation"
            mania_has_confirmation = has_any_mentions and (has_trusted_confirmation or has_high_conviction_mentions)
            execution_status = "executable_now" if snapshot.score >= 80 and mania_has_confirmation else "confirm_needed"
            risk_grade = "B" if execution_status == "executable_now" else "C"
            stop_reference = "tight_mania_stop"
        elif snapshot.setup_type == "base_breakout_weak":
            risk_band = "watchlist"
            execution_status = "confirm_needed"
            risk_grade = "B"
            stop_reference = "structure_low_7d"
        elif snapshot.score >= 78 and snapshot.taker_buy_sell_ratio >= 1.15:
            risk_band = "aggressive_best"
        elif snapshot.score >= 65:
            risk_band = "watchlist"
        basket.append(
            {
                "symbol": snapshot.symbol,
                "score": snapshot.score,
                "risk_band": risk_band,
                "execution_status": execution_status,
                "risk_grade": risk_grade,
                "stop_reference": stop_reference,
                "setup_type": snapshot.setup_type,
                "setup_quality": snapshot.setup_quality,
                "manipulation_stage": snapshot.manipulation_stage,
                "manipulation_quality": snapshot.manipulation_quality,
                "manipulation_flags": snapshot.manipulation_flags,
                "derivatives_flow": {
                    "regime": snapshot.derivatives_flow_regime,
                    "tags": list(snapshot.derivatives_flow_tags),
                    "confidence_adjustment": snapshot.derivatives_flow_confidence_adjustment,
                    "long_risk_score": snapshot.derivatives_flow_long_risk_score,
                    "short_risk_score": snapshot.derivatives_flow_short_risk_score,
                    "spot_confirmation": snapshot.derivatives_flow_spot_confirmation,
                    "explanation": snapshot.derivatives_flow_explanation,
                },
                "signal_risk_tags": list(snapshot.derivatives_flow_tags),
                "price_position_7d": snapshot.price_position_7d,
                "price_position_14d": snapshot.price_position_14d,
                "price_position_20d": snapshot.price_position_20d,
                "breakout_distance_7d": snapshot.breakout_distance_7d,
                "compression_score": snapshot.compression_score,
                "kline_lower_shadow_pct": snapshot.kline_lower_shadow_pct,
                "kline_upper_shadow_pct": snapshot.kline_upper_shadow_pct,
                "kline_body_pct": snapshot.kline_body_pct,
                "kline_direction": snapshot.kline_direction,
                "kline_volume_vs_ma5": snapshot.kline_volume_vs_ma5,
                "kline_range_vs_ma5_range": snapshot.kline_range_vs_ma5_range,
                "kline_close_position": snapshot.kline_close_position,
                "kline_consecutive_direction": snapshot.kline_consecutive_direction,
                "kline_atr_pct": snapshot.kline_atr_pct,
                "kline_ma20_distance_pct": snapshot.kline_ma20_distance_pct,
                "kline_volume_trend": snapshot.kline_volume_trend,
                "kline_max_drawdown_5d": snapshot.kline_max_drawdown_5d,
                "kline_pattern_bullish_count": snapshot.kline_pattern_bullish_count,
                "kline_pattern_bearish_count": snapshot.kline_pattern_bearish_count,
                "kline_pattern_flags": snapshot.kline_pattern_flags,
                "h4_close_above_ma20": snapshot.h4_close_above_ma20,
                "h4_ma20_slope_pct": snapshot.h4_ma20_slope_pct,
                "h4_price_vs_ma20_pct": snapshot.h4_price_vs_ma20_pct,
                "h4_close_position_20bars": snapshot.h4_close_position_20bars,
                "h4_low_break_recent_20bars": snapshot.h4_low_break_recent_20bars,
                "h4_high_break_recent_20bars": snapshot.h4_high_break_recent_20bars,
                "h4_atr_compression": snapshot.h4_atr_compression,
                "h4_lower_highs_count": snapshot.h4_lower_highs_count,
                "h4_higher_lows_count": snapshot.h4_higher_lows_count,
                "h4_structure_state": snapshot.h4_structure_state,
                "h4_price_location": snapshot.h4_price_location,
                "h4_pattern_type": snapshot.h4_pattern_type,
                "bb_price_position_96": snapshot.bb_price_position_96,
                "bb_price_position_210": snapshot.bb_price_position_210,
                "bb_price_position_672_15m": snapshot.bb_price_position_672_15m,
                "bb_price_position_672_4h": snapshot.bb_price_position_672_4h,
                "bb_touch_zone": snapshot.bb_touch_zone,
                "momo_structure": {
                    "bars": {
                        "15m": snapshot.momo_15m_bars,
                        "1m": snapshot.momo_1m_bars,
                        "4h": snapshot.momo_4h_bars,
                        "1d": snapshot.momo_1d_bars,
                    },
                    "daily": {
                        "ma7": snapshot.daily_ma7,
                        "ma20": snapshot.daily_ma20,
                        "ema21": snapshot.daily_ema21,
                        "bb365_position": snapshot.daily_price_position_365,
                        "price_vs_ma7_pct": snapshot.daily_price_vs_ma7_pct,
                        "price_vs_ma20_pct": snapshot.daily_price_vs_ma20_pct,
                    },
                    "h4": {
                        "bb105_position": snapshot.bb_price_position_105_4h,
                        "bb672_position": snapshot.bb_price_position_672_4h,
                        "bb1440_position": snapshot.bb_price_position_1440_4h,
                    },
                    "m15": {
                        "bb96_position": snapshot.bb_price_position_96,
                        "bb120_position": snapshot.bb_price_position_120,
                        "bb210_position": snapshot.bb_price_position_210,
                        "bb300_position": snapshot.bb_price_position_300,
                        "bb672_position": snapshot.bb_price_position_672_15m,
                        "bb1440_position": snapshot.bb_price_position_1440_15m,
                        "bb1920_position": snapshot.bb_price_position_1920_15m,
                        "bb2880_position": snapshot.bb_price_position_2880_15m,
                        "bb4404_position": snapshot.bb_price_position_4404_15m,
                    },
                    "m1": {
                        "bb105_position": snapshot.bb_price_position_105_1m,
                        "sma210": snapshot.sma210_1m,
                        "bb300_position": snapshot.bb_price_position_300_1m,
                        "bb672_position": snapshot.bb_price_position_672_1m,
                        "bb1920_position": snapshot.bb_price_position_1920_1m,
                    },
                    "williams_r": {
                        "15m": snapshot.williams_r_15m,
                        "15m_zone": snapshot.williams_r_15m_zone,
                        "15m_120": snapshot.williams_r_15m_120,
                        "15m_120_zone": snapshot.williams_r_15m_120_zone,
                        "1m": snapshot.williams_r_1m,
                        "1m_zone": snapshot.williams_r_1m_zone,
                    },
                },
                "funding_rate": snapshot.funding_rate,
                "oi_change_pct_4h": snapshot.oi_change_pct_4h,
                "price_change_pct_5m": snapshot.price_change_pct_5m,
                "price_change_pct_24h": snapshot.price_change_pct_24h,
                "sentiment_score": snapshot.sentiment_score,
                "sentiment_mentions": snapshot.sentiment_mentions,
                "sentiment_source": snapshot.sentiment_source,
                "rationale": rationale,
                "blocked_reasons": [reason for reason in snapshot.reasons if reason in {"sentiment_below_min", "funding_negative", "oi_not_expanding", "oi_reference_cooling", "taker_weak", "taker_cooling", "momentum_weak", "volume_thin"}],
            }
        )
    return basket


def _has_trusted_sentiment_confirmation(snapshot: MarketSnapshot) -> bool:
    source = (snapshot.sentiment_source or "").lower()
    return any(item in source for item in TRUSTED_SENTIMENT_SOURCES)


def _sentiment_source_count(snapshot: MarketSnapshot) -> int:
    return len({part.strip() for part in (snapshot.sentiment_source or "").split(",") if part.strip()})
