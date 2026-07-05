from __future__ import annotations

from trading_system.config import StrategyConfig
from trading_system.models import MarketSnapshot
from trading_system.v5_derivatives_flow import classify_derivatives_flow_context


LOW_INFO_SETUP_TYPES = {"oi_divergence_long_candidate", "base_breakout_confirmed"}
SHORT_SETUP_TYPES = {"squeeze_short_candidate", "base_breakdown_confirmed", "base_breakdown_watch"}
WATCH_ONLY_SETUP_TYPES = {"base_breakdown_watch"}
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
    "opentwitter_search",
    "opentwitter_official",
    "opennews_news",
    "opennews_listing",
    "opennews_onchain",
    "opennews_market",
)
WEAK_DISCOVERY_SOURCES = {"binance_top_movers", "coingecko_trending"}


def score_snapshot(
    snapshot: MarketSnapshot,
    config: StrategyConfig,
    regime: dict | None = None,
    predictive_model: dict | None = None,
    hf_quality: dict | None = None,
    pdf_context: dict | None = None,
) -> MarketSnapshot:
    reasons: list[str] = []
    weighted = 0.0

    hotness = _scale(snapshot.sentiment_score, 40, 100)
    weighted += hotness * 30
    if hotness > 0:
        reasons.append(f"hotness={snapshot.sentiment_score:.0f}")

    short_fuel_long = _is_short_fuel_long_setup(snapshot)
    short_setup = _is_short_setup(snapshot)
    funding = _scale(snapshot.funding_rate, 0.0, 0.0008)
    weighted += funding * 15
    if funding > 0:
        reasons.append(f"funding={snapshot.funding_rate:.5f}")
    if short_fuel_long:
        weighted += 8
        reasons.append("short_fuel")
    if short_setup and snapshot.funding_rate <= 0:
        short_funding = 8.0 + min(8.0, abs(snapshot.funding_rate) * 4000.0)
        weighted += short_funding
        reasons.append(f"short_funding={snapshot.funding_rate:.5f}")

    oi = _scale(snapshot.oi_change_pct_4h, 0, 20)
    weighted += oi * 20
    if oi > 0:
        reasons.append(f"oi4h={snapshot.oi_change_pct_4h:.1f}%")

    if short_setup:
        if snapshot.price_change_pct_24h < 0:
            momentum = _scale(abs(snapshot.price_change_pct_24h), 0, 25)
            weighted += momentum * 15
            if momentum > 0:
                reasons.append(f"short_24h={snapshot.price_change_pct_24h:.1f}%")
        else:
            overextension = _scale(snapshot.price_change_pct_24h, 20, max(80.0, config.anomaly_price_change_pct))
            weighted += overextension * 10
            if overextension > 0:
                reasons.append(f"short_overextension={snapshot.price_change_pct_24h:.1f}%")
    else:
        momentum = _scale(snapshot.price_change_pct_24h, 0, 25)
        weighted += momentum * 15
        if momentum > 0:
            reasons.append(f"24h={snapshot.price_change_pct_24h:.1f}%")

    volume = _scale(snapshot.quote_volume_usdt_24h, config.min_quote_volume_usdt, 150_000_000)
    weighted += volume * 10
    if volume > 0:
        reasons.append(f"vol24h={snapshot.quote_volume_usdt_24h/1_000_000:.1f}M")

    if short_setup:
        taker = _scale(1.0 - snapshot.taker_buy_sell_ratio, 0.0, 0.3)
        weighted += taker * 10
        if taker > 0:
            reasons.append(f"short_taker={snapshot.taker_buy_sell_ratio:.2f}")
    else:
        taker = _scale(snapshot.taker_buy_sell_ratio, 1.0, 1.35)
        weighted += taker * 10
        if taker > 0:
            reasons.append(f"taker={snapshot.taker_buy_sell_ratio:.2f}")

    listing_bonus = _listing_bonus(snapshot.listing_days, config)
    weighted += listing_bonus
    if listing_bonus > 0:
        reasons.append(f"listing={snapshot.listing_days}d")

    if snapshot.setup_type == "base_breakout":
        weighted += 12
        reasons.append(f"setup=base_breakout:{snapshot.setup_quality:.1f}")
        if snapshot.price_position_20d is not None:
            reasons.append(f"pos20d={snapshot.price_position_20d:.2f}")
        if snapshot.breakout_distance_7d is not None:
            reasons.append(f"break7d={snapshot.breakout_distance_7d:.3f}")
        if snapshot.compression_score is not None:
            reasons.append(f"compress={snapshot.compression_score:.2f}")
    elif snapshot.setup_type == "base_breakdown_confirmed":
        weighted += 18
        reasons.append(f"setup=base_breakdown_confirmed:{snapshot.setup_quality:.1f}")
    elif snapshot.setup_type == "base_breakdown_watch":
        weighted += 8
        reasons.append(f"setup=base_breakdown_watch:{snapshot.setup_quality:.1f}")
    elif snapshot.setup_type == "squeeze_short_candidate":
        weighted += 16
        reasons.append(f"setup=squeeze_short_candidate:{snapshot.setup_quality:.1f}")
    elif snapshot.setup_type == "mania_followthrough":
        weighted += 14
        reasons.append(f"setup=mania_followthrough:{snapshot.setup_quality:.1f}")
        if snapshot.price_change_pct_5m > 0:
            reasons.append(f"5m={snapshot.price_change_pct_5m:.1f}%")
        if snapshot.sentiment_mentions > 0:
            reasons.append(f"mentions={snapshot.sentiment_mentions}")
    elif snapshot.setup_type == "squeeze_anomaly":
        weighted -= 8
        reasons.append(f"setup=squeeze_anomaly:{snapshot.setup_quality:.1f}")
    elif snapshot.setup_type == "oi_divergence_long_candidate":
        weighted += 16
        reasons.append(f"setup=oi_divergence_long_candidate:{snapshot.setup_quality:.1f}")
        reasons.append(f"oi_div={snapshot.oi_price_divergence_score:.1f}")
        if snapshot.price_change_pct_4h:
            reasons.append(f"4h={snapshot.price_change_pct_4h:.1f}%")

    if snapshot.manipulation_stage and snapshot.manipulation_stage != "trend":
        reasons.append(f"stage={snapshot.manipulation_stage}:{snapshot.manipulation_quality:.1f}")
        if snapshot.manipulation_stage == "accumulation":
            weighted += min(10.0, max(0.0, snapshot.manipulation_quality) / 10.0)
        elif snapshot.manipulation_stage == "markup":
            weighted += min(6.0, max(0.0, snapshot.manipulation_quality) / 20.0)
        elif snapshot.manipulation_stage == "distribution":
            weighted -= min(24.0, max(8.0, snapshot.manipulation_quality / 4.0))
            reasons.append("distribution_risk")
        elif snapshot.manipulation_stage == "markdown":
            weighted -= min(18.0, max(6.0, snapshot.manipulation_quality / 5.0))
            reasons.append("markdown_risk")
    for flag in snapshot.manipulation_flags:
        reasons.append(f"flag={flag}")

    penalties = 0.0
    if snapshot.sentiment_score < config.min_sentiment_score:
        sentiment_penalty = 10
        if snapshot.setup_type == "mania_followthrough" and snapshot.sentiment_score >= config.mania_followthrough_min_sentiment_score:
            sentiment_penalty = 0
        penalties += sentiment_penalty
        if sentiment_penalty:
            reasons.append("sentiment_below_min")
    if snapshot.funding_rate < config.min_positive_funding_rate:
        if short_fuel_long:
            reasons.append("funding_negative_as_short_fuel")
        elif short_setup:
            reasons.append("funding_negative_as_short_signal")
        else:
            penalties += 20
            reasons.append("funding_negative")
    if snapshot.oi_change_pct_4h < config.min_oi_change_pct:
        oi_penalty = 8
        if snapshot.setup_type == "mania_followthrough":
            if snapshot.oi_change_pct_4h >= config.mania_followthrough_min_oi_change_pct_4h:
                oi_penalty = 0
                reasons.append("oi_reference_only")
            else:
                oi_penalty = 4
        penalties += oi_penalty
        if oi_penalty >= 8:
            reasons.append("oi_reference_cooling")
        elif oi_penalty > 0:
            reasons.append("oi_cooling")
    if snapshot.taker_buy_sell_ratio < config.min_taker_ratio:
        taker_penalty = 10
        if short_setup:
            if snapshot.taker_buy_sell_ratio <= 1.0:
                taker_penalty = 0
                reasons.append("taker_sell_pressure")
            else:
                taker_penalty = 4
        elif snapshot.setup_type == "mania_followthrough":
            if snapshot.taker_buy_sell_ratio >= config.mania_followthrough_min_taker_ratio:
                taker_penalty = 0
            else:
                taker_penalty = 4
        penalties += taker_penalty
        if taker_penalty >= 10:
            reasons.append("taker_weak")
        elif taker_penalty > 0:
            reasons.append("taker_cooling")
    if snapshot.price_change_pct_24h < config.min_price_change_pct:
        if short_setup and snapshot.price_change_pct_24h <= -abs(config.min_price_change_pct):
            reasons.append("momentum_bearish_short")
        else:
            penalties += 10
            reasons.append("momentum_weak")
    if snapshot.quote_volume_usdt_24h < config.min_quote_volume_usdt:
        penalties += 20
        reasons.append("volume_thin")
    has_trusted_confirmation = _has_trusted_sentiment_confirmation(snapshot)
    if snapshot.sentiment_mentions < config.absolute_min_sentiment_mentions:
        penalties += 10
        reasons.append("no_sentiment_mentions")
    min_mentions_required = 0 if has_trusted_confirmation else max(
        config.min_entry_sentiment_mentions,
        config.untrusted_entry_min_sentiment_mentions,
    )
    if snapshot.sentiment_mentions < min_mentions_required:
        penalties += 6
        reasons.append("unconfirmed_low_mentions")
    if (
        not has_trusted_confirmation
        and (
            _is_weak_single_source_untrusted(snapshot, config)
            or _is_discovery_only_untrusted(snapshot)
        )
    ):
        penalties += 4
        reasons.append("single_source_unconfirmed")
    if (
        snapshot.setup_type == "oi_divergence_long_candidate"
        and config.oi_divergence_require_trusted_confirmation
        and not has_trusted_confirmation
    ):
        penalties += 20
        reasons.append("oi_divergence_unconfirmed")
    if snapshot.setup_type == "oi_divergence_long_candidate":
        if snapshot.sentiment_mentions < config.oi_divergence_min_sentiment_mentions:
            penalties += 16
            reasons.append("oi_divergence_low_mentions")
        if snapshot.oi_price_divergence_score < config.oi_divergence_min_divergence_score:
            penalties += 18
            reasons.append("oi_divergence_signal_weak")
        if _sentiment_source_count(snapshot) < config.oi_divergence_min_source_count:
            penalties += 20
            reasons.append("oi_divergence_single_source")
    if snapshot.setup_type == "base_breakout_confirmed":
        if (
            config.base_breakout_require_trusted_confirmation
            and not has_trusted_confirmation
            and _is_weak_single_source_untrusted(snapshot, config)
        ):
            penalties += 18
            reasons.append("base_breakout_unconfirmed")
        if snapshot.sentiment_mentions < config.base_breakout_min_sentiment_mentions and not has_trusted_confirmation:
            penalties += 10
            reasons.append("base_breakout_low_mentions")
        if (
            not has_trusted_confirmation
            and _is_weak_single_source_untrusted(snapshot, config)
            and _sentiment_source_count(snapshot) < config.base_breakout_min_source_count
        ):
            penalties += 12
            reasons.append("base_breakout_single_source")
    if (
        snapshot.setup_type in LOW_INFO_SETUP_TYPES
        and snapshot.sentiment_mentions < config.low_info_min_sentiment_mentions
        and not has_trusted_confirmation
    ):
        penalties += 12
        reasons.append("low_info_setup")

    news_bonus = _news_confirmation_bonus(snapshot, config)
    if news_bonus > 0:
        weighted += news_bonus
        reasons.append(f"news_confirm=+{news_bonus:.1f}")

    # 极端正费率惩罚：>0.005(0.5%) 对 long 是负信号，庄家派发前兆
    EXTREME_FUNDING_THRESHOLD = 0.005
    EXTREME_FUNDING_THRESHOLD = 0.005
    if snapshot.funding_rate >= EXTREME_FUNDING_THRESHOLD and snapshot.setup_type not in ("squeeze_short_candidate", "base_breakdown_confirmed"):
        extreme_funding_penalty = min(18.0, (snapshot.funding_rate - EXTREME_FUNDING_THRESHOLD) * 3000.0)
        penalties += extreme_funding_penalty
        reasons.append("funding_extreme_positive")

    # Derivatives-flow signal adjustment: confidence/risk tags only.
    flow_side = "short" if short_setup else "long"
    derivatives_flow = classify_derivatives_flow_context(
        symbol=snapshot.symbol,
        side=flow_side,
        price_change_pct_24h=snapshot.price_change_pct_24h,
        price_change_pct_4h=snapshot.price_change_pct_4h,
        oi_change_pct_4h=snapshot.oi_change_pct_4h,
        funding_rate=snapshot.funding_rate,
        perp_volume_usdt=snapshot.quote_volume_usdt_24h,
        spot_volume_usdt=None,
        volume_trend=snapshot.kline_volume_trend,
        upper_wick_pct=snapshot.kline_upper_shadow_pct,
        candle_body_pct=snapshot.kline_body_pct,
        taker_buy_sell_ratio=snapshot.taker_buy_sell_ratio,
    )
    if derivatives_flow.confidence_adjustment:
        weighted += derivatives_flow.confidence_adjustment
        reasons.append(f"deriv_flow_adj={derivatives_flow.confidence_adjustment:+.1f}")
    for tag in derivatives_flow.risk_tags[:5]:
        reasons.append(f"deriv_flow={tag}")
    snapshot.derivatives_flow_regime = derivatives_flow.flow_regime
    snapshot.derivatives_flow_tags = derivatives_flow.risk_tags
    snapshot.derivatives_flow_confidence_adjustment = derivatives_flow.confidence_adjustment
    snapshot.derivatives_flow_long_risk_score = derivatives_flow.long_risk_score
    snapshot.derivatives_flow_short_risk_score = derivatives_flow.short_risk_score
    snapshot.derivatives_flow_spot_confirmation = derivatives_flow.spot_confirmation
    snapshot.derivatives_flow_explanation = derivatives_flow.explanation

    # Predictive model adjustment based on closed-loop trade history.
    if predictive_model and predictive_model.get("status") == "ok":
        symbol_stats = predictive_model.get("symbol_stats", {})
        symbol_upper = snapshot.symbol.upper()
        if symbol_upper in symbol_stats:
            hist_win_rate = float(symbol_stats[symbol_upper].get("win_rate", 0.5))
            hist_avg_pnl = float(symbol_stats[symbol_upper].get("avg_pnl", 0.0))
            predicted_adj = round((hist_win_rate - 0.5) * 30.0, 2)
            if hist_avg_pnl > 0.01 and hist_win_rate > 0.4:
                predicted_adj += min(10.0, hist_avg_pnl * 200.0)
            if hist_avg_pnl < -0.01 and hist_win_rate < 0.4:
                predicted_adj -= min(10.0, abs(hist_avg_pnl) * 200.0)
            if _is_explosive_probe_setup(snapshot) and predicted_adj < -5.0:
                predicted_adj = -5.0
                reasons.append("old_symbol_history_decayed")
            if predicted_adj != 0:
                weighted += predicted_adj
                reasons.append(f"pred_model_adj={predicted_adj:+.1f}")

        # 模型止损风险预警 + 胜率预测：优先实时推断，fallback 到查表
        win_predict_fn = predictive_model.get("_win_predict_fn")
        sl_predict_fn = predictive_model.get("_sl_predict_fn")
        expected_dim = int(predictive_model.get("feature_count") or 0)
        if win_predict_fn and sl_predict_fn:
            from trading_system.self_train_predictor import build_realtime_features
            _features = build_realtime_features(snapshot, hf_quality, pdf_context)
            if expected_dim > 0 and len(_features) != expected_dim:
                predictions = predictive_model.get("predictions", [])
                symbol_preds = [p for p in predictions if p.get("symbol", "").upper() == symbol_upper]
                sl_risk = symbol_preds[0].get("predicted_stop_loss_risk", 0) if symbol_preds else 0
                win_prob = symbol_preds[0].get("predicted_win_probability", 0.5) if symbol_preds else 0.5
            else:
                sl_risk = sl_predict_fn(_features)
                win_prob = win_predict_fn(_features)
        else:
            predictions = predictive_model.get("predictions", [])
            symbol_preds = [p for p in predictions if p.get("symbol", "").upper() == symbol_upper]
            sl_risk = symbol_preds[0].get("predicted_stop_loss_risk", 0) if symbol_preds else 0
            win_prob = symbol_preds[0].get("predicted_win_probability", 0.5) if symbol_preds else 0.5

        if sl_risk >= 0.6 and not _is_explosive_probe_setup(snapshot):
            weighted -= 12
            reasons.append(f"sl_risk_warn={sl_risk:.2f}")

        if win_prob < 0.35 and not _is_explosive_probe_setup(snapshot):
            weighted -= 15
            reasons.append(f"low_win_prob={win_prob:.2f}")

        factor_adj = _factor_backtest_adjustment(snapshot, predictive_model.get("factor_backtest") or {})
        if factor_adj:
            weighted += factor_adj
            reasons.append(f"factor_backtest_adj={factor_adj:+.1f}")

    # Regime 调节：在 flash_crash / risk_off 时提高入场门槛（等效降分）
    if regime:
        entry_mult = float(regime.get("entry_threshold_multiplier", 1.0))
        if entry_mult > 1.0:
            regime_reduction = (entry_mult - 1.0) * 45.0
            weighted -= regime_reduction
            reasons.append(f"regime_{regime.get('regime','unknown')}_tightening")

    score = max(0.0, min(100.0, round(weighted - penalties, 2)))
    if snapshot.setup_type in WATCH_ONLY_SETUP_TYPES:
        score = min(score, max(0.0, config.entry_threshold - 1.0))
        reasons.append("watch_only_no_entry")
    snapshot.score = score
    snapshot.reasons = reasons
    return snapshot


def is_entry_candidate(snapshot: MarketSnapshot, config: StrategyConfig) -> bool:
    return snapshot.score >= config.entry_threshold


def _scale(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    if value <= lower:
        return 0.0
    if value >= upper:
        return 1.0
    return (value - lower) / (upper - lower)


def _factor_backtest_adjustment(snapshot: MarketSnapshot, factor_backtest: dict) -> float:
    if factor_backtest.get("status") != "ok":
        return 0.0
    total = 0.0
    for item in (factor_backtest.get("recommendations") or [])[:24]:
        factor = str(item.get("factor") or "")
        bucket = str(item.get("bucket") or "")
        if _snapshot_factor_bucket(snapshot, factor) == bucket:
            total += float(item.get("score_adjustment") or 0.0)
    return round(max(-12.0, min(10.0, total)), 2)


def _snapshot_factor_bucket(snapshot: MarketSnapshot, factor: str) -> str:
    value = float(getattr(snapshot, factor, 0.0) or 0.0)
    if factor in {"price_change_pct_5m", "price_change_pct_24h", "oi_change_pct_4h"}:
        if value < 0:
            return "negative"
        if value < 3:
            return "flat"
        if value < 12:
            return "warm"
        return "hot"
    if factor == "taker_buy_sell_ratio":
        if value < 0.98:
            return "sell_dominant"
        if value < 1.03:
            return "neutral"
        if value < 1.12:
            return "buying"
        return "aggressive_buying"
    if factor == "funding_rate":
        if value < -0.001:
            return "negative_extreme"
        if value < 0:
            return "negative"
        if value < 0.001:
            return "normal_positive"
        return "positive_extreme"
    if factor == "sentiment_mentions":
        if value < 5:
            return "thin"
        if value < 20:
            return "moderate"
        if value < 80:
            return "active"
        return "crowded"
    return "missing"


def _listing_bonus(listing_days: int | None, config: StrategyConfig) -> float:
    if listing_days is None:
        return 0.0
    if listing_days < config.min_listing_days:
        return -5.0
    if listing_days > config.max_listing_days:
        return -5.0
    if listing_days <= 90:
        return 5.0
    return 2.0


def _has_trusted_sentiment_confirmation(snapshot: MarketSnapshot) -> bool:
    source = (snapshot.sentiment_source or "").lower()
    return any(item in source for item in TRUSTED_SENTIMENT_SOURCES)


def _news_confirmation_bonus(snapshot: MarketSnapshot, config: StrategyConfig) -> float:
    if not getattr(config, "news_confirmation_bonus_enabled", True):
        return 0.0
    tokens = {item.strip().lower() for item in (snapshot.sentiment_source or "").split(",") if item.strip()}
    if not tokens:
        return 0.0
    has_official_news = "opentwitter_official" in tokens
    if not has_official_news and snapshot.sentiment_score < float(getattr(config, "news_confirmation_min_score", 60.0)):
        return 0.0
    bonus = 0.0
    strong_news = {"opennews_news", "opennews_listing", "opennews_onchain"}
    weak_news = {"opennews_prediction", "opennews_meme"}
    market_news = {"opennews_market"}
    if has_official_news:
        bonus += 4.0
    if tokens & strong_news:
        bonus += 5.0
    if tokens & weak_news:
        bonus += 2.0
    if tokens & market_news:
        # Market alerts such as funding/OI/liquidation are useful context, but not
        # standalone project news. Keep the bonus intentionally small.
        bonus += 1.0
    if "opentwitter_search" in tokens and snapshot.sentiment_mentions >= int(getattr(config, "news_confirmation_twitter_min_mentions", 2)):
        bonus += 2.0
    if bonus <= 0:
        return 0.0
    return round(min(float(getattr(config, "news_confirmation_bonus_max", 8.0)), bonus), 2)


def _sentiment_source_count(snapshot: MarketSnapshot) -> int:
    return len({item.strip().lower() for item in (snapshot.sentiment_source or "").split(",") if item.strip()})


def _is_short_fuel_long_setup(snapshot: MarketSnapshot) -> bool:
    """Negative funding is bullish fuel for strong long meme/OI divergence structures."""
    return (
        snapshot.setup_type in {"oi_divergence_long_candidate", "mania_followthrough"}
        and snapshot.funding_rate < 0
        and snapshot.price_change_pct_24h > 0
        and snapshot.oi_change_pct_4h >= 20
        and snapshot.taker_buy_sell_ratio >= 1.03
    )


def _is_short_setup(snapshot: MarketSnapshot) -> bool:
    return snapshot.setup_type in SHORT_SETUP_TYPES


def _is_explosive_probe_setup(snapshot: MarketSnapshot) -> bool:
    """Current-market override: old symbol history must not hard-block explosive probe setups."""
    return (
        snapshot.setup_type in {"oi_divergence_long_candidate", "mania_followthrough"}
        and snapshot.setup_quality >= 90
        and snapshot.price_change_pct_24h >= 20
        and snapshot.oi_change_pct_4h >= 50
        and snapshot.taker_buy_sell_ratio >= 1.05
        and snapshot.quote_volume_usdt_24h >= 50_000_000
    )


def _is_weak_single_source_untrusted(snapshot: MarketSnapshot, config: StrategyConfig) -> bool:
    tokens = {item.strip().lower() for item in (snapshot.sentiment_source or "").split(",") if item.strip()}
    if not tokens:
        return True
    meaningful_tokens = tokens.difference(WEAK_DISCOVERY_SOURCES)
    if meaningful_tokens:
        return False
    return len(tokens) == 1


def _is_discovery_only_untrusted(snapshot: MarketSnapshot) -> bool:
    tokens = {item.strip().lower() for item in (snapshot.sentiment_source or "").split(",") if item.strip()}
    return bool(tokens) and tokens.issubset(WEAK_DISCOVERY_SOURCES)
