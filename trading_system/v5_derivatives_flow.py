from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DerivativesFlowContext:
    symbol: str = ""
    flow_regime: str = "neutral"
    risk_tags: list[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0
    long_risk_score: float = 0.0
    short_risk_score: float = 0.0
    spot_confirmation: str = "unknown"
    perp_dominance_score: float = 0.0
    oi_price_divergence_score: float = 0.0
    funding_extreme_score: float = 0.0
    orderbook_pressure_state: str = "unknown"
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_perp_spot_volume_ratio(perp_volume_usdt: float | None, spot_volume_usdt: float | None) -> float | None:
    perp = _finite_float(perp_volume_usdt, None)
    spot = _finite_float(spot_volume_usdt, None)
    if perp is None or spot is None or spot <= 0:
        return None
    return perp / spot


def classify_funding_regime(funding_rate: float | None) -> tuple[str, float]:
    funding = _finite_float(funding_rate, 0.0) or 0.0
    if funding <= -0.002:
        return "negative_extreme", min(5.0, abs(funding) * 1800.0)
    if funding < 0:
        return "negative", min(3.0, abs(funding) * 1500.0)
    if funding >= 0.005:
        return "positive_extreme", min(5.0, funding * 900.0)
    if funding >= 0.001:
        return "positive_hot", min(3.0, funding * 1000.0)
    return "neutral", 0.0


def classify_oi_price_divergence(
    *,
    price_change_pct_24h: float | None = None,
    price_change_pct_4h: float | None = None,
    oi_change_pct_4h: float | None = None,
    volume_trend: float | None = None,
) -> tuple[str, float]:
    price_24h = _finite_float(price_change_pct_24h, 0.0) or 0.0
    price_4h = _finite_float(price_change_pct_4h, 0.0) or 0.0
    oi_4h = _finite_float(oi_change_pct_4h, 0.0) or 0.0
    volume = _finite_float(volume_trend, 0.0) or 0.0
    price_hot = max(price_24h, price_4h * 3.0)
    if price_hot >= 18.0 and oi_4h <= 0.0:
        return "oi_price_divergence_top", min(10.0, price_hot / 4.0 + abs(oi_4h) / 2.0)
    if price_hot >= 25.0 and oi_4h < 5.0 and volume <= 0.0:
        return "oi_volume_fading_top", min(9.0, price_hot / 5.0 + max(0.0, 5.0 - oi_4h))
    if price_hot > 0.0 and oi_4h >= 15.0:
        return "oi_expansion_with_price", min(6.0, oi_4h / 5.0)
    if price_hot < -8.0 and oi_4h < -5.0:
        return "oi_unwind_down", min(6.0, abs(oi_4h) / 4.0)
    return "neutral", 0.0


def classify_orderbook_pressure(
    *,
    bid_ask_imbalance: float | None = None,
    upper_pressure: bool | None = None,
    order_rate_anomaly: float | None = None,
    upper_wick_pct: float | None = None,
    candle_body_pct: float | None = None,
    taker_buy_sell_ratio: float | None = None,
) -> tuple[str, float]:
    imbalance = _finite_float(bid_ask_imbalance, 0.0) or 0.0
    order_rate = _finite_float(order_rate_anomaly, 0.0) or 0.0
    upper_wick = _finite_float(upper_wick_pct, 0.0) or 0.0
    body = _finite_float(candle_body_pct, 0.0) or 0.0
    taker = _finite_float(taker_buy_sell_ratio, 1.0) or 1.0
    if upper_pressure or (upper_wick >= 1.0 and upper_wick > max(body, 0.05)):
        return "upper_pressure", min(5.0, upper_wick + max(0.0, 1.05 - taker) * 5.0)
    if order_rate >= 2.0:
        return "bot_order_rate_anomaly", min(4.0, order_rate)
    if imbalance >= 0.2 and taker >= 1.08:
        return "bid_absorption", min(4.0, imbalance * 8.0)
    if imbalance <= -0.2 or taker <= 0.95:
        return "sell_pressure", min(4.0, abs(imbalance) * 8.0 + max(0.0, 1.0 - taker) * 4.0)
    return "neutral", 0.0


def merge_derivatives_flow_tags(*tag_groups: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in tag_groups:
        for tag in group:
            if tag and tag not in seen:
                seen.add(tag)
                merged.append(str(tag))
    return merged


def classify_derivatives_flow_context(
    *,
    symbol: str = "",
    side: str = "long",
    price_change_pct_24h: float | None = None,
    price_change_pct_4h: float | None = None,
    oi_change_pct_4h: float | None = None,
    funding_rate: float | None = None,
    perp_volume_usdt: float | None = None,
    spot_volume_usdt: float | None = None,
    spot_volume_zscore: float | None = None,
    volume_trend: float | None = None,
    bid_ask_imbalance: float | None = None,
    upper_pressure: bool | None = None,
    order_rate_anomaly: float | None = None,
    upper_wick_pct: float | None = None,
    candle_body_pct: float | None = None,
    taker_buy_sell_ratio: float | None = None,
) -> DerivativesFlowContext:
    side = str(side or "long").lower()
    tags: list[str] = []
    notes: list[str] = []
    long_risk = 0.0
    short_risk = 0.0
    confidence_adjustment = 0.0

    price_24h = _finite_float(price_change_pct_24h, 0.0) or 0.0
    oi_4h = _finite_float(oi_change_pct_4h, 0.0) or 0.0
    ratio = compute_perp_spot_volume_ratio(perp_volume_usdt, spot_volume_usdt)
    spot_z = _finite_float(spot_volume_zscore, None)
    funding_regime, funding_score = classify_funding_regime(funding_rate)
    divergence_state, divergence_score = classify_oi_price_divergence(
        price_change_pct_24h=price_change_pct_24h,
        price_change_pct_4h=price_change_pct_4h,
        oi_change_pct_4h=oi_change_pct_4h,
        volume_trend=volume_trend,
    )
    pressure_state, pressure_score = classify_orderbook_pressure(
        bid_ask_imbalance=bid_ask_imbalance,
        upper_pressure=upper_pressure,
        order_rate_anomaly=order_rate_anomaly,
        upper_wick_pct=upper_wick_pct,
        candle_body_pct=candle_body_pct,
        taker_buy_sell_ratio=taker_buy_sell_ratio,
    )

    perp_score = 0.0
    spot_confirmation = "unknown"
    if ratio is not None:
        perp_score = min(10.0, max(0.0, ratio))
        if ratio >= 3.0:
            tags.append("perp_dominance_risk")
            long_risk += min(4.0, ratio / 2.0)
            notes.append(f"perp_spot_ratio={ratio:.2f}")
        elif ratio >= 1.5:
            tags.append("perp_dominance_watch")
            long_risk += 1.0
        spot_confirmation = "weak" if ratio >= 3.0 else "neutral"
    if spot_z is not None:
        if spot_z >= 1.0:
            spot_confirmation = "strong"
        elif spot_z <= -0.5:
            spot_confirmation = "weak"
            tags.append("spot_confirmation_missing")
            long_risk += 2.0

    if funding_regime in {"negative", "negative_extreme"} and price_24h > 0 and oi_4h >= 10.0:
        tags.append("negative_funding_squeeze")
        short_risk += 4.0 + funding_score
        notes.append(f"funding={_finite_float(funding_rate, 0.0):.6f}")
        if side == "short":
            confidence_adjustment -= min(5.0, 2.0 + funding_score)
        else:
            confidence_adjustment -= 1.0
    elif funding_regime == "positive_extreme":
        tags.append("funding_extreme_positive")
        long_risk += 2.0 + funding_score / 2.0
        if side != "short":
            confidence_adjustment -= min(4.0, 1.0 + funding_score / 2.0)

    if divergence_state in {"oi_price_divergence_top", "oi_volume_fading_top"}:
        tags.append("oi_price_divergence_top")
        long_risk += 4.0 + divergence_score / 3.0
        if side == "long":
            confidence_adjustment -= min(5.0, 2.0 + divergence_score / 3.0)
        else:
            confidence_adjustment += min(2.0, divergence_score / 5.0)
        notes.append(divergence_state)
    elif divergence_state == "oi_expansion_with_price":
        tags.append("oi_expansion_with_price")

    if pressure_state == "upper_pressure":
        tags.append("upper_pressure_active")
        long_risk += 2.0 + pressure_score
        if side == "long":
            confidence_adjustment -= min(4.0, 1.0 + pressure_score / 2.0)
    elif pressure_state == "bot_order_rate_anomaly":
        tags.append("bot_order_rate_anomaly")
        long_risk += 1.0
        short_risk += 1.0
    elif pressure_state == "bid_absorption":
        tags.append("bid_absorption")
        if side == "long":
            confidence_adjustment += min(2.0, pressure_score / 2.0)
    elif pressure_state == "sell_pressure":
        tags.append("sell_pressure")
        long_risk += 2.0

    if "oi_price_divergence_top" in tags and ("upper_pressure_active" in tags or "perp_dominance_risk" in tags):
        flow_regime = "distribution"
        tags.append("late_squeeze_distribution")
    elif "negative_funding_squeeze" in tags:
        flow_regime = "squeeze_active"
    elif "perp_dominance_risk" in tags and spot_confirmation == "weak":
        flow_regime = "trap"
    elif "bid_absorption" in tags and spot_confirmation in {"neutral", "strong"}:
        flow_regime = "accumulation"
    else:
        flow_regime = "neutral"

    if side == "long" and long_risk >= 6.0:
        tags.append("derivatives_long_risk")
    if side == "short" and short_risk >= 5.0:
        tags.append("derivatives_short_risk")

    confidence_adjustment = round(max(-5.0, min(5.0, confidence_adjustment)), 2)
    tags = merge_derivatives_flow_tags(tags)
    explanation = ", ".join(notes) if notes else flow_regime
    return DerivativesFlowContext(
        symbol=symbol,
        flow_regime=flow_regime,
        risk_tags=tags,
        confidence_adjustment=confidence_adjustment,
        long_risk_score=round(long_risk, 4),
        short_risk_score=round(short_risk, 4),
        spot_confirmation=spot_confirmation,
        perp_dominance_score=round(perp_score, 4),
        oi_price_divergence_score=round(divergence_score, 4),
        funding_extreme_score=round(funding_score, 4),
        orderbook_pressure_state=pressure_state,
        explanation=explanation,
    )


def _finite_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out
