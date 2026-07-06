from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


ANGLES: dict[str, float] = {"1x2": 0.5, "1x1": 1.0, "2x1": 2.0}


@dataclass(frozen=True)
class GannFeature:
    symbol: str
    zone: str
    bias: str
    confidence: float
    anchor_high: float
    anchor_low: float
    anchor_high_time: str
    anchor_low_time: str
    cycle_days: int
    scale: str
    line_values: dict[str, float]
    anchor_stability: float
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_daily_klines(rows: Iterable[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            try:
                normalized.append({
                    "time": str(row.get("time") or row.get("open_time") or row.get("timestamp") or row.get("date") or ""),
                    "open": float(row.get("open")),
                    "high": float(row.get("high")),
                    "low": float(row.get("low")),
                    "close": float(row.get("close")),
                })
            except (TypeError, ValueError):
                continue
        elif isinstance(row, (list, tuple)) and len(row) >= 5:
            try:
                normalized.append({
                    "time": str(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                })
            except (TypeError, ValueError):
                continue
    return normalized


def _anchor_for_index(
    rows: list[dict[str, Any]],
    idx: int,
    *,
    lookback_days: int,
    min_cycle_days: int,
    min_drawdown_pct: float,
) -> tuple[int, int] | None:
    if idx <= 0:
        return None
    start = max(0, idx - lookback_days)
    window = rows[start:idx]
    if not window:
        return None
    high_offset, high_row = max(enumerate(window), key=lambda item: float(item[1]["high"]))
    high_idx = start + high_offset
    low_window = rows[high_idx:idx + 1]
    if not low_window:
        return None
    low_offset, low_row = min(enumerate(low_window), key=lambda item: float(item[1]["low"]))
    low_idx = high_idx + low_offset
    cycle_days = low_idx - high_idx
    if cycle_days < min_cycle_days:
        return None
    high = float(high_row["high"])
    low = float(low_row["low"])
    if high <= 0 or low <= 0 or high <= low:
        return None
    drawdown_pct = (high - low) / high * 100.0
    if drawdown_pct < min_drawdown_pct:
        return None
    return high_idx, low_idx


def _line_values(rows: list[dict[str, Any]], idx: int, high_idx: int, low_idx: int) -> dict[str, float] | None:
    high = float(rows[high_idx]["high"])
    low = float(rows[low_idx]["low"])
    cycle_days = low_idx - high_idx
    days_since_low = idx - low_idx
    if high <= 0 or low <= 0 or cycle_days <= 0 or days_since_low < 0:
        return None
    base_slope = (math.log(high) - math.log(low)) / cycle_days
    return {
        angle: low * math.exp(multiplier * base_slope * days_since_low)
        for angle, multiplier in ANGLES.items()
    }


def classify_zone(close: float, lines: dict[str, float]) -> str:
    one_x_one = float(lines.get("1x1") or 0.0)
    one_x_two = float(lines.get("1x2") or 0.0)
    if close <= 0 or one_x_one <= 0 or one_x_two <= 0:
        return "unknown"
    if close >= one_x_one:
        return "above_1x1"
    if close >= one_x_two:
        return "1x2_to_1x1"
    return "below_1x2"


def bias_for_zone(zone: str) -> str:
    if zone == "above_1x1":
        return "support_long"
    if zone == "below_1x2":
        return "support_short"
    if zone == "1x2_to_1x1":
        return "neutral"
    return "unknown"


def anchor_stability_pct(
    rows: list[dict[str, Any]],
    idx: int,
    *,
    lookback_days: int,
    min_cycle_days: int,
    min_drawdown_pct: float,
    shift_days: int = 3,
) -> float:
    base_anchor = _anchor_for_index(
        rows,
        idx,
        lookback_days=lookback_days,
        min_cycle_days=min_cycle_days,
        min_drawdown_pct=min_drawdown_pct,
    )
    if not base_anchor:
        return 999.0
    base_lines = _line_values(rows, idx, *base_anchor)
    if not base_lines or float(base_lines.get("1x1") or 0.0) <= 0:
        return 999.0
    base_line = float(base_lines["1x1"])
    drifts: list[float] = []
    for shifted_idx in range(max(1, idx - shift_days), min(len(rows) - 1, idx + shift_days) + 1):
        anchor = _anchor_for_index(
            rows,
            shifted_idx,
            lookback_days=lookback_days,
            min_cycle_days=min_cycle_days,
            min_drawdown_pct=min_drawdown_pct,
        )
        if not anchor:
            continue
        lines = _line_values(rows, idx, *anchor)
        if not lines or float(lines.get("1x1") or 0.0) <= 0:
            continue
        drifts.append(abs(float(lines["1x1"]) - base_line) / base_line * 100.0)
    if not drifts:
        return 999.0
    return max(drifts)


def confidence_from_stability(stability_pct: float, *, strict_stability_pct: float = 2.0, loose_stability_pct: float = 5.0) -> float:
    if stability_pct <= strict_stability_pct:
        return 0.80
    if stability_pct <= loose_stability_pct:
        return 0.55
    return 0.0


def compute_gann_feature(
    symbol: str,
    daily_klines: Iterable[Any],
    *,
    asof_index: int | None = None,
    lookback_days: int = 365,
    min_cycle_days: int = 14,
    min_drawdown_pct: float = 10.0,
    stability_shift_days: int = 3,
) -> GannFeature | None:
    rows = normalize_daily_klines(daily_klines)
    if len(rows) < max(lookback_days // 2, min_cycle_days + 2):
        return None
    idx = len(rows) - 1 if asof_index is None else int(asof_index)
    if idx <= 0 or idx >= len(rows):
        return None
    anchor = _anchor_for_index(
        rows,
        idx,
        lookback_days=lookback_days,
        min_cycle_days=min_cycle_days,
        min_drawdown_pct=min_drawdown_pct,
    )
    if not anchor:
        return None
    high_idx, low_idx = anchor
    lines = _line_values(rows, idx, high_idx, low_idx)
    if not lines:
        return None
    close = float(rows[idx]["close"])
    zone = classify_zone(close, lines)
    stability = anchor_stability_pct(
        rows,
        idx,
        lookback_days=lookback_days,
        min_cycle_days=min_cycle_days,
        min_drawdown_pct=min_drawdown_pct,
        shift_days=stability_shift_days,
    )
    confidence = confidence_from_stability(stability)
    return GannFeature(
        symbol=str(symbol or "").upper(),
        zone=zone,
        bias=bias_for_zone(zone),
        confidence=confidence,
        anchor_high=round(float(rows[high_idx]["high"]), 8),
        anchor_low=round(float(rows[low_idx]["low"]), 8),
        anchor_high_time=str(rows[high_idx]["time"]),
        anchor_low_time=str(rows[low_idx]["time"]),
        cycle_days=low_idx - high_idx,
        scale="log",
        line_values={key: round(float(value), 8) for key, value in lines.items()},
        anchor_stability=round(stability, 6),
        stale=False,
    )


def gann_position_multiplier(feature: GannFeature | dict[str, Any] | None, side: str) -> float:
    if feature is None:
        return 1.0
    data = feature.to_dict() if isinstance(feature, GannFeature) else dict(feature)
    if data.get("stale"):
        return 1.0
    confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
    if confidence < 0.40:
        return 1.0
    bias = str(data.get("bias") or "")
    normalized_side = str(side or "").lower()
    supportive = (normalized_side == "long" and bias == "support_long") or (normalized_side == "short" and bias == "support_short")
    conflict = (normalized_side == "long" and bias == "support_short") or (normalized_side == "short" and bias == "support_long")
    if supportive:
        return round(1.0 + 0.25 * confidence, 6)
    if conflict:
        return round(1.0 - 0.50 * confidence, 6)
    return 1.0
