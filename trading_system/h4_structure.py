from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class H4StructureSignal:
    symbol: str
    computed_at: str
    h4_close_above_ma20: bool
    h4_ma20_slope_pct: float
    h4_price_vs_ma20_pct: float
    h4_close_position_20bars: float
    h4_low_break_recent_20bars: bool
    h4_high_break_recent_20bars: bool
    h4_atr_compression: float
    h4_lower_highs_count: int
    h4_higher_lows_count: int
    h4_structure_state: str
    h4_price_location: str
    h4_pattern_type: str
    # 结构止损绝对价 (2026-05-30): 入场即锚定, 不等破位确认
    h4_range_low: float = 0.0
    h4_recent_swing_low: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MIN_BARS = 21
_SWING_WINDOW = 3


def compute_h4_structure(symbol: str, klines: list, mark_price: float) -> H4StructureSignal:
    now_iso = datetime.now(timezone.utc).isoformat()

    if not klines or len(klines) < _MIN_BARS:
        return _unknown_signal(symbol, now_iso)

    # Use previous 20 completed bars for range/MA, exclude current bar
    prev_klines = klines[-_MIN_BARS:-1]
    closes = [float(k[4]) for k in prev_klines]
    highs = [float(k[2]) for k in prev_klines]
    lows = [float(k[3]) for k in prev_klines]

    close = mark_price if mark_price > 0 else float(klines[-1][4])

    ma20 = sum(closes) / len(closes)
    if ma20 == 0:
        return _unknown_signal(symbol, now_iso)

    h4_close_above_ma20 = close > ma20
    h4_price_vs_ma20_pct = (close - ma20) / ma20 * 100

    # MA20 slope: compare current ma20 vs ma20 of 5 bars earlier window
    # Need klines[-26:-6] for the earlier 20-bar window (shifted 5 bars back)
    if len(klines) >= 27:
        closes_5ago = [float(k[4]) for k in klines[-26:-6]]
        ma20_5ago = sum(closes_5ago) / len(closes_5ago) if closes_5ago else ma20
    else:
        ma20_5ago = ma20
    h4_ma20_slope_pct = (ma20 - ma20_5ago) / max(abs(ma20_5ago), 1e-9) * 100

    # 20-bar range
    range_high = max(highs)
    range_low = min(lows)
    range_span = range_high - range_low
    h4_close_position_20bars = (close - range_low) / range_span if range_span > 0 else 0.5

    # Break detection
    h4_low_break_recent_20bars = close < range_low
    h4_high_break_recent_20bars = close > range_high

    # ATR compression (using previous completed bars)
    atr_values = []
    for i in range(1, len(prev_klines)):
        h = float(prev_klines[i][2])
        l = float(prev_klines[i][3])
        prev_c = float(prev_klines[i - 1][4])
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        atr_values.append(tr)

    if len(atr_values) >= 5:
        atr_recent = sum(atr_values[-5:]) / 5
        atr_base = sum(atr_values) / len(atr_values)
        h4_atr_compression = atr_recent / atr_base if atr_base > 0 else 1.0
    else:
        h4_atr_compression = 1.0

    # Swing high/low detection
    swing_highs = _find_swing_highs(highs, _SWING_WINDOW)
    swing_lows = _find_swing_lows(lows, _SWING_WINDOW)

    h4_lower_highs_count = _count_lower_highs(swing_highs)
    h4_higher_lows_count = _count_higher_lows(swing_lows)

    # Structure state
    h4_structure_state = _classify_structure_state(
        h4_close_above_ma20, h4_ma20_slope_pct, h4_close_position_20bars, h4_low_break_recent_20bars, close, ma20
    )

    # Price location
    h4_price_location = _classify_price_location(
        h4_close_position_20bars, h4_low_break_recent_20bars, h4_high_break_recent_20bars
    )

    # Pattern type
    h4_pattern_type = _classify_pattern(
        h4_higher_lows_count, h4_lower_highs_count, h4_atr_compression,
        h4_low_break_recent_20bars, h4_high_break_recent_20bars, swing_lows
    )

    return H4StructureSignal(
        symbol=symbol,
        computed_at=now_iso,
        h4_close_above_ma20=h4_close_above_ma20,
        h4_ma20_slope_pct=round(h4_ma20_slope_pct, 4),
        h4_price_vs_ma20_pct=round(h4_price_vs_ma20_pct, 4),
        h4_close_position_20bars=round(h4_close_position_20bars, 4),
        h4_low_break_recent_20bars=h4_low_break_recent_20bars,
        h4_high_break_recent_20bars=h4_high_break_recent_20bars,
        h4_atr_compression=round(h4_atr_compression, 4),
        h4_lower_highs_count=h4_lower_highs_count,
        h4_higher_lows_count=h4_higher_lows_count,
        h4_structure_state=h4_structure_state,
        h4_price_location=h4_price_location,
        h4_pattern_type=h4_pattern_type,
        h4_range_low=round(range_low, 8),
        h4_recent_swing_low=round(swing_lows[-1] if swing_lows else range_low, 8),
    )


def _unknown_signal(symbol: str, computed_at: str) -> H4StructureSignal:
    return H4StructureSignal(
        symbol=symbol,
        computed_at=computed_at,
        h4_close_above_ma20=True,
        h4_ma20_slope_pct=0.0,
        h4_price_vs_ma20_pct=0.0,
        h4_close_position_20bars=0.5,
        h4_low_break_recent_20bars=False,
        h4_high_break_recent_20bars=False,
        h4_atr_compression=1.0,
        h4_lower_highs_count=0,
        h4_higher_lows_count=0,
        h4_structure_state="unknown",
        h4_price_location="middle",
        h4_pattern_type="unknown",
    )


def _find_swing_highs(highs: list[float], window: int) -> list[float]:
    results = []
    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) and all(
            highs[i] >= highs[i + j] for j in range(1, window + 1)
        ):
            results.append(highs[i])
    return results


def _find_swing_lows(lows: list[float], window: int) -> list[float]:
    results = []
    for i in range(window, len(lows) - window):
        if all(lows[i] <= lows[i - j] for j in range(1, window + 1)) and all(
            lows[i] <= lows[i + j] for j in range(1, window + 1)
        ):
            results.append(lows[i])
    return results


def _count_lower_highs(swing_highs: list[float]) -> int:
    if len(swing_highs) < 2:
        return 0
    count = 0
    for i in range(1, len(swing_highs)):
        if swing_highs[i] < swing_highs[i - 1]:
            count += 1
    return count


def _count_higher_lows(swing_lows: list[float]) -> int:
    if len(swing_lows) < 2:
        return 0
    count = 0
    for i in range(1, len(swing_lows)):
        if swing_lows[i] > swing_lows[i - 1]:
            count += 1
    return count


def _classify_structure_state(
    close_above_ma20: bool, ma20_slope_pct: float, close_position: float,
    low_break: bool, close: float, ma20: float
) -> str:
    if low_break or (close < ma20 and ma20_slope_pct < -1):
        return "broken"
    if close < ma20 or close_position < 0.25 or ma20_slope_pct < 0:
        return "weakening"
    if close >= ma20 and not low_break and close_position >= 0.25:
        return "valid"
    return "unknown"


def _classify_price_location(
    close_position: float, low_break: bool, high_break: bool
) -> str:
    if low_break:
        return "below_support"
    if high_break:
        return "above_resistance"
    if close_position > 0.65:
        return "upper_half"
    if close_position < 0.4:
        return "lower_half"
    return "middle"


def _classify_pattern(
    higher_lows: int, lower_highs: int, atr_compression: float,
    low_break: bool, high_break: bool, swing_lows: list[float]
) -> str:
    if low_break:
        return "breakdown"
    if high_break:
        return "breakout"
    if higher_lows >= 2 and lower_highs >= 2 and atr_compression < 0.85:
        return "triangle"
    if lower_highs >= 2 and atr_compression < 0.85:
        if len(swing_lows) >= 2 and swing_lows[-1] < swing_lows[-2]:
            return "wedge"
    if atr_compression < 0.85 and lower_highs < 2 and higher_lows < 2:
        return "range"
    return "unknown"
