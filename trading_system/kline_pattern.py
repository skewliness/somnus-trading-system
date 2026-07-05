"""K-line pattern detection: mathematical rules for candlestick patterns.

Input format: Binance kline = [timestamp, open, high, low, close, volume, ...]
All pattern rules derived from PDF methodology documents.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class PatternResult:
    name: str
    direction: str  # "bullish" / "bearish" / "neutral"
    strength: float  # 0-1
    context_valid: bool = True


def _parse_bar(kline) -> tuple[float, float, float, float, float]:
    """Extract (open, high, low, close, volume) from a Binance kline."""
    return float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[5])


def _bar_metrics(o: float, h: float, l: float, c: float) -> dict:
    range_ = max(h - l, 1e-9)
    body = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    return {
        "range": range_,
        "body": body,
        "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow,
        "body_ratio": body / range_,
        "upper_ratio": upper_shadow / range_,
        "lower_ratio": lower_shadow / range_,
        "is_bullish": c >= o,
        "mid_body": (o + c) / 2,
    }


# ---------------------------------------------------------------------------
# Single-bar patterns (6)
# ---------------------------------------------------------------------------

def _detect_hammer(o, h, l, c, m) -> PatternResult | None:
    if m["lower_ratio"] >= 0.6 and m["body_ratio"] <= 0.3 and m["upper_ratio"] <= 0.1:
        strength = min(m["lower_ratio"], 1.0)
        return PatternResult("hammer", "bullish", strength)
    return None


def _detect_inverted_hammer(o, h, l, c, m) -> PatternResult | None:
    if m["upper_ratio"] >= 0.6 and m["body_ratio"] <= 0.3 and m["lower_ratio"] <= 0.1:
        strength = min(m["upper_ratio"], 1.0)
        return PatternResult("inverted_hammer", "bullish", strength)
    return None


def _detect_doji(o, h, l, c, m) -> PatternResult | None:
    if m["body_ratio"] <= 0.05:
        strength = 1.0 - m["body_ratio"] / 0.05
        return PatternResult("doji", "neutral", strength)
    return None


def _detect_long_upper_shadow(o, h, l, c, m) -> PatternResult | None:
    if m["upper_ratio"] >= 0.6 and m["body_ratio"] <= 0.2:
        return PatternResult("long_upper_shadow", "bearish", m["upper_ratio"])
    return None


def _detect_long_lower_shadow(o, h, l, c, m) -> PatternResult | None:
    if m["lower_ratio"] >= 0.6 and m["body_ratio"] <= 0.2:
        return PatternResult("long_lower_shadow", "bullish", m["lower_ratio"])
    return None


def _detect_marubozu(o, h, l, c, m) -> PatternResult | None:
    if m["body_ratio"] >= 0.9:
        direction = "bullish" if m["is_bullish"] else "bearish"
        return PatternResult("marubozu", direction, m["body_ratio"])
    return None


_SINGLE_DETECTORS = [
    _detect_hammer, _detect_inverted_hammer, _detect_doji,
    _detect_long_upper_shadow, _detect_long_lower_shadow, _detect_marubozu,
]


# ---------------------------------------------------------------------------
# Double-bar patterns (4)
# ---------------------------------------------------------------------------

def _detect_bullish_engulfing(prev, curr, pm, cm) -> PatternResult | None:
    if not pm["is_bullish"] and cm["is_bullish"]:
        prev_o, prev_c = float(prev[1]), float(prev[4])
        curr_o, curr_c = float(curr[1]), float(curr[4])
        if curr_o <= min(prev_o, prev_c) and curr_c >= max(prev_o, prev_c):
            strength = cm["body"] / max(pm["body"], 1e-9)
            return PatternResult("bullish_engulfing", "bullish", min(strength / 2, 1.0))
    return None


def _detect_bearish_engulfing(prev, curr, pm, cm) -> PatternResult | None:
    if pm["is_bullish"] and not cm["is_bullish"]:
        prev_o, prev_c = float(prev[1]), float(prev[4])
        curr_o, curr_c = float(curr[1]), float(curr[4])
        if curr_o >= max(prev_o, prev_c) and curr_c <= min(prev_o, prev_c):
            strength = cm["body"] / max(pm["body"], 1e-9)
            return PatternResult("bearish_engulfing", "bearish", min(strength / 2, 1.0))
    return None


def _detect_dark_cloud_cover(prev, curr, pm, cm) -> PatternResult | None:
    if pm["is_bullish"] and not cm["is_bullish"]:
        prev_h = float(prev[2])
        prev_c = float(prev[4])
        curr_o, curr_c = float(curr[1]), float(curr[4])
        # Crypto rarely gaps; accept open >= prev_close (near prev high)
        if curr_o >= prev_c and curr_c < pm["mid_body"]:
            penetration = (pm["mid_body"] - curr_c) / max(pm["body"], 1e-9)
            strength = min(penetration, 1.0)
            if curr_o > prev_h:
                strength = min(strength + 0.2, 1.0)
            return PatternResult("dark_cloud_cover", "bearish", strength)
    return None


def _detect_piercing_line(prev, curr, pm, cm) -> PatternResult | None:
    if not pm["is_bullish"] and cm["is_bullish"]:
        prev_l = float(prev[3])
        prev_c = float(prev[4])
        curr_o, curr_c = float(curr[1]), float(curr[4])
        # Crypto rarely gaps; accept open <= prev_close (near prev low)
        if curr_o <= prev_c and curr_c > pm["mid_body"]:
            penetration = (curr_c - pm["mid_body"]) / max(pm["body"], 1e-9)
            strength = min(penetration, 1.0)
            if curr_o < prev_l:
                strength = min(strength + 0.2, 1.0)
            return PatternResult("piercing_line", "bullish", strength)
    return None


_DOUBLE_DETECTORS = [
    _detect_bullish_engulfing, _detect_bearish_engulfing,
    _detect_dark_cloud_cover, _detect_piercing_line,
]


# ---------------------------------------------------------------------------
# Triple-bar patterns (4)
# ---------------------------------------------------------------------------

def _detect_morning_star(k1, k2, k3, m1, m2, m3) -> PatternResult | None:
    if not m1["is_bullish"] and m3["is_bullish"] and m2["body_ratio"] <= 0.3:
        first_mid = (float(k1[1]) + float(k1[4])) / 2
        if float(k3[4]) > first_mid:
            return PatternResult("morning_star", "bullish", 0.8)
    return None


def _detect_evening_star(k1, k2, k3, m1, m2, m3) -> PatternResult | None:
    if m1["is_bullish"] and not m3["is_bullish"] and m2["body_ratio"] <= 0.3:
        first_mid = (float(k1[1]) + float(k1[4])) / 2
        if float(k3[4]) < first_mid:
            return PatternResult("evening_star", "bearish", 0.8)
    return None


def _detect_three_white_soldiers(k1, k2, k3, m1, m2, m3) -> PatternResult | None:
    c1, c2, c3 = float(k1[4]), float(k2[4]), float(k3[4])
    if m1["is_bullish"] and m2["is_bullish"] and m3["is_bullish"]:
        if c2 > c1 and c3 > c2 and m2["body"] >= m1["body"] * 0.7:
            return PatternResult("three_white_soldiers", "bullish", 0.9)
    return None


def _detect_three_black_crows(k1, k2, k3, m1, m2, m3) -> PatternResult | None:
    c1, c2, c3 = float(k1[4]), float(k2[4]), float(k3[4])
    if not m1["is_bullish"] and not m2["is_bullish"] and not m3["is_bullish"]:
        if c2 < c1 and c3 < c2 and m2["body"] >= m1["body"] * 0.7:
            return PatternResult("three_black_crows", "bearish", 0.9)
    return None


_TRIPLE_DETECTORS = [
    _detect_morning_star, _detect_evening_star,
    _detect_three_white_soldiers, _detect_three_black_crows,
]


# ---------------------------------------------------------------------------
# Trend/structure patterns (4) — require 20+ bars for MA calculation
# ---------------------------------------------------------------------------

def _compute_ma(klines: list, period: int) -> float | None:
    if len(klines) < period:
        return None
    closes = [float(k[4]) for k in klines[-period:]]
    return mean(closes)


def _detect_trend_patterns(klines: list) -> list[PatternResult]:
    results = []
    if len(klines) < 20:
        return results
    ma7 = _compute_ma(klines, 7)
    ma20 = _compute_ma(klines, 20)
    price = float(klines[-1][4])
    if ma7 is None or ma20 is None:
        return results

    if ma7 > ma20 and price > ma7:
        results.append(PatternResult("bullish_trend", "bullish", 0.7))
    elif ma7 < ma20 and price < ma7:
        results.append(PatternResult("bearish_trend", "bearish", 0.7))

    if len(klines) >= 21:
        prev_ma7 = mean([float(k[4]) for k in klines[-8:-1]])
        prev_ma20 = mean([float(k[4]) for k in klines[-21:-1]])
        if prev_ma7 <= prev_ma20 and ma7 > ma20:
            results.append(PatternResult("ma_golden_cross", "bullish", 0.85))
        elif prev_ma7 >= prev_ma20 and ma7 < ma20:
            results.append(PatternResult("ma_death_cross", "bearish", 0.85))

    return results


# ---------------------------------------------------------------------------
# Context-aware combo patterns (6) — pattern + position/volume
# ---------------------------------------------------------------------------

def _price_position_nd(klines: list, n: int) -> float:
    """Price position within last N bars' range (0=low, 1=high)."""
    if len(klines) < n:
        return 0.5
    highs = [float(k[2]) for k in klines[-n:]]
    lows = [float(k[3]) for k in klines[-n:]]
    high_n = max(highs)
    low_n = min(lows)
    rng = high_n - low_n
    if rng < 1e-9:
        return 0.5
    return (float(klines[-1][4]) - low_n) / rng


def _recent_return(klines: list, n: int) -> float:
    if len(klines) < n + 1:
        return 0.0
    c_now = float(klines[-1][4])
    c_prev = float(klines[-(n + 1)][4])
    return (c_now - c_prev) / max(abs(c_prev), 1e-9)


def _detect_context_patterns(klines: list, single_patterns: list[PatternResult]) -> list[PatternResult]:
    results = []
    if len(klines) < 20:
        return results

    pos_20 = _price_position_nd(klines, 20)
    ret_5 = _recent_return(klines, 5)
    pattern_names = {p.name for p in single_patterns}

    volumes = [float(k[5]) for k in klines[-5:]]
    ma5_vol = mean(volumes) if volumes else 1.0
    curr_vol = float(klines[-1][5])
    vol_ratio = curr_vol / max(ma5_vol, 1e-9)

    if "hammer" in pattern_names and pos_20 < 0.2:
        results.append(PatternResult("hammer_at_support", "bullish", 0.9, context_valid=True))

    if "bearish_engulfing" in pattern_names and pos_20 > 0.8:
        results.append(PatternResult("engulfing_at_resistance", "bearish", 0.9, context_valid=True))

    if "morning_star" in pattern_names and ret_5 < -0.03:
        results.append(PatternResult("morning_star_after_decline", "bullish", 0.9, context_valid=True))

    if "evening_star" in pattern_names and ret_5 > 0.03:
        results.append(PatternResult("evening_star_after_rally", "bearish", 0.9, context_valid=True))

    high_prev = max(float(k[2]) for k in klines[-20:-1]) if len(klines) >= 2 else 0.0
    price = float(klines[-1][4])
    if price > high_prev * 0.998 and vol_ratio > 2.0:
        results.append(PatternResult("volume_breakout", "bullish", min(vol_ratio / 3, 1.0), context_valid=True))

    if "long_upper_shadow" in pattern_names and pos_20 > 0.85 and vol_ratio > 1.5:
        results.append(PatternResult("exhaustion_at_top", "bearish", 0.85, context_valid=True))

    return results


# ---------------------------------------------------------------------------
# Main detection API
# ---------------------------------------------------------------------------

def detect_patterns(klines: list, mark_price: float = 0.0) -> list[PatternResult]:
    """Detect all candlestick patterns from a list of Binance klines.

    Args:
        klines: List of Binance kline arrays [ts, o, h, l, c, v, ...]
        mark_price: Current mark price (unused, kept for API consistency)

    Returns:
        List of detected PatternResult objects.
    """
    if not klines or len(klines) < 1:
        return []

    results: list[PatternResult] = []

    # Single-bar: analyze latest bar
    o, h, l, c, v = _parse_bar(klines[-1])
    m = _bar_metrics(o, h, l, c)
    for detector in _SINGLE_DETECTORS:
        p = detector(o, h, l, c, m)
        if p:
            results.append(p)

    # Double-bar
    if len(klines) >= 2:
        prev = klines[-2]
        curr = klines[-1]
        po, ph, pl, pc, _ = _parse_bar(prev)
        pm = _bar_metrics(po, ph, pl, pc)
        cm = m
        for detector in _DOUBLE_DETECTORS:
            p = detector(prev, curr, pm, cm)
            if p:
                results.append(p)

    # Triple-bar
    if len(klines) >= 3:
        k1, k2, k3 = klines[-3], klines[-2], klines[-1]
        o1, h1, l1, c1, _ = _parse_bar(k1)
        o2, h2, l2, c2, _ = _parse_bar(k2)
        o3, h3, l3, c3, _ = _parse_bar(k3)
        m1 = _bar_metrics(o1, h1, l1, c1)
        m2 = _bar_metrics(o2, h2, l2, c2)
        m3 = _bar_metrics(o3, h3, l3, c3)
        for detector in _TRIPLE_DETECTORS:
            p = detector(k1, k2, k3, m1, m2, m3)
            if p:
                results.append(p)

    # Trend/structure
    results.extend(_detect_trend_patterns(klines))

    # Context combos
    results.extend(_detect_context_patterns(klines, results))

    return results


# ---------------------------------------------------------------------------
# Feature output for model training
# ---------------------------------------------------------------------------

PATTERN_NAMES = [
    "hammer", "inverted_hammer", "doji", "long_upper_shadow",
    "long_lower_shadow", "marubozu", "bullish_engulfing", "bearish_engulfing",
    "dark_cloud_cover", "piercing_line", "morning_star", "evening_star",
    "three_white_soldiers", "three_black_crows", "bullish_trend", "bearish_trend",
    "ma_golden_cross", "ma_death_cross",
    "hammer_at_support", "engulfing_at_resistance", "morning_star_after_decline",
    "evening_star_after_rally", "volume_breakout", "exhaustion_at_top",
]

SEQUENCE_FEATURE_KEYS = []
for _i in range(1, 6):
    for _feat in ("body_ratio", "lower_shadow_ratio", "upper_shadow_ratio", "range_vs_prev", "volume_vs_prev"):
        SEQUENCE_FEATURE_KEYS.append(f"bar_n{_i}_{_feat}")

# Cross-bar relationship features for multi-bar pattern learning
CROSS_BAR_FEATURE_KEYS = [
    # Bar direction sequence (1=bullish, -1=bearish)
    "dir_n1", "dir_n2", "dir_n3", "dir_n4", "dir_n5",
    # Body engulfing ratios (curr_body / prev_body)
    "body_engulf_n1n2", "body_engulf_n2n3", "body_engulf_n3n4",
    # Close vs prev body midpoint (for piercing/dark cloud)
    "close_vs_prev_mid_n1", "close_vs_prev_mid_n2",
    # Open gap (curr_open vs prev_close)
    "gap_n1n2", "gap_n2n3",
    # Consecutive same direction count
    "consec_bullish", "consec_bearish",
    # Close progression (c2>c1, c3>c2 for soldiers/crows)
    "close_rising_3", "close_falling_3",
    # Body size progression
    "body_growing_3", "body_shrinking_3",
    # Price position in 20-bar range
    "price_position_20",
    # MA relationship features
    "close_vs_ma7", "close_vs_ma20", "ma7_vs_ma20",
    # MA slope (direction of MA change)
    "ma7_slope", "ma20_slope",
    # MA cross distance (how close MA7 and MA20 are — near zero = about to cross)
    "ma_cross_distance",
    # Previous bar MA relationship (for detecting the cross moment)
    "prev_ma7_vs_ma20",
    # Volume spike
    "vol_vs_ma5", "vol_vs_ma20",
]

ALL_FEATURE_KEYS = SEQUENCE_FEATURE_KEYS + CROSS_BAR_FEATURE_KEYS


def compute_pattern_features(klines: list, mark_price: float = 0.0) -> dict[str, float]:
    """Compute pattern labels + sequence features for model training.

    Returns dict with:
    - 24 pattern binary labels (pat_<name>: 0.0 or 1.0)
    - 25 sequence features (bar_n1_body_ratio, etc.)
    - pattern_bullish_count, pattern_bearish_count summary
    """
    features: dict[str, float] = {}

    # Pattern labels
    patterns = detect_patterns(klines, mark_price)
    detected = {p.name for p in patterns}
    bullish_count = 0
    bearish_count = 0
    for name in PATTERN_NAMES:
        features[f"pat_{name}"] = 1.0 if name in detected else 0.0
    for p in patterns:
        if p.direction == "bullish":
            bullish_count += 1
        elif p.direction == "bearish":
            bearish_count += 1
    features["pattern_bullish_count"] = float(bullish_count)
    features["pattern_bearish_count"] = float(bearish_count)

    # Sequence features: last 5 bars relative metrics
    for i in range(1, 6):
        idx = -i
        prefix = f"bar_n{i}_"
        if abs(idx) <= len(klines):
            o, h, l, c, v = _parse_bar(klines[idx])
            m = _bar_metrics(o, h, l, c)
            features[prefix + "body_ratio"] = round(m["body_ratio"], 4)
            features[prefix + "lower_shadow_ratio"] = round(m["lower_ratio"], 4)
            features[prefix + "upper_shadow_ratio"] = round(m["upper_ratio"], 4)
            # range vs prev
            if abs(idx) < len(klines) and idx - 1 >= -len(klines):
                _, ph, pl, _, _ = _parse_bar(klines[idx - 1])
                prev_range = max(ph - pl, 1e-9)
                features[prefix + "range_vs_prev"] = round(m["range"] / prev_range, 4)
            else:
                features[prefix + "range_vs_prev"] = 1.0
            # volume vs prev
            if abs(idx) < len(klines) and idx - 1 >= -len(klines):
                _, _, _, _, pv = _parse_bar(klines[idx - 1])
                features[prefix + "volume_vs_prev"] = round(v / max(pv, 1e-9), 4)
            else:
                features[prefix + "volume_vs_prev"] = 1.0
        else:
            features[prefix + "body_ratio"] = 0.0
            features[prefix + "lower_shadow_ratio"] = 0.0
            features[prefix + "upper_shadow_ratio"] = 0.0
            features[prefix + "range_vs_prev"] = 1.0
            features[prefix + "volume_vs_prev"] = 1.0

    # Cross-bar relationship features
    _compute_cross_bar_features(klines, features)

    return features


def _compute_cross_bar_features(klines: list, features: dict[str, float]) -> None:
    """Compute cross-bar relationship features for multi-bar pattern learning."""
    n = len(klines)

    # Parse last 5 bars
    bars = []
    for i in range(1, 6):
        idx = -i
        if abs(idx) <= n:
            o, h, l, c, v = _parse_bar(klines[idx])
            bars.append({"o": o, "h": h, "l": l, "c": c, "v": v, "m": _bar_metrics(o, h, l, c)})
        else:
            bars.append(None)

    # Direction sequence
    for i in range(5):
        if bars[i]:
            features[f"dir_n{i+1}"] = 1.0 if bars[i]["m"]["is_bullish"] else -1.0
        else:
            features[f"dir_n{i+1}"] = 0.0

    # Body engulfing ratios
    for i, key in enumerate(["body_engulf_n1n2", "body_engulf_n2n3", "body_engulf_n3n4"]):
        if bars[i] and bars[i+1] and bars[i+1]["m"]["body"] > 1e-9:
            features[key] = round(bars[i]["m"]["body"] / bars[i+1]["m"]["body"], 4)
        else:
            features[key] = 1.0

    # Close vs prev body midpoint
    for i, key in enumerate(["close_vs_prev_mid_n1", "close_vs_prev_mid_n2"]):
        if bars[i] and bars[i+1]:
            prev_mid = bars[i+1]["m"]["mid_body"]
            curr_close = bars[i]["c"]
            prev_range = bars[i+1]["m"]["range"]
            features[key] = round((curr_close - prev_mid) / max(prev_range, 1e-9), 4)
        else:
            features[key] = 0.0

    # Open gap
    for i, key in enumerate(["gap_n1n2", "gap_n2n3"]):
        if bars[i] and bars[i+1]:
            gap = bars[i]["o"] - bars[i+1]["c"]
            features[key] = round(gap / max(bars[i+1]["m"]["range"], 1e-9), 4)
        else:
            features[key] = 0.0

    # Consecutive direction
    consec_bull = 0
    consec_bear = 0
    for b in bars:
        if b is None:
            break
        if b["m"]["is_bullish"]:
            consec_bull += 1
        else:
            break
    for b in bars:
        if b is None:
            break
        if not b["m"]["is_bullish"]:
            consec_bear += 1
        else:
            break
    features["consec_bullish"] = float(consec_bull)
    features["consec_bearish"] = float(consec_bear)

    # Close progression (3-bar)
    if bars[0] and bars[1] and bars[2]:
        c1, c2, c3 = bars[2]["c"], bars[1]["c"], bars[0]["c"]
        features["close_rising_3"] = 1.0 if (c2 > c1 and c3 > c2) else 0.0
        features["close_falling_3"] = 1.0 if (c2 < c1 and c3 < c2) else 0.0
        b1, b2, b3 = bars[2]["m"]["body"], bars[1]["m"]["body"], bars[0]["m"]["body"]
        features["body_growing_3"] = 1.0 if (b2 >= b1 * 0.7 and b3 >= b2 * 0.7) else 0.0
        features["body_shrinking_3"] = 1.0 if (b2 <= b1 * 1.3 and b3 <= b2 * 0.7) else 0.0
    else:
        features["close_rising_3"] = 0.0
        features["close_falling_3"] = 0.0
        features["body_growing_3"] = 0.0
        features["body_shrinking_3"] = 0.0

    # Price position in 20-bar range
    if n >= 20:
        highs = [float(k[2]) for k in klines[-20:]]
        lows = [float(k[3]) for k in klines[-20:]]
        h20, l20 = max(highs), min(lows)
        rng20 = h20 - l20
        price = float(klines[-1][4])
        features["price_position_20"] = round((price - l20) / max(rng20, 1e-9), 4)
    else:
        features["price_position_20"] = 0.5

    # MA features
    if n >= 20:
        closes = [float(k[4]) for k in klines]
        ma7 = mean(closes[-7:]) if n >= 7 else closes[-1]
        ma20 = mean(closes[-20:])
        price = closes[-1]
        features["close_vs_ma7"] = round((price - ma7) / max(abs(ma7), 1e-9), 6)
        features["close_vs_ma20"] = round((price - ma20) / max(abs(ma20), 1e-9), 6)
        features["ma7_vs_ma20"] = round((ma7 - ma20) / max(abs(ma20), 1e-9), 6)

        # MA slope: compare current MA vs 3-bar-ago MA
        if n >= 10:
            prev_ma7 = mean(closes[-10:-3]) if n >= 10 else ma7
            features["ma7_slope"] = round((ma7 - prev_ma7) / max(abs(prev_ma7), 1e-9), 6)
        else:
            features["ma7_slope"] = 0.0
        if n >= 23:
            prev_ma20 = mean(closes[-23:-3])
            features["ma20_slope"] = round((ma20 - prev_ma20) / max(abs(prev_ma20), 1e-9), 6)
        else:
            features["ma20_slope"] = 0.0

        # MA cross distance: normalized gap between MA7 and MA20
        features["ma_cross_distance"] = round(abs(ma7 - ma20) / max(abs(ma20), 1e-9), 6)

        # Previous bar's MA7 vs MA20 (to detect the cross moment)
        if n >= 21:
            prev_ma7_val = mean(closes[-8:-1])
            prev_ma20_val = mean(closes[-21:-1])
            features["prev_ma7_vs_ma20"] = round((prev_ma7_val - prev_ma20_val) / max(abs(prev_ma20_val), 1e-9), 6)
        else:
            features["prev_ma7_vs_ma20"] = 0.0
    else:
        features["close_vs_ma7"] = 0.0
        features["close_vs_ma20"] = 0.0
        features["ma7_vs_ma20"] = 0.0
        features["ma7_slope"] = 0.0
        features["ma20_slope"] = 0.0
        features["ma_cross_distance"] = 0.0
        features["prev_ma7_vs_ma20"] = 0.0

    # Volume features
    if n >= 5:
        vols = [float(k[5]) for k in klines[-5:]]
        ma5_v = mean(vols)
        features["vol_vs_ma5"] = round(float(klines[-1][5]) / max(ma5_v, 1e-9), 4)
    else:
        features["vol_vs_ma5"] = 1.0
    if n >= 20:
        vols20 = [float(k[5]) for k in klines[-20:]]
        ma20_v = mean(vols20)
        features["vol_vs_ma20"] = round(float(klines[-1][5]) / max(ma20_v, 1e-9), 4)
    else:
        features["vol_vs_ma20"] = 1.0
