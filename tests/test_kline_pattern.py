"""Unit tests for kline_pattern.py — all 18 patterns + context combos + features."""
from __future__ import annotations

import pytest
from trading_system.kline_pattern import (
    detect_patterns, compute_pattern_features, PatternResult,
    PATTERN_NAMES, SEQUENCE_FEATURE_KEYS,
)


def _make_bar(o, h, l, c, v=1000.0, ts=0):
    return [ts, o, h, l, c, v]


class TestSingleBarPatterns:
    def test_hammer(self):
        bar = _make_bar(100, 101, 90, 100.5)
        patterns = detect_patterns([bar])
        names = [p.name for p in patterns]
        assert "hammer" in names

    def test_inverted_hammer(self):
        bar = _make_bar(90, 100, 89.5, 90.5)
        patterns = detect_patterns([bar])
        names = [p.name for p in patterns]
        assert "inverted_hammer" in names

    def test_doji(self):
        bar = _make_bar(100, 105, 95, 100.2)
        patterns = detect_patterns([bar])
        names = [p.name for p in patterns]
        assert "doji" in names

    def test_long_upper_shadow(self):
        bar = _make_bar(100, 110, 99, 100.5)
        patterns = detect_patterns([bar])
        names = [p.name for p in patterns]
        assert "long_upper_shadow" in names

    def test_long_lower_shadow(self):
        bar = _make_bar(100, 101, 90, 99.5)
        patterns = detect_patterns([bar])
        names = [p.name for p in patterns]
        assert "long_lower_shadow" in names

    def test_marubozu_bullish(self):
        bar = _make_bar(100, 110.2, 99.8, 110)
        patterns = detect_patterns([bar])
        names = [p.name for p in patterns]
        assert "marubozu" in names
        marubozu = [p for p in patterns if p.name == "marubozu"][0]
        assert marubozu.direction == "bullish"

    def test_marubozu_bearish(self):
        bar = _make_bar(110, 110.2, 99.8, 100)
        patterns = detect_patterns([bar])
        names = [p.name for p in patterns]
        assert "marubozu" in names
        marubozu = [p for p in patterns if p.name == "marubozu"][0]
        assert marubozu.direction == "bearish"


class TestDoubleBarPatterns:
    def test_bullish_engulfing(self):
        prev = _make_bar(105, 106, 100, 101)  # bearish
        curr = _make_bar(100, 107, 99, 106)   # bullish, body wraps prev
        patterns = detect_patterns([prev, curr])
        names = [p.name for p in patterns]
        assert "bullish_engulfing" in names

    def test_bearish_engulfing(self):
        prev = _make_bar(100, 105, 99, 104)   # bullish
        curr = _make_bar(105, 106, 98, 99)    # bearish, body wraps prev
        patterns = detect_patterns([prev, curr])
        names = [p.name for p in patterns]
        assert "bearish_engulfing" in names

    def test_dark_cloud_cover(self):
        prev = _make_bar(100, 108, 99, 107)   # bullish
        curr = _make_bar(109, 110, 102, 102.5)  # opens above prev high, closes below prev mid
        patterns = detect_patterns([prev, curr])
        names = [p.name for p in patterns]
        assert "dark_cloud_cover" in names

    def test_piercing_line(self):
        prev = _make_bar(107, 108, 100, 101)  # bearish
        curr = _make_bar(99, 106, 98, 105)    # opens below prev low, closes above prev mid
        patterns = detect_patterns([prev, curr])
        names = [p.name for p in patterns]
        assert "piercing_line" in names


class TestTripleBarPatterns:
    def test_morning_star(self):
        k1 = _make_bar(110, 111, 100, 101)    # bearish, big body
        k2 = _make_bar(101, 102, 100, 101.2)  # small body (doji-like)
        k3 = _make_bar(101, 108, 100, 107)    # bullish, closes above k1 mid
        patterns = detect_patterns([k1, k2, k3])
        names = [p.name for p in patterns]
        assert "morning_star" in names

    def test_evening_star(self):
        k1 = _make_bar(100, 110, 99, 109)     # bullish, big body
        k2 = _make_bar(109, 110, 108, 109.2)  # small body
        k3 = _make_bar(109, 110, 101, 102)    # bearish, closes below k1 mid
        patterns = detect_patterns([k1, k2, k3])
        names = [p.name for p in patterns]
        assert "evening_star" in names

    def test_three_white_soldiers(self):
        k1 = _make_bar(100, 105, 99, 104)
        k2 = _make_bar(104, 109, 103, 108)
        k3 = _make_bar(108, 113, 107, 112)
        patterns = detect_patterns([k1, k2, k3])
        names = [p.name for p in patterns]
        assert "three_white_soldiers" in names

    def test_three_black_crows(self):
        k1 = _make_bar(112, 113, 107, 108)
        k2 = _make_bar(108, 109, 103, 104)
        k3 = _make_bar(104, 105, 99, 100)
        patterns = detect_patterns([k1, k2, k3])
        names = [p.name for p in patterns]
        assert "three_black_crows" in names


class TestTrendPatterns:
    def _make_uptrend_klines(self, n=21):
        """Generate N bars in a clear uptrend."""
        klines = []
        for i in range(n):
            base = 100 + i * 2
            klines.append(_make_bar(base, base + 1.5, base - 0.5, base + 1))
        return klines

    def _make_downtrend_klines(self, n=21):
        klines = []
        for i in range(n):
            base = 200 - i * 2
            klines.append(_make_bar(base, base + 0.5, base - 1.5, base - 1))
        return klines

    def test_bullish_trend(self):
        klines = self._make_uptrend_klines()
        patterns = detect_patterns(klines)
        names = [p.name for p in patterns]
        assert "bullish_trend" in names

    def test_bearish_trend(self):
        klines = self._make_downtrend_klines()
        patterns = detect_patterns(klines)
        names = [p.name for p in patterns]
        assert "bearish_trend" in names

    def test_golden_cross(self):
        # 14 bars: clear downtrend so MA7 stays below MA20
        klines = []
        for i in range(14):
            base = 120 - i * 2
            klines.append(_make_bar(base, base + 0.5, base - 0.5, base - 0.3))
        # 7 bars: sharp rally so MA7 crosses above MA20
        for i in range(7):
            base = 94 + i * 5
            klines.append(_make_bar(base, base + 3, base - 0.5, base + 2.5))
        patterns = detect_patterns(klines)
        names = [p.name for p in patterns]
        assert "ma_golden_cross" in names


class TestContextPatterns:
    def test_hammer_at_support(self):
        # 20 bars: first 19 trending up, last bar drops to very bottom with hammer
        klines = []
        for i in range(19):
            klines.append(_make_bar(120 + i * 2, 125 + i * 2, 119 + i * 2, 122 + i * 2))
        # Last bar: hammer shape, close near 20-bar low
        # 20-bar high = 125+18*2 = 161, 20-bar low = 115 (this bar's low)
        # close = 120, pos = (120-115)/(161-115) = 5/46 = 0.108 < 0.2
        klines.append(_make_bar(120, 120.5, 115, 120.3))
        patterns = detect_patterns(klines)
        names = [p.name for p in patterns]
        assert "hammer_at_support" in names

    def test_volume_breakout(self):
        klines = []
        for i in range(19):
            klines.append(_make_bar(100, 102, 99, 101, v=100))
        # Last bar: closes at new 20-bar high with 3x volume
        klines.append(_make_bar(101, 106, 100, 105.5, v=350))
        patterns = detect_patterns(klines)
        names = [p.name for p in patterns]
        assert "volume_breakout" in names


class TestComputePatternFeatures:
    def test_output_keys(self):
        klines = [_make_bar(100 + i, 102 + i, 99 + i, 101 + i) for i in range(10)]
        features = compute_pattern_features(klines)
        for name in PATTERN_NAMES:
            assert f"pat_{name}" in features
        for key in SEQUENCE_FEATURE_KEYS:
            assert key in features
        assert "pattern_bullish_count" in features
        assert "pattern_bearish_count" in features

    def test_sequence_features_values(self):
        klines = [_make_bar(100, 110, 90, 105, v=1000) for _ in range(6)]
        features = compute_pattern_features(klines)
        assert features["bar_n1_body_ratio"] > 0
        assert features["bar_n1_range_vs_prev"] == pytest.approx(1.0, abs=0.01)
        assert features["bar_n1_volume_vs_prev"] == pytest.approx(1.0, abs=0.01)

    def test_empty_klines(self):
        features = compute_pattern_features([])
        # All values should be numeric defaults (0.0, 0.5, or 1.0)
        assert all(isinstance(v, float) for v in features.values())

    def test_single_bar(self):
        features = compute_pattern_features([_make_bar(100, 110, 90, 105)])
        assert "bar_n1_body_ratio" in features
