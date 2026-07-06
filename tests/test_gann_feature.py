from __future__ import annotations

from datetime import date, timedelta

import pytest

from trading_system.gann_feature import classify_zone, compute_gann_feature, gann_position_multiplier


def make_daily_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for idx in range(90):
        if idx < 10:
            close = 100.0 + idx
        elif idx == 10:
            close = 200.0
        elif idx <= 40:
            close = 200.0 - (idx - 10) * 3.0
        else:
            close = 110.0 + (idx - 40) * 1.0
        rows.append({
            "time": f"2026-01-{(idx % 28) + 1:02d}",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        })
    return rows


def test_gann_feature_computes_log_lines_and_zone() -> None:
    feature = compute_gann_feature(
        "BTCUSDT",
        make_daily_rows(),
        lookback_days=80,
        min_cycle_days=14,
        min_drawdown_pct=10.0,
    )

    assert feature is not None
    assert feature.symbol == "BTCUSDT"
    assert feature.scale == "log"
    assert feature.cycle_days >= 14
    assert set(feature.line_values) == {"1x2", "1x1", "2x1"}
    assert feature.zone in {"above_1x1", "1x2_to_1x1", "below_1x2", "unknown"}


def test_classify_zone_orders_price_against_1x2_and_1x1() -> None:
    assert classify_zone(120.0, {"1x2": 100.0, "1x1": 110.0}) == "above_1x1"
    assert classify_zone(105.0, {"1x2": 100.0, "1x1": 110.0}) == "1x2_to_1x1"
    assert classify_zone(95.0, {"1x2": 100.0, "1x1": 110.0}) == "below_1x2"


def test_gann_position_multiplier_is_macro_sizing_only() -> None:
    supportive = {
        "bias": "support_long",
        "confidence": 0.8,
        "stale": False,
    }
    conflict = {
        "bias": "support_short",
        "confidence": 0.8,
        "stale": False,
    }

    assert gann_position_multiplier(supportive, "long") > 1.0
    assert gann_position_multiplier(conflict, "long") < 1.0
    assert gann_position_multiplier({"bias": "support_long", "confidence": 0.2}, "long") == 1.0


# ---------------------------------------------------------------------------
# Fixture helpers for the structural tests below
# ---------------------------------------------------------------------------

def _iso(day_index: int) -> str:
    """Sequential sortable ISO dates so timestamp comparisons work as strings."""
    return (date(2025, 1, 1) + timedelta(days=day_index)).isoformat()


def _flat_rows(n: int, price: float = 800.0) -> list[dict[str, float | str]]:
    return [
        {"time": _iso(i), "open": price, "high": price, "low": price, "close": price}
        for i in range(n)
    ]


def _set_high(rows: list[dict[str, float | str]], idx: int, high: float) -> None:
    rows[idx]["high"] = high
    rows[idx]["open"] = rows[idx]["close"] = (high + float(rows[idx]["low"])) / 2.0


def _set_low(rows: list[dict[str, float | str]], idx: int, low: float) -> None:
    rows[idx]["low"] = low
    rows[idx]["open"] = rows[idx]["close"] = (low + float(rows[idx]["high"])) / 2.0


# ---------------------------------------------------------------------------
# NO FUTURE LEAK — the most important property of the feature
# ---------------------------------------------------------------------------

def test_no_future_leak_anchor_selected_only_from_past() -> None:
    """The global max high sits AFTER asof_index; anchors must ignore it."""
    rows: list[dict[str, float | str]] = []
    for i in range(120):
        if i < 10:
            close = 100.0 + i * 5.0
        elif i == 10:
            close = 200.0  # pre-asof peak (high = 202.0)
        elif i <= 40:
            close = 200.0 - (i - 10) * 3.0  # decline into the anchor low at i=40
        elif i <= 80:
            close = 110.0 + (i - 40) * 0.2
        else:
            close = 120.0 + (i - 80) * 10.0  # future blow-off: global max high AFTER asof
        rows.append({
            "time": _iso(i),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        })

    asof = 80
    global_max_high = max(float(r["high"]) for r in rows)
    past_max_high = max(float(r["high"]) for r in rows[: asof + 1])
    assert global_max_high > past_max_high  # test is meaningful: real leak bait exists

    feature = compute_gann_feature("BTCUSDT", rows, asof_index=asof, lookback_days=100)
    assert feature is not None

    # Anchor must come from the past window, not the future global max.
    assert feature.anchor_high == pytest.approx(202.0)
    assert feature.anchor_high < global_max_high
    past_times = {str(r["time"]) for r in rows[: asof + 1]}
    assert feature.anchor_high_time in past_times
    assert feature.anchor_low_time in past_times
    assert feature.anchor_high_time <= str(rows[asof]["time"])
    assert feature.anchor_low_time <= str(rows[asof]["time"])
    assert feature.anchor_high_time <= feature.anchor_low_time

    # Equivalence proof: computing on the truncated series (no future rows at
    # all) must produce the identical feature.
    truncated = compute_gann_feature("BTCUSDT", rows[: asof + 1], asof_index=asof, lookback_days=100)
    assert truncated is not None
    assert truncated.to_dict() == feature.to_dict()


# ---------------------------------------------------------------------------
# LOG-LINE MATH — hand-computed 1x2 / 1x1 / 2x1 values on the log axis
# ---------------------------------------------------------------------------

def test_log_line_values_match_hand_computation() -> None:
    # Anchor high 1000 at idx 5, anchor low 500 at idx 25 -> cycle_days = 20.
    # Evaluate at idx 35 -> days_since_low = 10.
    # base_slope = (ln 1000 - ln 500) / 20 = ln(2)/20
    # line(mult) = low * exp(mult * base_slope * 10) = 500 * 2**(mult/2)
    rows = _flat_rows(40)
    _set_high(rows, 5, 1000.0)
    _set_low(rows, 25, 500.0)

    feature = compute_gann_feature("ETHUSDT", rows, asof_index=35, lookback_days=60)
    assert feature is not None
    assert feature.anchor_high == pytest.approx(1000.0)
    assert feature.anchor_low == pytest.approx(500.0)
    assert feature.cycle_days == 20
    assert feature.scale == "log"

    assert feature.line_values["1x2"] == pytest.approx(500.0 * 2 ** 0.25, rel=1e-9)  # ~594.60
    assert feature.line_values["1x1"] == pytest.approx(500.0 * 2 ** 0.5, rel=1e-9)   # ~707.11
    assert feature.line_values["2x1"] == pytest.approx(1000.0, rel=1e-9)

    # close = 800 sits above 1x1 (707.11) -> above_1x1 / support_long.
    assert feature.zone == "above_1x1"
    assert feature.bias == "support_long"


# ---------------------------------------------------------------------------
# ZONE CLASSIFICATION — including exact-boundary semantics
# ---------------------------------------------------------------------------

def test_classify_zone_boundaries_are_inclusive_upward() -> None:
    lines = {"1x2": 100.0, "1x1": 110.0}
    # Exact boundaries: classify_zone uses >=, so a close sitting exactly ON a
    # line belongs to the zone ABOVE that line.
    assert classify_zone(110.0, lines) == "above_1x1"
    assert classify_zone(100.0, lines) == "1x2_to_1x1"
    assert classify_zone(99.999999, lines) == "below_1x2"
    # Degenerate inputs -> "unknown", never an exception.
    assert classify_zone(0.0, lines) == "unknown"
    assert classify_zone(105.0, {"1x2": 0.0, "1x1": 110.0}) == "unknown"
    assert classify_zone(105.0, {}) == "unknown"


# ---------------------------------------------------------------------------
# ANCHOR STABILITY — small drift on stable series, large when anchor jumps
# ---------------------------------------------------------------------------

def test_anchor_stability_small_on_stable_series() -> None:
    # Single unambiguous high; lookback covers the whole series so shifting
    # asof by +/-3 days never changes the selected anchor -> drift ~0.
    rows = _flat_rows(80)
    _set_high(rows, 10, 1000.0)
    _set_low(rows, 30, 500.0)

    feature = compute_gann_feature("SOLUSDT", rows, asof_index=70, lookback_days=100)
    assert feature is not None
    assert feature.anchor_stability == pytest.approx(0.0, abs=1e-9)
    assert feature.confidence == pytest.approx(0.80)


def test_anchor_stability_large_when_anchor_jumps() -> None:
    # Two near-equal highs 30 days apart (1000 @ idx 10, 999 @ idx 40) with a
    # short lookback (60): shifting asof from 70 to 71..73 pushes the first
    # high out of the window, so the anchor jumps to the second high and the
    # 1x1 line moves massively -> large drift, zero confidence.
    rows = _flat_rows(80)
    _set_high(rows, 10, 1000.0)
    _set_low(rows, 30, 500.0)
    _set_high(rows, 40, 999.0)
    _set_low(rows, 60, 700.0)

    feature = compute_gann_feature("SOLUSDT", rows, asof_index=70, lookback_days=60)
    assert feature is not None
    assert feature.anchor_high == pytest.approx(1000.0)  # base anchor = first high
    assert feature.anchor_stability > 10.0  # anchor jump reported as large drift
    assert feature.confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# STALE / INVALID — module's documented invalid representation is None
# ---------------------------------------------------------------------------
# compute_gann_feature never raises on bad input and never emits stale=True;
# every invalid/insufficient case returns None (callers treat None as "no
# Gann feature", and gann_position_multiplier(None, side) == 1.0).

def test_insufficient_data_returns_none() -> None:
    rows = _flat_rows(10)
    assert compute_gann_feature("BTCUSDT", rows) is None  # < lookback//2 rows
    # Fewer rows than min_cycle_days + 2 with a tiny lookback also fails.
    assert compute_gann_feature("BTCUSDT", _flat_rows(10), lookback_days=4, min_cycle_days=14) is None


def test_monotonic_series_without_drawdown_returns_none() -> None:
    rows = [
        {"time": _iso(i), "open": 100.0 + i, "high": 100.0 + i, "low": 100.0 + i, "close": 100.0 + i}
        for i in range(60)
    ]
    # Monotonic rise: the window high is always the latest bar -> cycle_days 0.
    assert compute_gann_feature("BTCUSDT", rows, lookback_days=60) is None


def test_shallow_drawdown_below_threshold_returns_none() -> None:
    rows = _flat_rows(40, price=995.0)
    _set_high(rows, 5, 1000.0)
    _set_low(rows, 25, 950.0)  # 5% drawdown < min_drawdown_pct=10
    assert compute_gann_feature("BTCUSDT", rows, asof_index=35, lookback_days=60) is None


def test_zero_or_negative_prices_return_none_without_raising() -> None:
    zero_rows = _flat_rows(40, price=0.0)
    assert compute_gann_feature("BTCUSDT", zero_rows, asof_index=35, lookback_days=60) is None

    negative_low_rows = _flat_rows(40, price=100.0)
    _set_high(negative_low_rows, 5, 200.0)
    negative_low_rows[25]["low"] = -5.0  # would blow up math.log if not guarded
    assert compute_gann_feature("BTCUSDT", negative_low_rows, asof_index=35, lookback_days=60) is None


def test_out_of_range_asof_index_returns_none() -> None:
    rows = _flat_rows(40)
    assert compute_gann_feature("BTCUSDT", rows, asof_index=0, lookback_days=60) is None
    assert compute_gann_feature("BTCUSDT", rows, asof_index=40, lookback_days=60) is None
