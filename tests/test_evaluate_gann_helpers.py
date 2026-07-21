from __future__ import annotations

import math

import pytest

from scripts.evaluate_gann_structural_features import (
    build_outcome_row,
    compute_future_return_pct,
    compute_mfe_mae_pct,
    profit_factor,
    summarize_rows,
    zone_ranking_verdict,
)


# ---------------------------------------------------------------------------
# Future return math
# ---------------------------------------------------------------------------

def test_future_return_pct_known_values() -> None:
    closes = [100.0, 110.0, 90.0, 120.0]
    assert compute_future_return_pct(closes, 0, 1) == pytest.approx(10.0)
    assert compute_future_return_pct(closes, 0, 2) == pytest.approx(-10.0)
    assert compute_future_return_pct(closes, 0, 3) == pytest.approx(20.0)
    assert compute_future_return_pct(closes, 1, 2) == pytest.approx((120.0 - 110.0) / 110.0 * 100.0)


def test_future_return_pct_unavailable_horizon_is_none() -> None:
    closes = [100.0, 110.0, 90.0]
    assert compute_future_return_pct(closes, 0, 3) is None  # past series end
    assert compute_future_return_pct(closes, 2, 1) is None  # last bar has no next day
    assert compute_future_return_pct(closes, -1, 1) is None
    assert compute_future_return_pct([0.0, 10.0], 0, 1) is None  # non-positive entry


# ---------------------------------------------------------------------------
# MFE / MAE
# ---------------------------------------------------------------------------

def test_mfe_mae_known_values() -> None:
    #             entry  d+1    d+2
    highs = [101.0, 105.0, 110.0]
    lows = [99.0, 95.0, 90.0]
    closes = [100.0, 100.0, 100.0]
    mfe, mae = compute_mfe_mae_pct(highs, lows, closes, 0, window_days=2)
    assert mfe == pytest.approx(10.0)   # max high 110 vs entry 100
    assert mae == pytest.approx(-10.0)  # min low 90 vs entry 100


def test_mfe_mae_excludes_entry_bar_itself() -> None:
    # Entry bar has the extreme high/low; the window starts at d+1 so it must
    # NOT count toward MFE/MAE.
    highs = [200.0, 105.0, 104.0]
    lows = [10.0, 96.0, 97.0]
    closes = [100.0, 100.0, 100.0]
    mfe, mae = compute_mfe_mae_pct(highs, lows, closes, 0, window_days=2)
    assert mfe == pytest.approx(5.0)
    assert mae == pytest.approx(-4.0)


def test_mfe_mae_requires_full_window() -> None:
    highs = [101.0, 105.0, 110.0]
    lows = [99.0, 95.0, 90.0]
    closes = [100.0, 100.0, 100.0]
    assert compute_mfe_mae_pct(highs, lows, closes, 0, window_days=3) == (None, None)
    assert compute_mfe_mae_pct(highs, lows, closes, 2, window_days=1) == (None, None)


# ---------------------------------------------------------------------------
# Profit factor
# ---------------------------------------------------------------------------

def test_profit_factor_known_values() -> None:
    assert profit_factor([10.0, -5.0, 5.0, -5.0]) == pytest.approx(1.5)
    assert profit_factor([-2.0, -3.0]) == 0.0
    assert profit_factor([4.0, 6.0]) == math.inf  # wins, no losses
    assert profit_factor([]) is None
    assert profit_factor([0.0, 0.0]) is None  # flat trades: no wins, no losses


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def _row(symbol: str, zone: str, r7: float | None, mfe: float | None = None, mae: float | None = None) -> dict:
    return {"symbol": symbol, "zone": zone, "future_return_7d": r7, "mfe_30d": mfe, "mae_30d": mae}


def test_summarize_rows_known_numbers() -> None:
    rows = [
        _row("BTCUSDT", "above_1x1", 10.0, mfe=12.0, mae=-3.0),
        _row("BTCUSDT", "above_1x1", -5.0, mfe=2.0, mae=-9.0),
        _row("BTCUSDT", "above_1x1", 4.0, mfe=None, mae=None),  # tail row: no 30d window
        _row("BTCUSDT", "below_1x2", -2.0, mfe=1.0, mae=-8.0),
    ]
    summary = summarize_rows(rows, horizons=(7,))
    assert len(summary) == 2

    above = next(e for e in summary if e["zone"] == "above_1x1")
    assert above["symbol"] == "BTCUSDT"
    assert above["horizon_days"] == 7
    assert above["n"] == 3
    assert above["avg_pct"] == pytest.approx(3.0)       # (10 - 5 + 4) / 3
    assert above["median_pct"] == pytest.approx(4.0)
    assert above["win_rate_pct"] == pytest.approx(200.0 / 3)  # 2 of 3
    assert above["profit_factor"] == pytest.approx(14.0 / 5.0)
    assert above["avg_mfe_pct"] == pytest.approx(7.0)   # None excluded
    assert above["avg_mae_pct"] == pytest.approx(-6.0)

    below = next(e for e in summary if e["zone"] == "below_1x2")
    assert below["n"] == 1
    assert below["avg_pct"] == pytest.approx(-2.0)
    assert below["win_rate_pct"] == pytest.approx(0.0)
    assert below["profit_factor"] == pytest.approx(0.0)


def test_summarize_rows_skips_horizons_without_data() -> None:
    rows = [_row("ETHUSDT", "1x2_to_1x1", None)]
    assert summarize_rows(rows, horizons=(7,)) == []


# ---------------------------------------------------------------------------
# Ranking verdict
# ---------------------------------------------------------------------------

def _summary_entry(symbol: str, zone: str, horizon: int, avg: float) -> dict:
    return {"symbol": symbol, "zone": zone, "horizon_days": horizon, "avg_pct": avg}


def test_zone_ranking_verdict_holds() -> None:
    summary = [
        _summary_entry("BTCUSDT", "above_1x1", 7, 5.0),
        _summary_entry("BTCUSDT", "1x2_to_1x1", 7, 2.0),
        _summary_entry("BTCUSDT", "below_1x2", 7, -1.0),
    ]
    assert zone_ranking_verdict(summary, "BTCUSDT", 7) is True


def test_zone_ranking_verdict_fails_when_order_broken() -> None:
    summary = [
        _summary_entry("BTCUSDT", "above_1x1", 7, 1.0),
        _summary_entry("BTCUSDT", "1x2_to_1x1", 7, 2.0),  # middle beats top
        _summary_entry("BTCUSDT", "below_1x2", 7, -1.0),
    ]
    assert zone_ranking_verdict(summary, "BTCUSDT", 7) is False


def test_zone_ranking_verdict_none_when_zone_missing() -> None:
    summary = [
        _summary_entry("BTCUSDT", "above_1x1", 7, 5.0),
        _summary_entry("BTCUSDT", "below_1x2", 7, -1.0),
        # 1x2_to_1x1 missing entirely
        _summary_entry("ETHUSDT", "1x2_to_1x1", 7, 2.0),  # other symbol must not leak in
    ]
    assert zone_ranking_verdict(summary, "BTCUSDT", 7) is None
    assert zone_ranking_verdict(summary, "BTCUSDT", 14) is None  # other horizon: no data


# ---------------------------------------------------------------------------
# Row assembly (feature dict -> outcome row)
# ---------------------------------------------------------------------------

def test_build_outcome_row_shapes_columns() -> None:
    feature_dict = {
        "symbol": "BTCUSDT",
        "zone": "above_1x1",
        "bias": "support_long",
        "confidence": 0.8,
        "anchor_high_time": "2025-01-05",
        "anchor_low_time": "2025-01-25",
        "cycle_days": 20,
        "line_values": {"1x2": 594.6, "1x1": 707.1, "2x1": 1000.0},
        "anchor_stability": 0.0,
    }
    closes = [100.0, 102.0, 104.0, 106.0]
    highs = [101.0, 103.0, 105.0, 107.0]
    lows = [99.0, 101.0, 103.0, 105.0]
    row = build_outcome_row(feature_dict, "2025-02-01", closes, highs, lows, 0, horizons=(1, 3, 7))

    assert row["symbol"] == "BTCUSDT"
    assert row["date"] == "2025-02-01"
    assert row["close"] == pytest.approx(100.0)
    assert row["line_1x1"] == pytest.approx(707.1)
    assert row["future_return_1d"] == pytest.approx(2.0)
    assert row["future_return_3d"] == pytest.approx(6.0)
    assert row["future_return_7d"] is None  # past series end -> empty
    assert row["mfe_30d"] is None and row["mae_30d"] is None  # no full 30d window
