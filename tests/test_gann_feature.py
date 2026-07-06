from __future__ import annotations

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
