from trading_system.v6_common.bb_utils import (
    band_position,
    near_band,
    nearest_band,
    snapshot,
    width_state,
)
from trading_system.v6_common.ma_utils import (
    broke_below_ma,
    describe_distance,
    held_above_ma,
    price_relation,
    reclaimed_ma,
    rejected_at_ma,
    slope_direction,
)
from trading_system.v6_common.mode_types import MODE_MIXED, is_known_mode
from trading_system.v6_common.reason_utils import dedupe_reasons, join_reasons, merge_reason_groups
from trading_system.v6_common.structure_utils import classify_label, summarize_labels


def test_ma_distance_and_relation():
    info = describe_distance(105.0, 100.0, tolerance_pct=0.2)

    assert round(info.distance_pct, 4) == 5.0
    assert info.relation == "above"
    assert price_relation(100.1, 100.0, tolerance_pct=0.2) == "near"


def test_ma_slope_and_cross_helpers():
    assert slope_direction(100.0, 101.0, tolerance_pct=0.2) == "up"
    assert slope_direction(100.0, 99.0, tolerance_pct=0.2) == "down"
    assert slope_direction(100.0, 100.1, tolerance_pct=0.2) == "flat"
    assert held_above_ma([99.0, 101.0, 102.0], [100.0, 100.0, 101.0], candles=2)
    assert broke_below_ma([101.0, 99.0], [100.0, 100.0], candles=1)
    assert reclaimed_ma(99.0, 101.0, 100.0)
    assert rejected_at_ma(101.0, 99.5, 100.0)


def test_bb_position_snapshot_and_width_state():
    assert band_position(105.0, 110.0, 100.0, 90.0) == 0.75
    assert nearest_band(96.0, 110.0, 100.0, 90.0) == "mid"
    assert near_band(100.2, 100.0, tolerance_pct=0.25)

    info = snapshot(105.0, 110.0, 100.0, 90.0)

    assert info.nearest == "upper"
    assert round(info.width_pct, 4) == 20.0
    assert width_state(10.0, 12.0, tolerance_pct=0.5) == "expanding"
    assert width_state(12.0, 10.0, tolerance_pct=0.5) == "compressing"
    assert width_state(10.0, 10.2, tolerance_pct=0.5) == "flat"


def test_structure_label_summary():
    assert classify_label("right-side-long") == "bullish"
    assert classify_label("resistance reject") == "bearish"
    assert classify_label("sideways") == "range"
    assert summarize_labels(["right_side_long", "support_hold", "neutral"]) == "bullish"
    assert summarize_labels(["right_side_short", "resistance_reject", "range"]) == "bearish"


def test_reason_helpers_preserve_order_and_remove_empty_duplicates():
    assert dedupe_reasons(["  A  ", "", "B", "A", None, "B"]) == ["A", "B"]
    assert join_reasons(["A", "A", "B"], separator=" | ") == "A | B"
    assert merge_reason_groups(["A", "B"], ["B", "C"]) == ["A", "B", "C"]


def test_mode_constants_are_descriptive_only():
    assert MODE_MIXED == "mixed"
    assert is_known_mode("mixed")
    assert not is_known_mode("trend")

