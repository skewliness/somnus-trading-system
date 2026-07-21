from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from trading_system.close_resolver import (
    CloseQuantityDecision,
    EMERGENCY_CLOSE_MARKERS,
    coalesce_close_intents,
    resolve_close_quantity,
)


@dataclass
class FakeIntent:
    symbol: str = "BTCUSDT"
    action: str = "close_long"
    close_fraction: float = 1.0
    risk_plan: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    target_name: str = ""


def resolve(**overrides: Any) -> CloseQuantityDecision:
    kwargs: dict[str, Any] = dict(
        symbol="BTCUSDT",
        side="long",
        requested_fraction=1.0,
        local_position={"position_qty": 5.0},
        exchange_position={"positionAmt": "4.7"},
        mark_price=60000.0,
        step_size=0.001,
        min_notional_usdt=5.0,
        dust_notional_usdt=20.0,
    )
    kwargs.update(overrides)
    return resolve_close_quantity(**kwargs)


def test_partial_close_leaving_dust_escalates_to_full_flatten() -> None:
    # 0.010 BTC @ 60k; 85% close would leave 0.002 (=120 USDT) < dust 150.
    decision = resolve(
        requested_fraction=0.85,
        exchange_position={"positionAmt": "0.010"},
        dust_notional_usdt=150.0,
    )
    assert decision.mode == "full_flatten"
    assert "dust" in decision.reason
    assert decision.quantity == "0.010"
    assert decision.quantity_float == 0.01
    assert decision.expected_remaining_qty == 0.0
    assert decision.expected_remaining_notional_usdt == 0.0


def test_full_close_uses_exchange_position_amt_not_stale_local() -> None:
    decision = resolve(
        requested_fraction=1.0,
        local_position={"position_qty": 5.0, "remaining_quantity": 5.0},
        exchange_position={"positionAmt": "4.7"},
    )
    assert decision.mode == "full_flatten"
    assert decision.quantity == "4.700"
    assert decision.quantity_float == 4.7


def test_short_side_negative_position_amt_uses_absolute_value() -> None:
    decision = resolve(side="short", exchange_position={"positionAmt": "-4.7"})
    assert decision.mode == "full_flatten"
    assert decision.quantity == "4.700"


def test_missing_exchange_position_never_submits() -> None:
    decision = resolve(exchange_position=None)
    assert decision.mode == "skip_missing_exchange_position"
    assert decision.quantity == "0"
    assert decision.quantity_float == 0.0
    assert decision.expected_remaining_qty == 0.0
    assert decision.expected_remaining_notional_usdt == 0.0


def test_flat_or_step_zero_exchange_position_skips_as_missing() -> None:
    flat = resolve(exchange_position={"positionAmt": "0.000"})
    assert flat.mode == "skip_missing_exchange_position"

    rounds_to_zero = resolve(exchange_position={"positionAmt": "0.0004"}, step_size=0.001)
    assert rounds_to_zero.mode == "skip_missing_exchange_position"
    assert rounds_to_zero.quantity == "0"


def test_partial_quantity_rounding_to_zero_skips() -> None:
    # 10% of 0.005 = 0.0005 -> floors to 0 at step 0.001. Remaining stays
    # well above the dust threshold so no escalation happens.
    decision = resolve(
        requested_fraction=0.1,
        exchange_position={"positionAmt": "0.005"},
        dust_notional_usdt=20.0,
    )
    assert decision.mode == "skip_zero"
    assert decision.quantity == "0"


def test_fraction_at_or_above_090_is_full_flatten() -> None:
    decision = resolve(requested_fraction=0.90)
    assert decision.mode == "full_flatten"
    assert decision.quantity == "4.700"
    assert decision.expected_remaining_qty == 0.0

    below = resolve(requested_fraction=0.5, dust_notional_usdt=0.0)
    assert below.mode == "partial"


def test_partial_close_fills_expected_remaining() -> None:
    decision = resolve(
        requested_fraction=0.5,
        exchange_position={"positionAmt": "4.000"},
        mark_price=100.0,
        dust_notional_usdt=10.0,
    )
    assert decision.mode == "partial"
    assert decision.quantity == "2.000"
    assert decision.quantity_float == 2.0
    assert decision.expected_remaining_qty == 2.0
    assert decision.expected_remaining_notional_usdt == 200.0


def test_step_size_formatting_no_artifacts_or_scientific_notation() -> None:
    # step 0.001 keeps three decimals with no trailing garbage.
    decision = resolve(exchange_position={"positionAmt": "4.7"}, step_size=0.001)
    assert decision.quantity == "4.700"

    # A float-artifact fraction (0.1 + 0.2) must not leak into the string.
    artifact = resolve(
        requested_fraction=0.1 + 0.2,
        exchange_position={"positionAmt": "1.0"},
        step_size=0.1,
        mark_price=1000.0,
        min_notional_usdt=0.0,
        dust_notional_usdt=0.0,
    )
    assert artifact.quantity == "0.3"
    assert "e" not in artifact.quantity.lower()

    # Tiny step sizes must not produce scientific notation.
    tiny = resolve(
        exchange_position={"positionAmt": "0.00012"},
        step_size=1e-05,
        mark_price=100000.0,
    )
    assert tiny.quantity == "0.00012"
    assert "e" not in tiny.quantity.lower()


def test_defensive_inputs_resolve_to_skip_zero() -> None:
    zero_mark = resolve(mark_price=0.0)
    assert zero_mark.mode == "skip_zero"
    assert zero_mark.reason == "invalid_mark_price"

    zero_step = resolve(step_size=0.0)
    assert zero_step.mode == "skip_zero"
    assert zero_step.reason == "invalid_step_size"

    negative_step = resolve(step_size=-0.001)
    assert negative_step.mode == "skip_zero"

    nan_fraction = resolve(requested_fraction=math.nan)
    assert nan_fraction.mode == "skip_zero"
    assert nan_fraction.reason == "invalid_requested_fraction"

    negative_fraction = resolve(requested_fraction=-0.5)
    assert negative_fraction.mode == "skip_zero"

    nan_mark = resolve(mark_price=math.nan)
    assert nan_mark.mode == "skip_zero"


def test_partial_below_min_notional_skips() -> None:
    # 10% of 0.010 BTC @ 1000 = 1 USDT order < min notional 5.
    decision = resolve(
        requested_fraction=0.1,
        exchange_position={"positionAmt": "0.010"},
        mark_price=1000.0,
        min_notional_usdt=5.0,
        dust_notional_usdt=0.0,
    )
    assert decision.mode == "skip_zero"
    assert decision.reason == "partial_notional_below_min_notional"


def test_repeated_tp_triggers_coalesce_into_one_intent() -> None:
    intents = [
        FakeIntent(close_fraction=0.25, rationale="tp stage 1", target_name="tp1"),
        FakeIntent(close_fraction=0.25, rationale="tp stage 2", target_name="tp2"),
        FakeIntent(close_fraction=0.5, rationale="tp stage 3", target_name="tp3"),
    ]
    result = coalesce_close_intents(intents)
    assert len(result) == 1
    merged = result[0]
    assert merged.close_fraction == 1.0
    assert "coalesced n=3" in merged.rationale
    assert "0.25+0.25+0.5" in merged.rationale
    # Carrier is the largest-fraction intent, returned as a copy.
    assert merged.target_name == "tp3"
    assert merged is not intents[2]
    # Originals are never mutated.
    assert intents[2].close_fraction == 0.5
    assert intents[2].rationale == "tp stage 3"


def test_emergency_protective_stop_beats_partial_tp() -> None:
    intents = [
        FakeIntent(close_fraction=0.25, rationale="take profit leg", target_name="tp1"),
        FakeIntent(close_fraction=1.0, rationale="protective stop hit", target_name="h4_protective_stop"),
    ]
    result = coalesce_close_intents(intents)
    assert len(result) == 1
    winner = result[0]
    assert winner.target_name == "h4_protective_stop"
    assert winner.close_fraction == 1.0
    assert "coalesced n=2" in winner.rationale


def test_coalesce_float_artifact_fractions_are_clean() -> None:
    intents = [
        FakeIntent(close_fraction=0.1, target_name="tp1"),
        FakeIntent(close_fraction=0.2, target_name="tp2"),
    ]
    merged = coalesce_close_intents(intents)[0]
    assert merged.close_fraction == 0.3  # not 0.30000000000000004
    assert "0.1+0.2" in merged.rationale


def test_coalesce_groups_by_symbol_and_action() -> None:
    intents = [
        FakeIntent(symbol="BTCUSDT", action="close_long", close_fraction=0.25),
        FakeIntent(symbol="ETHUSDT", action="close_long", close_fraction=0.25),
        FakeIntent(symbol="BTCUSDT", action="close_short", close_fraction=0.25),
        FakeIntent(symbol="BTCUSDT", action="close_long", close_fraction=0.25),
    ]
    result = coalesce_close_intents(intents)
    assert len(result) == 3
    assert [(item.symbol, item.action) for item in result] == [
        ("BTCUSDT", "close_long"),
        ("ETHUSDT", "close_long"),
        ("BTCUSDT", "close_short"),
    ]
    assert result[0].close_fraction == 0.5


def test_non_close_intents_pass_through_untouched_in_order() -> None:
    open_intent = FakeIntent(action="open_long", close_fraction=0.0)
    close_a = FakeIntent(close_fraction=0.25, target_name="tp1")
    close_b = FakeIntent(close_fraction=0.25, target_name="tp2")
    other = FakeIntent(symbol="ETHUSDT", action="open_short", close_fraction=0.0)
    result = coalesce_close_intents([open_intent, close_a, other, close_b])
    assert len(result) == 3
    assert result[0] is open_intent
    assert result[2] is other
    assert result[1].close_fraction == 0.5


def test_singleton_close_group_passes_through_unmodified() -> None:
    only = FakeIntent(close_fraction=0.25, rationale="tp stage 1")
    result = coalesce_close_intents([only])
    assert result == [only]
    assert result[0] is only
    assert only.rationale == "tp stage 1"


def test_emergency_markers_constant_is_extensible_frozenset() -> None:
    assert isinstance(EMERGENCY_CLOSE_MARKERS, frozenset)
    assert {"stop", "protective", "emergency", "disaster"} <= EMERGENCY_CLOSE_MARKERS
