"""Shared close-quantity resolution and close-intent coalescing.

This module is intentionally free of any import from
``trading_system.v6_execution`` so it can be wired into the execution
engine without creating a circular import. Intent objects passed to
:func:`coalesce_close_intents` are duck-typed via ``getattr``.

Design notes / semantics (mirrors v6_execution conventions):

- The exchange ``positionAmt`` (absolute value) from the Binance
  positionRisk row is the single authority for close quantities. The
  local ledger quantity is never used to size a close order — stale
  local state is exactly what caused repeated close submissions and
  residual dust in production.
- Quantities are floored to ``step_size`` using ``Decimal`` arithmetic
  (same approach as ``V6BinanceClient.round_down_to_step``) and
  formatted at step precision with no scientific notation and no float
  artifacts.
- ``min_notional_usdt`` guards partial closes only: a partial
  reduce-only order whose own notional is below the exchange minimum
  would be a useless submission, so it resolves to ``skip_zero``. Full
  flattens ignore ``min_notional_usdt`` — a full reduce-only close must
  always be allowed to go through.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any

# Fraction at or above which a requested close is treated as a full flatten.
FULL_CLOSE_FRACTION_THRESHOLD = 0.90

# Close actions this module coalesces; anything else passes through untouched.
CLOSE_ACTIONS: frozenset[str] = frozenset({"close_long", "close_short"})

# Markers (case-insensitive substring match on target_name / rationale) that
# identify stop-loss / emergency closes. Module-level so the integrator can
# extend the set without touching coalescing logic.
EMERGENCY_CLOSE_MARKERS: frozenset[str] = frozenset({
    "stop",
    "protective",
    "emergency",
    "disaster",
})


@dataclass(frozen=True)
class CloseQuantityDecision:
    quantity: str
    quantity_float: float
    mode: str  # "partial" | "full_flatten" | "skip_zero" | "skip_missing_exchange_position"
    reason: str
    expected_remaining_qty: float
    expected_remaining_notional_usdt: float


def _is_bad_number(value: float) -> bool:
    return math.isnan(value) or math.isinf(value)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _decimal_from_float(value: float) -> Decimal:
    return Decimal(str(value))


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    floored = (value / step).to_integral_value(rounding=ROUND_DOWN) * step
    # Decimal division normalizes exponents (4.7 / 0.001 -> 4.7E+3), so pin
    # the result back to the step's exponent for step-precision formatting
    # (e.g. step 0.001 -> "4.700").
    try:
        return floored.quantize(step, rounding=ROUND_DOWN)
    except InvalidOperation:
        return floored


def _format_quantity(value: Decimal) -> str:
    return format(value, "f")


def _skip(mode: str, reason: str) -> CloseQuantityDecision:
    return CloseQuantityDecision(
        quantity="0",
        quantity_float=0.0,
        mode=mode,
        reason=reason,
        expected_remaining_qty=0.0,
        expected_remaining_notional_usdt=0.0,
    )


def resolve_close_quantity(
    *,
    symbol: str,
    side: str,
    requested_fraction: float,
    local_position: dict[str, Any],
    exchange_position: dict[str, Any] | None,
    mark_price: float,
    step_size: float,
    min_notional_usdt: float,
    dust_notional_usdt: float,
) -> CloseQuantityDecision:
    """Resolve the quantity for a close/flatten order.

    The exchange ``positionAmt`` is authoritative; ``local_position`` is
    accepted only for signature completeness and is deliberately ignored
    for sizing (stale ledgers must never drive close quantities).
    """
    del symbol, side, local_position  # exchange positionAmt is the authority

    step_size = _safe_float(step_size)
    if _is_bad_number(step_size) or step_size <= 0:
        return _skip("skip_zero", "invalid_step_size")

    mark_price = _safe_float(mark_price)
    if _is_bad_number(mark_price) or mark_price <= 0:
        return _skip("skip_zero", "invalid_mark_price")

    requested_fraction = _safe_float(requested_fraction)
    if _is_bad_number(requested_fraction) or requested_fraction < 0:
        return _skip("skip_zero", "invalid_requested_fraction")
    requested_fraction = min(requested_fraction, 1.0)

    min_notional_usdt = _safe_float(min_notional_usdt)
    if _is_bad_number(min_notional_usdt) or min_notional_usdt < 0:
        min_notional_usdt = 0.0
    dust_notional_usdt = _safe_float(dust_notional_usdt)
    if _is_bad_number(dust_notional_usdt) or dust_notional_usdt < 0:
        dust_notional_usdt = 0.0

    if not isinstance(exchange_position, dict) or not exchange_position:
        return _skip("skip_missing_exchange_position", "exchange_position_missing")

    position_amt = _safe_float(exchange_position.get("positionAmt"))
    if _is_bad_number(position_amt):
        position_amt = 0.0
    exchange_qty = abs(position_amt)
    if exchange_qty <= 0:
        return _skip("skip_missing_exchange_position", "exchange_position_flat")

    step_dec = _decimal_from_float(step_size)
    try:
        full_qty_dec = _floor_to_step(_decimal_from_float(exchange_qty), step_dec)
    except (InvalidOperation, ZeroDivisionError):
        return _skip("skip_zero", "invalid_step_size")
    if full_qty_dec <= 0:
        return _skip("skip_missing_exchange_position", "exchange_position_rounds_to_zero_at_step")

    def full_flatten(reason: str) -> CloseQuantityDecision:
        return CloseQuantityDecision(
            quantity=_format_quantity(full_qty_dec),
            quantity_float=float(full_qty_dec),
            mode="full_flatten",
            reason=reason,
            expected_remaining_qty=0.0,
            expected_remaining_notional_usdt=0.0,
        )

    if requested_fraction >= FULL_CLOSE_FRACTION_THRESHOLD:
        return full_flatten("requested_fraction_full_flatten")

    partial_qty_dec = _floor_to_step(
        full_qty_dec * _decimal_from_float(requested_fraction), step_dec
    )
    remaining_dec = full_qty_dec - partial_qty_dec
    remaining_notional = float(remaining_dec) * mark_price

    if dust_notional_usdt > 0 and remaining_notional < dust_notional_usdt:
        return full_flatten("partial_remainder_below_dust_notional_escalated_to_full_flatten")

    if partial_qty_dec <= 0:
        return _skip("skip_zero", "partial_quantity_rounds_to_zero_at_step")

    if min_notional_usdt > 0 and float(partial_qty_dec) * mark_price < min_notional_usdt:
        return _skip("skip_zero", "partial_notional_below_min_notional")

    return CloseQuantityDecision(
        quantity=_format_quantity(partial_qty_dec),
        quantity_float=float(partial_qty_dec),
        mode="partial",
        reason="partial_close",
        expected_remaining_qty=round(float(remaining_dec), 12),
        expected_remaining_notional_usdt=round(remaining_notional, 8),
    )


def is_emergency_close_intent(intent: Any) -> bool:
    text = " ".join(
        str(getattr(intent, attr, "") or "") for attr in ("target_name", "rationale")
    ).lower()
    return any(marker in text for marker in EMERGENCY_CLOSE_MARKERS)


def _intent_fraction(intent: Any) -> float:
    value = _safe_float(getattr(intent, "close_fraction", 0.0))
    if _is_bad_number(value) or value < 0:
        return 0.0
    return value


def _format_fraction(value: float) -> str:
    text = format(_decimal_from_float(round(value, 12)).normalize(), "f")
    return text


def _append_rationale_note(intent: Any, note: str) -> None:
    rationale = str(getattr(intent, "rationale", "") or "")
    intent.rationale = f"{rationale} | {note}" if rationale else note


def _pick_largest_fraction(candidates: list[Any]) -> Any:
    # max() returns the first maximal element, keeping the result deterministic.
    return max(candidates, key=_intent_fraction)


def _coalesce_group(group: list[Any]) -> Any:
    if len(group) == 1:
        return group[0]

    fractions_note = "+".join(_format_fraction(_intent_fraction(item)) for item in group)

    emergencies = [item for item in group if is_emergency_close_intent(item)]
    if emergencies:
        carrier = copy.copy(_pick_largest_fraction(emergencies))
        _append_rationale_note(
            carrier,
            f"coalesced n={len(group)} emergency close prioritized fractions={fractions_note}",
        )
        return carrier

    fulls = [item for item in group if _intent_fraction(item) >= FULL_CLOSE_FRACTION_THRESHOLD]
    if fulls:
        carrier = copy.copy(_pick_largest_fraction(fulls))
        _append_rationale_note(
            carrier,
            f"coalesced n={len(group)} full close prioritized fractions={fractions_note}",
        )
        return carrier

    combined = min(1.0, round(sum(_intent_fraction(item) for item in group), 12))
    carrier = copy.copy(_pick_largest_fraction(group))
    carrier.close_fraction = combined
    _append_rationale_note(carrier, f"coalesced n={len(group)} fractions={fractions_note}")
    return carrier


def coalesce_close_intents(intents: list[Any]) -> list[Any]:
    """Collapse duplicate close intents so at most one close order is
    emitted per (symbol, action) group.

    Non-close intents pass through untouched in their original relative
    order. Each close group is emitted once, at the position of its first
    occurrence. Emergency closes (stop / protective / emergency /
    disaster markers in target_name or rationale) beat take-profits; a
    full close (fraction >= 0.90) beats partials; otherwise staged
    partial fractions are summed (capped at 1.0) onto a copy of the
    largest-fraction intent.
    """
    groups: dict[tuple[str, str], list[Any]] = {}
    layout: list[tuple[str, Any]] = []
    for intent in intents:
        action = str(getattr(intent, "action", "") or "")
        if action not in CLOSE_ACTIONS:
            layout.append(("intent", intent))
            continue
        key = (str(getattr(intent, "symbol", "") or "").upper(), action)
        if key not in groups:
            groups[key] = []
            layout.append(("group", key))
        groups[key].append(intent)

    result: list[Any] = []
    for kind, payload in layout:
        if kind == "intent":
            result.append(payload)
        else:
            result.append(_coalesce_group(groups[payload]))
    return result
