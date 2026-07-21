"""Resolve close quantities and collapse duplicate close intents.

The exchange position is authoritative. Local ledger quantities are retained
for audit only and are never used to size a reduce-only order.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any


FULL_CLOSE_FRACTION_THRESHOLD = 0.90
CLOSE_ACTIONS = frozenset({"close_long", "close_short"})
EMERGENCY_CLOSE_MARKERS = frozenset({"stop", "protective", "emergency", "disaster"})


@dataclass(frozen=True, slots=True)
class CloseQuantityDecision:
    quantity: str
    quantity_float: float
    mode: str
    reason: str
    expected_remaining_qty: float
    expected_remaining_notional_usdt: float


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _invalid_number(value: float) -> bool:
    return math.isnan(value) or math.isinf(value)


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    floored = (value / step).to_integral_value(rounding=ROUND_DOWN) * step
    try:
        return floored.quantize(step, rounding=ROUND_DOWN)
    except InvalidOperation:
        return floored


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
    requested_fraction: float,
    exchange_position: dict[str, Any] | None,
    mark_price: float,
    step_size: float,
    min_notional_usdt: float,
    dust_notional_usdt: float,
    symbol: str = "",
    side: str = "",
    local_position: dict[str, Any] | None = None,
) -> CloseQuantityDecision:
    # These fields complete the public resolver contract used by standalone
    # callers. They are intentionally ignored: exchange positionAmt is the
    # only quantity authority for reduce-only orders.
    del symbol, side, local_position

    step_size = _safe_float(step_size)
    mark_price = _safe_float(mark_price)
    requested_fraction = _safe_float(requested_fraction)
    min_notional_usdt = _safe_float(min_notional_usdt)
    dust_notional_usdt = _safe_float(dust_notional_usdt)

    if _invalid_number(step_size) or step_size <= 0:
        return _skip("skip_zero", "invalid_step_size")
    if _invalid_number(mark_price) or mark_price <= 0:
        return _skip("skip_zero", "invalid_mark_price")
    if _invalid_number(requested_fraction) or requested_fraction <= 0:
        return _skip("skip_zero", "invalid_requested_fraction")
    requested_fraction = min(requested_fraction, 1.0)
    if _invalid_number(min_notional_usdt) or min_notional_usdt < 0:
        min_notional_usdt = 0.0
    if _invalid_number(dust_notional_usdt) or dust_notional_usdt < 0:
        dust_notional_usdt = 0.0

    if not isinstance(exchange_position, dict) or not exchange_position:
        return _skip("skip_missing_exchange_position", "exchange_position_missing")
    position_amt = _safe_float(exchange_position.get("positionAmt"))
    if _invalid_number(position_amt) or abs(position_amt) <= 0:
        return _skip("skip_missing_exchange_position", "exchange_position_flat")

    step = _decimal(step_size)
    try:
        full_quantity = _floor_to_step(_decimal(abs(position_amt)), step)
    except (InvalidOperation, ZeroDivisionError):
        return _skip("skip_zero", "invalid_step_size")
    if full_quantity <= 0:
        return _skip("skip_missing_exchange_position", "exchange_position_rounds_to_zero_at_step")

    def full_flatten(reason: str) -> CloseQuantityDecision:
        return CloseQuantityDecision(
            quantity=format(full_quantity, "f"),
            quantity_float=float(full_quantity),
            mode="full_flatten",
            reason=reason,
            expected_remaining_qty=0.0,
            expected_remaining_notional_usdt=0.0,
        )

    if requested_fraction >= FULL_CLOSE_FRACTION_THRESHOLD:
        return full_flatten("requested_fraction_full_flatten")

    partial_quantity = _floor_to_step(full_quantity * _decimal(requested_fraction), step)
    remaining_quantity = full_quantity - partial_quantity
    remaining_notional = float(remaining_quantity) * mark_price
    if dust_notional_usdt > 0 and remaining_notional < dust_notional_usdt:
        return full_flatten("partial_remainder_below_dust_notional_escalated_to_full_flatten")
    if partial_quantity <= 0:
        return _skip("skip_zero", "partial_quantity_rounds_to_zero_at_step")
    if min_notional_usdt > 0 and float(partial_quantity) * mark_price < min_notional_usdt:
        return _skip("skip_zero", "partial_notional_below_min_notional")

    return CloseQuantityDecision(
        quantity=format(partial_quantity, "f"),
        quantity_float=float(partial_quantity),
        mode="partial",
        reason="partial_close",
        expected_remaining_qty=round(float(remaining_quantity), 12),
        expected_remaining_notional_usdt=round(remaining_notional, 8),
    )


def _intent_fraction(intent: Any) -> float:
    fraction = _safe_float(getattr(intent, "close_fraction", 0.0))
    if _invalid_number(fraction) or fraction < 0:
        return 0.0
    return min(fraction, 1.0)


def _is_emergency_close(intent: Any) -> bool:
    text = " ".join(str(getattr(intent, field, "") or "") for field in ("target_name", "rationale")).lower()
    return any(marker in text for marker in EMERGENCY_CLOSE_MARKERS)


def _append_note(intent: Any, note: str) -> None:
    rationale = str(getattr(intent, "rationale", "") or "")
    intent.rationale = f"{rationale} | {note}" if rationale else note


def _coalesce_group(group: list[Any]) -> Any:
    if len(group) == 1:
        return group[0]
    fractions = "+".join(format(_decimal(round(_intent_fraction(item), 12)).normalize(), "f") for item in group)
    emergency = [item for item in group if _is_emergency_close(item)]
    if emergency:
        carrier = copy.copy(max(emergency, key=_intent_fraction))
        _append_note(carrier, f"coalesced n={len(group)} emergency close prioritized fractions={fractions}")
        return carrier
    full = [item for item in group if _intent_fraction(item) >= FULL_CLOSE_FRACTION_THRESHOLD]
    if full:
        carrier = copy.copy(max(full, key=_intent_fraction))
        _append_note(carrier, f"coalesced n={len(group)} full close prioritized fractions={fractions}")
        return carrier
    carrier = copy.copy(max(group, key=_intent_fraction))
    carrier.close_fraction = min(1.0, round(sum(_intent_fraction(item) for item in group), 12))
    _append_note(carrier, f"coalesced n={len(group)} fractions={fractions}")
    return carrier


def coalesce_close_intents(intents: list[Any]) -> list[Any]:
    """Emit at most one close intent for each symbol and close action."""
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
        result.append(payload if kind == "intent" else _coalesce_group(groups[payload]))
    return result
