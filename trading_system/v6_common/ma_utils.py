"""Pure moving-average helpers for V6 extraction work."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class MaDistance:
    """Price position relative to a moving average."""

    price: float
    ma: float
    distance_pct: float
    relation: str


def _valid_number(value: float | int | None) -> bool:
    return value is not None and isfinite(float(value))


def distance_pct(price: float, ma: float) -> float:
    """Return percentage distance from MA.

    Positive means price is above MA. Negative means price is below MA.
    """

    if not _valid_number(price) or not _valid_number(ma) or float(ma) == 0.0:
        raise ValueError("price and ma must be finite numbers and ma must not be zero")
    return (float(price) - float(ma)) / float(ma) * 100.0


def price_relation(price: float, ma: float, *, tolerance_pct: float = 0.0) -> str:
    """Classify price as above, below, or near an MA."""

    pct = distance_pct(price, ma)
    tol = abs(float(tolerance_pct))
    if pct > tol:
        return "above"
    if pct < -tol:
        return "below"
    return "near"


def describe_distance(price: float, ma: float, *, tolerance_pct: float = 0.0) -> MaDistance:
    """Return a typed MA distance summary."""

    pct = distance_pct(price, ma)
    return MaDistance(
        price=float(price),
        ma=float(ma),
        distance_pct=pct,
        relation=price_relation(price, ma, tolerance_pct=tolerance_pct),
    )


def slope_direction(previous: float, current: float, *, tolerance_pct: float = 0.0) -> str:
    """Classify MA slope as up, down, or flat."""

    if not _valid_number(previous) or not _valid_number(current):
        raise ValueError("previous and current must be finite numbers")
    previous_f = float(previous)
    current_f = float(current)
    if previous_f == 0.0:
        delta_pct = 0.0 if current_f == 0.0 else float("inf")
    else:
        delta_pct = (current_f - previous_f) / abs(previous_f) * 100.0
    tol = abs(float(tolerance_pct))
    if delta_pct > tol:
        return "up"
    if delta_pct < -tol:
        return "down"
    return "flat"


def held_above_ma(closes: list[float], ma_values: list[float], *, candles: int = 2) -> bool:
    """Return True when the latest N closes are at or above their MA values."""

    if candles <= 0:
        raise ValueError("candles must be positive")
    if len(closes) < candles or len(ma_values) < candles:
        return False
    return all(float(c) >= float(m) for c, m in zip(closes[-candles:], ma_values[-candles:]))


def broke_below_ma(closes: list[float], ma_values: list[float], *, candles: int = 1) -> bool:
    """Return True when the latest N closes are below their MA values."""

    if candles <= 0:
        raise ValueError("candles must be positive")
    if len(closes) < candles or len(ma_values) < candles:
        return False
    return all(float(c) < float(m) for c, m in zip(closes[-candles:], ma_values[-candles:]))


def reclaimed_ma(previous_close: float, current_close: float, current_ma: float) -> bool:
    """Return True when price moves from below an MA to at/above it."""

    return float(previous_close) < float(current_ma) <= float(current_close)


def rejected_at_ma(high: float, close: float, ma: float) -> bool:
    """Return True when price traded above MA but closed below it."""

    return float(high) >= float(ma) and float(close) < float(ma)

