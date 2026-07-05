"""Pure Bollinger-band helpers for V6 extraction work."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class BandSnapshot:
    """Price position against a Bollinger band."""

    price: float
    upper: float
    mid: float
    lower: float
    position: float
    nearest: str
    width_pct: float


def _valid_number(value: float | int | None) -> bool:
    return value is not None and isfinite(float(value))


def validate_band(upper: float, mid: float, lower: float) -> tuple[float, float, float]:
    """Validate and normalize BB values."""

    if not all(_valid_number(v) for v in (upper, mid, lower)):
        raise ValueError("band values must be finite numbers")
    upper_f, mid_f, lower_f = float(upper), float(mid), float(lower)
    if upper_f < lower_f:
        raise ValueError("upper must be greater than or equal to lower")
    return upper_f, mid_f, lower_f


def band_position(price: float, upper: float, mid: float, lower: float) -> float:
    """Return normalized position inside the band.

    0 means lower band, 0.5 means middle band, and 1 means upper band.
    Values can be below 0 or above 1 when price is outside the band.
    """

    if not _valid_number(price):
        raise ValueError("price must be a finite number")
    upper_f, _mid_f, lower_f = validate_band(upper, mid, lower)
    width = upper_f - lower_f
    if width == 0.0:
        return 0.5
    return (float(price) - lower_f) / width


def distance_pct(price: float, level: float) -> float:
    """Return percentage distance from a BB level."""

    if not _valid_number(price) or not _valid_number(level) or float(level) == 0.0:
        raise ValueError("price and level must be finite numbers and level must not be zero")
    return (float(price) - float(level)) / float(level) * 100.0


def nearest_band(price: float, upper: float, mid: float, lower: float) -> str:
    """Return the nearest band name: upper, mid, or lower."""

    upper_f, mid_f, lower_f = validate_band(upper, mid, lower)
    price_f = float(price)
    distances = {
        "upper": abs(price_f - upper_f),
        "mid": abs(price_f - mid_f),
        "lower": abs(price_f - lower_f),
    }
    return min(distances, key=distances.get)


def width_pct(upper: float, mid: float, lower: float) -> float:
    """Return BB width as percentage of middle band."""

    upper_f, mid_f, lower_f = validate_band(upper, mid, lower)
    if mid_f == 0.0:
        raise ValueError("mid must not be zero")
    return (upper_f - lower_f) / abs(mid_f) * 100.0


def snapshot(price: float, upper: float, mid: float, lower: float) -> BandSnapshot:
    """Return a typed BB position summary."""

    upper_f, mid_f, lower_f = validate_band(upper, mid, lower)
    return BandSnapshot(
        price=float(price),
        upper=upper_f,
        mid=mid_f,
        lower=lower_f,
        position=band_position(price, upper_f, mid_f, lower_f),
        nearest=nearest_band(price, upper_f, mid_f, lower_f),
        width_pct=width_pct(upper_f, mid_f, lower_f),
    )


def near_band(price: float, level: float, *, tolerance_pct: float) -> bool:
    """Return True when price is within tolerance of the given band level."""

    return abs(distance_pct(price, level)) <= abs(float(tolerance_pct))


def width_state(previous_width_pct: float, current_width_pct: float, *, tolerance_pct: float = 0.0) -> str:
    """Classify BB width as expanding, compressing, or flat."""

    if not _valid_number(previous_width_pct) or not _valid_number(current_width_pct):
        raise ValueError("width values must be finite numbers")
    delta = float(current_width_pct) - float(previous_width_pct)
    tol = abs(float(tolerance_pct))
    if delta > tol:
        return "expanding"
    if delta < -tol:
        return "compressing"
    return "flat"

