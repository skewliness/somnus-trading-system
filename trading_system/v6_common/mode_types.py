"""Mode names for the future V6 product surface.

The constants are descriptive only. They do not create a strategy selector and
are not imported by the live execution engine in this phase.
"""

from __future__ import annotations

MODE_MIXED = "mixed"
MODE_LONG = "long"
MODE_SHORT = "short"
MODE_RANGE = "range"

V6_MODES = (MODE_MIXED, MODE_LONG, MODE_SHORT, MODE_RANGE)

MODE_DESCRIPTIONS = {
    MODE_MIXED: "Current combined V6 mode; the only active production mode.",
    MODE_LONG: "Future long-biased product mode; not implemented.",
    MODE_SHORT: "Future short-biased product mode; not implemented.",
    MODE_RANGE: "Future range/reversion product mode; not implemented.",
}


def is_known_mode(value: str) -> bool:
    """Return True when value is one of the documented V6 product modes."""

    return value in V6_MODES

