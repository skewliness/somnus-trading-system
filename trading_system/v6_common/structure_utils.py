"""Pure structure-label helpers for V6 extraction work."""

from __future__ import annotations

from collections.abc import Iterable

BULLISH_LABELS = {
    "bull",
    "bullish",
    "long",
    "long_right_side",
    "right_side_long",
    "up",
    "uptrend",
    "support_hold",
}

BEARISH_LABELS = {
    "bear",
    "bearish",
    "short",
    "short_right_side",
    "right_side_short",
    "down",
    "downtrend",
    "resistance_reject",
}

RANGE_LABELS = {
    "range",
    "mixed",
    "neutral",
    "wait",
    "sideways",
    "chop",
}

BROKEN_LABELS = {
    "broken",
    "failed",
    "invalid",
    "breakdown",
    "breakout_failed",
}


def normalize_label(value: str | None) -> str:
    """Normalize a free-form structure label."""

    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def classify_label(value: str | None) -> str:
    """Classify a single structure label."""

    label = normalize_label(value)
    if label in BULLISH_LABELS:
        return "bullish"
    if label in BEARISH_LABELS:
        return "bearish"
    if label in RANGE_LABELS:
        return "range"
    if label in BROKEN_LABELS:
        return "broken"
    return "unknown"


def summarize_labels(labels: Iterable[str | None]) -> str:
    """Summarize several structure labels into one coarse state."""

    counts = {"bullish": 0, "bearish": 0, "range": 0, "broken": 0, "unknown": 0}
    for label in labels:
        counts[classify_label(label)] += 1
    if counts["broken"] and counts["broken"] >= max(counts["bullish"], counts["bearish"], counts["range"]):
        return "broken"
    if counts["bullish"] > counts["bearish"] and counts["bullish"] >= counts["range"]:
        return "bullish"
    if counts["bearish"] > counts["bullish"] and counts["bearish"] >= counts["range"]:
        return "bearish"
    if counts["range"]:
        return "range"
    return "unknown"


def is_directional(state: str | None) -> bool:
    """Return True for bullish or bearish state."""

    return classify_label(state) in {"bullish", "bearish"}

