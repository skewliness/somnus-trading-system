"""Reason-string helpers for V6 report and extraction work."""

from __future__ import annotations

from collections.abc import Iterable


def normalize_reason(reason: object) -> str:
    """Normalize one reason fragment."""

    return " ".join(str(reason or "").strip().split())


def dedupe_reasons(reasons: Iterable[object]) -> list[str]:
    """De-duplicate reason fragments while preserving order."""

    seen: set[str] = set()
    output: list[str] = []
    for item in reasons:
        text = normalize_reason(item)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def join_reasons(reasons: Iterable[object], *, separator: str = "；") -> str:
    """Join normalized unique reasons into a stable string."""

    return separator.join(dedupe_reasons(reasons))


def merge_reason_groups(*groups: Iterable[object]) -> list[str]:
    """Merge multiple reason groups into one de-duplicated list."""

    merged: list[object] = []
    for group in groups:
        merged.extend(group)
    return dedupe_reasons(merged)

