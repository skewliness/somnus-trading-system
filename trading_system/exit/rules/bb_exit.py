from __future__ import annotations

from typing import Any


def matches(intent: Any) -> bool:
    risk_plan = getattr(intent, "risk_plan", {}) if intent is not None else {}
    partial = risk_plan.get("partial_take_profit") if isinstance(risk_plan, dict) else {}
    partial_type = str((partial or {}).get("type") or "")
    reason = str(getattr(intent, "rationale", "") or "").lower()
    return bool(partial_type in {"bb_take_profit", "roi_take_profit"} or "take profit touched" in reason)
