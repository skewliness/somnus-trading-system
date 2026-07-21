from __future__ import annotations

from typing import Any


def matches(intent: Any) -> bool:
    risk_plan = getattr(intent, "risk_plan", {}) if intent is not None else {}
    partial = risk_plan.get("partial_take_profit") if isinstance(risk_plan, dict) else {}
    partial_type = str((partial or {}).get("type") or "")
    reason = str(getattr(intent, "rationale", "") or "").lower()
    return bool(
        partial_type not in {"", "bb_take_profit", "roi_take_profit"}
        or (isinstance(risk_plan, dict) and risk_plan.get("profit_guard_full_exit"))
        or "profit guard" in reason
        or "smart-signal risk guard" in reason
        or "no-news momentum" in reason
    )
