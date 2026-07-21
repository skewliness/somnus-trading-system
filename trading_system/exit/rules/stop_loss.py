from __future__ import annotations

from typing import Any


def matches(intent: Any) -> bool:
    risk_plan = getattr(intent, "risk_plan", {}) if intent is not None else {}
    reason = str(getattr(intent, "rationale", "") or "").lower()
    return bool(
        "stop loss touched" in reason
        or "soft stop" in reason
        or (isinstance(risk_plan, dict) and risk_plan.get("v5_soft_stop_exit"))
    )
