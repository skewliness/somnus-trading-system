from __future__ import annotations

from typing import Any

from trading_system.exit.models import ExitRuleFamily, ExitSignal


PRIORITY = 100


def matches(intent: Any) -> bool:
    risk_plan = getattr(intent, "risk_plan", {}) if intent is not None else {}
    reason = str(getattr(intent, "rationale", "") or "").lower()
    trend_keys = {
        "logic_weakening_exit",
        "daily_ma7_short_reversal_exit",
        "daily_ma20_long_reversal_exit",
    }
    return bool(
        (isinstance(risk_plan, dict) and any(risk_plan.get(key) for key in trend_keys))
        or "max hold days reached" in reason
        or "opposite long probability" in reason
        or "opposite short probability" in reason
    )


def evaluate(intent: Any) -> ExitSignal:
    return ExitSignal(
        triggered=matches(intent),
        reason=str(getattr(intent, "rationale", "") or ""),
        priority=PRIORITY,
        family=ExitRuleFamily.TREND_EXIT,
        intent=intent,
    )
