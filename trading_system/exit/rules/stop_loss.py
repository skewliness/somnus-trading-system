from __future__ import annotations

from typing import Any

from trading_system.exit.models import ExitRuleContext, ExitRuleFamily, ExitSignal


PRIORITY = 400


def matches(intent: Any) -> bool:
    risk_plan = getattr(intent, "risk_plan", {}) if intent is not None else {}
    reason = str(getattr(intent, "rationale", "") or "").lower()
    return bool(
        "stop loss touched" in reason
        or "soft stop" in reason
        or (isinstance(risk_plan, dict) and risk_plan.get("v5_soft_stop_exit"))
    )


def evaluate(intent: Any) -> ExitSignal:
    return ExitSignal(
        triggered=matches(intent),
        reason=str(getattr(intent, "rationale", "") or ""),
        priority=PRIORITY,
        family=ExitRuleFamily.STOP_LOSS,
        intent=intent,
    )


def evaluate_hard_stop(context: ExitRuleContext) -> ExitSignal:
    triggered = bool(
        context.stop_price > 0
        and (
            (context.side == "short" and context.price >= context.stop_price)
            or (context.side != "short" and context.price <= context.stop_price)
        )
    )
    reason = "short stop loss touched" if context.side == "short" else "long stop loss touched"
    return ExitSignal(triggered, reason, 1000, ExitRuleFamily.STOP_LOSS)


def evaluate_v5_soft_stop(context: ExitRuleContext) -> ExitSignal:
    candidate = context.engine.v5_overlay_soft_stop_candidate(
        context.signal,
        context.position,
        context.risk_plan,
    )
    triggered = candidate is not None
    fallback = "short V5 soft stop confirmed" if context.side == "short" else "long V5 soft stop confirmed"
    return ExitSignal(
        triggered,
        str((candidate or {}).get("reason") or fallback),
        990,
        ExitRuleFamily.STOP_LOSS,
        payload={"v5_soft_stop_exit": candidate} if triggered else {},
    )
