from __future__ import annotations

from typing import Any

from trading_system.exit.models import ExitRuleContext, ExitRuleFamily, ExitSignal


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


def evaluate_max_hold(context: ExitRuleContext) -> ExitSignal:
    return ExitSignal(
        bool(context.hold_expired),
        "max hold days reached",
        1000,
        ExitRuleFamily.TREND_EXIT,
    )


def evaluate_daily_reversal(context: ExitRuleContext) -> ExitSignal:
    if context.side == "short":
        candidate = context.engine.daily_ma7_short_reversal_exit_candidate(
            context.signal,
            context.position,
            state=context.state,
        )
        key = "daily_ma7_short_reversal_exit"
        fallback = "short daily MA7 reversal"
    else:
        candidate = context.engine.daily_ma20_long_reversal_exit_candidate(
            context.signal,
            context.position,
            state=context.state,
        )
        key = "daily_ma20_long_reversal_exit"
        fallback = "long daily MA20 reversal"
    return ExitSignal(
        candidate is not None,
        str((candidate or {}).get("reason") or fallback),
        970,
        ExitRuleFamily.TREND_EXIT,
        payload={key: candidate} if candidate is not None else {},
    )


def evaluate_logic_weakening(context: ExitRuleContext) -> ExitSignal:
    candidate = context.engine.logic_weakening_exit_candidate(context.signal, context.position)
    fallback = "short original 4H logic weakened" if context.side == "short" else "long original 4H logic weakened"
    return ExitSignal(
        candidate is not None,
        str((candidate or {}).get("reason") or fallback),
        940,
        ExitRuleFamily.TREND_EXIT,
        payload={"logic_weakening_exit": candidate} if candidate is not None else {},
    )


def evaluate_opposite_probability(context: ExitRuleContext) -> ExitSignal:
    if context.engine.config.technical_entry_only_for(context.signal.symbol):
        return ExitSignal(False, "", 900, ExitRuleFamily.TREND_EXIT)
    if context.side == "short":
        triggered = context.signal.long_probability >= context.engine.opposite_exit_threshold_for_position(context.position)
        reason = "opposite long probability threshold"
    else:
        triggered = context.signal.short_probability >= context.engine.opposite_exit_threshold_for_position(context.position)
        reason = "opposite short probability threshold"
    return ExitSignal(bool(triggered), reason, 900, ExitRuleFamily.TREND_EXIT)
