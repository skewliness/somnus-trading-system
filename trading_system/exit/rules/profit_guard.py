from __future__ import annotations

from typing import Any

from trading_system.exit.models import ExitRuleContext, ExitRuleFamily, ExitSignal


PRIORITY = 300


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


def evaluate(intent: Any) -> ExitSignal:
    return ExitSignal(
        triggered=matches(intent),
        reason=str(getattr(intent, "rationale", "") or ""),
        priority=PRIORITY,
        family=ExitRuleFamily.PROFIT_GUARD,
        intent=intent,
    )


def _candidate_signal(
    candidate: dict[str, Any] | None,
    *,
    fallback: str,
    full_key: str,
    priority: int,
) -> ExitSignal:
    if candidate is None:
        return ExitSignal(False, "", priority, ExitRuleFamily.PROFIT_GUARD)
    payload_key = "profit_guard_take_profit" if candidate.get("kind") == "partial_take_profit" else full_key
    return ExitSignal(
        True,
        str(candidate.get("reason") or fallback),
        priority,
        ExitRuleFamily.PROFIT_GUARD,
        payload={payload_key: candidate},
    )


def evaluate_smart_signal(context: ExitRuleContext) -> ExitSignal:
    candidate = context.engine.smart_signal_risk_guard_candidate(
        context.signal,
        context.position,
        context.profit_tracker,
    )
    fallback = "short smart-signal risk guard" if context.side == "short" else "long smart-signal risk guard"
    return _candidate_signal(candidate, fallback=fallback, full_key="smart_signal_risk_exit", priority=950)


def evaluate_v5_no_news(context: ExitRuleContext) -> ExitSignal:
    candidate = context.engine.v5_no_news_momentum_take_profit_candidate(
        context.signal,
        context.position,
        context.profit_tracker,
    )
    fallback = (
        "V5 no-news momentum short take profit"
        if context.side == "short"
        else "V5 no-news momentum long take profit"
    )
    return _candidate_signal(candidate, fallback=fallback, full_key="profit_guard_full_exit", priority=945)


def evaluate_fifteen_m_giveback(context: ExitRuleContext) -> ExitSignal:
    candidate = context.engine.fifteen_m_h4_profit_giveback_candidate(
        context.signal,
        context.position,
        context.profit_tracker,
    )
    return _candidate_signal(candidate, fallback="long profit guard", full_key="profit_guard_full_exit", priority=943)


def evaluate_h4_lower_reversal(context: ExitRuleContext) -> ExitSignal:
    candidate = context.engine.h4_lower_reversal_guard_candidate(
        context.signal,
        context.position,
        context.profit_tracker,
    )
    return _candidate_signal(
        candidate,
        fallback="long h4 lower reversal guard",
        full_key="profit_guard_full_exit",
        priority=942,
    )
