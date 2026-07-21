from __future__ import annotations

from typing import Any

from trading_system.exit.models import ExitRuleContext, ExitRuleFamily, ExitSignal


PRIORITY = 200


def matches(intent: Any) -> bool:
    risk_plan = getattr(intent, "risk_plan", {}) if intent is not None else {}
    partial = risk_plan.get("partial_take_profit") if isinstance(risk_plan, dict) else {}
    partial_type = str((partial or {}).get("type") or "")
    reason = str(getattr(intent, "rationale", "") or "").lower()
    return bool(partial_type in {"bb_take_profit", "roi_take_profit"} or "take profit touched" in reason)


def evaluate(intent: Any) -> ExitSignal:
    return ExitSignal(
        triggered=matches(intent),
        reason=str(getattr(intent, "rationale", "") or ""),
        priority=PRIORITY,
        family=ExitRuleFamily.BB_EXIT,
        intent=intent,
    )


def evaluate_roi_take_profit(context: ExitRuleContext) -> ExitSignal:
    candidate = context.engine.next_roi_take_profit(context.signal, context.position)
    if candidate is None:
        return ExitSignal(False, "", 930, ExitRuleFamily.BB_EXIT)
    stages, close_fraction = candidate
    names = ",".join(str(stage.get("name") or "") for stage in stages)
    total_initial = sum(float(stage.get("fraction_of_initial") or 0.0) for stage in stages)
    max_roi = max(float(stage.get("trigger_roi_pct") or 0.0) for stage in stages)
    reason = (
        f"ROI {context.side} take profit threshold touched: {names} "
        f"roi={max_roi:.2f}% "
        f"close_initial={total_initial:.2f} "
        f"close_remaining={close_fraction:.4f}"
    )
    return ExitSignal(
        True,
        reason,
        930,
        ExitRuleFamily.BB_EXIT,
        payload={"roi_take_profit": candidate},
    )


def evaluate_staged_take_profit(context: ExitRuleContext) -> ExitSignal:
    candidate = context.engine.next_staged_take_profit(context.signal, context.position)
    if candidate is None:
        return ExitSignal(False, "", 920, ExitRuleFamily.BB_EXIT)
    stages, close_fraction = candidate
    names = ",".join(str(stage.get("name") or "") for stage in stages)
    total_initial = sum(
        float(stage.get("effective_fraction_of_initial") or stage.get("fraction_of_initial") or 0.0)
        for stage in stages
    )
    reason = (
        f"staged {context.side} take profit touched with current/delayed 15M volume confirmation: {names} "
        f"close_initial={total_initial:.2f} "
        f"close_remaining={close_fraction:.4f}"
    )
    return ExitSignal(
        True,
        reason,
        920,
        ExitRuleFamily.BB_EXIT,
        payload={"staged_take_profit": candidate},
    )


def evaluate_fixed_take_profit(context: ExitRuleContext) -> ExitSignal:
    has_stages = context.engine.ensure_position_take_profit_stages(context.position, context.signal)
    touched = bool(
        not has_stages
        and context.take_profit_price > 0
        and (
            (context.side == "short" and context.price <= context.take_profit_price)
            or (context.side != "short" and context.price >= context.take_profit_price)
        )
    )
    reason = "short take profit touched" if context.side == "short" else "long take profit touched"
    return ExitSignal(touched, reason, 910, ExitRuleFamily.BB_EXIT)
