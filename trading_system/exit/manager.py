from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from trading_system.exit.models import ExitRuleContext, ExitRuleFamily
from trading_system.exit.resolver import resolve_exit_rules, resolve_exit_signals
from trading_system.exit.rules.bb_exit import (
    evaluate_fixed_take_profit,
    evaluate_roi_take_profit,
    evaluate_staged_take_profit,
)
from trading_system.exit.rules.profit_guard import (
    evaluate_fifteen_m_giveback,
    evaluate_h4_lower_reversal,
    evaluate_smart_signal,
    evaluate_v5_no_news,
)
from trading_system.exit.rules import build_exit_signals, classify_exit_intent
from trading_system.exit.rules.stop_loss import evaluate_hard_stop, evaluate_v5_soft_stop
from trading_system.exit.rules.trend_exit import (
    evaluate_daily_reversal,
    evaluate_logic_weakening,
    evaluate_max_hold,
    evaluate_opposite_probability,
)
from trading_system.trading_context import TradingContext


LEGACY_EXIT_SOURCE_SHA256 = "95b4d8c73f3857d0c8559348a704ea64a6cc5d54f9143af9a54a44c3d84ee9a7"

SHORT_EXIT_RULES = (
    evaluate_max_hold,
    evaluate_hard_stop,
    evaluate_v5_soft_stop,
    evaluate_daily_reversal,
    evaluate_logic_weakening,
    evaluate_smart_signal,
    evaluate_v5_no_news,
    evaluate_roi_take_profit,
    evaluate_staged_take_profit,
    evaluate_fixed_take_profit,
    evaluate_opposite_probability,
)

LONG_EXIT_RULES = (
    evaluate_max_hold,
    evaluate_hard_stop,
    evaluate_v5_soft_stop,
    evaluate_daily_reversal,
    evaluate_v5_no_news,
    evaluate_fifteen_m_giveback,
    evaluate_h4_lower_reversal,
    evaluate_logic_weakening,
    evaluate_smart_signal,
    evaluate_roi_take_profit,
    evaluate_staged_take_profit,
    evaluate_fixed_take_profit,
    evaluate_opposite_probability,
)


class ExitManager:
    """Behavior-preserving owner of the V6 exit orchestration boundary."""

    def __init__(self, *, intent_factory: Callable[..., Any], utc_now: Callable[[], str]) -> None:
        self.intent_factory = intent_factory
        self.utc_now = utc_now

    @staticmethod
    def classify_intent(intent: Any) -> ExitRuleFamily:
        return classify_exit_intent(intent)

    def evaluate(
        self,
        engine: Any,
        signal: Any,
        position: dict[str, Any],
        *,
        state: dict[str, Any] | None = None,
    ) -> Any | None:
        context_factory = getattr(engine, "trading_context", None)
        context = (
            context_factory(signal, position)
            if callable(context_factory)
            else TradingContext.from_signal(signal, config=engine.config, position=position)
        )
        return self.evaluate_context(engine, context, state=state)

    def evaluate_context(
        self,
        engine: Any,
        context: TradingContext,
        *,
        state: dict[str, Any] | None = None,
    ) -> Any | None:
        signal = context.signal
        if context.position is None:
            raise TypeError("exit context requires a position")
        position = context.position
        side = str(position.get("side") or "").lower()
        price = float(signal.price)
        stop_price = float(position.get("stop_loss_price") or 0.0)
        take_profit_price = float(position.get("take_profit_price") or 0.0)
        risk_plan = position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {}
        hold_expired = engine.position_max_hold_expired(position)
        profit_tracker = engine.update_position_profit_tracker(signal, position) if side in {"long", "short"} else {}
        if engine.config.stop_only_manual_profit():
            stop_touched = bool(
                stop_price > 0
                and (
                    (side == "short" and price >= stop_price)
                    or (side == "long" and price <= stop_price)
                )
            )
            position["manual_profit_management"] = {
                "enabled": True,
                "mode": "stop_only_manual_profit",
                "updated_at": self.utc_now(),
                "current_price": round(price, 8),
                "stop_loss_price": round(stop_price, 8),
                "reference_take_profit_price": round(take_profit_price, 8),
                "stop_touched": stop_touched,
                "profit_tracker": dict(profit_tracker or {}),
                "note": "V6 may open, scale in, and stop out; profit-taking and reductions are operator managed.",
            }
            if side not in {"long", "short"} or not stop_touched:
                return None
            hold_expired = False
        rule_context = ExitRuleContext(
            engine=engine,
            trading_context=context,
            state=state,
            side=side,
            price=price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            risk_plan=risk_plan,
            profit_tracker=dict(profit_tracker or {}),
            hold_expired=hold_expired,
        )
        selected = resolve_exit_rules(
            rule_context,
            SHORT_EXIT_RULES if side == "short" else LONG_EXIT_RULES,
        )
        if selected is None:
            return None
        reason = selected.reason
        payload = selected.payload
        staged_take_profit = payload.get("staged_take_profit")
        roi_take_profit = payload.get("roi_take_profit")
        profit_guard_take_profit = payload.get("profit_guard_take_profit")
        profit_guard_full_exit = payload.get("profit_guard_full_exit")
        daily_ma7_short_reversal_exit = payload.get("daily_ma7_short_reversal_exit")
        daily_ma20_long_reversal_exit = payload.get("daily_ma20_long_reversal_exit")
        logic_weakening_exit = payload.get("logic_weakening_exit")
        smart_signal_risk_exit = payload.get("smart_signal_risk_exit")
        v5_soft_stop_exit = payload.get("v5_soft_stop_exit")
        if side == "short":
            action = "close_short"
            order_side = "BUY"
        else:
            action = "close_long"
            order_side = "SELL"
        manual_quarantine = position.get("manual_quarantine") if isinstance(position.get("manual_quarantine"), dict) else None
        quarantine_until = engine.parse_utc_datetime((manual_quarantine or {}).get("until"))
        if (manual_quarantine or {}).get("enabled") and quarantine_until and quarantine_until > datetime.now(timezone.utc):
            return None
        retry_block_until = engine.parse_utc_datetime(position.get("close_retry_block_until"))
        if retry_block_until and retry_block_until > datetime.now(timezone.utc) and "stop loss touched" not in reason:
            return None
        partial_risk_plan: dict[str, Any] = {}
        close_fraction = 1.0
        target_name = ""
        if profit_guard_take_profit is not None:
            stages, close_fraction = profit_guard_take_profit["partial"]
            target_name = ",".join(str(stage.get("name") or "") for stage in stages if stage.get("name"))
            total_initial = sum(float(stage.get("fraction_of_initial") or 0.0) for stage in stages)
            partial_type = str(profit_guard_take_profit.get("type") or "profit_giveback_guard")
            partial_risk_plan = {
                "partial_take_profit": {
                    "type": partial_type,
                    "stages": stages,
                    "stage": stages[0],
                    "fraction_of_initial": total_initial,
                    "close_fraction_of_remaining": close_fraction,
                    "remaining_fraction_before": float(position.get("remaining_fraction") or 1.0),
                    "trigger_roi_pct": (profit_guard_take_profit.get("trigger") or {}).get("current_roi_pct"),
                    "trigger_giveback_pct": (profit_guard_take_profit.get("trigger") or {}).get("giveback_pct"),
                    "trigger": dict(profit_guard_take_profit.get("trigger") or {}),
                }
            }
        if staged_take_profit is not None:
            stages, close_fraction = staged_take_profit
            target_name = ",".join(str(stage.get("name") or "") for stage in stages if stage.get("name"))
            total_initial = sum(
                float(stage.get("effective_fraction_of_initial") or stage.get("fraction_of_initial") or 0.0)
                for stage in stages
            )
            partial_risk_plan = {
                "partial_take_profit": {
                    "type": "bb_take_profit",
                    "stages": stages,
                    "stage": stages[0],
                    "fraction_of_initial": total_initial,
                    "close_fraction_of_remaining": close_fraction,
                    "remaining_fraction_before": float(position.get("remaining_fraction") or 1.0),
                }
            }
        if roi_take_profit is not None:
            stages, close_fraction = roi_take_profit
            target_name = ",".join(str(stage.get("name") or "") for stage in stages if stage.get("name"))
            total_initial = sum(float(stage.get("fraction_of_initial") or 0.0) for stage in stages)
            partial_risk_plan = {
                "partial_take_profit": {
                    "type": "roi_take_profit",
                    "stages": stages,
                    "stage": stages[0],
                    "fraction_of_initial": total_initial,
                    "close_fraction_of_remaining": close_fraction,
                    "remaining_fraction_before": float(position.get("remaining_fraction") or 1.0),
                    "roi_thresholds": [float(stage.get("roi_threshold_pct") or 0.0) for stage in stages],
                    "trigger_roi_pct": max(float(stage.get("trigger_roi_pct") or 0.0) for stage in stages),
                }
            }
        if logic_weakening_exit is not None:
            partial_risk_plan = {
                "logic_weakening_exit": logic_weakening_exit,
            }
        if daily_ma7_short_reversal_exit is not None:
            partial_risk_plan = {
                "daily_ma7_short_reversal_exit": daily_ma7_short_reversal_exit,
            }
        if daily_ma20_long_reversal_exit is not None:
            partial_risk_plan = {
                "daily_ma20_long_reversal_exit": daily_ma20_long_reversal_exit,
            }
        if v5_soft_stop_exit is not None:
            partial_risk_plan = {
                "v5_soft_stop_exit": v5_soft_stop_exit,
            }
        if profit_guard_full_exit is not None:
            partial_risk_plan = {
                "profit_guard_full_exit": profit_guard_full_exit,
            }
        intent = self.intent_factory(
            action=action,
            symbol=signal.symbol,
            side=order_side,
            quantity_hint_usdt=float(position.get("notional_usdt") or engine.config.base_notional_for(signal.symbol)),
            leverage=int(position.get("leverage") or engine.config.default_leverage),
            reduce_only=True,
            entry_price=price,
            stop_loss_price=stop_price,
            take_profit_price=take_profit_price,
            long_probability=signal.long_probability,
            short_probability=signal.short_probability,
            wait_probability=signal.wait_probability,
            rationale=reason,
            generated_at=self.utc_now(),
            risk_plan=partial_risk_plan,
            close_fraction=close_fraction,
            target_name=target_name,
        )
        return resolve_exit_signals(build_exit_signals(intent)).intent
