from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from trading_system.exit.models import ExitRuleFamily
from trading_system.exit.resolver import resolve_exit_signals
from trading_system.exit.rules import build_exit_signals, classify_exit_intent
from trading_system.trading_context import TradingContext


LEGACY_EXIT_SOURCE_SHA256 = "95b4d8c73f3857d0c8559348a704ea64a6cc5d54f9143af9a54a44c3d84ee9a7"


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
        should_exit = False
        reason = ""
        risk_plan = position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {}
        staged_take_profit: tuple[dict[str, Any], float] | None = None
        roi_take_profit: tuple[dict[str, Any], float] | None = None
        profit_guard_take_profit: dict[str, Any] | None = None
        profit_guard_full_exit: dict[str, Any] | None = None
        daily_ma7_short_reversal_exit: dict[str, Any] | None = None
        daily_ma20_long_reversal_exit: dict[str, Any] | None = None
        logic_weakening_exit: dict[str, Any] | None = None
        smart_signal_risk_exit: dict[str, Any] | None = None
        v5_soft_stop_exit: dict[str, Any] | None = None
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
            # The existing side branch now resolves only the hard stop. Prevent
            # max-hold from replacing the stop reason in the emitted intent.
            hold_expired = False
        if side == "short":
            if hold_expired:
                should_exit = True
                reason = "max hold days reached"
            elif stop_price > 0 and price >= stop_price:
                should_exit = True
                reason = "short stop loss touched"
            elif (v5_soft_stop_exit := engine.v5_overlay_soft_stop_candidate(signal, position, risk_plan)) is not None:
                should_exit = True
                reason = str(v5_soft_stop_exit.get("reason") or "short V5 soft stop confirmed")
            elif (
                daily_ma7_short_reversal_exit := engine.daily_ma7_short_reversal_exit_candidate(
                    signal,
                    position,
                    state=state,
                )
            ) is not None:
                should_exit = True
                reason = str(daily_ma7_short_reversal_exit.get("reason") or "short daily MA7 reversal")
            elif (logic_weakening_exit := engine.logic_weakening_exit_candidate(signal, position)) is not None:
                should_exit = True
                reason = str(logic_weakening_exit.get("reason") or "short original 4H logic weakened")
            elif (smart_signal_risk_exit := engine.smart_signal_risk_guard_candidate(signal, position, profit_tracker)) is not None:
                if smart_signal_risk_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = smart_signal_risk_exit
                    smart_signal_risk_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or smart_signal_risk_exit or {}).get("reason") or "short smart-signal risk guard")
            elif (profit_guard_full_exit := engine.v5_no_news_momentum_take_profit_candidate(signal, position, profit_tracker)) is not None:
                if profit_guard_full_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = profit_guard_full_exit
                    profit_guard_full_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or profit_guard_full_exit or {}).get("reason") or "V5 no-news momentum short take profit")
            elif (roi_take_profit := engine.next_roi_take_profit(signal, position)) is not None:
                stages, close_fraction = roi_take_profit
                names = ",".join(str(stage.get("name") or "") for stage in stages)
                total_initial = sum(float(stage.get("fraction_of_initial") or 0.0) for stage in stages)
                max_roi = max(float(stage.get("trigger_roi_pct") or 0.0) for stage in stages)
                should_exit = True
                reason = (
                    f"ROI short take profit threshold touched: {names} "
                    f"roi={max_roi:.2f}% "
                    f"close_initial={total_initial:.2f} "
                    f"close_remaining={close_fraction:.4f}"
                )
            elif (staged_take_profit := engine.next_staged_take_profit(signal, position)) is not None:
                stages, close_fraction = staged_take_profit
                names = ",".join(str(stage.get("name") or "") for stage in stages)
                total_initial = sum(
                    float(stage.get("effective_fraction_of_initial") or stage.get("fraction_of_initial") or 0.0)
                    for stage in stages
                )
                should_exit = True
                reason = (
                    f"staged short take profit touched with current/delayed 15M volume confirmation: {names} "
                    f"close_initial={total_initial:.2f} "
                    f"close_remaining={close_fraction:.4f}"
                )
            elif not engine.ensure_position_take_profit_stages(position, signal) and take_profit_price > 0 and price <= take_profit_price:
                should_exit = True
                reason = "short take profit touched"
            elif (
                not engine.config.technical_entry_only_for(signal.symbol)
                and signal.long_probability >= engine.opposite_exit_threshold_for_position(position)
            ):
                should_exit = True
                reason = "opposite long probability threshold"
            action = "close_short"
            order_side = "BUY"
        else:
            if hold_expired:
                should_exit = True
                reason = "max hold days reached"
            elif stop_price > 0 and price <= stop_price:
                should_exit = True
                reason = "long stop loss touched"
            elif (v5_soft_stop_exit := engine.v5_overlay_soft_stop_candidate(signal, position, risk_plan)) is not None:
                should_exit = True
                reason = str(v5_soft_stop_exit.get("reason") or "long V5 soft stop confirmed")
            elif (
                daily_ma20_long_reversal_exit := engine.daily_ma20_long_reversal_exit_candidate(
                    signal,
                    position,
                    state=state,
                )
            ) is not None:
                should_exit = True
                reason = str(daily_ma20_long_reversal_exit.get("reason") or "long daily MA20 reversal")
            elif (profit_guard_full_exit := engine.v5_no_news_momentum_take_profit_candidate(signal, position, profit_tracker)) is not None:
                if profit_guard_full_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = profit_guard_full_exit
                    profit_guard_full_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or profit_guard_full_exit or {}).get("reason") or "V5 no-news momentum long take profit")
            elif (profit_guard_full_exit := engine.fifteen_m_h4_profit_giveback_candidate(signal, position, profit_tracker)) is not None:
                if profit_guard_full_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = profit_guard_full_exit
                    profit_guard_full_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or profit_guard_full_exit or {}).get("reason") or "long profit guard")
            elif (profit_guard_full_exit := engine.h4_lower_reversal_guard_candidate(signal, position, profit_tracker)) is not None:
                if profit_guard_full_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = profit_guard_full_exit
                    profit_guard_full_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or profit_guard_full_exit or {}).get("reason") or "long h4 lower reversal guard")
            elif (logic_weakening_exit := engine.logic_weakening_exit_candidate(signal, position)) is not None:
                should_exit = True
                reason = str(logic_weakening_exit.get("reason") or "long original 4H logic weakened")
            elif (smart_signal_risk_exit := engine.smart_signal_risk_guard_candidate(signal, position, profit_tracker)) is not None:
                if smart_signal_risk_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = smart_signal_risk_exit
                    smart_signal_risk_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or smart_signal_risk_exit or {}).get("reason") or "long smart-signal risk guard")
            elif (roi_take_profit := engine.next_roi_take_profit(signal, position)) is not None:
                stages, close_fraction = roi_take_profit
                names = ",".join(str(stage.get("name") or "") for stage in stages)
                total_initial = sum(float(stage.get("fraction_of_initial") or 0.0) for stage in stages)
                max_roi = max(float(stage.get("trigger_roi_pct") or 0.0) for stage in stages)
                should_exit = True
                reason = (
                    f"ROI long take profit threshold touched: {names} "
                    f"roi={max_roi:.2f}% "
                    f"close_initial={total_initial:.2f} "
                    f"close_remaining={close_fraction:.4f}"
                )
            elif (staged_take_profit := engine.next_staged_take_profit(signal, position)) is not None:
                stages, close_fraction = staged_take_profit
                names = ",".join(str(stage.get("name") or "") for stage in stages)
                total_initial = sum(
                    float(stage.get("effective_fraction_of_initial") or stage.get("fraction_of_initial") or 0.0)
                    for stage in stages
                )
                should_exit = True
                reason = (
                    f"staged long take profit touched with current/delayed 15M volume confirmation: {names} "
                    f"close_initial={total_initial:.2f} "
                    f"close_remaining={close_fraction:.4f}"
                )
            elif not engine.ensure_position_take_profit_stages(position, signal) and take_profit_price > 0 and price >= take_profit_price:
                should_exit = True
                reason = "long take profit touched"
            elif (
                not engine.config.technical_entry_only_for(signal.symbol)
                and signal.short_probability >= engine.opposite_exit_threshold_for_position(position)
            ):
                should_exit = True
                reason = "opposite short probability threshold"
            action = "close_long"
            order_side = "SELL"
        if not should_exit:
            return None
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
