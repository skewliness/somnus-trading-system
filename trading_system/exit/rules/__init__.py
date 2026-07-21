from __future__ import annotations

from typing import Any

from trading_system.exit.models import ExitRuleFamily, ExitSignal
from trading_system.exit.rules.bb_exit import evaluate as evaluate_bb_exit
from trading_system.exit.rules.profit_guard import evaluate as evaluate_profit_guard
from trading_system.exit.rules.stop_loss import evaluate as evaluate_stop_loss
from trading_system.exit.rules.trend_exit import evaluate as evaluate_trend_exit
from trading_system.exit.resolver import resolve_exit_signals


def build_exit_signals(intent: Any) -> tuple[ExitSignal, ...]:
    signals = (
        evaluate_stop_loss(intent),
        evaluate_profit_guard(intent),
        evaluate_bb_exit(intent),
        evaluate_trend_exit(intent),
    )
    return signals + (
        ExitSignal(
            triggered=not any(signal.triggered for signal in signals),
            reason=str(getattr(intent, "rationale", "") or ""),
            priority=0,
            family=ExitRuleFamily.OTHER,
            intent=intent,
        ),
    )


def classify_exit_intent(intent: Any) -> ExitRuleFamily:
    return resolve_exit_signals(build_exit_signals(intent)).family


__all__ = ["build_exit_signals", "classify_exit_intent"]
