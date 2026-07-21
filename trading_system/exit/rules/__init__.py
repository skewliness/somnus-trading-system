from __future__ import annotations

from typing import Any

from trading_system.exit.models import ExitRuleFamily
from trading_system.exit.rules.bb_exit import matches as matches_bb_exit
from trading_system.exit.rules.profit_guard import matches as matches_profit_guard
from trading_system.exit.rules.stop_loss import matches as matches_stop_loss
from trading_system.exit.rules.trend_exit import matches as matches_trend_exit


def classify_exit_intent(intent: Any) -> ExitRuleFamily:
    if matches_stop_loss(intent):
        return ExitRuleFamily.STOP_LOSS
    if matches_profit_guard(intent):
        return ExitRuleFamily.PROFIT_GUARD
    if matches_bb_exit(intent):
        return ExitRuleFamily.BB_EXIT
    if matches_trend_exit(intent):
        return ExitRuleFamily.TREND_EXIT
    return ExitRuleFamily.OTHER


__all__ = ["classify_exit_intent"]
