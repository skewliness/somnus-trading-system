from __future__ import annotations

from enum import Enum


class ExitRuleFamily(str, Enum):
    STOP_LOSS = "stop_loss"
    BB_EXIT = "bb_exit"
    PROFIT_GUARD = "profit_guard"
    TREND_EXIT = "trend_exit"
    OTHER = "other"
