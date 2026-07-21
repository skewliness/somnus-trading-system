from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExitRuleFamily(str, Enum):
    STOP_LOSS = "stop_loss"
    BB_EXIT = "bb_exit"
    PROFIT_GUARD = "profit_guard"
    TREND_EXIT = "trend_exit"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ExitSignal:
    """Structured result emitted by an exit rule adapter."""

    triggered: bool
    reason: str
    priority: int
    family: ExitRuleFamily
    intent: Any
