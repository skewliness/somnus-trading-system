from __future__ import annotations

from dataclasses import dataclass, field
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
    intent: Any = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExitRuleContext:
    """Shared immutable-by-convention input for one ordered exit evaluation."""

    engine: Any
    trading_context: Any
    state: dict[str, Any] | None
    side: str
    price: float
    stop_price: float
    take_profit_price: float
    risk_plan: dict[str, Any]
    profit_tracker: dict[str, Any]
    hold_expired: bool

    @property
    def signal(self) -> Any:
        return self.trading_context.signal

    @property
    def position(self) -> dict[str, Any]:
        position = self.trading_context.position
        if position is None:
            raise TypeError("exit context requires a position")
        return position
