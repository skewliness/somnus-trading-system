from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from trading_system.trading_context import TradingContext


class ContextDecisionEngine(Protocol):
    def intents_for_context(self, context: TradingContext, state: dict[str, Any]) -> list[Any]: ...


@dataclass(slots=True)
class ReplayDecisionAdapter:
    """Feed replay data through the same context decision path used in production."""

    engine: ContextDecisionEngine

    def decide(self, context: TradingContext, state: dict[str, Any]) -> list[Any]:
        return self.engine.intents_for_context(context, state)
