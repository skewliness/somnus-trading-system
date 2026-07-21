from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class TradingContext:
    """Immutable container that preserves references to the current V6 inputs."""

    symbol: str
    price: float
    position: dict[str, Any] | None
    market_regime: str
    indicators: dict[str, Any]
    features: dict[str, Any]
    config: Any
    signal: Any

    @classmethod
    def from_signal(
        cls,
        signal: Any,
        *,
        config: Any,
        position: dict[str, Any] | None = None,
    ) -> "TradingContext":
        return cls(
            symbol=str(getattr(signal, "symbol", "") or ""),
            price=float(getattr(signal, "price", 0.0) or 0.0),
            position=position,
            market_regime=str(getattr(signal, "side_bias", "") or ""),
            indicators=getattr(signal, "indicators", {}),
            features=getattr(signal, "levels", {}),
            config=config,
            signal=signal,
        )

    def with_position(self, position: dict[str, Any] | None) -> "TradingContext":
        return replace(self, position=position)
