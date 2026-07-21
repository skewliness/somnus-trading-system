from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


def _regime_name(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("regime")
    return str(value or "").strip()


def _market_regime_from_signal(signal: Any) -> str:
    explicit = _regime_name(getattr(signal, "market_regime", None))
    if explicit:
        return explicit
    for attribute in ("indicators", "levels"):
        container = getattr(signal, attribute, None)
        if isinstance(container, Mapping):
            nested = _regime_name(container.get("market_regime"))
            if nested:
                return nested
    return "unknown"


@dataclass(frozen=True, slots=True)
class TradingContext:
    """Immutable container that preserves references to the current V6 inputs."""

    symbol: str
    price: float
    position: dict[str, Any] | None
    side_bias: str
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
            side_bias=str(getattr(signal, "side_bias", "") or "neutral"),
            market_regime=_market_regime_from_signal(signal),
            indicators=getattr(signal, "indicators", {}),
            features=getattr(signal, "levels", {}),
            config=config,
            signal=signal,
        )

    def with_position(self, position: dict[str, Any] | None) -> "TradingContext":
        return replace(self, position=position)
