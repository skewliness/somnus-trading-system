from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    allowed: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RiskRecommendation:
    """Advisory risk output that must never mutate an order decision."""

    risk_score: float
    risk_level: str
    position_recommendation: str
    position_size_multiplier: float
    volatility_adjustment: float
    max_risk_budget_usdt: float
    budget_utilization_pct: float
    advisory_only: bool = True
    reasons: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "position_recommendation": self.position_recommendation,
            "position_size_multiplier": self.position_size_multiplier,
            "volatility_adjustment": self.volatility_adjustment,
            "max_risk_budget_usdt": self.max_risk_budget_usdt,
            "budget_utilization_pct": self.budget_utilization_pct,
            "advisory_only": self.advisory_only,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }
