from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class GatePolicyDecision:
    reason: str
    classification: str
    action: str
    notional_multiplier: float
    penalty_score: float = 0.0
    mode: str = "record_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "classification": self.classification,
            "action": self.action,
            "notional_multiplier": self.notional_multiplier,
            "penalty_score": self.penalty_score,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class TransitionGatePolicy:
    """Fable5 transition gate policy.

    Disabled by default. When enabled, selected over-gating reasons are not
    treated as hard rejects; they are recorded and converted into sizing/score
    penalties so shadow or small-size rollout can be tested safely.
    """

    enabled: bool = False
    soft_multiplier: float = 0.75
    sizing_multiplier: float = 0.50
    watch_multiplier: float = 0.50
    hard_reasons: frozenset[str] = field(default_factory=lambda: frozenset({
        "entry_blocked_stale_market_data",
        "technical_entry_only_no_indicator_level",
        "short_blocked_by_smart_signal_short_squeeze",
        "long_blocked_by_smart_signal_long_crowding",
        "short_blocked_market_daily_ma20_reclaim",
        "long_blocked_daily_ma20_loss",
        "generic_indicator_1m_ma105_ma300_not_confirmed",
    }))
    soft_score_reasons: frozenset[str] = field(default_factory=lambda: frozenset({
        "symbol_bb_entry_profile_selected_level_blocked",
        "symbol_bb_entry_profile_selected_level_not_allowed",
        "symbol_bb_entry_profile_no_allowed_level",
        "fib_structure_confirmation_missing",
        "short_support_confluence_hold",
        "long_resistance_confluence_hold",
        "short_low_chase_without_15m_rejection",
        "long_high_chase_without_15m_reclaim",
    }))
    sizing_reasons: frozenset[str] = field(default_factory=lambda: frozenset({
        "macro_direction_mismatch",
        "macro_direction_neutral",
        "short_blocked_15m_ma20_trend_alive",
        "long_blocked_15m_ma20_downtrend_alive",
        "trend_big_blocked_without_4h_right_side",
        "trend_big_blocked_without_1d_ma7_rebreak",
        "short_15m_blocked_by_4h_right_side_long",
        "short_blocked_4h_lower_band_chase",
        "zec_short_blocked_confirmed_4h_low_chase",
    }))
    watch_reasons: frozenset[str] = field(default_factory=lambda: frozenset({
        "four_h_trend_waiting_for_15m_selector",
        "technical_entry_only_unconfirmed_15m_trigger",
        "v5_wait_clean_15m_ma20_pullback",
        "v5_wait_mixed_structure_resolution",
        "v5_long_upper_band_breakout_wait_retest",
        "v5_short_lower_band_breakdown_wait_retest",
        "v5_long_wait_15m_ma20_pullback",
        "v5_short_wait_15m_ma20_retest",
        "v5_long_wait_below_15m_resistance",
        "v5_short_wait_above_15m_support",
    }))

    @classmethod
    def from_env(cls) -> "TransitionGatePolicy":
        return cls(
            enabled=_env_bool("TS_V6_TRANSITION_GATE_POLICY_ENABLED", False),
            soft_multiplier=_env_float("TS_V6_TRANSITION_GATE_SOFT_MULTIPLIER", 0.75),
            sizing_multiplier=_env_float("TS_V6_TRANSITION_GATE_SIZING_MULTIPLIER", 0.50),
            watch_multiplier=_env_float("TS_V6_TRANSITION_GATE_WATCH_MULTIPLIER", 0.50),
        )

    def decision_for(self, block: dict[str, Any] | None) -> GatePolicyDecision | None:
        if not block:
            return None
        reason = str(block.get("reason") or "")
        if not reason or reason in self.hard_reasons:
            return None
        if reason in self.soft_score_reasons:
            return GatePolicyDecision(
                reason=reason,
                classification="soft_score",
                action="downgrade_to_score_penalty",
                notional_multiplier=max(0.05, min(1.0, self.soft_multiplier)),
                penalty_score=-15.0,
                mode="transition_gate",
            )
        if reason in self.sizing_reasons:
            return GatePolicyDecision(
                reason=reason,
                classification="sizing",
                action="allow_with_reduced_size",
                notional_multiplier=max(0.05, min(1.0, self.sizing_multiplier)),
                penalty_score=-25.0,
                mode="transition_gate",
            )
        if reason in self.watch_reasons or bool(block.get("watchable")):
            return GatePolicyDecision(
                reason=reason,
                classification="watch",
                action="allow_as_watchlist_probe",
                notional_multiplier=max(0.05, min(1.0, self.watch_multiplier)),
                penalty_score=-20.0,
                mode="transition_gate",
            )
        return None

    def should_downgrade(self, block: dict[str, Any] | None) -> bool:
        return self.enabled and self.decision_for(block) is not None


def append_gate_decision(levels: dict[str, Any], block: dict[str, Any], decision: GatePolicyDecision) -> None:
    events = levels.get("transition_gate_policy")
    if not isinstance(events, list):
        events = []
    events.append({
        "block": dict(block or {}),
        "decision": decision.to_dict(),
    })
    levels["transition_gate_policy"] = events


def transition_notional_multiplier(levels: dict[str, Any]) -> float:
    events = levels.get("transition_gate_policy")
    if not isinstance(events, list):
        return 1.0
    multipliers: list[float] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
        try:
            multipliers.append(float(decision.get("notional_multiplier") or 1.0))
        except (TypeError, ValueError):
            continue
    if not multipliers:
        return 1.0
    return max(0.05, min(1.0, min(multipliers)))
