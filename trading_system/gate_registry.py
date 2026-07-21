from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VALID_GATE_CLASSIFICATIONS = {"hard", "watch", "soft", "support"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class GateDefinition:
    reason: str
    classification: str
    description: str
    notional_multiplier: float = 1.0
    risk_exit: bool = False

    def __post_init__(self) -> None:
        if self.classification not in VALID_GATE_CLASSIFICATIONS:
            raise ValueError(f"unsupported gate classification: {self.classification}")
        if not 0.0 <= float(self.notional_multiplier) <= 1.0:
            raise ValueError("gate notional_multiplier must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GateRegistry:
    """Auditable catalog for V6 entry gates.

    Runtime predicates remain in the execution engine because they need live
    signal state. This registry owns their policy classification, sizing
    defaults, and risk-exit semantics so the policy is no longer hidden in a
    long serial chain of ad-hoc ``if`` statements.
    """

    def __init__(self, definitions: Iterable[GateDefinition] = ()) -> None:
        self._definitions: dict[str, GateDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: GateDefinition) -> None:
        reason = str(definition.reason or "").strip()
        if not reason:
            raise ValueError("gate reason is required")
        self._definitions[reason] = definition

    def get(self, reason: str) -> GateDefinition | None:
        return self._definitions.get(str(reason or "").strip())

    def resolve(
        self,
        reason: str,
        *,
        fallback_classification: str,
        fallback_multiplier: float = 1.0,
    ) -> GateDefinition:
        known = self.get(reason)
        if known is not None:
            return known
        return GateDefinition(
            reason=str(reason or "entry_gate"),
            classification=fallback_classification,
            description="Unregistered runtime gate; caller classification retained.",
            notional_multiplier=max(0.0, min(1.0, float(fallback_multiplier))),
        )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {reason: definition.to_dict() for reason, definition in sorted(self._definitions.items())}


def default_entry_gate_registry() -> GateRegistry:
    hard = {
        "entry_blocked_stale_market_data": "Market data is stale or incomplete.",
        "technical_entry_only_no_indicator_level": "No auditable technical entry level exists.",
        "mixed_structure_macro_direction_mismatch": (
            "Long and short structures are both active; only the macro-aligned side may enter."
        ),
        "mixed_structure_macro_direction_neutral": (
            "Long and short structures are both active, but the macro direction is neutral."
        ),
        "mixed_structure_probability_fallback_loss_streak": (
            "Probability fallback is paused after three consecutive losing completed trades."
        ),
        "short_blocked_by_smart_signal_short_squeeze": "Short-squeeze structure creates asymmetric short risk.",
        "long_blocked_by_smart_signal_long_crowding": "Crowded-long structure creates asymmetric long risk.",
        "short_blocked_market_daily_ma20_reclaim": "Daily short thesis has been invalidated.",
        "long_blocked_daily_ma20_loss": "Daily long thesis has been invalidated.",
        "manager_reentry_lock_active": "The position manager invalidated this side; a fresh same-side structure confirmation is required.",
        "short_blocked_by_15m_bb96_210_lower_second_test_support": "Confirmed second-test support opposes the short.",
        "long_blocked_by_15m_bb96_210_upper_second_test_pressure": "Confirmed second-test pressure opposes the long.",
    }
    watch = {
        "manager_reversal_observation_active": (
            "A 4H manager exit occurred; wait for a complete new 4H candle, opposite right-side confirmation, and a 15M selector."
        ),
        "mixed_structure_cross_timeframe_4h_direction_active": (
            "Daily and 4H directions conflict; retain the active 4H side until two closed 4H candles weaken it."
        ),
        "mixed_structure_cross_timeframe_4h_logic_weakened": (
            "The active 4H thesis weakened for two closed 4H candles; stop adding in the old direction."
        ),
        "mixed_structure_cross_timeframe_wait_key_structure_break": (
            "The 4H thesis weakened, but the opposite side still needs a confirmed 15M/4H structure break and retest."
        ),
        "four_h_trend_waiting_for_15m_selector": "4H zone exists but the 15M execution selector is not ready.",
        "technical_entry_only_unconfirmed_15m_trigger": "The selected 15M level has not been touched or confirmed.",
        "short_low_chase_without_15m_rejection": "Short would chase into nearby support without rejection.",
        "long_high_chase_without_15m_reclaim": "Long would chase into nearby resistance without reclaim.",
        "generic_indicator_1m_ma105_ma300_not_confirmed": "1M timing confirmation is absent.",
        "v5_wait_clean_15m_ma20_pullback": "V5 needs a clean 15M MA20 pullback.",
        "v5_wait_mixed_structure_resolution": "V5 structure remains mixed.",
        "v5_long_upper_band_breakout_wait_retest": "V5 long must wait for an upper-band retest.",
        "v5_short_lower_band_breakdown_wait_retest": "V5 short must wait for a lower-band retest.",
        "v5_long_wait_15m_ma20_pullback": "V5 long is extended from 15M MA20.",
        "v5_short_wait_15m_ma20_retest": "V5 short is extended from 15M MA20.",
        "v5_long_wait_below_15m_resistance": "V5 long is below nearby resistance.",
        "v5_short_wait_above_15m_support": "V5 short is above nearby support.",
    }
    soft = {
        "macro_direction_mismatch": ("Technical direction conflicts with the macro prior.", 0.50),
        "macro_direction_neutral": ("Macro prior is neutral.", 1.00),
        "symbol_bb_entry_profile_selected_level_blocked": ("Backtest profile dislikes the level; retain as a penalized prior.", 0.85),
        "symbol_bb_entry_profile_selected_level_not_allowed": (
            "Level is outside the profile allow-list; retain as a penalized prior.",
            0.85,
        ),
        "symbol_bb_entry_profile_no_allowed_level": ("Profile has no preferred active level.", 0.75),
        "short_blocked_15m_ma20_trend_alive": ("Short conflicts with the live 15M MA20 trend.", 0.65),
        "long_blocked_15m_ma20_downtrend_alive": ("Long conflicts with the live 15M MA20 trend.", 0.65),
        "trend_big_blocked_without_4h_right_side": ("4H right-side trend confirmation is incomplete.", 0.50),
        "trend_big_blocked_without_1d_ma7_rebreak": ("1D MA7 rebreak confirmation is incomplete.", 0.50),
        "short_15m_blocked_by_4h_right_side_long": ("Countertrend short requires a qualified small probe.", 0.30),
        "short_blocked_4h_lower_band_chase": ("Short is close to 4H lower support.", 0.50),
        "zec_short_blocked_confirmed_4h_low_chase": ("ZEC short is already extended near 4H lows.", 0.50),
        "short_15m_requires_three_line_or_4h_pressure": ("15M short quality confirmation is incomplete.", 0.60),
        "fib_structure_confirmation_missing": ("Fib confirmation is missing; Fib remains an auxiliary prior.", 0.90),
        "short_support_confluence_hold": ("Multi-line support is holding against the short.", 0.50),
        "long_resistance_confluence_hold": ("Multi-line resistance is holding against the long.", 0.50),
    }
    support = {
        "directional_level_touched": "Selected structure level has been touched.",
        "three_line_structure_confirmation": "Three or more directional structure lines confirm the entry.",
        "gann_macro_support": "Stable BTC Gann zone supports the trade direction.",
    }
    definitions = [
        *(GateDefinition(reason, "hard", description) for reason, description in hard.items()),
        *(GateDefinition(reason, "watch", description) for reason, description in watch.items()),
        *(GateDefinition(reason, "soft", description, multiplier) for reason, (description, multiplier) in soft.items()),
        *(GateDefinition(reason, "support", description) for reason, description in support.items()),
    ]
    return GateRegistry(definitions)


class GateEventRecorder:
    """Append-only compatibility recorder for standalone gate events."""

    def __init__(self, path: Path, *, enabled: bool = False) -> None:
        self.path = path
        self.enabled = enabled

    @classmethod
    def from_env(cls, root: Path | None = None) -> "GateEventRecorder":
        enabled = _env_bool("TS_V6_GATE_EVENT_RECORDER_ENABLED", False)
        configured = os.getenv("TS_V6_GATE_EVENT_FILE", "data/trading_system/v6_execution/gate_events.jsonl")
        path = Path(configured)
        if root is not None and not path.is_absolute():
            path = root / path
        return cls(path, enabled=enabled)

    def record_entry_block(self, signal: Any, block: dict[str, Any]) -> None:
        if not self.enabled:
            return
        event = {
            "event_type": "entry_structure_block",
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "symbol": str(getattr(signal, "symbol", "") or ""),
            "side_bias": str(getattr(signal, "side_bias", "") or ""),
            "price": getattr(signal, "price", None),
            "long_probability": getattr(signal, "long_probability", None),
            "short_probability": getattr(signal, "short_probability", None),
            "wait_probability": getattr(signal, "wait_probability", None),
            "computed_at": getattr(signal, "computed_at", None),
            "block": dict(block or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
