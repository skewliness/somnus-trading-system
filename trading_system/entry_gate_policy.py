from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from trading_system.gate_registry import GateRegistry


VALID_GATE_SEVERITIES = {"hard", "watch", "soft", "support"}
DIAGNOSTIC_PAYLOAD_KEYS = {
    "enabled",
    "blocked",
    "watchable",
    "action",
    "side",
    "reason",
    "message",
    "symbol",
    "selected_entry_level",
    "market_data_freshness",
    "smart_signal",
    "ma20_trend",
    "daily_ma7_trend",
    "market_daily_ma7_reversal",
    "h4_right_side",
    "triggers",
    "boundary",
    "generic_indicator",
    "fib_structure_4h",
    "confluence",
    "structure_conflict",
    "effective_bias",
    "macro_source",
    "macro_state_direction",
    "macro_state_source",
    "probability_fallback",
    "selected_probability",
    "opposite_probability",
    "consecutive_losses",
    "max_consecutive_losses",
    "daily_direction",
    "h4_direction",
    "required_closed_4h_candles",
    "checked_closed_4h_candles",
    "weakened_count",
    "h4_logic_weakened",
    "reversal_confirmation",
    "precedence",
    "original_classification",
    "arbitration_status",
    "arbitration_allowed_action",
    "required_15m_pressure_lines",
    "required_rejection_lines",
    "required_trigger_count",
}


@dataclass(frozen=True, slots=True)
class EntryGateFinding:
    reason: str
    severity: str
    message: str = ""
    notional_multiplier: float = 1.0
    payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.severity not in VALID_GATE_SEVERITIES:
            raise ValueError(f"unsupported entry gate severity: {self.severity}")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "reason": self.reason,
            "severity": self.severity,
            "message": self.message,
            "notional_multiplier": round(float(self.notional_multiplier), 6),
        }
        if self.payload:
            result["payload"] = self.payload
        return result


@dataclass(frozen=True, slots=True)
class EntryGateDecision:
    state: str
    allowed: bool
    notional_multiplier: float
    findings: tuple[EntryGateFinding, ...]

    @property
    def blocking_finding(self) -> EntryGateFinding | None:
        for severity in ("hard", "watch"):
            for finding in self.findings:
                if finding.severity == severity:
                    return finding
        return None

    def to_dict(self) -> dict[str, Any]:
        blocking = self.blocking_finding
        return {
            "version": "v2",
            "state": self.state,
            "allowed": self.allowed,
            "notional_multiplier": round(float(self.notional_multiplier), 6),
            "blocking_reason": blocking.reason if blocking else "",
            "hard_reasons": [item.reason for item in self.findings if item.severity == "hard"],
            "watch_reasons": [item.reason for item in self.findings if item.severity == "watch"],
            "soft_reasons": [item.reason for item in self.findings if item.severity == "soft"],
            "support_reasons": [item.reason for item in self.findings if item.severity == "support"],
            "findings": [item.to_dict() for item in self.findings],
        }


def finding_from_block(
    block: dict[str, Any] | None,
    *,
    severity: str | None = None,
    notional_multiplier: float | None = None,
    registry: GateRegistry | None = None,
) -> EntryGateFinding | None:
    if not block:
        return None
    reason = str(block.get("reason") or "entry_gate")
    fallback_severity = severity or "watch"
    fallback_multiplier = 1.0 if notional_multiplier is None else float(notional_multiplier)
    definition = (
        registry.resolve(
            reason,
            fallback_classification=fallback_severity,
            fallback_multiplier=fallback_multiplier,
        )
        if registry is not None
        else None
    )
    effective_severity = severity or (definition.classification if definition else fallback_severity)
    effective_multiplier = (
        float(notional_multiplier)
        if notional_multiplier is not None
        else float(definition.notional_multiplier if definition else fallback_multiplier)
    )
    payload = {
        key: value
        for key, value in block.items()
        if key in DIAGNOSTIC_PAYLOAD_KEYS
    }
    if definition is not None:
        payload["gate_definition"] = definition.to_dict()
    return EntryGateFinding(
        reason=reason,
        severity=effective_severity,
        message=str(block.get("message") or block.get("reason") or ""),
        notional_multiplier=effective_multiplier,
        payload=payload,
    )


def evaluate_entry_gate_findings(
    findings: Iterable[EntryGateFinding | None],
    *,
    minimum_notional_multiplier: float = 0.60,
) -> EntryGateDecision:
    collected = tuple(item for item in findings if item is not None)
    hard = next((item for item in collected if item.severity == "hard"), None)
    watch = next((item for item in collected if item.severity == "watch"), None)

    multiplier = 1.0
    for finding in collected:
        if finding.severity == "soft":
            multiplier *= max(0.0, min(1.0, float(finding.notional_multiplier)))
    multiplier = max(0.0, min(1.0, multiplier))
    if collected:
        multiplier = max(max(0.0, min(1.0, minimum_notional_multiplier)), multiplier)

    if hard is not None:
        return EntryGateDecision("blocked", False, multiplier, collected)
    if watch is not None:
        return EntryGateDecision("watch", False, multiplier, collected)
    return EntryGateDecision("allowed", True, multiplier, collected)
