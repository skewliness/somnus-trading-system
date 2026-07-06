from __future__ import annotations

from pathlib import Path

from trading_system.gate_registry import GateEventRecorder
from trading_system.transition_gate_policy import TransitionGatePolicy, transition_notional_multiplier
from trading_system.v6_execution import V6ExecutionConfig, V6ExecutionEngine, V6Signal


def make_signal() -> V6Signal:
    return V6Signal(
        symbol="BTCUSDT",
        computed_at="2026-07-06T00:00:00+00:00",
        price=60000.0,
        side_bias="short",
        long_probability=0.2,
        short_probability=0.6,
        wait_probability=0.2,
        model_quality_score=0.7,
        reasons=[],
        h4={},
        fifteen_m={},
        one_m={},
        mid_retest={},
        levels={},
        indicators={},
    )


def make_engine(tmp_path: Path, *, policy_enabled: bool) -> V6ExecutionEngine:
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        state_dir=tmp_path,
        base_notional_usdt=300.0,
        min_notional_usdt=300.0,
        transition_gate_min_notional_usdt=5.0,
    )
    engine.transition_gate_policy = TransitionGatePolicy(enabled=policy_enabled)
    engine.gate_recorder = GateEventRecorder(tmp_path / "gate_events.jsonl", enabled=True)
    return engine


def test_transition_gate_policy_disabled_keeps_block(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, policy_enabled=False)
    signal = make_signal()
    block = {"reason": "macro_direction_mismatch", "message": "blocked"}

    result = engine.maybe_downgrade_entry_block(signal, block)

    assert result == block
    assert "transition_gate_policy" not in signal.levels


def test_transition_gate_policy_enabled_records_and_allows_probe(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, policy_enabled=True)
    signal = make_signal()
    block = {"reason": "macro_direction_mismatch", "message": "blocked"}

    result = engine.maybe_downgrade_entry_block(signal, block)

    assert result is None
    assert signal.levels["transition_gate_policy"][0]["decision"]["classification"] == "sizing"
    assert transition_notional_multiplier(signal.levels) == 0.5
    assert "transition gate downgraded macro_direction_mismatch" in "; ".join(signal.reasons)


def test_transition_gate_multiplier_reduces_open_notional_below_v6_min(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, policy_enabled=True)
    engine.stop_take_profit = lambda signal, action: (59000.0, 63000.0, {})
    engine.v6_smart_signal_context = lambda signal, side: {}
    engine.apply_candidate_metadata_to_risk_plan = lambda symbol, risk_plan: None
    signal = make_signal()
    signal.levels["transition_gate_policy"] = [{
        "decision": {
            "notional_multiplier": 0.5,
            "classification": "sizing",
            "reason": "macro_direction_mismatch",
        }
    }]

    intent = engine.open_intent(signal, "open_short")

    assert intent.quantity_hint_usdt == 150.0
    assert intent.risk_plan["position_sizing_overlays"]["transition_gate_multiplier"] == 0.5


def test_transition_gate_can_probe_unconfirmed_technical_15m_trigger(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, policy_enabled=True)
    engine.config.entry_structure_filter_enabled = False
    signal = make_signal()
    signal.symbol = "TESTUSDT"
    signal.levels = {
        "structure_entry_scan": {
            "long_candidates": [{
                "name": "bb96_lower_15m",
                "price": 59000.0,
                "distance_pct": 1.7,
                "abs_distance_pct": 1.7,
                "period": 96,
                "band": "lower",
                "timeframe": "15m",
                "structure_signal": "long_lower_support_rebound",
            }]
        },
        "all_supports": [("bb96_lower_15m", 59000.0)],
    }

    ready, block, _ = engine.technical_entry_ready(signal, "open_long")

    assert ready is True
    assert block is None
    assert signal.levels["transition_gate_policy"][0]["decision"]["classification"] == "watch"
