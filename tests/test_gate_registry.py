from __future__ import annotations

import json
from pathlib import Path

from trading_system.gate_registry import GateEventRecorder
from trading_system.v6_execution import V6ExecutionEngine, V6Signal


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
    )


def test_gate_event_recorder_disabled_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "gate_events.jsonl"
    recorder = GateEventRecorder(path, enabled=False)

    recorder.record_entry_block(make_signal(), {"reason": "blocked"})

    assert not path.exists()


def test_gate_event_recorder_enabled_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "gate_events.jsonl"
    recorder = GateEventRecorder(path, enabled=True)

    recorder.record_entry_block(make_signal(), {"reason": "blocked"})

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["event_type"] == "entry_structure_block"
    assert event["symbol"] == "BTCUSDT"
    assert event["block"]["reason"] == "blocked"


def test_apply_entry_structure_block_records_without_changing_decision_path(tmp_path: Path) -> None:
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.gate_recorder = GateEventRecorder(tmp_path / "gate_events.jsonl", enabled=True)
    signal = make_signal()

    engine.apply_entry_structure_block(signal, {"reason": "profile_blocked", "message": "selected level blocked"})

    assert signal.levels["entry_structure_filter"]["reason"] == "profile_blocked"
    assert "selected level blocked" in signal.reasons
    event = json.loads((tmp_path / "gate_events.jsonl").read_text(encoding="utf-8"))
    assert event["block"]["message"] == "selected level blocked"
