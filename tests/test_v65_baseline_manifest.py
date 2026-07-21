from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/v6_0_behavioral_baseline.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_baseline_manifest_is_complete_and_recoverable() -> None:
    manifest = load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["tag"] == "v6.0-baseline-20260721"
    assert manifest["commit"].startswith("0aaaaa2")
    assert manifest["source"]["path"] == "trading_system/v6_execution.py"
    assert len(manifest["source"]["sha256"]) == 64
    assert manifest["config"]["field_count"] == 248
    assert len(manifest["config"]["default_expression_sha256"]) == 64


def test_baseline_manifest_covers_every_required_policy_surface() -> None:
    manifest = load_manifest()
    rules = manifest["rules"]

    required = {
        "intent_for_signal",
        "intent_for_technical_entry_signal",
        "entry_structure_block",
        "macro_direction_entry_block",
        "open_intent",
        "scale_in_intent_if_needed",
        "exit_intent_if_needed",
        "stop_take_profit",
        "apply_intent_to_state",
        "reconcile_state_with_exchange",
    }
    assert set(rules) == required
    assert all(len(value) == 64 for value in rules.values())
    assert {
        "base_notional_usdt",
        "default_leverage",
        "max_positions",
        "stop_loss_pct",
        "take_profit_pct",
        "long_entry_threshold",
        "short_entry_threshold",
        "exit_opposite_threshold",
        "model_path",
    }.issubset(manifest["config"]["key_defaults"])

