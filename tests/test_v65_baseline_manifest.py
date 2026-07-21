from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/v6_0_behavioral_baseline.json"
V6_EXECUTION = ROOT / "trading_system/v6_execution.py"
V65_ARCHITECTURE_CONFIG_FIELDS = {
    "config_version",
    "model_version",
    "order_state_file",
    "strategy_version",
    "trade_event_logging_enabled",
    "trade_event_strategy_version",
    "trade_events_file",
    "trade_events_max_bytes",
}


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


def test_every_v60_config_default_is_unchanged_in_v65() -> None:
    manifest = load_manifest()
    tree = ast.parse(V6_EXECUTION.read_text(encoding="utf-8"))
    config_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "V6ExecutionConfig"
    )
    defaults = {
        node.target.id: ast.unparse(node.value) if node.value is not None else None
        for node in config_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    architecture_fields = set(defaults) - V65_ARCHITECTURE_CONFIG_FIELDS
    baseline_defaults = {name: defaults[name] for name in architecture_fields}
    payload = json.dumps(baseline_defaults, sort_keys=True, separators=(",", ":"))

    assert set(defaults) - architecture_fields == V65_ARCHITECTURE_CONFIG_FIELDS
    assert len(baseline_defaults) == manifest["config"]["field_count"]
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == manifest["config"]["default_expression_sha256"]
