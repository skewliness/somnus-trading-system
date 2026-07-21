from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from trading_system.events import EventWriter
from trading_system.v6_execution import V6ExecutionConfig, V6ExecutionEngine, V6OrderIntent
from trading_system.versioning import RuntimeVersionInfo, config_fingerprint, config_fingerprint_payload, model_artifact_version


def sample_intent() -> V6OrderIntent:
    return V6OrderIntent(
        action="open_long",
        symbol="BTCUSDT",
        side="BUY",
        quantity_hint_usdt=100.0,
        leverage=8,
        reduce_only=False,
        entry_price=60000.0,
        stop_loss_price=59000.0,
        take_profit_price=62000.0,
        long_probability=0.7,
        short_probability=0.2,
        wait_probability=0.1,
        rationale="structure entry",
        generated_at="2026-07-21T00:00:00+00:00",
    )


def test_config_fingerprint_is_deterministic_and_credential_independent(tmp_path: Path) -> None:
    first = V6ExecutionConfig(
        state_dir=tmp_path / "first",
        demo_api_key="first-key",
        demo_api_secret="first-secret",
        long_entry_threshold=0.72,
    )
    second = V6ExecutionConfig(
        state_dir=tmp_path / "second",
        demo_api_key="second-key",
        demo_api_secret="second-secret",
        long_entry_threshold=0.72,
    )

    assert config_fingerprint(first) == config_fingerprint(second)
    assert "first-key" not in json.dumps(config_fingerprint_payload(first))
    second.long_entry_threshold = 0.73
    assert config_fingerprint(first) != config_fingerprint(second)


def test_model_version_hashes_artifact_and_honors_explicit_version(tmp_path: Path) -> None:
    model = tmp_path / "model.pkl"
    model.write_bytes(b"model-a")
    first = model_artifact_version(model)
    model.write_bytes(b"model-b")

    assert first.startswith("sha256:")
    assert first != model_artifact_version(model)
    assert model_artifact_version(model, "release-42") == "release-42"


def test_model_version_read_failure_is_non_fatal(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "model.pkl"
    model.write_bytes(b"model")

    def fail_open(*_args, **_kwargs):
        raise PermissionError("temporarily unavailable")

    monkeypatch.setattr(Path, "open", fail_open)

    assert model_artifact_version(model) == "unreadable:model.pkl"


def test_runtime_version_info_contains_all_required_fields(tmp_path: Path) -> None:
    model = tmp_path / "model.pkl"
    model.write_bytes(b"model")
    config = V6ExecutionConfig(
        model_path=model,
        strategy_version="v6.5-structure",
        config_version="config-20260721",
    )

    versions = RuntimeVersionInfo.from_config(config).to_dict()

    assert set(versions) == {"strategy_version", "config_version", "model_version", "config_hash"}
    assert versions["strategy_version"] == "v6.5-structure"
    assert versions["config_version"] == "config-20260721"
    assert versions["model_version"].startswith("sha256:")


def test_every_annotated_intent_and_lot_preserves_runtime_versions(tmp_path: Path) -> None:
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        model_path=tmp_path / "missing-model.pkl",
        strategy_version="v6.5-structure",
        config_version="config-1",
    )
    engine.risk_manager = None
    intent = sample_intent()

    engine.annotate_intent_execution_risk(intent)
    versions = intent.risk_plan["runtime_versions"]
    lot = engine.lot_from_open_result(
        intent,
        {"submitted_at": "2026-07-21T00:00:01+00:00", "executed_quantity": 0.01},
        fill_price=60000.0,
        notional=600.0,
    )

    assert versions["strategy_version"] == "v6.5-structure"
    assert lot["runtime_versions"] == versions
    assert lot["risk_plan"]["runtime_versions"] == versions


def test_event_envelope_carries_complete_runtime_versions(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(
        path,
        enabled=True,
        strategy_version="strategy-1",
        config_version="config-1",
        model_version="model-1",
        config_hash="sha256:abc",
    )
    summary = {
        "generated_at": "2026-07-21T00:00:00+00:00",
        "signals": [{"symbol": "BTCUSDT", "computed_at": "2026-07-21T00:00:00+00:00"}],
        "intents": [],
        "results": [],
        "positions": [],
    }

    assert writer.emit_run_summary(summary) == 1
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["strategy_version"] == "strategy-1"
    assert row["config_version"] == "config-1"
    assert row["model_version"] == "model-1"
    assert row["config_hash"] == "sha256:abc"


def test_from_env_reads_version_labels(monkeypatch) -> None:
    monkeypatch.setenv("TS_V6_STRATEGY_VERSION", "strategy-env")
    monkeypatch.setenv("TS_V6_CONFIG_VERSION", "config-env")
    monkeypatch.setenv("TS_V6_MODEL_VERSION", "model-env")

    config = V6ExecutionConfig.from_env()

    assert config.strategy_version == "strategy-env"
    assert config.config_version == "config-env"
    assert config.model_version == "model-env"
