#!/usr/bin/env python3
"""Capture deterministic V6 decisions for the V6/V6.5 compatibility test.

The script intentionally imports ``trading_system.v6_execution`` from a
caller-provided repository root.  That makes the same harness runnable against
the frozen V6 tag and the V6.5 candidate without copying either implementation
into the test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "v65_behavioral_compatibility_input.json"


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def canonical(value: Any) -> Any:
    """Return a JSON-safe, deterministic representation."""

    payload = json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return json.loads(payload)


def fixture_sha256(fixture: dict[str, Any]) -> str:
    payload = json.dumps(fixture, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_sha256(repo_root: Path) -> str:
    source = repo_root / "trading_system" / "v6_execution.py"
    # Normalize checkout line endings so the provenance hash is identical on
    # Windows and macOS and matches the frozen baseline manifest.
    text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_v6_execution(repo_root: Path) -> Any:
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("trading_system.v6_execution")


def _supported_kwargs(dataclass_type: Any, values: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(dataclass_type)}
    return {name: value for name, value in values.items() if name in names}


def _runtime_paths(root: Path) -> dict[str, Path]:
    return {
        "state_dir": root,
        "state_file": root / "state.json",
        "signals_file": root / "signals.jsonl",
        "entry_gate_events_file": root / "entry_gate_events.jsonl",
        "orders_file": root / "orders.jsonl",
        "order_state_file": root / "order_states.json",
        "wal_file": root / "execution_wal.jsonl",
        "latest_summary_file": root / "latest_run_summary.json",
        "closed_trade_review_queue_file": root / "closed_trade_review_queue.jsonl",
        "post_trade_training_samples_file": root / "post_trade_training_samples.jsonl",
        "manual_exchange_action_audit_file": root / "manual_exchange_actions.jsonl",
        "trade_events_file": root / "trade_events.jsonl",
        "kline_cache_dir": root / "kline_cache",
        "model_path": root / "missing-model.pkl",
        "telegram_whale_bridge_file": root / "missing-telegram-whale.json",
    }


def build_engine(v6: Any, config_values: dict[str, Any], root: Path, fixed_now: str) -> Any:
    v6.utc_now_iso = lambda: fixed_now
    values: dict[str, Any] = {
        "symbols": ["BTCUSDT", "ETHUSDT", "HYPEUSDT"],
        "order_mode": "dry_run",
        "instance_name": "behavioral-compatibility",
        "kline_cache_enabled": False,
        "telegram_whale_fallback_enabled": False,
        "smart_signal_risk_guard_enabled": False,
        "entry_gate_event_recording_enabled": False,
        "trade_event_logging_enabled": False,
        "mixed_structure_macro_arbitration_enabled": False,
        "gann_position_sizing_enabled": False,
        "close_resolver_enabled": False,
        "close_intent_coalescing_enabled": False,
    }
    values.update(_runtime_paths(root))
    values.update(config_values)
    config = v6.V6ExecutionConfig(**_supported_kwargs(v6.V6ExecutionConfig, values))
    return v6.V6ExecutionEngine(config)


def make_signal(v6: Any, raw: dict[str, Any]) -> Any:
    return v6.V6Signal(**_supported_kwargs(v6.V6Signal, raw))


def canonical_intent(intent: Any | None) -> dict[str, Any] | None:
    if intent is None:
        return None
    if hasattr(intent, "to_dict"):
        return canonical(intent.to_dict())
    return canonical(intent)


def _stub_market_client(engine: Any, market: dict[str, Any]) -> None:
    klines = market["klines"]
    engine.client.reset_market_data_freshness = lambda symbol: None
    engine.client.fetch_klines = lambda symbol, interval, limit: list(klines.get(interval) or [])
    engine.client.supplement_4h_klines_for_bb2880 = lambda symbol, rows, limit: list(rows)
    engine.client.mark_price = lambda symbol: float(market["mark_price"])
    engine.client.ticker_24h = lambda symbol: dict(market["ticker_24h"])
    engine.client.market_data_freshness = lambda symbol: dict(market.get("market_data_freshness") or {})
    engine.telegram_whale_bridge.snapshot = lambda *args, **kwargs: {}


def capture_market_signal(v6: Any, fixture: dict[str, Any], root: Path) -> dict[str, Any]:
    market = fixture["historical_market_snapshot"]
    engine = build_engine(v6, dict(market.get("config") or {}), root, fixture["fixed_now"])
    try:
        _stub_market_client(engine, market)
        return canonical(engine.build_signal(str(market["symbol"])))
    finally:
        engine.close()


def capture_case(v6: Any, fixture: dict[str, Any], case: dict[str, Any], root: Path) -> dict[str, Any]:
    engine = build_engine(v6, dict(case.get("config") or {}), root, fixture["fixed_now"])
    try:
        signal = make_signal(v6, dict(case["signal"]))
        state = canonical(case.get("state") or {"positions": []})
        kind = str(case["kind"])
        entry_intent = None
        exit_intent = None
        selected_intent = None
        if kind == "entry":
            selected_intent = engine.intent_for_signal(signal, state)
            entry_intent = canonical_intent(selected_intent)
        elif kind == "exit":
            position = canonical(case["position"])
            selected_intent = engine.exit_intent_if_needed(signal, position, state=state)
            exit_intent = canonical_intent(selected_intent)
        else:
            raise ValueError(f"unsupported behavioral case kind: {kind}")

        return {
            "id": str(case["id"]),
            "kind": kind,
            "signal": canonical(signal.to_dict()),
            "entry_intent": entry_intent,
            "exit_intent": exit_intent,
            "position_size_usdt": None if selected_intent is None else float(selected_intent.quantity_hint_usdt),
            "trade_reason": None if selected_intent is None else str(selected_intent.rationale),
        }
    finally:
        engine.close()


def capture(
    v6: Any,
    fixture: dict[str, Any],
    *,
    repo_root: Path,
    work_root: Path,
    label: str,
    commit: str,
) -> dict[str, Any]:
    original_utc_now_iso = v6.utc_now_iso
    try:
        market_root = work_root / "market-signal"
        cases = [
            capture_case(v6, fixture, case, work_root / "cases" / str(case["id"]))
            for case in fixture["cases"]
        ]
        return {
            "schema_version": 1,
            "baseline": {
                "label": label,
                "commit": commit,
                "source_path": "trading_system/v6_execution.py",
                "source_sha256": source_sha256(repo_root),
            },
            "fixture_sha256": fixture_sha256(fixture),
            "market_signal": capture_market_signal(v6, fixture, market_root),
            "cases": cases,
        }
    finally:
        v6.utc_now_iso = original_utc_now_iso


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--commit", default="unknown")
    args = parser.parse_args(argv)

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    v6 = import_v6_execution(args.repo_root)
    with tempfile.TemporaryDirectory(prefix="v65-behavior-") as temporary:
        result = capture(
            v6,
            fixture,
            repo_root=args.repo_root,
            work_root=Path(temporary),
            label=args.label,
            commit=args.commit,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
