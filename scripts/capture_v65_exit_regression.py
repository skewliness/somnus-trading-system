from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import types
from dataclasses import fields
from pathlib import Path
from typing import Any


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, float):
        return round(value, 10)
    return value


def fixture_sha256(fixture: dict[str, Any]) -> str:
    payload = json.dumps(fixture, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_sha256(repo_root: Path) -> str:
    source = repo_root / "trading_system" / "v6_execution.py"
    text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_v6_execution(repo_root: Path) -> Any:
    if os.name == "nt" and "fcntl" not in sys.modules:
        fcntl = types.ModuleType("fcntl")
        fcntl.LOCK_EX = 2
        fcntl.LOCK_NB = 4
        fcntl.LOCK_UN = 8
        fcntl.flock = lambda *_args, **_kwargs: None
        sys.modules["fcntl"] = fcntl
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("trading_system.v6_execution")


def supported_kwargs(dataclass_type: Any, values: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(dataclass_type)}
    return {name: value for name, value in values.items() if name in names}


def runtime_paths(root: Path) -> dict[str, Path]:
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


def build_engine(v6: Any, root: Path, fixed_now: str, config_values: dict[str, Any]) -> Any:
    v6.utc_now_iso = lambda: fixed_now
    values: dict[str, Any] = {
        "symbols": ["BTCUSDT", "ETHUSDT", "HYPEUSDT"],
        "order_mode": "dry_run",
        "instance_name": "exit-regression",
        "kline_cache_enabled": False,
        "telegram_whale_fallback_enabled": False,
        "smart_signal_risk_guard_enabled": False,
        "entry_gate_event_recording_enabled": False,
        "trade_event_logging_enabled": False,
        "gann_position_sizing_enabled": False,
        "close_resolver_enabled": False,
        "close_intent_coalescing_enabled": False,
    }
    values.update(runtime_paths(root))
    values.update(config_values)
    config = v6.V6ExecutionConfig(**supported_kwargs(v6.V6ExecutionConfig, values))
    return v6.V6ExecutionEngine(config)


def make_signal(v6: Any, raw: dict[str, Any]) -> Any:
    return v6.V6Signal(**supported_kwargs(v6.V6Signal, raw))


def tuple_candidate(raw: Any) -> tuple[list[dict[str, Any]], float] | None:
    if not isinstance(raw, dict):
        return None
    stages = [dict(item) for item in raw.get("stages") or [] if isinstance(item, dict)]
    if not stages:
        return None
    return stages, float(raw.get("close_fraction") or 1.0)


def install_case_stubs(engine: Any, stubs: dict[str, Any]) -> None:
    engine.position_max_hold_expired = lambda position: bool(stubs.get("hold_expired"))
    engine.update_position_profit_tracker = lambda signal, position: dict(stubs.get("profit_tracker") or {})
    engine.v5_overlay_soft_stop_candidate = lambda signal, position, risk_plan: stubs.get("v5_soft_stop")
    engine.daily_ma7_short_reversal_exit_candidate = lambda signal, position, state=None: stubs.get("daily_reversal")
    engine.daily_ma20_long_reversal_exit_candidate = lambda signal, position, state=None: stubs.get("daily_reversal")
    engine.logic_weakening_exit_candidate = lambda signal, position: stubs.get("logic_weakening")
    engine.smart_signal_risk_guard_candidate = lambda signal, position, tracker: stubs.get("smart_signal")
    engine.v5_no_news_momentum_take_profit_candidate = lambda signal, position, tracker: stubs.get("v5_no_news")
    engine.fifteen_m_h4_profit_giveback_candidate = lambda signal, position, tracker: stubs.get("fifteen_m_giveback")
    engine.h4_lower_reversal_guard_candidate = lambda signal, position, tracker: stubs.get("h4_lower_reversal")
    engine.next_roi_take_profit = lambda signal, position: tuple_candidate(stubs.get("roi_take_profit"))
    engine.next_staged_take_profit = lambda signal, position: tuple_candidate(stubs.get("staged_take_profit"))
    engine.ensure_position_take_profit_stages = lambda position, signal: bool(stubs.get("has_take_profit_stages"))
    engine.opposite_exit_threshold_for_position = lambda position: float(stubs.get("opposite_threshold") or 0.62)


def selected_rule(intent: Any) -> tuple[str, int]:
    reason = str(getattr(intent, "rationale", "") or "").lower()
    risk_plan = getattr(intent, "risk_plan", {}) if intent is not None else {}
    partial = risk_plan.get("partial_take_profit") if isinstance(risk_plan, dict) else {}
    partial_type = str((partial or {}).get("type") or "")
    if "max hold days reached" in reason:
        return "max_hold", 1000
    if "stop loss touched" in reason:
        return "hard_stop", 1000
    if "soft stop" in reason or (isinstance(risk_plan, dict) and risk_plan.get("v5_soft_stop_exit")):
        return "v5_soft_stop", 990
    if isinstance(risk_plan, dict) and (
        risk_plan.get("daily_ma7_short_reversal_exit") or risk_plan.get("daily_ma20_long_reversal_exit")
    ):
        return "daily_reversal", 970
    if isinstance(risk_plan, dict) and risk_plan.get("logic_weakening_exit"):
        return "logic_weakening", 940
    if "smart-signal risk guard" in reason:
        return "smart_signal", 950
    if partial_type == "roi_take_profit":
        return "roi_take_profit", 930
    if partial_type == "bb_take_profit":
        return "staged_take_profit", 920
    if "take profit touched" in reason:
        return "fixed_take_profit", 910
    if "opposite long probability" in reason or "opposite short probability" in reason:
        return "opposite_probability", 900
    raise AssertionError(f"unclassified exit intent: {getattr(intent, 'rationale', '')}")


def capture_case(v6: Any, fixture: dict[str, Any], case: dict[str, Any], root: Path) -> dict[str, Any]:
    engine = build_engine(v6, root, fixture["fixed_now"], dict(case.get("config") or {}))
    try:
        install_case_stubs(engine, dict(case.get("stubs") or {}))
        signal = make_signal(v6, dict(case["signal"]))
        position = canonical(case["position"])
        intent = engine.exit_intent_if_needed(signal, position, state=canonical(case.get("state") or {}))
        if intent is None:
            raise AssertionError(f"exit case did not produce an intent: {case['id']}")
        rule, priority = selected_rule(intent)
        return {
            "id": str(case["id"]),
            "exit_reason": str(intent.rationale),
            "exit_priority": priority,
            "exit_price": round(float(intent.entry_price), 10),
            "rule": rule,
        }
    finally:
        engine.close()


def capture_current(repo_root: Path, fixture: dict[str, Any], *, work_root: Path) -> dict[str, Any]:
    v6 = import_v6_execution(repo_root)
    original_utc_now_iso = v6.utc_now_iso
    try:
        cases = []
        for index, case in enumerate(fixture["cases"]):
            cases.append(capture_case(v6, fixture, case, work_root / f"case-{index:02d}"))
        return {
            "schema_version": 1,
            "fixture_sha256": fixture_sha256(fixture),
            "source_sha256": source_sha256(repo_root),
            "cases": cases,
        }
    finally:
        v6.utc_now_iso = original_utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture deterministic V6 exit behavior for regression comparison.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--source-ref", default="")
    parser.add_argument("--source-sha", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    result = capture_current(args.repo_root, fixture, work_root=args.work_root)
    result["source_ref"] = args.source_ref
    result["source_commit"] = args.source_sha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
