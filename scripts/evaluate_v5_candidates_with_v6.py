#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


load_local_env(ROOT / ".env.local")

from trading_system.persistence import atomic_write_json  # noqa: E402
from trading_system.v6_execution import V6ExecutionConfig, V6ExecutionEngine  # noqa: E402


DEFAULT_REPORT = Path("data/trading_system/pipeline_run_report.json")
DEFAULT_OUTPUT_DIR = Path("data/trading_system/v5_v6_overlay_shadow")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_symbols(report: dict[str, Any], limit: int) -> list[str]:
    symbols: list[str] = []
    for row in report.get("candidate_basket") or []:
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol or symbol in symbols:
            continue
        symbols.append(symbol)
        if limit > 0 and len(symbols) >= limit:
            break
    return symbols


def reset_shadow_state(config: V6ExecutionConfig) -> None:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        config.state_file,
        {
            "schema_version": 1,
            "positions": [],
            "updated_at": utc_now(),
            "note": "V5 candidates evaluated through V6 trading system in dry-run shadow mode.",
        },
    )


def compact_signal(signal: dict[str, Any], v5_row: dict[str, Any], intent: dict[str, Any] | None) -> dict[str, Any]:
    levels = signal.get("levels") if isinstance(signal.get("levels"), dict) else {}
    scan = levels.get("structure_entry_scan") if isinstance(levels.get("structure_entry_scan"), dict) else {}
    risk_plan = intent.get("risk_plan") if isinstance(intent, dict) and isinstance(intent.get("risk_plan"), dict) else {}
    context = risk_plan.get("entry_context") if isinstance(risk_plan.get("entry_context"), dict) else {}
    selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
    return {
        "symbol": signal.get("symbol"),
        "v5_score": v5_row.get("score"),
        "v5_status": v5_row.get("execution_status"),
        "v5_risk_band": v5_row.get("risk_band"),
        "v5_blocked_reasons": v5_row.get("blocked_reasons") or v5_row.get("blocked_reason") or [],
        "v6_bias": signal.get("side_bias"),
        "long_probability": signal.get("long_probability"),
        "short_probability": signal.get("short_probability"),
        "wait_probability": signal.get("wait_probability"),
        "long_candidate_count": scan.get("long_candidate_count") or len(scan.get("long_candidates") or []),
        "short_candidate_count": scan.get("short_candidate_count") or len(scan.get("short_candidates") or []),
        "intent_action": intent.get("action") if intent else None,
        "entry_price": intent.get("entry_price") if intent else None,
        "stop_loss_price": intent.get("stop_loss_price") if intent else None,
        "take_profit_price": intent.get("take_profit_price") if intent else None,
        "horizon": risk_plan.get("horizon") if risk_plan else None,
        "entry_timeframe": risk_plan.get("entry_timeframe") if risk_plan else None,
        "selected_level": selected.get("name") if selected else None,
        "selected_level_price": selected.get("price") if selected else None,
        "selected_level_distance_pct": selected.get("distance_pct") if selected else None,
        "rationale": intent.get("rationale") if intent else "",
    }


def build_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# V5 candidates through V6 trading system",
        "",
        f"- generated_at: {report.get('generated_at')}",
        f"- mode: {summary.get('mode')}",
        f"- candidates: {summary.get('candidate_count')}",
        f"- v6_signals: {summary.get('signal_count')}",
        f"- v6_intents: {summary.get('intent_count')}",
        f"- signal_failures: {summary.get('signal_failure_count')}",
        "",
        "## Current V5 Paper Baseline",
    ]
    baseline = report.get("v5_performance_dashboard") or {}
    if baseline:
        lines.extend(
            [
                f"- closed_trades: {baseline.get('closed_trades')}",
                f"- win_rate: {baseline.get('win_rate')}%",
                f"- total_pnl_usdt: {baseline.get('total_pnl_usdt')}",
                f"- total_return_pct: {baseline.get('total_return_pct')}%",
            ]
        )
    lines.extend(["", "## Overlay Decisions"])
    for row in report.get("rows") or []:
        intent = row.get("intent_action") or "no_trade"
        detail = ""
        if row.get("intent_action"):
            detail = (
                f" entry={row.get('entry_price')} stop={row.get('stop_loss_price')} "
                f"tp={row.get('take_profit_price')} horizon={row.get('horizon')} "
                f"level={row.get('selected_level')}"
            )
        lines.append(
            "- "
            f"{row.get('symbol')}: V5={row.get('v5_status')} score={row.get('v5_score')} | "
            f"V6={row.get('v6_bias')} 多={row.get('long_probability')} "
            f"空={row.get('short_probability')} 等={row.get('wait_probability')} | "
            f"{intent}{detail}"
        )
    failures = report.get("signal_failures") or []
    if failures:
        lines.extend(["", "## Signal Failures"])
        for item in failures:
            lines.append(f"- {item.get('symbol')}: {item.get('error')}")
    return "\n".join(lines) + "\n"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    source_report_file = Path(args.report)
    source_report = load_json(source_report_file)
    symbols = candidate_symbols(source_report, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not symbols:
        payload = {
            "generated_at": utc_now(),
            "mode": "v5_v6_overlay_shadow",
            "source_report_file": str(source_report_file),
            "summary": {
                "mode": "dry_run_shadow_no_candidates",
                "candidate_count": 0,
                "signal_count": 0,
                "intent_count": 0,
                "signal_failure_count": 0,
            },
            "rows": [],
            "signal_failures": [],
            "v5_performance_dashboard": source_report.get("performance_dashboard") or {},
        }
        atomic_write_json(output_dir / "latest_overlay_report.json", payload)
        (output_dir / "latest_overlay_report.md").write_text(build_markdown(payload), encoding="utf-8")
        return payload

    config = V6ExecutionConfig.from_env()
    config.symbols = symbols
    config.order_mode = "dry_run"
    config.commit_dry_run_state = False
    config.state_dir = output_dir
    config.state_file = output_dir / "state.json"
    config.signals_file = output_dir / "signals.jsonl"
    config.orders_file = output_dir / "orders.jsonl"
    config.wal_file = output_dir / "execution_wal.jsonl"
    config.latest_summary_file = output_dir / "latest_run_summary.json"
    config.closed_trade_review_queue_file = output_dir / "closed_trade_review_queue.jsonl"
    config.post_trade_training_samples_file = output_dir / "post_trade_training_samples.jsonl"
    if args.technical_only:
        config.technical_entry_only_symbols = tuple(symbols)
    reset_shadow_state(config)

    engine = V6ExecutionEngine(config)
    try:
        v6_summary = engine.run_once(submit=False, commit_dry_run_state=False)
    finally:
        engine.close()

    candidate_map = {str(row.get("symbol") or "").upper(): row for row in source_report.get("candidate_basket") or []}
    intent_map = {str(item.get("symbol") or "").upper(): item for item in v6_summary.get("intents") or []}
    rows = [
        compact_signal(signal, candidate_map.get(str(signal.get("symbol") or "").upper(), {}), intent_map.get(str(signal.get("symbol") or "").upper()))
        for signal in v6_summary.get("signals") or []
    ]
    failures = [
        item for item in v6_summary.get("results") or []
        if str(item.get("status") or "") == "signal_failed"
    ]
    actions: dict[str, int] = {}
    for item in v6_summary.get("intents") or []:
        action = str(item.get("action") or "unknown")
        actions[action] = actions.get(action, 0) + 1
    payload = {
        "generated_at": utc_now(),
        "mode": "v5_v6_overlay_shadow",
        "source_report_file": str(source_report_file),
        "v6_summary_file": str(config.latest_summary_file),
        "summary": {
            "mode": "dry_run_shadow",
            "candidate_count": len(symbols),
            "signal_count": len(v6_summary.get("signals") or []),
            "intent_count": len(v6_summary.get("intents") or []),
            "signal_failure_count": len(failures),
            "actions": actions,
            "technical_only": bool(args.technical_only),
        },
        "v5_performance_dashboard": source_report.get("performance_dashboard") or {},
        "rows": rows,
        "signal_failures": failures,
    }
    atomic_write_json(output_dir / "latest_overlay_report.json", payload)
    (output_dir / "latest_overlay_report.md").write_text(build_markdown(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate V5 candidate basket through the V6 trading system in dry-run shadow mode.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="V5 pipeline report JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for shadow V6 state and report.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum V5 candidates to evaluate; 0 means all.")
    parser.add_argument("--technical-only", action=argparse.BooleanOptionalAction, default=True, help="Treat V5 candidates as V6 technical-entry-only symbols.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args(argv)
    report = evaluate(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report.get("summary") or {}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
