#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


os.umask(0o077)

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

from trading_system.persistence import append_text_line, atomic_write_json, file_lock, load_json, timed_file_lock
from trading_system.redaction import redact_sensitive_text
from trading_system.v6_execution import V6BinanceClient, V6ExecutionConfig, V6ExecutionEngine, V6ExecutionError, utc_now_iso


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def v6_execution_lock_context(config: V6ExecutionConfig):
    if _env_bool("TS_V6_EXECUTION_LOCK_HELD", False):
        return nullcontext()
    lock_target = Path(os.getenv("TS_V6_EXECUTION_LOCK_TARGET", str(config.state_dir / "v6_execution")))
    return timed_file_lock(
        lock_target,
        timeout_seconds=float(os.getenv("TS_V6_LEDGER_EXECUTION_LOCK_TIMEOUT_SECONDS", "5")),
        poll_seconds=float(os.getenv("TS_V6_EXECUTION_LOCK_POLL_SECONDS", "0.25")),
    )


def normalize_state(payload: Any) -> dict[str, Any]:
    state = payload if isinstance(payload, dict) else {}
    state.setdefault("schema_version", 1)
    state.setdefault("engine", "v6_execution")
    state.setdefault("positions", [])
    state.setdefault("position_lots", [])
    state.setdefault("journal", [])
    state.setdefault("entry_cooldowns", {})
    return state


def lot_qty(lot: dict[str, Any]) -> float:
    try:
        return max(0.0, float(lot.get("remaining_quantity") if lot.get("remaining_quantity") is not None else lot.get("quantity") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def position_key(symbol: str, side: str) -> str:
    return f"{str(symbol or '').upper()}:{str(side or '').lower()}"


def summarize_local_positions(state: dict[str, Any], engine: V6ExecutionEngine | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in state.get("positions") or []:
        if not isinstance(position, dict):
            continue
        if engine is not None and engine.is_strategy_dust_position(position):
            continue
        lots = [lot for lot in position.get("lots") or [] if isinstance(lot, dict)]
        rows.append(
            {
                "key": position_key(position.get("symbol", ""), position.get("side", "")),
                "symbol": str(position.get("symbol") or "").upper(),
                "side": str(position.get("side") or "").lower(),
                "position_qty": round(abs(float(position.get("position_qty") or position.get("remaining_quantity") or 0.0)), 12),
                "lot_qty": round(sum(lot_qty(lot) for lot in lots), 12),
                "lot_count": len([lot for lot in lots if lot_qty(lot) > 0]),
                "entry_price": float(position.get("entry_price") or 0.0),
                "mark_price": float(position.get("current_price") or 0.0),
                "horizon": position.get("horizon") or "",
                "entry_timeframe": position.get("entry_timeframe") or "",
                "last_exchange_sync_at": position.get("last_exchange_sync_at") or "",
            }
        )
    return rows


def summarize_exchange_rows(rows: list[dict[str, Any]], client: V6BinanceClient | None = None) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in rows:
        snapshot = V6BinanceClient.exchange_position_snapshot(row)
        if client is not None and client.is_dust_exchange_snapshot(snapshot):
            continue
        symbol = str(snapshot.get("symbol") or "").upper()
        side = str(snapshot.get("side") or "").lower()
        qty = abs(float(snapshot.get("position_amt") or snapshot.get("abs_position_qty") or 0.0))
        if not symbol or not side or qty <= 0:
            continue
        summary.append(
            {
                "key": position_key(symbol, side),
                "symbol": symbol,
                "side": side,
                "position_qty": round(qty, 12),
                "entry_price": float(snapshot.get("entry_price") or 0.0),
                "mark_price": float(snapshot.get("mark_price") or 0.0),
                "notional": abs(float(snapshot.get("notional") or 0.0)),
                "leverage": int(snapshot.get("leverage") or 0),
                "margin_type": snapshot.get("margin_type") or "",
            }
        )
    return sorted(summary, key=lambda item: item["key"])


def compare_local_exchange(local_rows: list[dict[str, Any]], exchange_rows: list[dict[str, Any]], *, tolerance: float) -> list[dict[str, Any]]:
    local_by_key = {row["key"]: row for row in local_rows}
    exchange_by_key = {row["key"]: row for row in exchange_rows}
    keys = sorted(set(local_by_key) | set(exchange_by_key))
    mismatches: list[dict[str, Any]] = []
    for key in keys:
        local = local_by_key.get(key)
        exchange = exchange_by_key.get(key)
        if not local:
            mismatches.append({"key": key, "type": "missing_local", "exchange": exchange})
            continue
        if not exchange:
            mismatches.append({"key": key, "type": "missing_exchange", "local": local})
            continue
        local_qty = float(local.get("position_qty") or 0.0)
        lot_qty_value = float(local.get("lot_qty") or 0.0)
        exchange_qty = float(exchange.get("position_qty") or 0.0)
        if abs(local_qty - exchange_qty) > tolerance or abs(lot_qty_value - exchange_qty) > tolerance:
            mismatches.append(
                {
                    "key": key,
                    "type": "quantity_mismatch",
                    "local_position_qty": local_qty,
                    "local_lot_qty": lot_qty_value,
                    "exchange_qty": exchange_qty,
                }
            )
    return mismatches


def run_reconcile(*, check_only: bool, tolerance: float) -> dict[str, Any]:
    config = V6ExecutionConfig.from_env()
    config.ensure_dirs()
    report_file = Path(os.getenv("TS_V6_LEDGER_RECONCILE_REPORT_FILE", str(config.state_dir / "ledger_reconcile_latest.json")))
    history_file = Path(os.getenv("TS_V6_LEDGER_RECONCILE_HISTORY_FILE", str(config.state_dir / "ledger_reconcile.jsonl")))
    engine = V6ExecutionEngine(config)
    started = time.time()
    try:
        if config.order_mode == "dry_run":
            report = {
                "event": "v6_ledger_reconcile",
                "status": "skipped",
                "reason": "dry_run_order_mode",
                "generated_at": utc_now_iso(),
                "state_file": str(config.state_file),
            }
            atomic_write_json(report_file, report)
            append_text_line(history_file, json.dumps(report, ensure_ascii=False))
            return report

        exchange_rows_before = engine.client.actual_position_rows()
        exchange_before = summarize_exchange_rows(exchange_rows_before, engine.client)
        with v6_execution_lock_context(config):
            with file_lock(config.state_file):
                state = normalize_state(load_json(config.state_file, default={}, lock=False))
                local_before = summarize_local_positions(state, engine)
                before_mismatches = compare_local_exchange(local_before, exchange_before, tolerance=tolerance)
                reconcile_result = (
                    {
                        "event": "exchange_position_reconcile",
                        "status": "check_only",
                        "reason": "check_only",
                    }
                    if check_only
                    else engine.reconcile_state_with_exchange(
                        state,
                        actual_rows=exchange_rows_before,
                        sync_exchange_side_effects=False,
                    )
                )
                if not check_only and reconcile_result and reconcile_result.get("status") == "ok":
                    state["updated_at"] = utc_now_iso()
                    state["last_ledger_reconcile"] = {
                        "status": "ok",
                        "synced_at": utc_now_iso(),
                        "script": "scripts/run_v6_ledger_reconcile.py",
                    }
                    atomic_write_json(config.state_file, state, lock=False)
                local_after = summarize_local_positions(state, engine)

        exchange_after = summarize_exchange_rows(engine.client.actual_position_rows(), engine.client)
        after_mismatches = compare_local_exchange(local_after, exchange_after, tolerance=tolerance)
        status = "ok"
        if reconcile_result and reconcile_result.get("status") == "failed":
            status = "failed"
        elif after_mismatches:
            status = "mismatch"
        elif before_mismatches and not check_only:
            status = "repaired"
        report = {
            "event": "v6_ledger_reconcile",
            "status": status,
            "check_only": check_only,
            "generated_at": utc_now_iso(),
            "elapsed_seconds": round(time.time() - started, 3),
            "order_mode": config.order_mode,
            "state_file": str(config.state_file),
            "local_before": local_before,
            "exchange_before": exchange_before,
            "local_after": local_after,
            "exchange_after": exchange_after,
            "before_mismatches": before_mismatches,
            "after_mismatches": after_mismatches,
            "reconcile_result": reconcile_result or {},
        }
        atomic_write_json(report_file, report)
        append_text_line(history_file, json.dumps(report, ensure_ascii=False))
        return report
    except TimeoutError as exc:
        report = {
            "event": "v6_ledger_reconcile",
            "status": "skipped",
            "reason": "v6_execution_lock_busy",
            "generated_at": utc_now_iso(),
            "elapsed_seconds": round(time.time() - started, 3),
            "order_mode": config.order_mode,
            "state_file": str(config.state_file),
            "error": f"{type(exc).__name__}: {redact_sensitive_text(exc)}",
            "note": "Skipped ledger reconciliation because another V6 execution flow held the execution lock.",
        }
        atomic_write_json(report_file, report)
        append_text_line(history_file, json.dumps(report, ensure_ascii=False))
        return report
    except V6ExecutionError as exc:
        report = {
            "event": "v6_ledger_reconcile",
            "status": "exchange_unavailable",
            "reason": "binance_position_risk_unavailable",
            "generated_at": utc_now_iso(),
            "elapsed_seconds": round(time.time() - started, 3),
            "order_mode": config.order_mode,
            "state_file": str(config.state_file),
            "error": f"{type(exc).__name__}: {redact_sensitive_text(exc)}",
            "note": "Skipped ledger reconciliation because exchange positionRisk could not be read. Local state was not modified.",
        }
        atomic_write_json(report_file, report)
        append_text_line(history_file, json.dumps(report, ensure_ascii=False))
        return report
    finally:
        engine.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconcile V6 local lot ledger with Binance positionRisk.")
    parser.add_argument("--check-only", action="store_true", help="Only report mismatches; do not mutate V6 state.")
    parser.add_argument("--tolerance", type=float, default=float(os.getenv("TS_V6_LEDGER_RECONCILE_QTY_TOLERANCE", "0.00000001")))
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_reconcile(check_only=bool(args.check_only), tolerance=max(0.0, float(args.tolerance)))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "generated_at": report.get("generated_at"),
                    "status": report.get("status"),
                    "before_mismatch_count": len(report.get("before_mismatches") or []),
                    "after_mismatch_count": len(report.get("after_mismatches") or []),
                    "state_file": report.get("state_file"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 1 if report.get("status") in {"failed", "mismatch"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
