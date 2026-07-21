#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys
import time
from datetime import datetime, timezone
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

from trading_system.config import StrategyConfig  # noqa: E402
from trading_system.persistence import atomic_write_json, file_lock, load_json  # noqa: E402
from trading_system.redaction import redact_sensitive_text  # noqa: E402
from trading_system.v6_execution import V6BinanceClient, V6ExecutionEngine  # noqa: E402
from scripts.run_v5_v6_execution import configure_v6_with_v5_keys, v5_execution_lock  # noqa: E402


DEFAULT_STATE_DIR = Path("data/trading_system/v5_v6_execution")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def position_signature(state: dict[str, Any]) -> list[tuple[str, str, float, float, float, bool, bool]]:
    rows: list[tuple[str, str, float, float, float, bool, bool]] = []
    for item in state.get("positions") or []:
        if not isinstance(item, dict):
            continue
        try:
            qty = round(abs(float(item.get("position_qty") or item.get("remaining_quantity") or 0.0)), 12)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            entry = round(float(item.get("entry_price") or 0.0), 8)
        except (TypeError, ValueError):
            entry = 0.0
        try:
            stop = round(float(item.get("stop_loss_price") or 0.0), 8)
        except (TypeError, ValueError):
            stop = 0.0
        risk_plan = item.get("risk_plan") if isinstance(item.get("risk_plan"), dict) else {}
        overlay = risk_plan.get("v5_stop_overlay") if isinstance(risk_plan.get("v5_stop_overlay"), dict) else {}
        rows.append((
            str(item.get("symbol") or "").upper(),
            str(item.get("side") or "").lower(),
            qty,
            entry,
            stop,
            bool(overlay.get("enabled")),
            bool(item.get("v5_overlay_protective_stop_synced_at")),
        ))
    return sorted(rows)


def exchange_snapshot_summary(engine: V6ExecutionEngine, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rows is None:
        rows = engine.client.actual_position_rows()
    active: list[dict[str, Any]] = []
    dust: list[dict[str, Any]] = []
    for row in rows:
        snapshot = V6BinanceClient.exchange_position_snapshot(row)
        if not snapshot:
            continue
        item = {
            "symbol": snapshot.get("symbol"),
            "side": snapshot.get("side"),
            "position_qty": snapshot.get("abs_position_qty"),
            "entry_price": snapshot.get("entry_price"),
            "mark_price": snapshot.get("mark_price"),
            "notional": round(engine.client.exchange_snapshot_notional(snapshot), 8),
            "unrealized_profit": snapshot.get("unrealized_profit"),
            "leverage": snapshot.get("leverage"),
            "margin_type": snapshot.get("margin_type"),
            "position_side": snapshot.get("position_side"),
            "update_time": snapshot.get("update_time"),
        }
        if engine.client.is_dust_exchange_snapshot(snapshot):
            dust.append(item)
        else:
            active.append(item)
    return {
        "active": sorted(active, key=lambda item: (str(item.get("symbol") or ""), str(item.get("side") or ""))),
        "dust": sorted(dust, key=lambda item: (str(item.get("symbol") or ""), str(item.get("side") or ""))),
    }


def synthetic_exchange_row(position: dict[str, Any]) -> dict[str, Any]:
    side = str(position.get("side") or "").lower()
    quantity = abs(float(position.get("position_qty") or position.get("remaining_quantity") or 0.0))
    signed_quantity = -quantity if side == "short" else quantity
    entry = float(position.get("entry_price") or 0.0)
    mark = float(position.get("current_price") or entry or 0.0)
    return {
        "symbol": str(position.get("symbol") or "").upper(),
        "positionAmt": str(signed_quantity),
        "positionSide": "SHORT" if side == "short" else "LONG",
        "entryPrice": str(entry),
        "markPrice": str(mark),
        "notional": str(signed_quantity * mark),
        "unRealizedProfit": str(position.get("unrealized_profit") or 0.0),
        "leverage": str(position.get("leverage") or 0),
        "marginType": str(position.get("margin_type") or "isolated"),
        "_somnus_transient_absence_guard": True,
    }


def guard_transient_absences(
    engine: V6ExecutionEngine,
    state: dict[str, Any],
    actual_rows: list[dict[str, Any]],
    confirmation_state: dict[str, Any],
    *,
    now_ts: float,
    required_confirmations: int,
    minimum_elapsed_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Keep a local position until repeated exchange-flat snapshots confirm it."""
    rows = [dict(row) for row in actual_rows if isinstance(row, dict)]
    actual_keys = {engine.position_key_from_risk_row(row) for row in rows}
    local_positions = [item for item in state.get("positions") or [] if isinstance(item, dict)]
    local_keys = {engine.position_key_from_state(item) for item in local_positions}
    records = confirmation_state.get("missing") if isinstance(confirmation_state.get("missing"), dict) else {}
    next_records: dict[str, dict[str, Any]] = {}
    protected: list[str] = []
    confirmed_absent: list[str] = []

    for position in local_positions:
        key = engine.position_key_from_state(position)
        key_text = f"{key[0]}:{key[1]}"
        if key in actual_keys:
            continue
        if engine.is_strategy_dust_position(position):
            confirmed_absent.append(key_text)
            continue
        previous = records.get(key_text) if isinstance(records.get(key_text), dict) else {}
        first_seen = float(previous.get("first_seen_ts") or now_ts)
        count = int(previous.get("count") or 0) + 1
        record = {
            "first_seen_ts": first_seen,
            "last_seen_ts": now_ts,
            "count": count,
        }
        elapsed = max(0.0, now_ts - first_seen)
        confirmed = count >= max(1, int(required_confirmations)) and elapsed >= max(0.0, float(minimum_elapsed_seconds))
        record["confirmed"] = confirmed
        next_records[key_text] = record
        if confirmed:
            confirmed_absent.append(key_text)
            continue
        synthetic = synthetic_exchange_row(position)
        if abs(float(synthetic.get("positionAmt") or 0.0)) > 0:
            rows.append(synthetic)
            protected.append(key_text)

    next_records = {key: value for key, value in next_records.items() if key in {f"{item[0]}:{item[1]}" for item in local_keys}}
    next_state = {
        "updated_at": utc_now(),
        "missing": next_records,
    }
    return rows, next_state, {
        "protected": sorted(protected),
        "confirmed_absent": sorted(confirmed_absent),
        "required_confirmations": max(1, int(required_confirmations)),
        "minimum_elapsed_seconds": max(0.0, float(minimum_elapsed_seconds)),
    }


def reconcile_once(
    engine: V6ExecutionEngine,
    state_file: Path,
    summary_file: Path,
    events_file: Path,
    *,
    state_dir: Path | None = None,
    confirmation_state_file: Path | None = None,
    absence_required_confirmations: int = 2,
    absence_minimum_elapsed_seconds: float = 5.0,
    lock_timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    started = time.time()
    lock_ctx = (
        v5_execution_lock(state_dir, lock_timeout_seconds)
        if state_dir is not None
        else nullcontext()
    )
    with lock_ctx:
        # Read the exchange snapshot only after the execution lock is held. A
        # pre-lock snapshot can resurrect a position that was closed while this
        # watcher waited for the lock.
        actual_rows = engine.client.actual_position_rows()
        exchange_positions = exchange_snapshot_summary(engine, actual_rows)
        with file_lock(state_file):
            state = load_json(state_file, default={}, lock=False)
            if not isinstance(state, dict):
                state = {}
            confirmation_state_path = confirmation_state_file or state_file.with_name("position_reconcile_confirmation_state.json")
            confirmation_state = load_json(confirmation_state_path, default={}, lock=False)
            if not isinstance(confirmation_state, dict):
                confirmation_state = {}
            previous_confirmation_state = confirmation_state
            guarded_rows, confirmation_state, absence_guard = guard_transient_absences(
                engine,
                state,
                actual_rows,
                confirmation_state,
                now_ts=time.time(),
                required_confirmations=absence_required_confirmations,
                minimum_elapsed_seconds=absence_minimum_elapsed_seconds,
            )
            before = position_signature(state)
            result = engine.reconcile_state_with_exchange(
                state,
                actual_rows=guarded_rows,
                sync_exchange_side_effects=False,
            )
            after = position_signature(state)
            changed = (
                before != after
                or bool((result or {}).get("removed"))
                or bool((result or {}).get("recovered"))
                or bool((result or {}).get("protective_stop_sync"))
                or bool((result or {}).get("v5_overlay_migrations"))
            )
            if changed:
                atomic_write_json(state_file, state, lock=False)
            previous_missing = previous_confirmation_state.get("missing") if isinstance(previous_confirmation_state.get("missing"), dict) else {}
            current_missing = confirmation_state.get("missing") if isinstance(confirmation_state.get("missing"), dict) else {}
            if current_missing != previous_missing:
                atomic_write_json(confirmation_state_path, confirmation_state, lock=False)
    latency_ms = round((time.time() - started) * 1000, 3)
    payload = {
        "generated_at": utc_now(),
        "mode": "v5_position_reconcile_watcher",
        "status": (result or {}).get("status") or "ok",
        "changed": changed,
        "latency_ms": latency_ms,
        "position_count": len(state.get("positions") or []),
        "before": before,
        "after": after,
        "reconcile": result or {"event": "exchange_position_reconcile", "status": "skipped"},
        "absence_guard": absence_guard,
        "exchange_positions": exchange_positions,
    }
    atomic_write_json(summary_file, payload)
    if changed or payload["status"] != "ok":
        append_jsonl(events_file, payload)
    return payload


def build_lightweight_reconcile_engine(config: Any) -> V6ExecutionEngine:
    config.binance_request_timeout_seconds = min(
        float(config.binance_request_timeout_seconds),
        max(1.0, float(os.getenv("TS_V5_POSITION_RECONCILE_REQUEST_TIMEOUT_SECONDS", "3"))),
    )
    config.ensure_dirs()
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = config
    engine.client = V6BinanceClient(config)
    engine.scorer = None
    engine._market_daily_ma7_reversal_cache = None
    full_protective_stop_sync = V6ExecutionEngine.sync_protective_stops_with_exchange.__get__(engine, V6ExecutionEngine)
    protective_stop_sync_interval = max(
        0.0,
        float(os.getenv("TS_V5_PROTECTIVE_STOP_SYNC_INTERVAL_SECONDS", "30")),
    )
    engine._v5_next_full_protective_stop_sync_at = 0.0

    def sync_only_recent_v5_overlay_migrations(positions: list[dict[str, Any]], actual_by_key: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
        now = time.time()
        if now >= float(getattr(engine, "_v5_next_full_protective_stop_sync_at", 0.0) or 0.0):
            engine._v5_next_full_protective_stop_sync_at = now + protective_stop_sync_interval
            return full_protective_stop_sync(positions, actual_by_key)
        pending = [
            item for item in positions
            if isinstance(item, dict)
            and item.get("v5_overlay_migrated_at")
            and not item.get("v5_overlay_protective_stop_synced_at")
        ]
        if not pending:
            return []
        actions = V6ExecutionEngine.sync_protective_stops_with_exchange(engine, pending, actual_by_key)
        synced_at = utc_now()
        for item in pending:
            item["v5_overlay_protective_stop_synced_at"] = synced_at
            item["v5_overlay_protective_stop_sync_actions"] = [
                action for action in actions
                if isinstance(action, dict) and str(action.get("symbol") or "").upper() == str(item.get("symbol") or "").upper()
            ][-10:]
        return actions

    # Keep the 1s watcher light. It only performs signed protective-stop sync once
    # for existing positions migrated into the V5 overlay stop model.
    engine.sync_protective_stops_with_exchange = sync_only_recent_v5_overlay_migrations
    return engine


def run_loop(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    state_file = state_dir / "state.json"
    summary_file = state_dir / "latest_position_reconcile_watch.json"
    events_file = state_dir / "position_reconcile_events.jsonl"
    confirmation_state_file = state_dir / "position_reconcile_confirmation_state.json"
    config = configure_v6_with_v5_keys(StrategyConfig(), state_dir, [])
    engine = build_lightweight_reconcile_engine(config)
    interval = max(0.25, float(args.interval_seconds))
    errors = 0
    try:
        while True:
            started = time.time()
            cycle_failed = False
            try:
                payload = reconcile_once(
                    engine,
                    state_file,
                    summary_file,
                    events_file,
                    state_dir=state_dir,
                    confirmation_state_file=confirmation_state_file,
                    absence_required_confirmations=int(args.absence_required_confirmations),
                    absence_minimum_elapsed_seconds=float(args.absence_minimum_elapsed_seconds),
                    lock_timeout_seconds=float(args.lock_timeout_seconds),
                )
                errors = 0
                if args.verbose or payload.get("changed"):
                    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
            except TimeoutError:
                errors = 0
                payload = {
                    "generated_at": utc_now(),
                    "mode": "v5_position_reconcile_watcher",
                    "status": "skipped_lock_busy",
                    "changed": False,
                    "reason": "v5_execution_in_progress",
                }
                atomic_write_json(summary_file, payload)
                if args.verbose:
                    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
            except Exception as exc:
                errors += 1
                cycle_failed = True
                payload = {
                    "generated_at": utc_now(),
                    "mode": "v5_position_reconcile_watcher",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {redact_sensitive_text(exc)}",
                    "consecutive_errors": errors,
                }
                atomic_write_json(summary_file, payload)
                append_jsonl(events_file, payload)
                print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
                if errors >= int(args.max_consecutive_errors):
                    return 2
            if args.once:
                return 0
            elapsed = time.time() - started
            if cycle_failed:
                time.sleep(max(interval, min(5.0, float(errors))))
            else:
                time.sleep(max(0.0, interval - elapsed))
    finally:
        engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Poll V5 Binance positions and reconcile V5 local ledger.")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("TS_V5_POSITION_RECONCILE_INTERVAL_SECONDS", "1")))
    parser.add_argument(
        "--lock-timeout-seconds",
        type=float,
        default=float(os.getenv("TS_V5_POSITION_RECONCILE_LOCK_TIMEOUT_SECONDS", "0.20")),
    )
    parser.add_argument(
        "--absence-required-confirmations",
        type=int,
        default=int(os.getenv("TS_V5_POSITION_RECONCILE_ABSENCE_CONFIRMATIONS", "2")),
    )
    parser.add_argument(
        "--absence-minimum-elapsed-seconds",
        type=float,
        default=float(os.getenv("TS_V5_POSITION_RECONCILE_ABSENCE_MIN_SECONDS", "5")),
    )
    parser.add_argument("--max-consecutive-errors", type=int, default=int(os.getenv("TS_V5_POSITION_RECONCILE_MAX_ERRORS", "30")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return run_loop(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
