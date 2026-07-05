#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_LIMITS = {"1m": 4707, "15m": 4707, "4h": 4707, "1d": 4707}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def inside_root(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path is outside repo root: {path}")
    return resolved


def row_open_time(row: Any) -> int | None:
    try:
        if isinstance(row, dict):
            value = row.get("open_time") or row.get("openTime") or row.get("t")
        else:
            value = row[0]
        return int(value)
    except Exception:
        return None


def dedupe_sorted(rows: list[Any]) -> list[Any]:
    by_open: dict[int, Any] = {}
    for row in rows:
        open_time = row_open_time(row)
        if open_time is not None:
            by_open[open_time] = row
    return [by_open[key] for key in sorted(by_open)]


def load_rows(path: Path) -> tuple[Any, list[Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, payload, "list"
    if isinstance(payload, dict):
        rows = payload.get("klines")
        if isinstance(rows, list):
            return payload, rows, "klines"
        rows = payload.get("data")
        if isinstance(rows, list):
            return payload, rows, "data"
    return payload, [], "unknown"


def write_rows(path: Path, payload: Any, rows: list[Any], key: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if key == "list":
        body = rows
    elif isinstance(payload, dict) and key in {"klines", "data"}:
        body = dict(payload)
        body[key] = rows
        body["limit"] = len(rows)
        if rows:
            body["first_open_time"] = row_open_time(rows[0])
            body["last_open_time"] = row_open_time(rows[-1])
        body["updated_at"] = utc_now_iso()
    else:
        return
    tmp.write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def prune_execution_cache(cache_dir: Path, *, dry_run: bool) -> dict[str, int]:
    result = {"files_seen": 0, "files_pruned": 0, "rows_removed": 0, "errors": 0}
    if not cache_dir.exists():
        return result
    for interval, limit in EXECUTION_LIMITS.items():
        interval_dir = cache_dir / interval
        if not interval_dir.exists():
            continue
        for path in interval_dir.glob("*.json"):
            result["files_seen"] += 1
            try:
                payload, rows, key = load_rows(path)
                normalized = dedupe_sorted(rows)
                pruned = normalized[-limit:]
                removed = max(0, len(normalized) - len(pruned))
                if removed <= 0 and len(normalized) == len(rows):
                    continue
                result["files_pruned"] += 1
                result["rows_removed"] += removed
                if not dry_run:
                    write_rows(path, payload, pruned, key)
            except Exception:
                result["errors"] += 1
    return result


def remove_empty_dirs(root: Path) -> int:
    removed = 0
    if not root.exists():
        return removed
    for path in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def clean_expired_files(cache_dir: Path, *, ttl_days: float, dry_run: bool) -> dict[str, int]:
    result = {"files_seen": 0, "files_deleted": 0, "bytes_deleted": 0, "errors": 0, "empty_dirs_removed": 0}
    if not cache_dir.exists():
        return result
    cutoff = time.time() - float(ttl_days) * 86400.0
    for path in cache_dir.rglob("*.json"):
        result["files_seen"] += 1
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            result["files_deleted"] += 1
            result["bytes_deleted"] += path.stat().st_size
            if not dry_run:
                path.unlink()
        except Exception:
            result["errors"] += 1
    if not dry_run:
        result["empty_dirs_removed"] = remove_empty_dirs(cache_dir)
    return result


def clean_stale_execution_request_locks(request_dir: Path, *, ttl_days: float, dry_run: bool) -> dict[str, int]:
    result = {"files_seen": 0, "files_deleted": 0, "bytes_deleted": 0, "errors": 0}
    if not request_dir.exists():
        return result
    cutoff = time.time() - float(ttl_days) * 86400.0
    for path in request_dir.glob("*.lock"):
        result["files_seen"] += 1
        try:
            stat = path.stat()
            if stat.st_mtime >= cutoff:
                continue
            result["files_deleted"] += 1
            result["bytes_deleted"] += stat.st_size
            if not dry_run:
                path.unlink()
        except Exception:
            result["errors"] += 1
    return result


def active_symbols_from_json(path: Path) -> set[str]:
    symbols: set[str] = set()
    if not path.exists():
        return symbols
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return symbols
    for item in payload.get("symbols") or []:
        if isinstance(item, str) and item:
            symbols.add(item.upper())
    for position in payload.get("positions") or []:
        if isinstance(position, dict):
            symbol = str(position.get("symbol") or "").upper()
            if symbol:
                symbols.add(symbol)
    return symbols


def collect_active_symbols() -> set[str]:
    files = [
        ROOT / "data/trading_system/v6_execution/latest_run_summary.json",
        ROOT / "data/trading_system/v6_execution/state.json",
        ROOT / "data/trading_system/v5_v6_execution/latest_run_summary.json",
        ROOT / "data/trading_system/v5_v6_execution/state.json",
    ]
    symbols: set[str] = set()
    for path in files:
        symbols.update(active_symbols_from_json(path))
    return symbols


def clean_inactive_execution_symbols(cache_dir: Path, *, active_symbols: set[str], inactive_days: float, dry_run: bool) -> dict[str, int]:
    result = {"files_seen": 0, "files_deleted": 0, "bytes_deleted": 0, "errors": 0, "empty_dirs_removed": 0}
    if not cache_dir.exists() or not active_symbols:
        return result
    cutoff = time.time() - float(inactive_days) * 86400.0
    for interval_dir in [cache_dir / item for item in EXECUTION_LIMITS]:
        if not interval_dir.exists():
            continue
        for path in interval_dir.glob("*.json"):
            result["files_seen"] += 1
            try:
                symbol = path.stem.upper()
                if symbol in active_symbols or path.stat().st_mtime >= cutoff:
                    continue
                result["files_deleted"] += 1
                result["bytes_deleted"] += path.stat().st_size
                if not dry_run:
                    path.unlink()
            except Exception:
                result["errors"] += 1
    if not dry_run:
        result["empty_dirs_removed"] = remove_empty_dirs(cache_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-cache-dir", default=os.getenv("TS_EXECUTION_KLINE_CACHE_DIR", "data/trading_system/cache/execution_klines"))
    parser.add_argument("--legacy-execution-cache-dir", default="data/trading_system/v6_execution/kline_cache")
    parser.add_argument("--chart-cache-dir", default=os.getenv("TS_CHART_KLINE_CACHE_DIR", "data/trading_system/cache/chart_klines"))
    parser.add_argument("--legacy-chart-cache-dir", default="data/trading_system/telegram_monitor/kline_cache")
    parser.add_argument("--legacy-mixed-cache-dir", default="data/trading_system/cache/binance_klines")
    parser.add_argument("--execution-requests-dir", default=os.getenv("TS_EXECUTION_REQUESTS_DIR", "data/trading_system/execution_requests"))
    parser.add_argument("--chart-ttl-days", type=float, default=float(os.getenv("TS_CHART_KLINE_CACHE_TTL_DAYS", "7")))
    parser.add_argument("--inactive-execution-days", type=float, default=float(os.getenv("TS_INACTIVE_EXECUTION_KLINE_CACHE_DAYS", "14")))
    parser.add_argument("--inactive-legacy-mixed-days", type=float, default=float(os.getenv("TS_INACTIVE_LEGACY_MIXED_KLINE_CACHE_DAYS", "30")))
    parser.add_argument("--execution-request-lock-ttl-days", type=float, default=float(os.getenv("TS_EXECUTION_REQUEST_LOCK_TTL_DAYS", "2")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    execution_dir = inside_root(ROOT / args.execution_cache_dir)
    legacy_execution_dir = inside_root(ROOT / args.legacy_execution_cache_dir)
    chart_dir = inside_root(ROOT / args.chart_cache_dir)
    legacy_chart_dir = inside_root(ROOT / args.legacy_chart_cache_dir)
    legacy_mixed_dir = inside_root(ROOT / args.legacy_mixed_cache_dir)
    execution_requests_dir = inside_root(ROOT / args.execution_requests_dir)
    active_symbols = collect_active_symbols()

    summary = {
        "generated_at": utc_now_iso(),
        "dry_run": bool(args.dry_run),
        "active_symbols": sorted(active_symbols),
        "execution_cache": prune_execution_cache(execution_dir, dry_run=args.dry_run),
        "legacy_execution_cache": prune_execution_cache(legacy_execution_dir, dry_run=args.dry_run),
        "inactive_execution_cache": clean_inactive_execution_symbols(
            execution_dir,
            active_symbols=active_symbols,
            inactive_days=args.inactive_execution_days,
            dry_run=args.dry_run,
        ),
        "inactive_legacy_mixed_cache": clean_inactive_execution_symbols(
            legacy_mixed_dir,
            active_symbols=active_symbols,
            inactive_days=args.inactive_legacy_mixed_days,
            dry_run=args.dry_run,
        ),
        "chart_cache": clean_expired_files(chart_dir, ttl_days=args.chart_ttl_days, dry_run=args.dry_run),
        "legacy_chart_cache": clean_expired_files(legacy_chart_dir, ttl_days=args.chart_ttl_days, dry_run=args.dry_run),
        "execution_request_locks": clean_stale_execution_request_locks(
            execution_requests_dir,
            ttl_days=args.execution_request_lock_ttl_days,
            dry_run=args.dry_run,
        ),
    }
    out = ROOT / "data/trading_system/cache/kline_cache_maintenance_latest.json"
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
