#!/usr/bin/env python3
"""Rotate and prune Somnus maintenance files without deleting live state."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

MB = 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "data" / "trading_system" / "maintenance_archives" / "jsonl"
LOCK_PATH = ARCHIVE_ROOT / "jsonl_rotate.lock"


@dataclass(frozen=True)
class Target:
    path: Path
    max_mb: int
    keep_lines: int


TARGETS = (
    Target(ROOT / "data/trading_system/v6_execution/execution_wal.jsonl", 32, 100),
    Target(ROOT / "data/trading_system/v6_execution/signals.jsonl", 64, 1000),
    Target(ROOT / "data/trading_system/v6_execution/ledger_reconcile.jsonl", 32, 500),
    Target(ROOT / "data/trading_system/whale_overview/snapshot_ledger.jsonl", 64, 1000),
)

PRUNE_LOG_PATTERNS = (
    "data/trading_system/logs/*.log",
)

PRUNE_CAPTURE_PATTERNS = (
    "data/trading_system/model_training/whale_scan/live_distill/captures/*.json",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def archive_dir() -> Path:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = ARCHIVE_ROOT / today
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as fh:
        for _ in fh:
            total += 1
    return total


def copy_mode(src: Path, dst: Path) -> None:
    try:
        st = src.stat()
        os.chmod(dst, st.st_mode & 0o777)
    except OSError:
        pass


def rotate_target(target: Target, dry_run: bool = False) -> dict:
    path = target.path
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "status": "missing"}

    size = path.stat().st_size
    threshold = target.max_mb * MB
    if size <= threshold:
        return {
            "path": str(path.relative_to(ROOT)),
            "status": "kept",
            "size_mb": round(size / MB, 2),
            "threshold_mb": target.max_mb,
        }

    total = count_lines(path)
    if total <= target.keep_lines:
        return {
            "path": str(path.relative_to(ROOT)),
            "status": "too_few_lines",
            "size_mb": round(size / MB, 2),
            "lines": total,
            "keep_lines": target.keep_lines,
        }

    rotate_lines = total - target.keep_lines
    rel = path.relative_to(ROOT)
    safe_name = "__".join(rel.parts).replace(".jsonl", "")
    archive_path = archive_dir() / f"{safe_name}.{utc_stamp()}.head{rotate_lines}.jsonl.gz"
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")

    if dry_run:
        return {
            "path": str(rel),
            "status": "would_rotate",
            "size_mb": round(size / MB, 2),
            "lines": total,
            "archive_lines": rotate_lines,
            "keep_lines": target.keep_lines,
            "archive": str(archive_path.relative_to(ROOT)),
        }

    archived = 0
    kept = 0
    try:
        with path.open("rb") as src, gzip.open(archive_path, "wb", compresslevel=6) as gz, tmp_path.open("wb") as tmp:
            for idx, line in enumerate(src, start=1):
                if idx <= rotate_lines:
                    gz.write(line)
                    archived += 1
                else:
                    tmp.write(line)
                    kept += 1
        copy_mode(path, tmp_path)
        os.replace(tmp_path, path)
        return {
            "path": str(rel),
            "status": "rotated",
            "before_mb": round(size / MB, 2),
            "after_mb": round(path.stat().st_size / MB, 2),
            "lines": total,
            "archived_lines": archived,
            "kept_lines": kept,
            "archive": str(archive_path.relative_to(ROOT)),
            "archive_mb": round(archive_path.stat().st_size / MB, 2),
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def prune_archives(max_days: int) -> dict:
    if max_days <= 0 or not ARCHIVE_ROOT.exists():
        return {"status": "disabled"}
    now = datetime.now(timezone.utc).timestamp()
    cutoff = max_days * 24 * 3600
    removed = []
    for path in ARCHIVE_ROOT.rglob("*.jsonl.gz"):
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age > cutoff:
            size = path.stat().st_size
            path.unlink()
            removed.append({"path": str(path.relative_to(ROOT)), "mb": round(size / MB, 2)})
    for directory in sorted((p for p in ARCHIVE_ROOT.rglob("*") if p.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    return {"status": "ok", "removed_count": len(removed), "removed_mb": round(sum(x["mb"] for x in removed), 2)}


def prune_patterns(name: str, patterns: Sequence[str], max_days: int, dry_run: bool = False) -> dict:
    if max_days <= 0:
        return {"name": name, "status": "disabled"}
    now = datetime.now(timezone.utc).timestamp()
    cutoff = max_days * 24 * 3600
    removed_count = 0
    removed_bytes = 0
    matched_count = 0
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            matched_count += 1
            try:
                st = path.stat()
            except OSError:
                continue
            if now - st.st_mtime <= cutoff:
                continue
            removed_count += 1
            removed_bytes += st.st_size
            if not dry_run:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
    return {
        "name": name,
        "status": "ok",
        "matched_count": matched_count,
        "removed_count": removed_count,
        "removed_mb": round(removed_bytes / MB, 2),
        "dry_run": dry_run,
        "max_days": max_days,
    }


def acquire_lock():
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_fh = LOCK_PATH.open("w")
    if fcntl is not None:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
    return lock_fh


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-archive-days", type=int, default=3)
    parser.add_argument("--max-log-days", type=int, default=3)
    parser.add_argument("--max-capture-days", type=int, default=3)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    with acquire_lock():
        results = [rotate_target(target, dry_run=args.dry_run) for target in TARGETS]
        archive_prune = prune_archives(args.max_archive_days) if not args.dry_run else {"status": "skipped_dry_run"}
        log_prune = prune_patterns("logs", PRUNE_LOG_PATTERNS, args.max_log_days, dry_run=args.dry_run)
        capture_prune = prune_patterns("raw_captures", PRUNE_CAPTURE_PATTERNS, args.max_capture_days, dry_run=args.dry_run)
    print(json.dumps({"ok": True, "results": results, "archive_prune": archive_prune, "log_prune": log_prune, "capture_prune": capture_prune}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
