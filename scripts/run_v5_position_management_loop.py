#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


os.umask(0o077)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = ROOT / "data/trading_system/v5_v6_execution"
DEFAULT_REPORT = DEFAULT_STATE_DIR / "latest_v5_position_management_report.json"
DEFAULT_MAIN_EXECUTION_FLAG = DEFAULT_STATE_DIR / "v5_main_execution_requested.flag"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_heartbeat(args: argparse.Namespace, status: dict[str, object]) -> None:
    heartbeat = Path(args.state_dir) / "latest_position_management_loop.json"
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def main_execution_requested(args: argparse.Namespace) -> bool:
    flag = Path(args.main_execution_flag)
    if not flag.exists():
        return False
    try:
        age_seconds = time.time() - flag.stat().st_mtime
    except OSError:
        return False
    return age_seconds <= float(args.main_execution_flag_max_age_seconds)


def write_skip_heartbeat(args: argparse.Namespace, reason: str) -> dict[str, object]:
    status = {
        "generated_at": iso_now(),
        "mode": "v5_position_management_loop",
        "returncode": 0,
        "skipped": True,
        "reason": reason,
        "interval_seconds": float(args.interval_seconds),
        "min_rest_seconds": float(args.min_rest_seconds),
    }
    write_heartbeat(args, status)
    return status


def run_once(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    if main_execution_requested(args):
        return 0, write_skip_heartbeat(args, "main_execution_requested")

    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_v5_v6_execution.py"),
        "--positions-only",
        "--state-dir",
        str(args.state_dir),
        "--output-report",
        str(args.output_report),
        "--no-archive-v5-requests",
        "--lock-timeout-seconds",
        str(args.lock_timeout_seconds),
    ]
    if args.submit:
        cmd.append("--submit")
    if args.json:
        cmd.append("--json")
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=float(args.child_timeout_seconds),
        )
        returncode = proc.returncode
        stdout_tail = proc.stdout.strip()[-2000:] if proc.stdout.strip() else ""
        stderr_tail = proc.stderr.strip()[-2000:] if proc.stderr.strip() else ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stdout_tail = str(stdout).strip()[-2000:]
        stderr_tail = (str(stderr).strip()[-2000:] + "\nchild process timed out").strip()
        timed_out = True
    elapsed_ms = round((time.time() - started) * 1000, 3)
    status = {
        "generated_at": iso_now(),
        "mode": "v5_position_management_loop",
        "last_elapsed_ms": elapsed_ms,
        "returncode": returncode,
        "timed_out": timed_out,
        "child_timeout_seconds": float(args.child_timeout_seconds),
        "interval_seconds": float(args.interval_seconds),
        "min_rest_seconds": float(args.min_rest_seconds),
    }
    if stdout_tail:
        status["stdout_tail"] = stdout_tail
    if stderr_tail:
        status["stderr_tail"] = stderr_tail
    write_heartbeat(args, status)
    if returncode != 0 or args.verbose:
        print(json.dumps(status, ensure_ascii=False, separators=(",", ":")), flush=True)
    return returncode, status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V5 existing-position management loop.")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--output-report", default=str(DEFAULT_REPORT))
    parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("TS_V5_POSITION_MANAGEMENT_INTERVAL_SECONDS", "5")))
    parser.add_argument("--min-rest-seconds", type=float, default=float(os.getenv("TS_V5_POSITION_MANAGEMENT_MIN_REST_SECONDS", "1")))
    parser.add_argument(
        "--lock-timeout-seconds",
        type=float,
        default=float(os.getenv("TS_V5_POSITION_MANAGEMENT_LOCK_TIMEOUT_SECONDS", os.getenv("TS_V5_EXECUTION_LOCK_TIMEOUT_SECONDS", "30"))),
    )
    parser.add_argument("--child-timeout-seconds", type=float, default=float(os.getenv("TS_V5_POSITION_MANAGEMENT_CHILD_TIMEOUT_SECONDS", "180")))
    parser.add_argument("--main-execution-flag", default=os.getenv("TS_V5_MAIN_EXECUTION_FLAG", str(DEFAULT_MAIN_EXECUTION_FLAG)))
    parser.add_argument("--main-execution-flag-max-age-seconds", type=float, default=float(os.getenv("TS_V5_MAIN_EXECUTION_FLAG_MAX_AGE_SECONDS", "900")))
    parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    interval = max(1.0, float(args.interval_seconds))
    min_rest = max(0.0, float(args.min_rest_seconds))
    consecutive_errors = 0
    while True:
        started = time.time()
        code, status = run_once(args)
        if code == 0:
            consecutive_errors = 0
        else:
            consecutive_errors += 1
            if consecutive_errors >= 10:
                return code
        elapsed = time.time() - started
        sleep_seconds = max(min_rest, interval - elapsed)
        status["next_sleep_seconds"] = round(sleep_seconds, 3)
        status["next_run_not_before"] = datetime.fromtimestamp(time.time() + sleep_seconds, timezone.utc).isoformat()
        write_heartbeat(args, status)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
