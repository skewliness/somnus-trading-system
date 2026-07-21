#!/usr/bin/env python3
"""
Observation-period health check for the V4 shadow/testnet data loop.

Checks whether the pipeline is alive and whether core datasets are still
receiving fresh samples. This does not modify pipeline/model state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = ROOT / "data/trading_system/observation_health.json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc)
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": st.st_size,
        "mtime": mtime.isoformat(),
        "age_minutes": round((utc_now() - mtime).total_seconds() / 60.0, 2),
    }


def read_jsonl_stats(path: Path, timestamp_keys: tuple[str, ...], *, day_start: datetime) -> dict[str, Any]:
    info = file_info(path)
    if not path.exists():
        return {**info, "line_count": 0, "today_count": 0, "latest_sample_at": None, "latest_sample_age_minutes": None}
    line_count = 0
    today_count = 0
    latest_sample_at: datetime | None = None
    malformed = 0
    decision_labels = Counter()
    execution_status = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                row = json.loads(line)
            except Exception:
                malformed += 1
                continue
            ts = None
            for key in timestamp_keys:
                ts = parse_dt(row.get(key))
                if ts is not None:
                    break
            if ts is not None:
                if latest_sample_at is None or ts > latest_sample_at:
                    latest_sample_at = ts
                if ts >= day_start:
                    today_count += 1
            if row.get("decision_label") is not None:
                decision_labels[str(row.get("decision_label") or "")] += 1
            if row.get("execution_status") is not None:
                execution_status[str(row.get("execution_status") or "")] += 1
    latest_age = None
    if latest_sample_at is not None:
        latest_age = round((utc_now() - latest_sample_at).total_seconds() / 60.0, 2)
    return {
        **info,
        "line_count": line_count,
        "today_count": today_count,
        "latest_sample_at": latest_sample_at.isoformat() if latest_sample_at else None,
        "latest_sample_age_minutes": latest_age,
        "malformed_count": malformed,
        "decision_label_counts": dict(decision_labels),
        "execution_status_counts": dict(execution_status),
    }


def pid_status(pid_file: Path) -> dict[str, Any]:
    info = file_info(pid_file)
    pid_text = ""
    if pid_file.exists():
        pid_text = pid_file.read_text(encoding="utf-8", errors="ignore").strip()
    pid = int(pid_text) if pid_text.isdigit() else None
    running = False
    command = ""
    if pid is not None:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,etime=,stat=,command="],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            running = True
            command = result.stdout.strip()
    return {**info, "pid": pid, "running": running, "command": command}


def latest_log_info(log_dir: Path, pattern: str) -> dict[str, Any]:
    matches = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if not matches:
        return {"exists": False, "path": "", "pattern": pattern}
    return file_info(matches[0]) | {"pattern": pattern}


def status_rank(status: str) -> int:
    return {"OK": 0, "WARN": 1, "CRITICAL": 2}.get(status, 2)


def build_health(args: argparse.Namespace) -> dict[str, Any]:
    now = utc_now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    paths = {
        "decision_samples": ROOT / "data/trading_system/model_training/paper_replay_autolearn/paper_decision_samples.jsonl",
        "closed_trades": ROOT / "data/trading_system/model_training/paper_replay_autolearn/paper_closed_trades.jsonl",
        "pipeline_report": ROOT / "data/trading_system/pipeline_run_report.json",
        "paper_state": ROOT / "data/trading_system/paper_state.json",
        "pid_file": ROOT / "data/trading_system/pipeline_loop.pid",
        "lock_dir": ROOT / "data/trading_system/pipeline_loop.lock",
        "log_dir": ROOT / "data/trading_system/logs",
        "execution_results_dir": ROOT / "data/trading_system/execution_results",
    }

    decision = read_jsonl_stats(paths["decision_samples"], ("captured_at", "timestamp"), day_start=day_start)
    closed = read_jsonl_stats(paths["closed_trades"], ("exit_timestamp", "timestamp", "captured_at"), day_start=day_start)
    pid = pid_status(paths["pid_file"])
    report = file_info(paths["pipeline_report"])
    state = file_info(paths["paper_state"])
    pipeline_log = latest_log_info(paths["log_dir"], "pipeline_24h_*.log")
    trade_log = latest_log_info(paths["log_dir"], "trade_activity_*.log")

    execution_results_today = 0
    latest_execution_result = None
    latest_execution_mtime = None
    if paths["execution_results_dir"].exists():
        for path in paths["execution_results_dir"].glob("*_result.json"):
            mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            if latest_execution_mtime is None or mtime > latest_execution_mtime:
                latest_execution_mtime = mtime
                latest_execution_result = path
            if mtime >= day_start:
                execution_results_today += 1

    checks: list[dict[str, Any]] = []

    def add_check(name: str, status: str, detail: str, remediation: str = "") -> None:
        checks.append({"name": name, "status": status, "detail": detail, "remediation": remediation})

    if not pid["running"]:
        add_check("pipeline_process", "CRITICAL", "pipeline loop pid is not running", "Run scripts/start_pipeline_24h_background.sh")
    else:
        add_check("pipeline_process", "OK", f"pipeline loop running pid={pid['pid']}")

    if not decision["exists"]:
        add_check("decision_dataset_exists", "CRITICAL", "paper_decision_samples.jsonl missing")
    elif decision["latest_sample_age_minutes"] is None:
        add_check("decision_dataset_freshness", "CRITICAL", "no valid timestamps in decision dataset")
    elif decision["latest_sample_age_minutes"] > args.critical_stale_minutes:
        add_check(
            "decision_dataset_freshness",
            "CRITICAL",
            f"latest decision sample age {decision['latest_sample_age_minutes']}m exceeds {args.critical_stale_minutes}m",
            "Inspect pipeline logs and restart the loop if needed.",
        )
    elif decision["latest_sample_age_minutes"] > args.warn_stale_minutes:
        add_check(
            "decision_dataset_freshness",
            "WARN",
            f"latest decision sample age {decision['latest_sample_age_minutes']}m exceeds {args.warn_stale_minutes}m",
        )
    else:
        add_check("decision_dataset_freshness", "OK", f"latest decision sample age {decision['latest_sample_age_minutes']}m")

    if decision["today_count"] < args.min_decisions_today:
        add_check(
            "decision_dataset_today_volume",
            "WARN",
            f"today decision rows {decision['today_count']} below {args.min_decisions_today}",
            "If early in the day this may be normal; otherwise inspect pipeline cadence.",
        )
    else:
        add_check("decision_dataset_today_volume", "OK", f"today decision rows {decision['today_count']}")

    for label, info in (("pipeline_report", report), ("paper_state", state), ("pipeline_log", pipeline_log), ("trade_log", trade_log)):
        if not info.get("exists"):
            add_check(label, "WARN", f"{label} missing")
        elif float(info.get("age_minutes") or 999999) > args.critical_stale_minutes:
            add_check(label, "WARN", f"{label} age {info.get('age_minutes')}m exceeds {args.critical_stale_minutes}m")
        else:
            add_check(label, "OK", f"{label} age {info.get('age_minutes')}m")

    if closed.get("exists"):
        add_check("closed_trade_dataset", "OK", f"closed trades total {closed['line_count']}, today {closed['today_count']}")
    else:
        add_check("closed_trade_dataset", "WARN", "closed trade dataset missing or not created yet")

    if latest_execution_result is not None:
        age = round((now - latest_execution_mtime).total_seconds() / 60.0, 2) if latest_execution_mtime else None
        add_check("execution_results", "OK", f"today results {execution_results_today}, latest age {age}m")
    else:
        add_check("execution_results", "WARN", "no execution result files found")

    overall = max((check["status"] for check in checks), key=status_rank)
    return {
        "generated_at": now.isoformat(),
        "overall_status": overall,
        "thresholds": {
            "warn_stale_minutes": args.warn_stale_minutes,
            "critical_stale_minutes": args.critical_stale_minutes,
            "min_decisions_today": args.min_decisions_today,
        },
        "checks": checks,
        "datasets": {
            "decision_samples": decision,
            "closed_trades": closed,
        },
        "runtime": {
            "pid": pid,
            "pipeline_report": report,
            "paper_state": state,
            "pipeline_log": pipeline_log,
            "trade_log": trade_log,
            "lock_dir_exists": paths["lock_dir"].exists(),
            "execution_results_today": execution_results_today,
            "latest_execution_result": str(latest_execution_result) if latest_execution_result else "",
        },
    }


def send_macos_notification(title: str, message: str) -> None:
    script = f'display notification {json.dumps(message)} with title {json.dumps(title)}'
    subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True)


def print_human(report: dict[str, Any]) -> None:
    status = report["overall_status"]
    print(f"观察期健康检查: {status}")
    print(f"时间: {report['generated_at']}")
    print("")
    for check in report["checks"]:
        prefix = {"OK": "[OK]", "WARN": "[WARN]", "CRITICAL": "[严重]"}.get(check["status"], "[?]")
        print(f"{prefix} {check['name']}: {check['detail']}")
        if check.get("remediation"):
            print(f"      处理建议: {check['remediation']}")
    decision = report["datasets"]["decision_samples"]
    closed = report["datasets"]["closed_trades"]
    print("")
    print("数据概览:")
    print(f"- decision 总行数: {decision.get('line_count', 0)}")
    print(f"- decision 今日新增: {decision.get('today_count', 0)}")
    print(f"- decision 最新样本: {decision.get('latest_sample_at')}")
    print(f"- closed trades 总行数: {closed.get('line_count', 0)}")
    print(f"- closed trades 今日新增: {closed.get('today_count', 0)}")
    print(f"- execution results 今日新增: {report['runtime']['execution_results_today']}")
    print("")
    if status == "OK":
        print("结论: 数据链路正在正常积累。")
    elif status == "WARN":
        print("结论: 数据链路有轻微异常或低于预期，建议今天内检查。")
    else:
        print("结论: 数据链路可能中断，需要现在处理。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check V4 observation-period data health.")
    parser.add_argument("--warn-stale-minutes", type=float, default=30.0)
    parser.add_argument("--critical-stale-minutes", type=float, default=90.0)
    parser.add_argument("--min-decisions-today", type=int, default=50)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--notify-on-problem", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_health(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)
        print("")
        print(f"JSON: {args.output_json}")
    if args.notify_on_problem and report["overall_status"] != "OK":
        send_macos_notification(
            f"交易观察健康检查: {report['overall_status']}",
            "pipeline 或数据积累出现异常，请查看 observation_health.json",
        )
    return 0 if report["overall_status"] != "CRITICAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
