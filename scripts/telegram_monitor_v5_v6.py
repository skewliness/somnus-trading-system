#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import os
import re
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


os.umask(0o077)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.bb_structure import compute_bb
from trading_system.binance_public_guard import (
    BinancePublicRateLimitError,
    assert_binance_public_not_banned,
    note_binance_public_response,
    throttle_binance_public_request,
)
try:
    from trading_system.coin_scout_4h_labeler import CoinScout4HLabeler, INTERVAL_MS
except ModuleNotFoundError:
    from trading_system.optional_integrations import CoinScout4HLabeler, INTERVAL_MS
from trading_system.config import StrategyConfig
from trading_system.v6_execution import V6BinanceClient, V6ExecutionConfig, V6ExecutionEngine, canonical_order_mode

DEFAULT_STATE_FILE = ROOT / "data/trading_system/telegram_monitor/state.json"
V5_PID_FILE = ROOT / "data/trading_system/pipeline_loop.pid"
V5_REPORT_FILE = ROOT / "data/trading_system/pipeline_run_report.json"
V5_MACRO_FILE = ROOT / "data/trading_system/macro_trend_cache.json"
V5_PAPER_STATE_FILE = ROOT / "data/trading_system/paper_state.json"
V5_V6_EXECUTION_REPORT_FILE = ROOT / "data/trading_system/v5_v6_execution/latest_v5_v6_execution_report.json"
V5_POSITION_MANAGEMENT_REPORT_FILE = ROOT / "data/trading_system/v5_v6_execution/latest_v5_position_management_report.json"
V5_V6_EXECUTION_SUMMARY_FILE = ROOT / "data/trading_system/v5_v6_execution/latest_run_summary.json"
V5_V6_EXECUTION_STATE_FILE = ROOT / "data/trading_system/v5_v6_execution/state.json"
V6_PID_FILE = ROOT / "data/trading_system/v6_execution/v6_execution_loop.pid"
V6_SUMMARY_FILE = ROOT / "data/trading_system/v6_execution/latest_run_summary.json"
V6_DISTILL_REPORT_FILE = ROOT / "data/trading_system/model_training/v6_continuous_distill/latest_v6_distill_report.json"
V6_DISTILL_HEARTBEAT_FILE = ROOT / "data/trading_system/model_training/v6_continuous_distill/heartbeat.json"
CHART_DIR = ROOT / "data/trading_system/telegram_monitor/charts"
CHART_KLINE_CACHE_DIR = ROOT / os.getenv("TS_CHART_KLINE_CACHE_DIR", "data/trading_system/cache/chart_klines")
CHART_RENDER_SCRIPT = ROOT / "scripts/render_v6_chart.py"
REPORT_SIGNAL_CACHE: dict[str, dict[str, Any]] = {}
COMMAND_KLINE_MEMORY_CACHE: dict[tuple[str, str, int], tuple[float, list[dict[str, Any]]]] = {}
COMMAND_KLINE_MEMORY_TTL_SECONDS = float(os.getenv("TS_TELEGRAM_COMMAND_KLINE_MEMORY_TTL_SECONDS", "60"))
REPORT_SIGNAL_SUMMARY_MAX_AGE_SECONDS = float(os.getenv("TS_TELEGRAM_REPORT_SIGNAL_SUMMARY_MAX_AGE_SECONDS", "120"))
MARK_PRICE_CACHE: dict[str, tuple[float, float]] = {}
MARK_PRICE_CACHE_TTL_SECONDS = float(os.getenv("TS_TELEGRAM_MARK_PRICE_CACHE_TTL_SECONDS", "60"))
MARKET_FLOW_REPORT_CACHE: dict[str, tuple[float, str]] = {}
MARKET_FLOW_REPORT_CACHE_TTL_SECONDS = float(os.getenv("TS_TELEGRAM_MARKET_FLOW_CACHE_TTL_SECONDS", "120"))
SMART_SIGNAL_FILE = ROOT / os.getenv("TS_TELEGRAM_SMART_SIGNAL_FILE", "data/trading_system/skill_bridges/binance-smart-signal.json")
SMART_SIGNAL_REPORT_CACHE: dict[str, tuple[float, str]] = {}
SMART_SIGNAL_REPORT_CACHE_TTL_SECONDS = float(os.getenv("TS_TELEGRAM_SMART_SIGNAL_CACHE_TTL_SECONDS", "120"))
SMART_SIGNAL_MAX_AGE_SECONDS = float(os.getenv("TS_TELEGRAM_SMART_SIGNAL_MAX_AGE_SECONDS", "7200"))
BEIJING_TZ = timezone(timedelta(hours=8))
BOT_MENTION_RE = re.compile(r"@([A-Za-z0-9_]{5,32})")


def _local_http_proxy_available(proxy_url: str, *, timeout: float = 0.25) -> bool:
    try:
        parsed = urllib.parse.urlparse(proxy_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host or port <= 0:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_default_proxy_url() -> str:
    configured = (
        os.getenv("TELEGRAM_PROXY_URL")
        or os.getenv("TS_TELEGRAM_PROXY_URL")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or ""
    ).strip()
    if configured.lower() in {"0", "false", "off", "none", "direct"}:
        return ""
    if configured:
        return configured
    local_proxy = (os.getenv("TS_LOCAL_PROXY_URL") or os.getenv("LOCAL_PROXY_URL") or "http://127.0.0.1:7897").strip()
    if local_proxy and _local_http_proxy_available(local_proxy):
        return local_proxy
    return ""


def _telegram_ssl_context() -> ssl.SSLContext | None:
    disabled = str(os.getenv("TS_TELEGRAM_SSL_NO_VERIFY", "")).lower() in {"1", "true", "yes", "on"}
    return ssl._create_unverified_context() if disabled else None


def _build_url_opener() -> urllib.request.OpenerDirector:
    proxy_url = _resolve_default_proxy_url()
    handlers: list[Any] = []
    if proxy_url:
        handlers.append(urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
    ssl_context = _telegram_ssl_context()
    if ssl_context is not None:
        handlers.append(urllib.request.HTTPSHandler(context=ssl_context))
    return urllib.request.build_opener(*handlers)


URL_OPENER = _build_url_opener()


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def open_url(request: str | urllib.request.Request, *, timeout: float):
    retries = max(1, int(os.getenv("TS_URL_OPEN_RETRIES", "3") or "3"))
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return URL_OPENER.open(request, timeout=timeout)
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(0.6 * (2 ** attempt))
    raise last_exc if last_exc is not None else RuntimeError("url open failed before response")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def file_age_seconds(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except Exception:
        return None


def latest_matching_file_age_seconds(pattern: str) -> tuple[float | None, str]:
    try:
        matches = sorted(
            ROOT.glob(pattern),
            key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
            reverse=True,
        )
    except Exception:
        return None, ""
    if not matches:
        return None, ""
    return file_age_seconds(matches[0]), str(matches[0])


def min_age_seconds(*values: float | None) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return min(valid) if valid else None


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "缺失"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}秒"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}分钟"
    hours = minutes // 60
    return f"{hours}小时{minutes % 60}分钟"


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except Exception:
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def systemd_service_state(service_name: str) -> tuple[bool, int | None]:
    try:
        result = subprocess.run(
            ["systemctl", "show", service_name, "--property=ActiveState,MainPID", "--no-pager"],
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False, None
    active_state = ""
    main_pid: int | None = None
    for line in result.stdout.splitlines():
        if line.startswith("ActiveState="):
            active_state = line.split("=", 1)[1].strip()
        elif line.startswith("MainPID="):
            try:
                parsed_pid = int(line.split("=", 1)[1].strip() or "0")
            except Exception:
                parsed_pid = 0
            main_pid = parsed_pid if parsed_pid > 0 else None
    # systemd reports Restart=always services as "activating" during RestartSec.
    # Treat that short planned restart window as running to avoid false down alerts.
    return active_state in {"active", "activating", "reloading"}, main_pid


def format_bytes_short(value: Any) -> str:
    try:
        size = float(value)
    except Exception:
        return "--"
    if size < 0 or size >= 18446744073709551615:
        return "--"
    if size >= 1024 ** 3:
        return f"{size / 1024 ** 3:.2f}G"
    if size >= 1024 ** 2:
        return f"{size / 1024 ** 2:.0f}M"
    if size >= 1024:
        return f"{size / 1024:.0f}K"
    return f"{size:.0f}B"


def system_swap_total_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("SwapTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except Exception:
        return 0
    return 0


def format_bytes_with_percent(value: Any, total: int) -> str:
    text = format_bytes_short(value)
    try:
        size = float(value)
    except Exception:
        return text
    if total <= 0 or size < 0:
        return text
    return f"{text}({size / total * 100:.2f}%)"


def format_percent_of_total(value: Any, total: int) -> str:
    try:
        size = float(value)
    except Exception:
        return "--"
    if total <= 0 or size < 0:
        return "--"
    return f"{size / total * 100:.2f}%"


def systemd_service_resources(service_name: str) -> dict[str, Any]:
    props = (
        "MemoryCurrent,MemoryPeak,MemorySwapCurrent,MemorySwapPeak,"
        "NRestarts,Result"
    )
    try:
        result = subprocess.run(
            ["systemctl", "show", service_name, f"--property={props}", "--no-pager"],
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    data: dict[str, Any] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        raw = raw.strip()
        if key in {"MemoryCurrent", "MemoryPeak", "MemorySwapCurrent", "MemorySwapPeak", "NRestarts"}:
            try:
                data[key] = int(raw or "0")
            except Exception:
                data[key] = 0
        else:
            data[key] = raw
    return data


def systemd_service_oom_count(service_name: str, since: str = "24 hours ago") -> int | None:
    try:
        result = subprocess.run(
            ["journalctl", "-u", service_name, "--since", since, "--no-pager"],
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except Exception:
        return None
    text = result.stdout.lower()
    return text.count("oom-kill") + text.count("oom killer")


def pgrep_count(pattern: str) -> int:
    stdout_chunks: list[str] = []
    for command in (["pgrep", "-af", pattern], ["pgrep", "-fl", pattern]):
        try:
            result = subprocess.run(
                command,
                cwd=str(ROOT),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue
        if result.returncode == 0 and result.stdout.strip():
            stdout_chunks.append(result.stdout)
            if any(pattern in line for line in result.stdout.splitlines()):
                break
    stdout = chr(10).join(stdout_chunks)
    if not any(pattern in line for line in stdout.splitlines()):
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,command="],
                cwd=str(ROOT),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            stdout = chr(10).join(line for line in result.stdout.splitlines() if pattern in line)
        except Exception:
            return 0
    lines = []
    for line in stdout.splitlines():
        if "telegram_monitor_v5_v6.py" in line:
            continue
        if pattern in line:
            lines.append(line)
    return len(lines)


def pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "--"


def number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def money(value: Any) -> str:
    try:
        return f"{float(value):+.3f}U"
    except Exception:
        return "--"


ZH_VALUE_MAP = {
    "bearish": "看空",
    "bullish": "看多",
    "neutral": "中性",
    "short": "空头",
    "long": "多头",
    "wait": "等待",
    "risk_off": "避险",
    "risk_on": "风险偏好",
    "mixed": "震荡",
    "strong": "强",
    "medium": "中等",
    "low": "低",
    "weak": "弱",
    "high": "高",
    "broken": "破位",
    "healthy": "健康",
    "lower_half": "下半区",
    "upper_half": "上半区",
    "below_support": "跌破支撑",
    "above_resistance": "突破压力",
    "near_support": "靠近支撑",
    "near_resistance": "靠近压力",
    "middle": "中部",
    "unknown": "未知",
    "paper": "模拟",
    "demo": "模拟",
    "live": "实盘",
    "buy": "买入",
    "sell": "卖出",
    "submitted": "已提交",
    "failed": "失败",
    "partial": "部分执行",
    "empty": "无订单",
    "new": "已挂单",
    "filled": "已成交",
    "canceled": "已取消",
    "true": "是",
    "false": "否",
}

ACTION_ZH_MAP = {
    "open_long": "开多",
    "close_long": "平多",
    "open_short": "开空",
    "close_short": "平空",
    "exit": "平仓",
    "take_profit_partial": "减仓",
    "partial_take_profit": "减仓",
    "partial_close_long": "减多",
    "partial_close_short": "减空",
    "reduce_long": "减多",
    "reduce_short": "减空",
    "add_long": "加多",
    "add_short": "加空",
    "pyramid_long": "加多",
    "pyramid_short": "加空",
    "stop_loss_long": "止损平多",
    "stop_loss_short": "止损平空",
    "hold": "持有",
    "wait": "等待",
}

V5_TRADE_ACTIVITY_ACTIONS = {
    "open_long",
    "open_short",
    "close_long",
    "close_short",
    "take_profit_partial",
    "partial_take_profit",
    "partial_close_long",
    "partial_close_short",
    "reduce_long",
    "reduce_short",
    "add_long",
    "add_short",
    "pyramid_long",
    "pyramid_short",
    "stop_loss_long",
    "stop_loss_short",
}

V5_TRADE_ACTIVITY_STATUSES = {
    "submitted",
    "partial",
    "filled",
    "success",
    "succeeded",
    "executed",
}

V5_PAPER_JOURNAL_ACTIONS = {
    "open_long",
    "open_short",
    "exit",
    "close_long",
    "close_short",
    "take_profit_partial",
    "partial_take_profit",
    "partial_close_long",
    "partial_close_short",
    "reduce_long",
    "reduce_short",
    "stop_loss_long",
    "stop_loss_short",
}


def zh_value(value: Any) -> str:
    if value is None or value == "":
        return "--"
    key = str(value).strip().lower()
    return ZH_VALUE_MAP.get(key, str(value))


def zh_action(value: Any) -> str:
    if value is None or value == "":
        return "--"
    key = str(value).strip().lower()
    return ACTION_ZH_MAP.get(key, zh_value(value))


def action_icon(value: Any) -> str:
    action = str(value or "").lower()
    if "long" in action or action in {"buy"}:
        return "💹"
    if "short" in action or action in {"sell"}:
        return "📉"
    return "⚪"


def execution_status_text(status: Any) -> str:
    raw = str(status or "").strip().lower()
    ok_statuses = {
        "submitted", "success", "ok", "new", "filled", "partially_filled",
        "partial", "accepted", "done", "executed", "closed", "open_submitted",
        "close_submitted", "reduce_submitted", "submitted_open", "submitted_close",
        "submitted_unprotected", "submitted_no_position_delta", "retried_submitted",
    }
    fail_statuses = {
        "failed", "failure", "error", "rejected", "reject", "skipped", "invalid_price",
        "closed_after_protective_stop_failed", "signal_failed",
    }
    if raw in ok_statuses or "submitted" in raw or "filled" in raw:
        return "✅"
    if raw in fail_statuses or "fail" in raw or "error" in raw or "reject" in raw or "skipped" in raw:
        return "❌"
    return zh_value(status)


def compact_fill_text(qty: Any, avg_price: Any) -> str:
    has_qty = qty not in {"", None, "--"}
    has_avg = avg_price not in {"", None, "--"}
    if has_qty and has_avg:
        return f"{qty}@{format_price(avg_price)}"
    if has_qty:
        return str(qty)
    if has_avg:
        return f"@{format_price(avg_price)}"
    return ""


def zh_error(value: Any) -> str:
    text = str(value or "")
    replacements = [
        ("Binance execution failed", "Binance执行失败"),
        ("Invalid symbol.", "无效交易对。"),
        ("insufficient balance", "余额不足"),
        ("Order would immediately trigger", "订单会立即触发"),
        ("Filter failure", "交易所规则校验失败"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text[:800] if text else "--"


def is_silent_execution_error(err: dict[str, Any]) -> bool:
    text = str(err.get("error") or "").lower()
    status = str(err.get("status") or "").lower()
    return (
        status in {"status_unknown", "status_unknown_reconciled", "status_unknown_not_found"}
        or '"code":-1121' in text
        or '"code": -1121' in text
        or '"code":-1007' in text
        or '"code": -1007' in text
        or "timeout waiting for response from backend server" in text
        or "execution status unknown" in text
        or "send status unknown" in text
        or "invalid symbol" in text
        or "symbol not available" in text
        or "not available on binance testnet_order endpoint" in text
        or "?????" in text
    )


def is_invalid_symbol_error_text(text: str) -> bool:
    normalized = str(text or "").lower()
    return (
        '"code":-1121' in normalized
        or '"code": -1121' in normalized
        or "invalid symbol" in normalized
        or "无效交易对" in normalized
        or "symbol not available" in normalized
        or "not available on binance testnet_order endpoint" in normalized
        or "insufficient kline history" in normalized
    )


def is_silent_v6_execution_result(result: dict[str, Any]) -> bool:
    status = str(result.get("status") or "").lower()
    if result.get("telegram_silent") is True or result.get("maintenance_only") is True:
        return True
    if status in {
        "skipped_percent_price_retry_deviation",
        "skipped_percent_price_retry_counterparty_filter",
        "skipped_percent_price_retry_failed",
        "deferred_percent_price_close_deviation",
        "deferred_percent_price_close_counterparty_filter",
        "deferred_percent_price_close_failed",
    }:
        return True
    if status == "submitted_no_position_delta":
        return True
    if status == "reconciled_missing_exchange_position":
        return True
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    action = str(result.get("action") or intent.get("action") or "").lower()
    error_text = " ".join(
        str(result.get(key) or "")
        for key in ("error", "original_error", "retry_error", "message", "msg")
    ).lower()
    if is_invalid_symbol_error_text(error_text):
        return True
    reconcile = result.get("position_reconcile") if isinstance(result.get("position_reconcile"), dict) else {}
    if status in {"submitted", "submitted_unprotected", "submitted_no_position_delta"} and action in {"open_long", "open_short", "close_long", "close_short"} and reconcile:
        try:
            delta_qty = abs(float(reconcile.get("delta_qty_abs") or 0.0))
        except Exception:
            delta_qty = 0.0
        if delta_qty <= 0:
            return True
    return action in {"close_long", "close_short"} and "no exchange position found" in error_text


def visible_v6_activity(
    intents: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visible_intents: list[dict[str, Any]] = []
    visible_results: list[dict[str, Any]] = []
    max_len = max(len(intents), len(results))
    for idx in range(max_len):
        intent = intents[idx] if idx < len(intents) and isinstance(intents[idx], dict) else None
        result = results[idx] if idx < len(results) and isinstance(results[idx], dict) else None
        if result is not None and is_silent_v6_execution_result(result):
            continue
        if intent is not None:
            visible_intents.append(intent)
        if result is not None:
            visible_results.append(result)
    return visible_intents, visible_results


def is_visible_v5_trade_request(request: dict[str, Any]) -> bool:
    action = str(request.get("action") or "").strip().lower()
    status = str(request.get("status") or request.get("submit_status") or "").strip().lower()
    if action not in V5_TRADE_ACTIVITY_ACTIONS:
        return False
    if status not in V5_TRADE_ACTIVITY_STATUSES:
        return False
    if str(request.get("symbol") or "").strip().upper() in {"", "--"}:
        return False
    return True


def visible_v5_trade_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        if is_visible_v5_trade_request(request):
            visible.append(request)
    return visible


def is_visible_v5_paper_journal_event(event: dict[str, Any]) -> bool:
    action = str(event.get("action") or "").strip().lower()
    symbol = str(event.get("symbol") or "").strip().upper()
    return bool(symbol and action in V5_PAPER_JOURNAL_ACTIONS)


def visible_v5_paper_journal_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if is_visible_v5_paper_journal_event(event):
            visible.append(event)
    return visible


def is_fresh_file(path: Path, max_age_seconds: float = 1800.0) -> bool:
    age = file_age_seconds(path)
    return age is not None and age <= max_age_seconds


def first_position_list(*payloads: dict[str, Any]) -> list[dict[str, Any]]:
    for payload in payloads:
        positions = payload.get("positions") if isinstance(payload, dict) else None
        if isinstance(positions, list):
            return [item for item in positions if isinstance(item, dict)]
    return []


def collect_v5_overlay_ledger() -> dict[str, Any]:
    report = load_json(V5_V6_EXECUTION_REPORT_FILE)
    management_report = load_json(V5_POSITION_MANAGEMENT_REPORT_FILE)
    summary = load_json(V5_V6_EXECUTION_SUMMARY_FILE)
    state = load_json(V5_V6_EXECUTION_STATE_FILE)
    fresh = is_fresh_file(V5_V6_EXECUTION_REPORT_FILE)
    management_fresh = is_fresh_file(V5_POSITION_MANAGEMENT_REPORT_FILE)
    mode = str(report.get("mode") or summary.get("mode") or "").strip()
    management_mode = str(management_report.get("mode") or "").strip()
    active = (fresh and mode == "v5_key_v6_trading_system") or (management_fresh and management_mode == "v5_key_v6_trading_system")
    positions = first_position_list(state, management_report, report, summary)
    visible_pairs = []
    for source in (report, management_report):
        raw_intents = [item for item in source.get("intents") or [] if isinstance(item, dict)]
        raw_results = [item for item in source.get("results") or [] if isinstance(item, dict)]
        visible_pairs.extend(
            (intent, result)
            for intent, result in pair_execution_intents_results(raw_intents, raw_results)
            if result is None or not is_silent_v6_execution_result(result)
        )
    intents = [intent for intent, _result in visible_pairs if intent is not None]
    results = [result for _intent, result in visible_pairs if result is not None]
    return {
        "active": active,
        "fresh": fresh or management_fresh,
        "mode": mode or management_mode,
        "age": min_age_seconds(file_age_seconds(V5_V6_EXECUTION_REPORT_FILE), file_age_seconds(V5_POSITION_MANAGEMENT_REPORT_FILE)),
        "report_file": str(V5_V6_EXECUTION_REPORT_FILE),
        "management_report_file": str(V5_POSITION_MANAGEMENT_REPORT_FILE),
        "positions": positions,
        "intents": intents,
        "results": results,
    }


def short_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class TelegramClient:
    def __init__(self, token: str, chat_id: str, *, timeout: float = 15.0, dry_run: bool = False) -> None:
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.dry_run = dry_run
        self._me: dict[str, Any] | None = None

    def api_post(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.token:
            raise RuntimeError("缺少 TELEGRAM_BOT_TOKEN")
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        data = urllib.parse.urlencode(params).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        with open_url(request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception as exc:
            raise RuntimeError(f"Telegram返回内容无法解析：{body[:200]}") from exc
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram请求失败：{body[:500]}")
        return payload

    def get_me(self) -> dict[str, Any]:
        if self._me is None:
            result = self.api_post("getMe", {}).get("result") or {}
            self._me = result if isinstance(result, dict) else {}
        return self._me

    def username(self) -> str:
        return normalize_bot_username(self.get_me().get("username"))

    def api_multipart_post(
        self,
        method: str,
        fields: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, Any]:
        if not self.token:
            raise RuntimeError("缺少 TELEGRAM_BOT_TOKEN")
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        boundary = f"somnus-{int(time.time() * 1000)}-{hashlib.sha1(os.urandom(16)).hexdigest()[:12]}"
        chunks: list[bytes] = []
        for key, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\n".encode("utf-8"),
                    f'Content-Disposition: form-data; name="{key}"\n\n'.encode("utf-8"),
                    str(value).encode("utf-8"),
                    b"\n",
                ]
            )
        for key, (filename, content, content_type) in files.items():
            chunks.extend(
                [
                    f"--{boundary}\n".encode("utf-8"),
                    (
                        f'Content-Disposition: form-data; name="{key}"; '
                        f'filename="{filename}"\n'
                    ).encode("utf-8"),
                    f"Content-Type: {content_type}\n\n".encode("utf-8"),
                    content,
                    b"\n",
                ]
            )
        chunks.append(f"--{boundary}--\n".encode("utf-8"))
        body = b"".join(chunks)
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with open_url(request, timeout=max(self.timeout, 30.0)) as response:
            response_body = response.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(response_body)
        except Exception as exc:
            raise RuntimeError(f"Telegram返回内容无法解析：{response_body[:200]}") from exc
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram请求失败：{response_body[:500]}")
        return payload

    def send(self, text: str, chat_id: str | int | None = None) -> None:
        if self.dry_run:
            print("\n--- Telegram模拟发送 ---")
            print(text)
            print("--- 结束 ---")
            return
        target_chat_id = chat_id if chat_id is not None else self.chat_id
        if not target_chat_id:
            raise RuntimeError("缺少 TELEGRAM_CHAT_ID")
        remaining = str(text or "")
        chunks: list[str] = []
        while remaining:
            if len(remaining) <= 3900:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, 3900)
            if split_at < 1200:
                split_at = 3900
            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        for chunk in chunks or [""]:
            self.api_post(
                "sendMessage",
                {
                    "chat_id": str(target_chat_id),
                    "text": chunk,
                    "disable_web_page_preview": "true",
                },
            )

    def send_photo(self, photo_path: Path, caption: str = "", chat_id: str | int | None = None) -> None:
        if self.dry_run:
            print("\n--- Telegram模拟发送图片 ---")
            print(f"图片={photo_path}")
            print(caption)
            print("--- 结束 ---")
            return
        target_chat_id = chat_id if chat_id is not None else self.chat_id
        if not target_chat_id:
            raise RuntimeError("缺少 TELEGRAM_CHAT_ID")
        if not photo_path.exists():
            raise RuntimeError(f"图片文件不存在：{photo_path}")
        content_type = mimetypes.guess_type(str(photo_path))[0] or "image/png"
        self.api_multipart_post(
            "sendPhoto",
            {
                "chat_id": str(target_chat_id),
                "caption": caption[:1000],
            },
            {
                "photo": (photo_path.name, photo_path.read_bytes(), content_type),
            },
        )

    def get_updates(self, offset: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timeout": "0",
            "allowed_updates": json.dumps(["message"], ensure_ascii=False),
        }
        if offset is not None:
            params["offset"] = str(offset)
        payload = self.api_post("getUpdates", params)
        result = payload.get("result")
        return result if isinstance(result, list) else []


def collect_status(args: argparse.Namespace) -> dict[str, Any]:
    v5_pid = read_pid(V5_PID_FILE)
    v6_pid = read_pid(V6_PID_FILE)
    v5_systemd_running, v5_systemd_pid = systemd_service_state("v5-trading.service")
    v6_systemd_running, v6_systemd_pid = systemd_service_state("v6-execution.service")
    v6_distill_systemd_running, v6_distill_systemd_pid = systemd_service_state("v6-distill.service")
    v5_effective_pid = v5_systemd_pid or v5_pid
    v6_effective_pid = v6_systemd_pid or v6_pid
    v5_report = load_json(V5_REPORT_FILE)
    v5_macro = load_json(V5_MACRO_FILE) or v5_report.get("macro_trend") or {}
    v6_summary = load_json(V6_SUMMARY_FILE)
    v6_distill = load_json(V6_DISTILL_REPORT_FILE)

    v5_report_age = file_age_seconds(V5_REPORT_FILE)
    v5_macro_age = file_age_seconds(V5_MACRO_FILE)
    v6_summary_age = file_age_seconds(V6_SUMMARY_FILE)
    v6_distill_age = file_age_seconds(V6_DISTILL_REPORT_FILE)
    v6_distill_heartbeat_age = file_age_seconds(V6_DISTILL_HEARTBEAT_FILE)
    v5_trade_log_age, v5_trade_log_path = latest_matching_file_age_seconds("data/trading_system/logs/trade_activity_*.log")
    v5_pipeline_log_age, v5_pipeline_log_path = latest_matching_file_age_seconds("data/trading_system/logs/pipeline_24h_*.log")
    v6_distill_log_age, v6_distill_log_path = latest_matching_file_age_seconds("data/trading_system/logs/v6_continuous_distill_*.log")

    v5_perf = v5_report.get("performance_dashboard") or {}
    v5_exec = v5_report.get("execution_results_summary") or {}
    v5_bridge = v5_report.get("execution_bridge_summary") or {}
    raw_v5_errors = v5_exec.get("latest_errors") or []
    v5_errors = [err for err in raw_v5_errors if not is_silent_execution_error(err)]
    v5_recent_requests = v5_exec.get("latest_requests") or []
    v5_trade_requests = visible_v5_trade_requests(v5_recent_requests)
    v5_state_path = Path(str(v5_perf.get("state_file") or V5_PAPER_STATE_FILE))
    if not v5_state_path.is_absolute():
        v5_state_path = ROOT / v5_state_path
    v5_state = load_json(v5_state_path)
    v5_overlay = collect_v5_overlay_ledger()
    v5_overlay_active = bool(v5_overlay.get("active"))
    v5_paper_journal_events = [] if v5_overlay_active else visible_v5_paper_journal_events(v5_state.get("journal") or [])
    if v5_overlay_active:
        v5_trade_requests = []
        v5_positions = v5_overlay.get("positions") or []
        v5_state_source = "v5_v6_execution"
    else:
        v5_positions = v5_state.get("positions") or []
        v5_state_source = "paper_state"

    v6_signals = v6_summary.get("signals") or []
    v6_signal_by_symbol = {str(item.get("symbol")): item for item in v6_signals}
    raw_v6_intents = v6_summary.get("intents") or []
    raw_v6_results = v6_summary.get("results") or []
    v6_intents, v6_results = visible_v6_activity(raw_v6_intents, raw_v6_results)
    v6_metrics = (((v6_distill.get("metrics") or {}).get("v6") or {}).get("mixed_undersampled") or {})
    live_ops = v6_distill.get("live_operations") or {}

    return {
        "generated_at": iso_now(),
        "v5": {
            "pid": v5_effective_pid,
            "pid_file_pid": v5_pid,
            "systemd_pid": v5_systemd_pid,
            "running": v5_systemd_running or pid_alive(v5_pid),
            "running_source": "systemd" if v5_systemd_running else "pid_file",
            "report_age": v5_report_age,
            "macro_age": v5_macro_age,
            "activity_age": min_age_seconds(v5_report_age, v5_trade_log_age, v5_pipeline_log_age),
            "trade_log_age": v5_trade_log_age,
            "trade_log_path": v5_trade_log_path,
            "pipeline_log_age": v5_pipeline_log_age,
            "pipeline_log_path": v5_pipeline_log_path,
            "run_id": v5_report.get("run_id", ""),
            "mode": v5_report.get("mode", ""),
            "execution_mode": v5_report.get("execution_mode", ""),
            "market_regime": (v5_report.get("market_regime") or {}).get("regime", ""),
            "macro": v5_macro,
            "performance": v5_perf,
            "intent_count": (v5_bridge.get("intent_count") or 0),
            "request_mode": v5_bridge.get("request_mode", ""),
            "latest_errors": v5_errors[:5],
            "latest_requests": v5_recent_requests[:5],
            "trade_requests": v5_trade_requests[:10],
            "paper_journal_events": v5_paper_journal_events[-20:],
            "intents": (v5_overlay.get("intents") or [])[:10] if v5_overlay_active else [],
            "results": (v5_overlay.get("results") or [])[:10] if v5_overlay_active else [],
            "positions": v5_positions,
            "shadow_positions": v5_state.get("shadow_positions") or [],
            "state_file": str(v5_state_path),
            "state_source": v5_state_source,
            "overlay": v5_overlay,
            "issue_count": v5_report.get("issue_count", 0),
        },
        "v6": {
            "pid": v6_effective_pid,
            "pid_file_pid": v6_pid,
            "systemd_pid": v6_systemd_pid,
            "running": v6_systemd_running or pid_alive(v6_pid),
            "running_source": "systemd" if v6_systemd_running else "pid_file",
            "distill_pid": v6_distill_systemd_pid,
            "distill_running": v6_distill_systemd_running or pgrep_count("run_v6_continuous_distill") > 0,
            "distill_running_source": "systemd" if v6_distill_systemd_running else "pgrep",
            "summary_age": v6_summary_age,
            "distill_age": v6_distill_age,
            "distill_activity_age": min_age_seconds(v6_distill_age, v6_distill_log_age, v6_distill_heartbeat_age),
            "distill_log_age": v6_distill_log_age,
            "distill_heartbeat_age": v6_distill_heartbeat_age,
            "distill_log_path": v6_distill_log_path,
            "generated_at": v6_summary.get("generated_at", ""),
            "order_mode": v6_summary.get("order_mode", ""),
            "submit": v6_summary.get("submit"),
            "signals": v6_signal_by_symbol,
            "intent_count": len(v6_intents),
            "result_count": len(v6_results),
            "positions": v6_summary.get("positions") or [],
            "intents": v6_intents[:10],
            "results": v6_results[:10],
            "metrics": v6_metrics,
            "live_ops_total": live_ops.get("total", 0),
        },
        "thresholds": {
            "v5_stale_seconds": args.v5_stale_seconds,
            "v6_stale_seconds": args.v6_stale_seconds,
            "v6_distill_stale_seconds": args.v6_distill_stale_seconds,
        },
    }


V6_DIRECTION_CACHE: dict[str, tuple[float, str]] = {}
V6_DIRECTION_CACHE_TTL_SECONDS = 600
V6_DIRECTION_INTERVAL_MS = {
    "15m": 900000,
    "4h": 14400000,
    "1d": 86400000,
    "2d": 172800000,
    "3d": 259200000,
    "1w": 604800000,
    "2w": 1209600000,
}
V6_DIRECTION_TOLERANCE = {
    "15m": 0.0015,
    "4h": 0.004,
    "1d": 0.008,
    "2d": 0.010,
    "3d": 0.012,
    "1w": 0.015,
    "2w": 0.020,
}
V6_DIRECTION_RULES = [
    {
        "horizon": "24H",
        "direction": "\u7a7a",
        "win": 73.2,
        "samples": 41,
        "features": ("15m:BB1440:below_mid", "4h:BB300:mid_reject", "1d:BB105:below_mid"),
        "reason": "15M BB1440<\u4e2d + 4H BB300\u53cd\u62bd\u4e0d\u8fc7 + 1D BB105<\u4e2d",
    },
    {
        "horizon": "24H",
        "direction": "\u591a",
        "win": 72.4,
        "samples": 58,
        "features": ("15m:BB210:lower_bounce", "4h:BB1440:above_mid", "1d:MA7:support"),
        "reason": "15M BB210\u4e0b\u8f68\u53cd\u5f39 + 4H BB1440>\u4e2d + 1D MA7\u652f\u6491",
    },
    {
        "horizon": "48H",
        "direction": "\u591a",
        "win": 76.9,
        "samples": 26,
        "features": ("15m:BB96:above_mid", "4h:BB300:lower_bounce", "2d:BB105:above_mid"),
        "reason": "15M BB96>\u4e2d + 4H BB300\u4e0b\u8f68\u53cd\u5f39 + 2D BB105>\u4e2d",
    },
    {
        "horizon": "48H",
        "direction": "\u7a7a",
        "win": 75.5,
        "samples": 49,
        "features": ("15m:BB1440:below_mid", "4h:BB300:mid_reject", "2d:MA7:below_slope_down"),
        "reason": "15M BB1440<\u4e2d + 4H BB300\u53cd\u62bd\u4e0d\u8fc7 + 2D MA7\u4e0b\u659c\u7387",
    },
    {
        "horizon": "72H",
        "direction": "\u591a",
        "win": 81.5,
        "samples": 27,
        "features": ("15m:BB2880:above_mid", "4h:BB672:above_mid", "3d:MA96:support"),
        "reason": "15M BB2880>\u4e2d + 4H BB672>\u4e2d + 3D MA96\u652f\u6491",
    },
    {
        "horizon": "72H",
        "direction": "\u7a7a",
        "win": 72.7,
        "samples": 44,
        "features": ("15m:MA20:reject", "4h:MA20:reject", "3d:MA20:cross_down"),
        "reason": "15M MA20\u53cd\u62bd\u4e0d\u8fc7 + 4H MA20\u53cd\u62bd\u4e0d\u8fc7 + 3D MA20\u4e0b\u7a7f",
    },
]

V6_DIRECTION_FEATURE_LABELS = {
    "15m:BB1440:below_mid": "15M BB1440<\u4e2d",
    "4h:BB300:mid_reject": "4H BB300\u53cd\u62bd\u4e0d\u8fc7",
    "1d:BB105:below_mid": "1D BB105<\u4e2d",
    "15m:BB210:lower_bounce": "15M BB210\u4e0b\u8f68\u53cd\u5f39",
    "4h:BB1440:above_mid": "4H BB1440>\u4e2d",
    "1d:MA7:support": "1D MA7\u652f\u6491",
    "15m:BB96:above_mid": "15M BB96>\u4e2d",
    "4h:BB300:lower_bounce": "4H BB300\u4e0b\u8f68\u53cd\u5f39",
    "2d:BB105:above_mid": "2D BB105>\u4e2d",
    "2d:MA7:below_slope_down": "2D MA7\u4e0b\u659c\u7387",
    "15m:BB2880:above_mid": "15M BB2880>\u4e2d",
    "4h:BB672:above_mid": "4H BB672>\u4e2d",
    "3d:MA96:support": "3D MA96\u652f\u6491",
    "15m:MA20:reject": "15M MA20\u53cd\u62bd\u4e0d\u8fc7",
    "4h:MA20:reject": "4H MA20\u53cd\u62bd\u4e0d\u8fc7",
    "3d:MA20:cross_down": "3D MA20\u4e0b\u7a7f",
}



V6_MACRO_4H_LOOKBACK = 4720
V6_MACRO_LONG_LOOKBACK_1D = 1900
V6_MACRO_4H_RULES = {
    "BTCUSDT": [
        {
            "source": "\u504f\u7a7a",
            "summary_slot": "4H",
            "summary_when_active": "\u504f\u7a7a",
            "display": "MA60/96\u4e0b\u659c\u7387 + BB300\u4e2d\u8f68\u538b\u5236",
            "alternatives": [
                {"features": ("4h:MA60:below_slope_down", "4h:BB300:mid_reject"), "win": 63.6, "samples": 140, "text": "MA60\u4e0b\u659c\u7387 + BB300\u4e2d\u8f68\u538b\u5236"},
                {"features": ("4h:MA96:below_slope_down", "4h:BB300:mid_reject"), "win": 63.8, "samples": 138, "text": "MA96\u4e0b\u659c\u7387 + BB300\u4e2d\u8f68\u538b\u5236"},
            ],
        },
        {
            "source": "\u504f\u591a",
            "summary_slot": "4H",
            "summary_when_active": "\u504f\u591a",
            "display": "MA7\u652f\u6491/\u4e0a\u659c\u7387 + BB300\u4e0b\u8f68\u53cd\u5f39\uff1b\u6216 MA105\u652f\u6491 + BB2880\u4e0a\u8f68\u7a81\u7834",
            "alternatives": [
                {"features": ("4h:MA7:support", "4h:BB300:lower_bounce"), "win": 65.9, "samples": 82, "text": "MA7\u652f\u6491 + BB300\u4e0b\u8f68\u53cd\u5f39"},
                {"features": ("4h:MA7:above_slope_up", "4h:BB300:lower_bounce"), "win": 67.9, "samples": 53, "text": "MA7\u4e0a\u659c\u7387 + BB300\u4e0b\u8f68\u53cd\u5f39"},
                {"features": ("4h:MA105:support", "4h:BB2880:upper_breakout"), "win": 64.6, "samples": 113, "text": "MA105\u652f\u6491 + BB2880\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA120:above_slope_up", "4h:BB105:mid_support", "4h:BB2880:upper_breakout"), "win": 68.3, "samples": 101, "text": "MA120\u4e0a\u659c\u7387 + BB105\u4e2d\u8f68\u652f\u6491 + BB2880\u4e0a\u8f68\u7a81\u7834"},
            ],
        },
        {
            "source": "\u5f3a\u7a7a",
            "summary_slot": "48H",
            "summary_when_active": "\u5f3a\u7a7a",
            "display": "MA300\u4e0b\u659c\u7387 + BB120\u4e0a\u8f68\u538b\u5236",
            "alternatives": [
                {"features": ("4h:MA300:below_slope_down", "4h:BB120:upper_reject"), "win": 77.8, "samples": 90, "text": "MA300\u4e0b\u659c\u7387 + BB120\u4e0a\u8f68\u538b\u5236"},
            ],
        },
        {
            "source": "\u5f3a\u591a",
            "summary_slot": "48H",
            "summary_when_active": "\u5f3a\u591a",
            "display": "MA105/120/300\u4e0a\u659c\u7387 + BB105\u4e2d\u8f68\u652f\u6491 + BB2880\u4e0a\u8f68\u7a81\u7834",
            "alternatives": [
                {"features": ("4h:MA105:above_slope_up", "4h:BB105:mid_support", "4h:BB2880:upper_breakout"), "win": 80.7, "samples": 88, "text": "MA105\u4e0a\u659c\u7387 + BB105\u4e2d\u8f68\u652f\u6491 + BB2880\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA300:above_slope_up", "4h:BB105:mid_support", "4h:BB2880:upper_breakout"), "win": 77.1, "samples": 109, "text": "MA300\u4e0a\u659c\u7387 + BB105\u4e2d\u8f68\u652f\u6491 + BB2880\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA120:above_slope_up", "4h:BB105:mid_support", "4h:BB2880:upper_breakout"), "win": 77.2, "samples": 101, "text": "MA120\u4e0a\u659c\u7387 + BB105\u4e2d\u8f68\u652f\u6491 + BB2880\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA105:support", "4h:BB1920:upper_breakout", "4h:BB2880:upper_breakout"), "win": 76.2, "samples": 105, "text": "MA105\u652f\u6491 + BB1920\u4e0a\u8f68\u7a81\u7834 + BB2880\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA105:support", "4h:BB1440:upper_breakout", "4h:BB2880:upper_breakout"), "win": 74.1, "samples": 85, "text": "MA105\u652f\u6491 + BB1440\u4e0a\u8f68\u7a81\u7834 + BB2880\u4e0a\u8f68\u7a81\u7834"},
            ],
        },
        {
            "source": "\u8f85\u52a9\u786e\u8ba4",
            "summary_slot": "48H",
            "summary_when_active": "\u8d8b\u52bf\u786e\u8ba4",
            "display": "MA60/96/105/120/300\u4e0a\u659c\u7387 + BB300\u4e0a\u8f68\u7a81\u7834 + BB1440\u4e0a\u8f68\u7a81\u7834",
            "alternatives": [
                {"features": ("4h:MA60:above_slope_up", "4h:BB300:upper_breakout", "4h:BB1440:upper_breakout"), "win": 66.7, "samples": 481, "text": "MA60\u4e0a\u659c\u7387 + BB300\u4e0a\u8f68\u7a81\u7834 + BB1440\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA96:above_slope_up", "4h:BB300:upper_breakout", "4h:BB1440:upper_breakout"), "win": 66.7, "samples": 481, "text": "MA96\u4e0a\u659c\u7387 + BB300\u4e0a\u8f68\u7a81\u7834 + BB1440\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA105:above_slope_up", "4h:BB300:upper_breakout", "4h:BB1440:upper_breakout"), "win": 66.7, "samples": 481, "text": "MA105\u4e0a\u659c\u7387 + BB300\u4e0a\u8f68\u7a81\u7834 + BB1440\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA120:above_slope_up", "4h:BB300:upper_breakout", "4h:BB1440:upper_breakout"), "win": 66.7, "samples": 481, "text": "MA120\u4e0a\u659c\u7387 + BB300\u4e0a\u8f68\u7a81\u7834 + BB1440\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA300:above_slope_up", "4h:BB300:upper_breakout", "4h:BB1440:upper_breakout"), "win": 66.7, "samples": 481, "text": "MA300\u4e0a\u659c\u7387 + BB300\u4e0a\u8f68\u7a81\u7834 + BB1440\u4e0a\u8f68\u7a81\u7834"},
            ],
        },
    ],
    "ETHUSDT": [
        {
            "source": "\u504f\u7a7a",
            "summary_slot": "4H",
            "summary_when_active": "\u504f\u7a7a",
            "display": "MA\u538b\u5236/\u4e0b\u659c\u7387 + BB\u4e2d\u8f68\u538b\u5236",
            "alternatives": [
                {"features": ("4h:MA60:reject", "4h:BB300:mid_reject"), "win": 72.4, "samples": 58, "text": "MA60\u538b\u5236 + BB300\u4e2d\u8f68\u538b\u5236"},
                {"features": ("4h:MA105:reject", "4h:BB300:mid_reject"), "win": 72.0, "samples": 50, "text": "MA105\u538b\u5236 + BB300\u4e2d\u8f68\u538b\u5236"},
                {"features": ("4h:MA96:reject", "4h:BB300:mid_reject"), "win": 70.7, "samples": 58, "text": "MA96\u538b\u5236 + BB300\u4e2d\u8f68\u538b\u5236"},
                {"features": ("4h:MA7:below_slope_down", "4h:MA120:below_slope_down", "4h:BB120:mid_reject"), "win": 70.1, "samples": 127, "text": "MA7\u4e0b\u659c\u7387 + MA120\u4e0b\u659c\u7387 + BB120\u4e2d\u8f68\u538b\u5236"},
            ],
        },
        {
            "source": "\u504f\u591a",
            "summary_slot": "4H",
            "summary_when_active": "\u504f\u591a",
            "display": "MA\u4e0a\u659c\u7387/\u652f\u6491 + BB\u652f\u6491/\u4e0a\u8f68\u7a81\u7834",
            "alternatives": [
                {"features": ("4h:MA300:above_slope_up", "4h:BB1440:mid_support"), "win": 71.0, "samples": 62, "text": "MA300\u4e0a\u659c\u7387 + BB1440\u4e2d\u8f68\u652f\u6491"},
                {"features": ("4h:MA7:support", "4h:BB300:upper_breakout", "4h:BB4707:upper_breakout"), "win": 69.7, "samples": 66, "text": "MA7\u652f\u6491 + BB300\u4e0a\u8f68\u7a81\u7834 + BB4707\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA96:above_slope_up", "4h:BB1440:mid_support"), "win": 67.6, "samples": 68, "text": "MA96\u4e0a\u659c\u7387 + BB1440\u4e2d\u8f68\u652f\u6491"},
                {"features": ("4h:MA105:above_slope_up", "4h:BB1440:mid_support"), "win": 67.6, "samples": 68, "text": "MA105\u4e0a\u659c\u7387 + BB1440\u4e2d\u8f68\u652f\u6491"},
                {"features": ("4h:MA120:above_slope_up", "4h:BB1440:mid_support"), "win": 67.6, "samples": 68, "text": "MA120\u4e0a\u659c\u7387 + BB1440\u4e2d\u8f68\u652f\u6491"},
                {"features": ("4h:MA7:above_slope_up", "4h:MA20:support", "4h:BB2880:upper_breakout"), "win": 66.7, "samples": 75, "text": "MA7\u4e0a\u659c\u7387 + MA20\u652f\u6491 + BB2880\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA7:above_slope_up", "4h:BB300:upper_breakout", "4h:BB4707:upper_breakout"), "win": 62.8, "samples": 113, "text": "MA7\u4e0a\u659c\u7387 + BB300\u4e0a\u8f68\u7a81\u7834 + BB4707\u4e0a\u8f68\u7a81\u7834"},
                {"features": ("4h:MA60:above_slope_up", "4h:BB300:upper_breakout", "4h:BB4707:upper_breakout"), "win": 64.3, "samples": 171, "text": "MA60\u4e0a\u659c\u7387 + BB300\u4e0a\u8f68\u7a81\u7834 + BB4707\u4e0a\u8f68\u7a81\u7834"},
            ],
        },
        {
            "source": "\u5f3a\u7a7a",
            "summary_slot": "48H",
            "summary_when_active": "\u5f3a\u7a7a",
            "display": "MA\u538b\u5236/\u4e0b\u659c\u7387 + BB\u4e2d\u8f68\u538b\u5236",
            "alternatives": [
                {"features": ("4h:MA105:reject", "4h:BB300:mid_reject"), "win": 76.0, "samples": 50, "text": "MA105\u538b\u5236 + BB300\u4e2d\u8f68\u538b\u5236"},
                {"features": ("4h:MA120:reject", "4h:BB672:mid_reject"), "win": 75.0, "samples": 52, "text": "MA120\u538b\u5236 + BB672\u4e2d\u8f68\u538b\u5236"},
                {"features": ("4h:MA60:reject", "4h:MA300:below_slope_down", "4h:BB120:mid_reject"), "win": 72.5, "samples": 91, "text": "MA60\u538b\u5236 + MA300\u4e0b\u659c\u7387 + BB120\u4e2d\u8f68\u538b\u5236"},
                {"features": ("4h:MA7:below_slope_down", "4h:MA300:below_slope_down", "4h:BB120:mid_reject"), "win": 72.4, "samples": 116, "text": "MA7\u4e0b\u659c\u7387 + MA300\u4e0b\u659c\u7387 + BB120\u4e2d\u8f68\u538b\u5236"},
                {"features": ("4h:MA105:below_slope_down", "4h:MA300:below_slope_down", "4h:BB1920:mid_reject"), "win": 70.6, "samples": 68, "text": "MA105\u4e0b\u659c\u7387 + MA300\u4e0b\u659c\u7387 + BB1920\u4e2d\u8f68\u538b\u5236"},
                {"features": ("4h:MA120:below_slope_down", "4h:MA300:below_slope_down", "4h:BB1920:mid_reject"), "win": 70.6, "samples": 68, "text": "MA120\u4e0b\u659c\u7387 + MA300\u4e0b\u659c\u7387 + BB1920\u4e2d\u8f68\u538b\u5236"},
            ],
        },
        {
            "source": "\u5f3a\u591a",
            "summary_slot": "48H",
            "summary_when_active": "\u5f3a\u591a",
            "display": "MA\u4e0a\u659c\u7387 + BB\u652f\u6491/\u4e0a\u8f68\u7a81\u7834",
            "alternatives": [
                {"features": ("4h:MA60:above_slope_up", "4h:BB2880:mid_support"), "win": 76.0, "samples": 50, "text": "MA60\u4e0a\u659c\u7387 + BB2880\u4e2d\u8f68\u652f\u6491"},
                {"features": ("4h:MA7:support", "4h:BB4707:mid_support", "4h:MA20:above", "4h:MA60:above_slope_up"), "win": 73.9, "samples": 23, "text": "MA7\u652f\u6491 + BB4707\u4e2d\u8f68\u652f\u6491 + MA20\u4e0a\u65b9 + MA60\u4e0a\u659c\u7387"},
            ],
        },
    ],
}

V6_MACRO_LONG_RULES = {
    "BTCUSDT": [
        {
            "source": "1W偏多",
            "summary_slot": "1W",
            "summary_when_active": "偏多",
            "display": "BB20下轨反弹",
            "alternatives": [
                {"features": ("1w:BB20:lower_bounce",), "win": 71.4, "samples": 28, "text": "BB20下轨反弹（预测1周）"},
            ],
        },
        {
            "source": "1W偏空",
            "summary_slot": "1W",
            "summary_when_active": "偏空",
            "display": "MA20压制 / BB20中轨压制",
            "alternatives": [
                {"features": ("1w:MA20:reject",), "win": 64.5, "samples": 31, "text": "MA20压制（预测2周）"},
                {"features": ("1w:BB20:mid_reject",), "win": 64.5, "samples": 31, "text": "BB20中轨压制（预测2周）"},
            ],
        },
        {
            "source": "2W偏多",
            "summary_slot": "2W",
            "summary_when_active": "偏多",
            "display": "BB20中轨支撑 + BB60>中",
            "alternatives": [
                {"features": ("2w:BB20:mid_support", "2w:BB60:above_mid"), "win": 72.7, "samples": 11, "text": "BB20中轨支撑 + BB60>中（样本少）"},
            ],
        },
        {
            "source": "2W偏空",
            "summary_slot": "2W",
            "summary_when_active": "偏空",
            "display": "MA7下穿",
            "alternatives": [
                {"features": ("2w:MA7:cross_down",), "win": 63.6, "samples": 11, "text": "MA7下穿（样本少）"},
            ],
        },
    ],
    "ETHUSDT": [
        {
            "source": "1W偏多",
            "summary_slot": "1W",
            "summary_when_active": "观察多",
            "display": "MA120/BB120中轨支撑",
            "alternatives": [
                {"features": ("1w:MA120:support",), "win": 60.0, "samples": 25, "text": "MA120支撑（弱信号）"},
                {"features": ("1w:BB120:mid_support",), "win": 60.0, "samples": 25, "text": "BB120中轨支撑（弱信号）"},
            ],
        },
        {
            "source": "1W偏空",
            "summary_slot": "1W",
            "summary_when_active": "偏空",
            "display": "MA20压制 / BB20中轨压制",
            "alternatives": [
                {"features": ("1w:MA20:reject",), "win": 75.0, "samples": 28, "text": "MA20压制（预测1周）"},
                {"features": ("1w:BB20:mid_reject",), "win": 75.0, "samples": 28, "text": "BB20中轨压制（预测1周）"},
            ],
        },
        {
            "source": "2W偏多",
            "summary_slot": "2W",
            "summary_when_active": "观察多",
            "display": "MA7上穿",
            "alternatives": [
                {"features": ("2w:MA7:cross_up",), "win": 63.6, "samples": 11, "text": "MA7上穿（样本少）"},
            ],
        },
        {
            "source": "2W偏空",
            "summary_slot": "2W",
            "summary_when_active": "偏空",
            "display": "MA7下穿 / BB20上轨压制",
            "alternatives": [
                {"features": ("2w:MA7:cross_down",), "win": 81.8, "samples": 11, "text": "MA7下穿（样本少）"},
                {"features": ("2w:BB20:upper_reject",), "win": 80.0, "samples": 10, "text": "BB20上轨压制（样本少）"},
            ],
        },
    ],
}


V6_MACRO_4H_FEATURE_LABELS = {
    "4h:MA60:below_slope_down": "MA60\u4e0b\u659c\u7387",
    "4h:MA96:below_slope_down": "MA96\u4e0b\u659c\u7387",
    "4h:MA7:support": "MA7\u652f\u6491",
    "4h:MA7:above_slope_up": "MA7\u4e0a\u659c\u7387",
    "4h:MA7:below_slope_down": "MA7\u4e0b\u659c\u7387",
    "4h:MA20:above": "MA20\u4e0a\u65b9",
    "4h:MA20:support": "MA20\u652f\u6491",
    "4h:MA60:above_slope_up": "MA60\u4e0a\u659c\u7387",
    "4h:MA60:reject": "MA60\u538b\u5236",
    "4h:MA96:above_slope_up": "MA96\u4e0a\u659c\u7387",
    "4h:MA96:reject": "MA96\u538b\u5236",
    "4h:MA105:support": "MA105\u652f\u6491",
    "4h:MA105:above_slope_up": "MA105\u4e0a\u659c\u7387",
    "4h:MA105:reject": "MA105\u538b\u5236",
    "4h:MA105:below_slope_down": "MA105\u4e0b\u659c\u7387",
    "4h:MA120:above_slope_up": "MA120\u4e0a\u659c\u7387",
    "4h:MA120:reject": "MA120\u538b\u5236",
    "4h:MA120:below_slope_down": "MA120\u4e0b\u659c\u7387",
    "4h:MA300:below_slope_down": "MA300\u4e0b\u659c\u7387",
    "4h:MA300:above_slope_up": "MA300\u4e0a\u659c\u7387",
    "4h:BB105:mid_support": "BB105\u4e2d\u8f68\u652f\u6491",
    "4h:BB120:mid_reject": "BB120\u4e2d\u8f68\u538b\u5236",
    "4h:BB300:mid_reject": "BB300\u4e2d\u8f68\u538b\u5236",
    "4h:BB300:mid_support": "BB300\u4e2d\u8f68\u652f\u6491",
    "4h:BB300:lower_bounce": "BB300\u4e0b\u8f68\u53cd\u5f39",
    "4h:BB300:upper_breakout": "BB300\u4e0a\u8f68\u7a81\u7834",
    "4h:BB120:upper_reject": "BB120\u4e0a\u8f68\u538b\u5236",
    "4h:BB672:mid_reject": "BB672\u4e2d\u8f68\u538b\u5236",
    "4h:BB1440:upper_breakout": "BB1440\u4e0a\u8f68\u7a81\u7834",
    "4h:BB1440:mid_support": "BB1440\u4e2d\u8f68\u652f\u6491",
    "4h:BB1920:mid_reject": "BB1920\u4e2d\u8f68\u538b\u5236",
    "4h:BB1920:upper_breakout": "BB1920\u4e0a\u8f68\u7a81\u7834",
    "4h:BB2880:mid_support": "BB2880\u4e2d\u8f68\u652f\u6491",
    "4h:BB2880:upper_breakout": "BB2880\u4e0a\u8f68\u7a81\u7834",
    "4h:BB4707:mid_support": "BB4707\u4e2d\u8f68\u652f\u6491",
    "4h:BB4707:upper_breakout": "BB4707\u4e0a\u8f68\u7a81\u7834",
    "4h:BB4707:upper_reject": "BB4707\u4e0a\u8f68\u538b\u5236",
    "1w:MA20:reject": "1W MA20压制",
    "1w:MA120:support": "1W MA120支撑",
    "1w:BB20:lower_bounce": "1W BB20下轨反弹",
    "1w:BB20:mid_reject": "1W BB20中轨压制",
    "1w:BB120:mid_support": "1W BB120中轨支撑",
    "2w:MA7:cross_up": "2W MA7上穿",
    "2w:MA7:cross_down": "2W MA7下穿",
    "2w:BB20:mid_support": "2W BB20中轨支撑",
    "2w:BB20:upper_reject": "2W BB20上轨压制",
    "2w:BB60:above_mid": "2W BB60>中",
}


def signal_line(symbol: str, signal: dict[str, Any]) -> str:
    if symbol in V6_MACRO_4H_RULES:
        return v6_macro_4h_signal_line(symbol, signal)
    if not signal:
        return f"{symbol}: \u6682\u65e0\u4fe1\u53f7"
    return v6_direction_signal_line(symbol, signal)


def v6_macro_4h_signal_line(symbol: str, signal: dict[str, Any]) -> str:
    now = time.time()
    cached = V6_DIRECTION_CACHE.get(symbol)
    if cached and now - cached[0] < V6_DIRECTION_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        data = v6_macro_4h_indicator_data(symbol)
        price = float(signal.get("price") or data.get("price") or 0.0)
        states = [v6_macro_4h_rule_state(rule, data) for rule in V6_MACRO_4H_RULES.get(symbol, [])]
        long_states = [v6_macro_4h_rule_state(rule, data) for rule in V6_MACRO_LONG_RULES.get(symbol, [])]
        summary = v6_macro_4h_summary(symbol, states)
        long_summary = v6_macro_long_summary(long_states)
        source_lines = [
            v6_macro_4h_source_text(item)
            for item in [*states, *long_states]
            if item.get("active")
        ]
        line = f"📈 4H/长周期信号：价格={number(price, 2)}｜{summary}｜{long_summary}"
        if source_lines:
            line += "\n  " + "\n  ".join(source_lines)
        else:
            waiting_lines = v6_macro_4h_waiting_lines(states)
            if waiting_lines:
                line += "\n  " + "\n  ".join(waiting_lines)
    except Exception as exc:
        price = signal.get("price") if signal else None
        line = f"{symbol}: \u4ef7\u683c={number(price, 2)} 4H\u9884\u6d4b=\u8ba1\u7b97\u5931\u8d25({type(exc).__name__})"
    V6_DIRECTION_CACHE[symbol] = (now, line)
    return line


def v6_macro_4h_indicator_data(symbol: str) -> dict[str, Any]:
    rows_4h = v6_direction_closed_rows(fetch_command_klines(symbol, "4h", V6_MACRO_4H_LOOKBACK))
    rows_1d = v6_direction_closed_rows(fetch_command_klines(symbol, "1d", V6_MACRO_LONG_LOOKBACK_1D))
    rows_1w = v6_direction_aggregate_rows(rows_1d, 7)
    rows_2w = v6_direction_aggregate_rows(rows_1d, 14)
    prepared = v6_direction_prepare_tf(rows_4h)
    price = float(rows_4h[-1].get("close") or 0.0) if rows_4h else 0.0
    return {
        "4h": prepared,
        "1d": v6_direction_prepare_tf(rows_1d),
        "1w": v6_direction_prepare_tf(rows_1w),
        "2w": v6_direction_prepare_tf(rows_2w),
        "price": price,
    }


def v6_macro_4h_rule_state(rule: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for alt in rule.get("alternatives") or []:
        features = tuple(alt.get("features") or ())
        matched = [feature for feature in features if v6_direction_feature_active(feature, data)]
        missing = [feature for feature in features if feature not in matched]
        item = {
            "rule": rule,
            "alternative": alt,
            "features": features,
            "matched": matched,
            "missing": missing,
            "active": not missing and bool(features),
        }
        if best is None:
            best = item
            continue
        best_score = (1 if best["active"] else 0, len(best["matched"]), float(best["alternative"].get("win") or 0.0))
        item_score = (1 if item["active"] else 0, len(item["matched"]), float(item["alternative"].get("win") or 0.0))
        if item_score > best_score:
            best = item
    return best or {"rule": rule, "alternative": {}, "features": (), "matched": [], "missing": [], "active": False}


def v6_macro_4h_summary(symbol: str, states: list[dict[str, Any]]) -> str:
    active_sources = {str(item["rule"].get("source")): item for item in states if item.get("active")}
    if symbol == "BTCUSDT":
        weak_short = "\u504f\u7a7a" in active_sources
        strong_short = "\u5f3a\u7a7a" in active_sources
        weak_long = "\u504f\u591a" in active_sources
        strong_long = "\u5f3a\u591a" in active_sources
        if (weak_long or strong_long) and (weak_short or strong_short):
            h4 = "\u51b2\u7a81"
        elif weak_long or strong_long:
            h4 = "\u504f\u591a"
        elif weak_short or strong_short:
            h4 = "\u504f\u7a7a"
        else:
            h4 = "\u7b49\u5f85"
        if strong_long and strong_short:
            h48 = "\u51b2\u7a81"
        elif strong_long:
            h48 = "\u5f3a\u591a"
        elif strong_short:
            h48 = "\u5f3a\u7a7a"
        elif weak_long:
            h48 = "\u504f\u591a"
        elif weak_short:
            h48 = "\u504f\u7a7a"
        else:
            h48 = "\u7b49\u5f85"
        return f"4H{h4}\uff0c48H{h48}"
    if symbol == "ETHUSDT":
        weak_short = "\u504f\u7a7a" in active_sources
        strong_short = "\u5f3a\u7a7a" in active_sources
        weak_long = "\u504f\u591a" in active_sources
        strong_long = "\u5f3a\u591a" in active_sources
        if (weak_long or strong_long) and (weak_short or strong_short):
            h4 = "\u51b2\u7a81"
        elif weak_short or strong_short:
            h4 = "\u504f\u7a7a"
        elif weak_long or strong_long:
            h4 = "\u504f\u591a"
        else:
            h4 = "\u7b49\u5f85"
        if (strong_long and (weak_short or strong_short)) or (strong_short and (weak_long or strong_long)):
            h48 = "\u51b2\u7a81"
        elif strong_long:
            h48 = "\u5f3a\u591a"
        elif strong_short:
            h48 = "\u5f3a\u7a7a"
        elif weak_long:
            h48 = "\u504f\u591a"
        elif weak_short:
            h48 = "\u504f\u7a7a"
        else:
            h48 = "\u7b49\u5f85"
        return f"4H{h4}\uff0c48H{h48}"
    if active_sources:
        return "\uff0c".join(f"{name}\u89e6\u53d1" for name in active_sources)
    return "4H\u7b49\u5f85\uff0c48H\u7b49\u5f85"


def v6_macro_long_summary(states: list[dict[str, Any]]) -> str:
    active_sources = {str(item["rule"].get("source")): item for item in states if item.get("active")}

    def slot_summary(prefix: str) -> str:
        long_item = active_sources.get(f"{prefix}偏多")
        short_item = active_sources.get(f"{prefix}偏空")
        if long_item and short_item:
            return "冲突"
        if long_item:
            return str(long_item["rule"].get("summary_when_active") or "偏多")
        if short_item:
            return str(short_item["rule"].get("summary_when_active") or "偏空")
        return "等待"

    return f"1W{slot_summary('1W')}，2W{slot_summary('2W')}"


def v6_macro_4h_source_text(state: dict[str, Any]) -> str:
    rule = state.get("rule") or {}
    alt = state.get("alternative") or {}
    source = str(rule.get("source") or "\u4fe1\u53f7")
    display = str((alt.get("text") if state.get("active") else rule.get("display")) or alt.get("text") or "--")
    for noise in ("（预测1周）", "（预测2周）", "（样本少）", "（弱信号）"):
        display = display.replace(noise, "")
    win = float(alt.get("win") or 0.0)
    samples = int(alt.get("samples") or 0)
    if state.get("active"):
        return f"{source}={display}\uff08{win:.1f}%,n={samples}\uff09"
    matched = len(state.get("matched") or [])
    total = len(state.get("features") or [])
    missing = state.get("missing") or []
    missing_text = ",".join(V6_MACRO_4H_FEATURE_LABELS.get(item, item) for item in missing[:2]) or "--"
    return f"{source}=\u672a\u89e6\u53d1 {display}\uff08{matched}/{total}\uff0c\u7f3a:{missing_text}\uff09"


def v6_macro_4h_waiting_lines(states: list[dict[str, Any]], limit: int = 2) -> list[str]:
    candidates = [
        state for state in states
        if not state.get("active") and state.get("matched")
    ]
    candidates.sort(
        key=lambda state: (
            len(state.get("matched") or []),
            float((state.get("alternative") or {}).get("win") or 0.0),
        ),
        reverse=True,
    )
    lines: list[str] = []
    for state in candidates[:max(1, int(limit))]:
        rule = state.get("rule") or {}
        source = str(rule.get("source") or "信号")
        matched = [
            V6_MACRO_4H_FEATURE_LABELS.get(item, item)
            for item in (state.get("matched") or [])
        ]
        missing = [
            V6_MACRO_4H_FEATURE_LABELS.get(item, item)
            for item in (state.get("missing") or [])
        ]
        lines.append(
            f"{source}近触发：已满足={','.join(matched[:3]) or '--'}；缺={','.join(missing[:3]) or '--'}"
        )
    return lines


def v6_direction_signal_line(symbol: str, signal: dict[str, Any]) -> str:
    now = time.time()
    cached = V6_DIRECTION_CACHE.get(symbol)
    if cached and now - cached[0] < V6_DIRECTION_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        data = v6_direction_indicator_data(symbol)
        parts = []
        for horizon in ("24H", "48H", "72H"):
            parts.append(v6_direction_horizon_text(horizon, data))
        line = f"{symbol}: \u4ef7\u683c={number(signal.get('price'), 2)} " + " ".join(parts)
    except Exception as exc:
        line = f"{symbol}: \u4ef7\u683c={number(signal.get('price'), 2)} \u5927\u65b9\u5411=\u8ba1\u7b97\u5931\u8d25({type(exc).__name__})"
    V6_DIRECTION_CACHE[symbol] = (now, line)
    return line


def v6_direction_horizon_text(horizon: str, data: dict[str, Any]) -> str:
    rules = [rule for rule in V6_DIRECTION_RULES if rule["horizon"] == horizon]
    evaluated = []
    for rule in rules:
        matched = [feature for feature in rule["features"] if v6_direction_feature_active(feature, data)]
        missing = [feature for feature in rule["features"] if feature not in matched]
        evaluated.append((rule, matched, missing))
    if not evaluated:
        return f"{horizon}=\u89c2\u5bdf(\u65e0\u5019\u9009\u7ec4\u5408)"

    active = [(rule, matched, missing) for rule, matched, missing in evaluated if not missing]
    if active:
        active.sort(key=lambda item: float(item[0]["win"]), reverse=True)
        rule = active[0][0]
        return f"{horizon}=\u770b{rule['direction']}({rule['win']:.1f}%,n={rule['samples']}) \u4f9d\u636e:{rule['reason']}"

    long_text = ""
    short_text = ""
    long_count = 0
    short_count = 0
    for rule, matched, missing in evaluated:
        item_text = v6_direction_rule_progress_text(rule, matched, missing)
        if rule.get("direction") == "\u591a":
            long_text = item_text
            long_count = len(matched)
        elif rule.get("direction") == "\u7a7a":
            short_text = item_text
            short_count = len(matched)
    if long_count == 0 and short_count == 0:
        bias = "\u7b49\u5f85"
    elif long_count == short_count:
        bias = "\u51b2\u7a81"
    elif long_count > short_count:
        bias = "\u504f\u591a"
    else:
        bias = "\u504f\u7a7a"
    return f"{horizon}=\u89c2\u5bdf({long_text} | {short_text} => {bias})"


def v6_direction_rule_progress_text(rule: dict[str, Any], matched: list[str], missing: list[str]) -> str:
    direction = str(rule.get("direction") or "--")
    missing_text = ",".join(v6_direction_feature_label(item) for item in missing[:2]) or "--"
    return f"{direction}{len(matched)}/{len(rule['features'])}({float(rule['win']):.1f}%,\u7f3a:{missing_text})"


def v6_direction_feature_label(feature: str) -> str:
    return V6_DIRECTION_FEATURE_LABELS.get(feature, feature)


def v6_direction_indicator_data(symbol: str) -> dict[str, Any]:
    rows_15m = v6_direction_closed_rows(fetch_command_klines(symbol, "15m", 2900))
    rows_4h = v6_direction_closed_rows(fetch_command_klines(symbol, "4h", 1460))
    rows_1d = v6_direction_closed_rows(fetch_command_klines(symbol, "1d", V6_MACRO_LONG_LOOKBACK_1D))
    rows_2d = v6_direction_aggregate_rows(rows_1d, 2)
    rows_3d = v6_direction_aggregate_rows(rows_1d, 3)
    rows_1w = v6_direction_aggregate_rows(rows_1d, 7)
    rows_2w = v6_direction_aggregate_rows(rows_1d, 14)
    return {
        "15m": v6_direction_prepare_tf(rows_15m),
        "4h": v6_direction_prepare_tf(rows_4h),
        "1d": v6_direction_prepare_tf(rows_1d),
        "2d": v6_direction_prepare_tf(rows_2d),
        "3d": v6_direction_prepare_tf(rows_3d),
        "1w": v6_direction_prepare_tf(rows_1w),
        "2w": v6_direction_prepare_tf(rows_2w),
    }


def v6_direction_closed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = [row for row in rows if float(row.get("close") or 0.0) > 0]
    return clean[:-1] if len(clean) > 2 else clean


def v6_direction_aggregate_rows(rows: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    interval_ms = 86400000 * days
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        key = int(row.get("open_time") or 0) // interval_ms * interval_ms
        buckets.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda item: int(item.get("open_time") or 0))
        if len(group) < days:
            continue
        output.append({
            "open_time": key,
            "open": float(group[0].get("open") or group[0].get("close") or 0.0),
            "high": max(float(item.get("high") or item.get("close") or 0.0) for item in group),
            "low": min(float(item.get("low") or item.get("close") or 0.0) for item in group),
            "close": float(group[-1].get("close") or 0.0),
        })
    return output


def v6_direction_prepare_tf(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closes = [float(row.get("close") or 0.0) for row in rows]
    return {"rows": rows, "closes": closes}


def v6_direction_sma(closes: list[float], period: int, offset: int = 0) -> float:
    end = len(closes) - offset
    start = end - period
    if period <= 0 or start < 0 or end > len(closes):
        return 0.0
    return sum(closes[start:end]) / period


def v6_direction_feature_active(feature: str, data: dict[str, Any]) -> bool:
    timeframe, indicator, condition = feature.split(":", 2)
    tf = data.get(timeframe) or {}
    rows = tf.get("rows") or []
    closes = tf.get("closes") or []
    if len(rows) < 3 or len(closes) < 3:
        return False
    row = rows[-1]
    prev_row = rows[-2]
    high = float(row.get("high") or row.get("close") or 0.0)
    low = float(row.get("low") or row.get("close") or 0.0)
    close = float(row.get("close") or 0.0)
    prev_close = float(prev_row.get("close") or 0.0)
    tolerance = V6_DIRECTION_TOLERANCE.get(timeframe, 0.004)
    if indicator.startswith("MA"):
        period = int(indicator[2:])
        value = v6_direction_sma(closes, period)
        prev_value = v6_direction_sma(closes, period, offset=1)
        older_value = v6_direction_sma(closes, period, offset=3)
        if value <= 0 or prev_value <= 0:
            return False
        if condition == "cross_up":
            return prev_close <= prev_value and close > value
        if condition == "cross_down":
            return prev_close >= prev_value and close < value
        if condition == "support":
            return low <= value * (1.0 + tolerance) and close >= value * (1.0 - tolerance) and close >= value
        if condition == "reject":
            return high >= value * (1.0 - tolerance) and close <= value * (1.0 + tolerance) and close <= value
        if condition == "above":
            return close > value
        if condition == "above_slope_up":
            return older_value > 0 and close > value and value > older_value
        if condition == "below_slope_down":
            return older_value > 0 and close < value and value < older_value
        return False
    if indicator.startswith("BB"):
        period = int(indicator[2:])
        if len(closes) < period + 1:
            return False
        upper, mid, lower = compute_bb(closes, period)
        _prev_upper, prev_mid, _prev_lower = compute_bb(closes[:-1], period)
        if condition == "above_mid":
            return close > mid
        if condition == "below_mid":
            return close < mid
        if condition == "lower_bounce":
            return low <= lower * (1.0 + tolerance) and close >= lower
        if condition == "upper_reject":
            return high >= upper * (1.0 - tolerance) and close <= upper
        if condition == "upper_breakout":
            return close > upper
        if condition == "mid_support":
            return low <= mid * (1.0 + tolerance) and close >= mid
        if condition == "mid_reject":
            return high >= mid * (1.0 - tolerance) and close <= mid
        if condition == "mid_reclaim":
            return prev_close <= prev_mid and close > mid
        if condition == "mid_loss":
            return prev_close >= prev_mid and close < mid
        return False
    return False


def v6_signal_levels(signal: dict[str, Any]) -> dict[str, Any]:
    levels = signal.get("levels")
    return levels if isinstance(levels, dict) else {}


def v6_signal_reasons_text(signal: dict[str, Any]) -> str:
    reasons = signal.get("reasons")
    if isinstance(reasons, list):
        return " ".join(str(item) for item in reasons)
    return str(reasons or "")


def v6_signal_15m_direction(signal: dict[str, Any]) -> str:
    reasons = v6_signal_reasons_text(signal)
    if "15M MA20 held for 2 candles" in reasons:
        return "偏多"
    if "15M MA20 rejected for 2 candles" in reasons:
        return "偏空"
    if "15M MA20 trend guard is broken or mixed" in reasons:
        return "震荡"

    scan = v6_signal_levels(signal).get("structure_entry_scan") or {}
    try:
        long_count = int(scan.get("long_candidate_count") or 0)
    except Exception:
        long_count = 0
    try:
        short_count = int(scan.get("short_candidate_count") or 0)
    except Exception:
        short_count = 0
    if long_count > 0 and short_count > 0:
        return "震荡"
    if long_count > 0:
        return "偏多"
    if short_count > 0:
        return "偏空"
    return "震荡"


def v6_signal_4h_direction(signal: dict[str, Any]) -> str:
    levels = v6_signal_levels(signal)
    reasons = v6_signal_reasons_text(signal)
    h4 = levels.get("h4_right_side") or {}
    h4_state = str(h4.get("state") or "").lower()
    if h4.get("long_confirmed") or h4_state == "long_right_side":
        return "多头右侧"
    if h4.get("short_confirmed") or h4_state == "short_right_side":
        return "空头右侧"

    ma20 = levels.get("ma20_4h_pressure") or {}
    if ma20.get("support_active"):
        return "支撑震荡"
    if ma20.get("pressure_active"):
        return "压力震荡"
    if "4H above MA20 with positive MA20 slope" in reasons:
        return "偏多"
    if "4H below MA20 with negative MA20 slope" in reasons:
        return "偏空"
    return "震荡"


def v6_signal_1d_direction(signal: dict[str, Any]) -> str:
    levels = v6_signal_levels(signal)
    reasons = v6_signal_reasons_text(signal)
    daily = levels.get("daily_ma7_trend") or {}
    daily_state = str(daily.get("state") or "").lower()
    if daily.get("bearish_stack") or daily_state == "bearish_stack_ma20_above_ma7" or "1D MA20 is above MA7; daily trend remains bearish until MA7 is reclaimed" in reasons:
        return "空头"
    if daily.get("short_confirmed") or daily_state == "short_ma7_rebreak" or "1D MA7 breakdown-retest-rebreak trend short is confirmed" in reasons:
        return "空头"
    if daily.get("long_confirmed") or daily_state == "long_ma7_rebreak" or "1D MA7 breakout-retest-rebreak trend long is confirmed" in reasons:
        return "多头"
    return "震荡"


def v6_signal_status_conclusion(d15: str, d4h: str, d1d: str, signal: dict[str, Any]) -> str:
    if d1d == "空头":
        if "多头" in d4h or d4h == "偏多":
            return "日线空头，4H反弹，短线谨慎"
        if "空头" in d4h or d4h == "偏空":
            return "空头延续"
        if d15 == "偏空":
            return "日线空头，15M偏空"
        return "日线空头，短线等确认"
    if d1d == "多头":
        if "空头" in d4h or d4h == "偏空":
            return "日线多头，4H回调，短线谨慎"
        if "多头" in d4h or d4h == "偏多":
            return "多头延续"
        if d15 == "偏多":
            return "日线多头，15M偏多"
        return "日线多头，短线等确认"
    if ("空头" in d4h or d4h == "偏空") and d15 == "偏空":
        return "偏空"
    if ("多头" in d4h or d4h == "偏多") and d15 == "偏多":
        return "偏多"
    side_bias = zh_value(signal.get("side_bias"))
    if side_bias in {"多头", "空头"}:
        return f"{side_bias}倾向"
    return "震荡"


TIMEFRAME_ALIASES = {
    "1": "1m",
    "1m": "1m",
    "1min": "1m",
    "1minute": "1m",
    "1分钟": "1m",
    "5": "5m",
    "5m": "5m",
    "5min": "5m",
    "5minute": "5m",
    "5分钟": "5m",
    "15": "15m",
    "15m": "15m",
    "15min": "15m",
    "15minute": "15m",
    "15分钟": "15m",
    "1h": "1h",
    "60m": "1h",
    "60min": "1h",
    "1hour": "1h",
    "1小时": "1h",
    "4h": "4h",
    "240m": "4h",
    "240min": "4h",
    "4hour": "4h",
    "1d": "1d",
    "1day": "1d",
    "daily": "1d",
    "4小时": "4h",
}

BB_PERIODS_BY_INTERVAL = {
    "1m": [105, 210, 300, 672, 1920],
    "5m": [96, 105, 120, 210, 300, 672, 1440],
    "15m": [96, 105, 120, 210, 300, 672, 1440, 1920, 2880, 4707],
    "1h": [96, 105, 120, 210, 300, 672, 1440],
    "4h": [96, 105, 120, 210, 300, 672, 1440, 1920, 2880, 4707],
    "1d": [96, 105, 120, 210, 300, 672, 1440, 1920, 2880, 4707],
}

TOLERANCE_15M = 0.0015
V6_ACTIVITY_ALERT_STATE_TTL_SECONDS = 3600
DEFAULT_ALERT_STATE_TTL_SECONDS = 7 * 24 * 3600

CHART_COMMANDS = {"fc", "chart", "图", "图表", "图片", "kline", "k线", "行情图"}
CHART_MARKERS = CHART_COMMANDS | {"image", "png"}

REASON_ZH_MAP = {
    "4H above MA20 with positive MA20 slope": "4H站上MA20，且MA20斜率为正",
    "4H below MA20 with negative MA20 slope": "4H低于MA20，且MA20斜率为负",
    "4H structure is broken": "4H结构破位",
    "price is already near the lower side of the 4H 20-bar range": "价格已靠近4H近20根K线区间下沿",
    "15M long-cycle structure is failed; use 4H as primary frame": "15M长周期结构失效，主参考切到4H",
    "price is far below multiple 4H mids; avoid chasing": "价格远低于多条4H中轨，避免追空",
    "1M upper pressure is active": "1M上轨压力有效",
    "V6 model quality score is low; increase wait weight": "V6模型质量分偏低，提高等待权重",
    "V6 model quality score supports following the structure": "V6模型质量分支持顺结构跟随",
    "1D MA20 is above MA7; daily trend remains bearish until MA7 is reclaimed": "1D MA20在MA7上方，日线仍为空头，第一关是MA7",
    "1D MA7 is reclaimed; first daily bearish checkpoint is under test": "价格站上1D MA7，正在测试日线空头第一关",
}


def command_help() -> str:
    return "\n".join(
        [
            "[Somnus命令]",
            "V5/V6交易bot只播报开仓、加仓、减仓、平仓和错误告警。",
            "持仓p、图表fc、宏观report、新闻news、判断解释已交给雪见。",
        ]
    )


def normalize_symbol(raw: str) -> str:
    symbol = str(raw or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if not symbol:
        return ""
    # Command arguments may be natural-language Chinese such as ``相关币种``.
    # Do not blindly append USDT to non-exchange tokens; that produces invalid
    # Binance symbols like ``相关币种USDT`` and makes report/fc look unresponsive.
    if not re.fullmatch(r"[A-Z0-9]{2,30}", symbol):
        return ""
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    return symbol


def normalize_interval(raw: str) -> str:
    value = str(raw or "15m").strip().lower()
    return TIMEFRAME_ALIASES.get(value, "")


def parse_focus_period(raw: Any) -> int | None:
    text = str(raw or "").strip().lower()
    if not text:
        return None
    if text.startswith("bb"):
        text = text[2:]
    try:
        value = int(text)
    except Exception:
        return None
    return value if value > 0 else None


def split_command_parts(text: str) -> tuple[list[str], str, bool]:
    parts = [part for part in str(text or "").strip().split() if part]
    if not parts:
        return [], "", False
    first = parts[0].strip()
    has_slash = first.startswith("/")
    if has_slash:
        first = first[1:]
    command = first.split("@", 1)[0].lower()
    return parts, command, has_slash


def normalize_bot_username(value: Any) -> str:
    return str(value or "").strip().lstrip("@").lower()


def text_mentions_bot(text: str, bot_username: str) -> bool:
    username = normalize_bot_username(bot_username)
    if not username:
        return False
    return any(normalize_bot_username(match.group(1)) == username for match in BOT_MENTION_RE.finditer(str(text or "")))


def message_mentions_bot(message: dict[str, Any], bot_username: str) -> bool:
    text = str(message.get("text") or "")
    return text_mentions_bot(text, bot_username)


def text_mentions_any_bot(text: str) -> bool:
    return BOT_MENTION_RE.search(str(text or "")) is not None


def strip_bot_mentions(text: str) -> str:
    cleaned = BOT_MENTION_RE.sub("", str(text or ""))
    return re.sub(r"\s+", " ", cleaned).strip()


def position_target_matches_scope(scope: str, target: str) -> bool:
    current_scope = str(scope or "all").strip().lower()
    current_target = str(target or "").strip().lower()
    return current_scope == "all" or current_scope == current_target


def parse_command_text(text: str) -> tuple[str, str] | None:
    parts, command, has_slash = split_command_parts(text)
    if not parts:
        return None
    if command in {"start", "help", "命令", "帮助"}:
        return ("HELP", "")
    if command == "bb":
        if len(parts) < 2:
            return ("HELP", "")
        symbol = normalize_symbol(parts[1])
        interval = normalize_interval(parts[2] if len(parts) >= 3 else "15m")
    else:
        symbol = normalize_symbol(command)
        interval = normalize_interval(parts[1] if len(parts) >= 2 else "15m")
    if not symbol or not interval:
        return ("HELP", "")
    return symbol, interval


def parse_chart_command_text(text: str) -> tuple[str, str, int | None] | None:
    parts, command, has_slash = split_command_parts(text)
    if not parts:
        return None
    if (
        not has_slash
        and command not in CHART_COMMANDS
        and command not in {"start", "help", "命令", "帮助", "bb"}
    ):
        return None
    args = parts[1:]

    if command in CHART_COMMANDS:
        if not args:
            return ("HELP", "", None)
        symbol = normalize_symbol(args[0])
        interval = "15m"
        focus_period: int | None = None
        for token in args[1:]:
            marker = token.strip().lower()
            if marker in CHART_MARKERS:
                continue
            normalized_interval = normalize_interval(marker)
            if normalized_interval:
                interval = normalized_interval
                continue
            period = parse_focus_period(marker)
            if period:
                focus_period = period
        if not symbol:
            return ("HELP", "", None)
        return symbol, interval, focus_period

    if command in {"start", "help", "命令", "帮助"}:
        return None

    if command == "bb":
        if not any(token.strip().lower() in CHART_MARKERS for token in args):
            return None
        if not args:
            return ("HELP", "", None)
        symbol = normalize_symbol(args[0])
        interval = "15m"
        focus_period = None
        for token in args[1:]:
            marker = token.strip().lower()
            if marker in CHART_MARKERS:
                continue
            normalized_interval = normalize_interval(marker)
            if normalized_interval:
                interval = normalized_interval
                continue
            period = parse_focus_period(marker)
            if period:
                focus_period = period
        if not symbol:
            return ("HELP", "", None)
        return symbol, interval, focus_period

    if not any(token.strip().lower() in CHART_MARKERS for token in args):
        return None

    symbol = normalize_symbol(command)
    interval = "15m"
    focus_period = None
    for token in args:
        marker = token.strip().lower()
        if marker in CHART_MARKERS:
            continue
        normalized_interval = normalize_interval(marker)
        if normalized_interval:
            interval = normalized_interval
            continue
        period = parse_focus_period(marker)
        if period:
            focus_period = period
    if not symbol:
        return ("HELP", "", None)
    return symbol, interval, focus_period


def parse_position_command_text(text: str) -> str | None:
    parts, command, _has_slash = split_command_parts(text)
    if not parts:
        return None
    if command not in {"p", "pos", "position", "positions", "持仓"}:
        return None
    if len(parts) < 2:
        return "SELF"
    target = str(parts[1]).strip().lower()
    if target in {"5", "v5"}:
        return "v5"
    if target in {"6", "v6"}:
        return "v6"
    return "HELP"


def parse_report_command_text(text: str) -> list[str] | None:
    parts, command, _has_slash = split_command_parts(text)
    if not parts:
        return None
    if command not in {"report", "rpt", "macro", "报告", "宏观"}:
        return None
    symbols: list[str] = []
    for raw in parts[1:]:
        token = str(raw or "").strip().lower()
        if token in {"15m", "15min", "4h", "1d", "all", "全部"}:
            continue
        symbol = normalize_symbol(raw)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols[:4] if symbols else ["BTCUSDT", "ETHUSDT"]


def pct_distance(price: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return (price - reference) / reference * 100.0


def latest_summary_price(symbol: str) -> float:
    symbol = normalize_symbol(symbol)
    age = file_age_seconds(V6_SUMMARY_FILE)
    if age is None or age > REPORT_SIGNAL_SUMMARY_MAX_AGE_SECONDS:
        return 0.0
    summary = load_json(V6_SUMMARY_FILE)
    for item in summary.get("signals") or []:
        if isinstance(item, dict) and normalize_symbol(str(item.get("symbol") or "")) == symbol:
            try:
                price = float(item.get("price") or 0.0)
            except Exception:
                price = 0.0
            if price > 0:
                return price
    return 0.0


def fetch_mark_price(symbol: str, *, force_refresh: bool = False) -> float:
    symbol = normalize_symbol(symbol)
    if not force_refresh:
        cached = MARK_PRICE_CACHE.get(symbol)
        if cached and time.time() - cached[0] < MARK_PRICE_CACHE_TTL_SECONDS:
            return cached[1]
        summary_price = latest_summary_price(symbol)
        if summary_price > 0:
            MARK_PRICE_CACHE[symbol] = (time.time(), summary_price)
            return summary_price
    url = "https://fapi.binance.com/fapi/v1/premiumIndex?" + urllib.parse.urlencode({"symbol": symbol})
    try:
        assert_binance_public_not_banned(f"tg:/fapi/v1/premiumIndex:{symbol}")
        throttle_binance_public_request()
        with open_url(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        price = float(payload.get("markPrice") or 0.0)
        if price > 0:
            MARK_PRICE_CACHE[symbol] = (time.time(), price)
        return price
    except BinancePublicRateLimitError:
        pass
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        note_binance_public_response(exc.code, body, f"tg:/fapi/v1/premiumIndex:{symbol}")
    except Exception:
        pass
    if force_refresh:
        summary_price = latest_summary_price(symbol)
        if summary_price > 0:
            MARK_PRICE_CACHE[symbol] = (time.time(), summary_price)
            return summary_price
    return 0.0

def command_kline_open_time(row: dict[str, Any]) -> int:
    try:
        return int(row.get("open_time") or row.get("openTime") or row.get("t") or 0)
    except Exception:
        return 0


def load_recent_command_kline_cache(symbol: str, interval: str, bars: int, now: datetime) -> list[dict[str, Any]] | None:
    interval_ms = INTERVAL_MS[interval]
    max_stale_ms = max(interval_ms * 2, 5 * 60 * 1000)
    candidates = sorted(
        CHART_KLINE_CACHE_DIR.glob(f"{symbol}_{interval}_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    now_ms = int(now.timestamp() * 1000)
    for path in candidates[:5]:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                continue
            clean = [row for row in rows if isinstance(row, dict) and command_kline_open_time(row) > 0]
            clean.sort(key=command_kline_open_time)
            if len(clean) < bars:
                continue
            last_open = command_kline_open_time(clean[-1])
            if now_ms - last_open > max_stale_ms:
                continue
            return clean[-bars:]
        except Exception:
            continue
    return None


def fetch_command_klines(
    symbol: str,
    interval: str,
    max_period: int,
    *,
    force_refresh: bool = True,
) -> list[dict[str, Any]]:
    symbol = normalize_symbol(symbol)
    now = utc_now()
    interval_ms = INTERVAL_MS[interval]
    bars = max_period + 20
    cache_key = (symbol, interval, bars)
    if not force_refresh:
        cached = COMMAND_KLINE_MEMORY_CACHE.get(cache_key)
        if cached and time.time() - cached[0] < COMMAND_KLINE_MEMORY_TTL_SECONDS:
            return list(cached[1])
        rows = load_recent_command_kline_cache(symbol, interval, bars, now)
        if rows is not None:
            COMMAND_KLINE_MEMORY_CACHE[cache_key] = (time.time(), list(rows))
            return rows

    start = now - timedelta(milliseconds=interval_ms * bars)
    labeler = CoinScout4HLabeler(cache_dir=str(CHART_KLINE_CACHE_DIR))
    rows = labeler.fetch_klines(symbol, interval, start, now, force_refresh=force_refresh)
    rows = sorted(
        {command_kline_open_time(row): row for row in rows if command_kline_open_time(row) > 0}.values(),
        key=command_kline_open_time,
    )
    if force_refresh and 0 < len(rows) < bars:
        expanded_start = now - timedelta(milliseconds=interval_ms * max(bars + 20, bars * 2))
        expanded_rows = labeler.fetch_klines(symbol, interval, expanded_start, now, force_refresh=True)
        expanded_rows = sorted(
            {command_kline_open_time(row): row for row in expanded_rows if command_kline_open_time(row) > 0}.values(),
            key=command_kline_open_time,
        )
        if len(expanded_rows) > len(rows):
            rows = expanded_rows
    if rows:
        rows = rows[-bars:]
    COMMAND_KLINE_MEMORY_CACHE[cache_key] = (time.time(), list(rows))
    return rows


def bb_line_name(period: int, band: str) -> str:
    band_zh = {"upper": "上轨", "mid": "中轨", "lower": "下轨"}[band]
    return f"BB{period}{band_zh}"


def format_price(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "--"
    if numeric >= 1000:
        return f"{numeric:.2f}"
    if numeric >= 10:
        return f"{numeric:.3f}"
    return f"{numeric:.5f}"


def format_pct_signed(value: Any) -> str:
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return "--"


def format_gap_from_price(price: float, value: float) -> str:
    if price <= 0 or value <= 0:
        return "--"
    gap = abs(value - price) / price * 100.0
    side = "上方" if value > price else "下方"
    return f"{side}{gap:.2f}%"


def build_bb_rows(symbol: str, interval: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    periods = BB_PERIODS_BY_INTERVAL[interval]
    klines = fetch_command_klines(symbol, interval, max(periods))
    if not klines:
        raise RuntimeError(f"没有抓到 {symbol} {interval} K线")
    closes = [float(item["close"]) for item in klines if float(item.get("close") or 0) > 0]
    api_symbol = str(klines[-1].get("binance_symbol") or symbol)
    mark_price = fetch_mark_price(api_symbol, force_refresh=True)
    price = mark_price if mark_price > 0 else float(closes[-1])
    rows: list[dict[str, Any]] = []
    for period in periods:
        if len(closes) < period:
            rows.append({"period": period, "ready": False, "available": len(closes)})
            continue
        upper, mid, lower = compute_bb(closes, period)
        width = upper - lower
        position = (price - lower) / width if width > 0 else 0.5
        rows.append(
            {
                "period": period,
                "ready": True,
                "upper": upper,
                "mid": mid,
                "lower": lower,
                "position": position,
                "dist_upper": pct_distance(price, upper),
                "dist_mid": pct_distance(price, mid),
                "dist_lower": pct_distance(price, lower),
            }
        )
    context = {
        "symbol": symbol,
        "api_symbol": api_symbol,
        "interval": interval,
        "price": price,
        "kline_count": len(closes),
        "last_open_time": klines[-1].get("open_time"),
        "kline_source": "binance实时刷新",
        "kline_refreshed_at": utc_now().isoformat(),
        "closes": closes,
        "klines": klines,
    }
    return rows, context


def collect_level_candidates(rows: list[dict[str, Any]], price: float) -> list[tuple[str, float, float]]:
    candidates: list[tuple[str, float, float]] = []
    for row in rows:
        if not row.get("ready"):
            continue
        period = int(row["period"])
        for band in ("upper", "mid", "lower"):
            value = float(row[band])
            if value <= 0:
                continue
            candidates.append((bb_line_name(period, band), value, pct_distance(price, value)))
    return candidates


def format_nearest_levels(rows: list[dict[str, Any]], price: float) -> list[str]:
    candidates = collect_level_candidates(rows, price)
    if not candidates:
        return ["- 最近轨道=--", "- 下方防线=--", "- 上方压力=--"]
    nearest = min(candidates, key=lambda item: abs(item[2]))
    supports = sorted([item for item in candidates if item[1] < price], key=lambda item: price - item[1])[:3]
    resistances = sorted([item for item in candidates if item[1] > price], key=lambda item: item[1] - price)[:3]

    def fmt_items(items: list[tuple[str, float, float]]) -> str:
        if not items:
            return "--"
        return "；".join(f"{name} {format_price(value)}({format_gap_from_price(price, value)})" for name, value, _dist in items)

    return [
        f"- 最近轨道={nearest[0]} {format_price(nearest[1])}（{format_gap_from_price(price, nearest[1])}）",
        f"- 下方防线={fmt_items(supports)}",
        f"- 上方压力={fmt_items(resistances)}",
    ]


def sma_from_tail(closes: list[float], period: int, offset: int = 0) -> float:
    end = len(closes) - offset
    start = end - period
    if period <= 0 or start < 0 or end > len(closes):
        return 0.0
    return sum(closes[start:end]) / period


def ma20_snapshot(closes: list[float], price: float) -> dict[str, Any]:
    ma20 = sma_from_tail(closes, 20)
    prev = sma_from_tail(closes, 20, offset=1)
    slope = pct_distance(ma20, prev) if ma20 > 0 and prev > 0 else 0.0
    return {
        "ma20": ma20,
        "distance_pct": pct_distance(price, ma20) if ma20 > 0 else 0.0,
        "slope_pct": slope,
    }


def fifteen_m_ma20_hold_state(klines: list[dict[str, Any]], tolerance_pct: float = 0.05) -> str:
    rows = [row for row in klines if float(row.get("close") or 0) > 0]
    if len(rows) < 22:
        return "数据不足"
    closes = [float(row["close"]) for row in rows]
    checked = []
    tolerance = tolerance_pct / 100.0
    for index in range(len(rows) - 2, len(rows)):
        ma20 = sum(closes[index - 19:index + 1]) / 20.0
        low = float(rows[index].get("low") or rows[index].get("close") or 0.0)
        high = float(rows[index].get("high") or rows[index].get("close") or 0.0)
        close = float(rows[index].get("close") or 0.0)
        checked.append(
            {
                "low_hold": low >= ma20 * (1.0 - tolerance) and close >= ma20 * (1.0 - tolerance),
                "high_reject": high <= ma20 * (1.0 + tolerance) and close <= ma20 * (1.0 + tolerance),
            }
        )
    if all(item["low_hold"] for item in checked):
        return "2根不破MA20，短线趋势还在"
    if all(item["high_reject"] for item in checked):
        return "2根不过MA20，短线偏空"
    return "MA20附近震荡，等确认"


def report_level_items(rows: list[dict[str, Any]], price: float) -> tuple[list[tuple[str, float]], list[tuple[str, float]], tuple[str, float] | None]:
    candidates = collect_level_candidates(rows, price)
    supports = [(name, value) for name, value, _dist in sorted([item for item in candidates if item[1] < price], key=lambda item: price - item[1])[:3]]
    resistances = [(name, value) for name, value, _dist in sorted([item for item in candidates if item[1] > price], key=lambda item: item[1] - price)[:3]]
    nearest = None
    if candidates:
        nearest_item = min(candidates, key=lambda item: abs(item[2]))
        nearest = (nearest_item[0], nearest_item[1])
    return supports, resistances, nearest


def format_report_levels(items: list[tuple[str, float]], price: float) -> str:
    if not items:
        return "--"
    return "；".join(f"{name} {format_price(value)}({format_gap_from_price(price, value)})" for name, value in items)


def classify_report_bias(interval: str, rows: list[dict[str, Any]], price: float, ma20: dict[str, Any], ma20_state: str) -> tuple[str, str]:
    ready = [row for row in rows if row.get("ready")]
    if not ready:
        return "数据不足", "BB数据不足"
    above_mids = sum(1 for row in ready if price >= float(row.get("mid") or 0.0))
    below_mids = len(ready) - above_mids
    above_uppers = sum(1 for row in ready if price >= float(row.get("upper") or 0.0))
    below_lowers = sum(1 for row in ready if price <= float(row.get("lower") or 0.0))
    long_score = 0
    short_score = 0
    if ma20.get("ma20", 0.0) > 0:
        if ma20["distance_pct"] > 0:
            long_score += 1
        elif ma20["distance_pct"] < 0:
            short_score += 1
        if ma20["slope_pct"] > 0:
            long_score += 1
        elif ma20["slope_pct"] < 0:
            short_score += 1
    if above_mids >= max(1, int(len(ready) * 0.6)):
        long_score += 2
    if below_mids >= max(1, int(len(ready) * 0.6)):
        short_score += 2
    if above_uppers >= 2:
        long_score += 1
    if below_lowers >= 2:
        short_score += 1
    if interval == "15m":
        if "2根不破" in ma20_state:
            long_score += 2
        elif "2根不过" in ma20_state:
            short_score += 2
    if long_score - short_score >= 2:
        bias = "偏多"
    elif short_score - long_score >= 2:
        bias = "偏空"
    else:
        bias = "震荡/等待"
    detail = f"中轨上方={above_mids}/{len(ready)} 上轨突破={above_uppers} 下轨跌破={below_lowers}"
    return bias, detail


def build_interval_report(symbol: str, interval: str) -> tuple[list[str], float]:
    rows, context = build_bb_rows(symbol, interval)
    price = float(context["price"])
    closes = list(context.get("closes") or [])
    klines = list(context.get("klines") or [])
    ma20 = ma20_snapshot(closes, price)
    ma20_state = fifteen_m_ma20_hold_state(klines) if interval == "15m" else (
        "站上MA20" if ma20.get("distance_pct", 0.0) > 0 else "低于MA20" if ma20.get("ma20", 0.0) > 0 else "MA20缺失"
    )
    bias, detail = classify_report_bias(interval, rows, price, ma20, ma20_state)
    supports, resistances, nearest = report_level_items(rows, price)
    support_text = format_report_levels(supports, price)
    resistance_text = format_report_levels(resistances, price)
    nearest_text = "--" if not nearest else f"{nearest[0]} {format_price(nearest[1])}({format_gap_from_price(price, nearest[1])})"
    lines = [
        f"{interval.upper()}：方向={bias} 价格={format_price(price)} K线={context['kline_count']} 数据源={context['api_symbol']}｜K线来源={context.get('kline_source', 'binance')}",
        f"- MA20={format_price(ma20.get('ma20'))} 距离={format_pct_signed(ma20.get('distance_pct'))} 斜率={format_pct_signed(ma20.get('slope_pct'))}；{ma20_state}",
        f"- 全BB扫描：{detail}；最近={nearest_text}",
        f"- 支撑={support_text}",
        f"- 压力={resistance_text}",
    ]
    return lines, price


def report_symbol_label(symbol: str) -> str:
    text = str(symbol or "").upper()
    return text[:-4] if text.endswith("USDT") else text


def format_report_model_line(symbol: str) -> str:
    signal = find_v6_signal(symbol)
    if not signal:
        return "🧠 模型方向=暂无\n🧠 模型概率=多-- 空-- 等--"
    return (
        f"🧠 模型方向={zh_value(signal.get('side_bias'))}\n"
        f"🧠 模型概率=多{number(float(signal.get('long_probability') or 0) * 100, 2)}% "
        f"空{number(float(signal.get('short_probability') or 0) * 100, 2)}% "
        f"等={number(float(signal.get('wait_probability') or 0) * 100, 2)}%"
    )


def current_v6_summary_positions() -> list[dict[str, Any]]:
    summary = load_json(V6_SUMMARY_FILE)
    positions = summary.get("positions") or []
    return [position for position in positions if isinstance(position, dict)]


def current_report_v6_positions() -> tuple[list[dict[str, Any]], str, str]:
    """Return V6 positions for macro reports, preferring exchange actual PnL.

    The macro report must not silently substitute mark-price estimates for an
    open position's PnL.  When exchange positionRisk cannot be read, keep local
    positions only as position context and return an explicit error note for the
    PnL field.
    """
    local_positions = current_v6_summary_positions()
    try:
        positions, source = fetch_v6_exchange_positions(local_positions)
        return positions, source, ""
    except Exception as exc:
        return local_positions, "本地V6状态", f"未获取到交易所实际盈亏：{zh_error(exc)}"


def position_exchange_actual_pnl(position: dict[str, Any]) -> tuple[float | None, float | None]:
    source = str(position.get("exchange_source") or position.get("pnl_source") or "").lower()
    actual_source = (
        bool(position.get("exchange_pnl_actual"))
        or source.startswith("binance")
        or "positionrisk" in source
        or source in {"exchange", "交易所"}
    )
    if not actual_source:
        return None, None
    pnl_usdt: float | None = None
    for pnl_key in ("unrealized_pnl_usdt", "unRealizedProfit", "unrealizedProfit", "pnl_usdt"):
        raw_pnl = position.get(pnl_key)
        if raw_pnl is None or raw_pnl in {"", "--"}:
            continue
        try:
            pnl_usdt = float(raw_pnl)
            break
        except Exception:
            continue
    if pnl_usdt is None:
        return None, None
    notional = position_notional(position)
    margin = float(position.get("position_initial_margin_usdt") or 0.0)
    denominator = margin if margin > 0 else notional
    pnl_pct = (pnl_usdt / denominator * 100.0) if denominator > 0 else None
    return pnl_usdt, pnl_pct


def format_exchange_actual_pnl_text(position: dict[str, Any], pnl_error_note: str = "") -> str:
    pnl_usdt, pnl_pct = position_exchange_actual_pnl(position)
    if pnl_usdt is None:
        text = "未获取到交易所实际盈亏"
        if pnl_error_note:
            text += f"（{pnl_error_note[:120]}）"
        return text
    if pnl_pct is None:
        return "盈亏比缺失"
    try:
        return f"{int(float(pnl_pct))}%"
    except Exception:
        return "盈亏比缺失"


def format_report_v6_position_line(
    symbol: str,
    positions: list[dict[str, Any]],
    *,
    pnl_error_note: str = "",
) -> str:
    normalized = normalize_symbol(symbol)
    for position in positions:
        if str(position.get("symbol") or "").upper() != normalized:
            continue
        side = infer_position_side(position)
        side_text = "🔴空" if side == "short" else "🟢多"
        entry_price = position_entry_price(position)
        mark_price = position_mark_price(position)
        stop_loss = position_stop_loss(position)
        horizon = str(position.get("horizon") or "--")
        entry_tf = str(position.get("entry_timeframe") or "--")
        pnl_text = format_exchange_actual_pnl_text(position, pnl_error_note)
        return (
            f"📌 V6持仓={side_text}｜开仓={format_price(entry_price)}｜标记={format_price(mark_price)}｜"
            f"止损={format_price(stop_loss)}｜周期={horizon}/{entry_tf}｜盈亏={pnl_text}"
        )
    if pnl_error_note:
        return f"📌 V6持仓=未确认（{pnl_error_note[:120]}）"
    return "📌 V6持仓=无"


def format_report_daily_checkpoint_line(symbol: str) -> str:
    rows = [row for row in fetch_command_klines(symbol, "1d", 60) if float(row.get("close") or 0.0) > 0]
    if len(rows) < 20:
        return "📅 日线方向=数据不足"
    closes = [float(row["close"]) for row in rows]
    api_symbol = str(rows[-1].get("binance_symbol") or normalize_symbol(symbol))
    price = fetch_mark_price(api_symbol, force_refresh=True) or closes[-1]
    ma7 = sma_from_tail(closes, 7)
    ma20 = sma_from_tail(closes, 20)
    if ma7 <= 0 or ma20 <= 0:
        return "📅 日线方向=数据不足"
    price_vs_ma7 = pct_distance(price, ma7)
    price_vs_ma20 = pct_distance(price, ma20)
    if ma20 > ma7 and price < ma7:
        direction = "空头"
        state = "MA20在MA7上方，未站回MA7"
    elif ma20 > ma7 and price >= ma7:
        direction = "空头反弹"
        state = "MA20仍在MA7上方，正在测试MA7/MA20"
    elif price >= ma7 and price >= ma20 and ma7 >= ma20:
        direction = "多头右侧"
        state = "价格在MA7/MA20上方"
    elif price < ma7 and price < ma20:
        direction = "空头右侧"
        state = "价格跌回MA7/MA20下方"
    else:
        direction = "震荡"
        state = "MA7/MA20附近拉锯"
    return (
        f"📅 日线方向={direction}｜价格={format_price(price)}｜MA7={format_price(ma7)} "
        f"距MA7={format_pct_signed(price_vs_ma7)}｜MA20={format_price(ma20)} "
        f"距MA20={format_pct_signed(price_vs_ma20)}｜{state}｜K线=实时刷新"
    )


def find_shared_ma_level(signal: dict[str, Any], name: str) -> float:
    levels = v6_signal_levels(signal)
    shared = levels.get("shared_ma_levels")
    if not isinstance(shared, list):
        return 0.0
    target = str(name or "").lower()
    for item in shared:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        if str(item[0]).lower() != target:
            continue
        try:
            return float(item[1])
        except Exception:
            return 0.0
    return 0.0


def format_report_short_4h_line(symbol: str) -> str:
    rows = [row for row in fetch_command_klines(symbol, "4h", 80) if float(row.get("close") or 0.0) > 0]
    if len(rows) < 60:
        return "🕓 短周期4H=数据不足"
    closes = [float(row["close"]) for row in rows]
    api_symbol = str(rows[-1].get("binance_symbol") or normalize_symbol(symbol))
    price = fetch_mark_price(api_symbol, force_refresh=True) or closes[-1]
    ma20 = sma_from_tail(closes, 20)
    ma20_prev = sma_from_tail(closes, 20, offset=1)
    ma60 = sma_from_tail(closes, 60)
    if ma20 <= 0:
        return "🕓 短周期4H=数据不足"
    slope = pct_distance(ma20, ma20_prev) if ma20_prev > 0 else 0.0
    if price >= ma20 and slope > 0:
        direction = "多头右侧"
    elif price < ma20 and slope < 0:
        direction = "空头右侧"
    elif price >= ma20:
        direction = "等待多头右侧"
    else:
        direction = "等待空头右侧"
    parts = [
        f"🕓 短周期4H={direction}",
        f"价格={format_price(price)}",
        f"MA20={format_price(ma20)}",
        f"距MA20={format_pct_signed(pct_distance(price, ma20))}",
        f"斜率={format_pct_signed(slope)}",
    ]
    if ma60 > 0:
        parts.append(f"MA60={format_price(ma60)}")
        parts.append(f"距MA60={format_pct_signed(pct_distance(price, ma60))}")
    parts.append("K线=实时刷新")
    return "｜".join(parts)


def format_report_15m_fallback_line(symbol: str) -> str:
    signal = find_v6_signal(symbol)
    if not signal:
        return "⏱️ 15M方向=K线暂不可用"
    return f"⏱️ 15M方向={v6_signal_15m_direction(signal)}｜K线暂不可用"


def parse_float(value: Any) -> float | None:
    try:
        return float(str(value).strip().replace("%", "").replace(",", ""))
    except Exception:
        return None


def format_signed_pct_value(value: Any, digits: int = 2) -> str:
    numeric = parse_float(value)
    if numeric is None:
        return "--"
    return f"{numeric:+.{digits}f}%"


def format_funding_pct(value: Any) -> str:
    numeric = parse_float(value)
    if numeric is None:
        return "--"
    return f"{numeric * 100:+.4f}%"


def compact_usd(value: Any) -> str:
    numeric = parse_float(value)
    if numeric is None:
        return "--"
    sign = "-" if numeric < 0 else ""
    amount = abs(numeric)
    if amount >= 1_000_000_000:
        return f"{sign}{amount / 1_000_000_000:.2f}B"
    if amount >= 1_000_000:
        return f"{sign}{amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        return f"{sign}{amount / 1_000:.1f}K"
    return f"{numeric:.0f}"


def market_flow_label(score: float, threshold: float = 0.75) -> str:
    if score >= threshold:
        return "偏多"
    if score <= -threshold:
        return "偏空"
    return "混合"


def market_flow_score_from_label(label: str) -> float:
    text = str(label or "").lower()
    if text in {"偏多", "多头", "long", "long_bias", "bullish"}:
        return 1.0
    if text in {"偏空", "空头", "short", "short_bias", "bearish"}:
        return -1.0
    return 0.0


def binance_public_json(path: str, params: dict[str, Any], timeout: float = 8.0) -> Any:
    query = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
    url = f"https://fapi.binance.com{path}"
    if query:
        url = f"{url}?{query}"
    assert_binance_public_not_banned(f"tg:{path}")
    throttle_binance_public_request()
    request = urllib.request.Request(url, headers={"User-Agent": "SomnusTelegramMonitor/1.0"})
    try:
        with open_url(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        note_binance_public_response(exc.code, body, f"tg:{path}")
        raise


def build_binance_market_flow(symbol: str) -> tuple[float | None, str]:
    normalized = normalize_symbol(symbol)
    try:
        ticker = binance_public_json("/fapi/v1/ticker/24hr", {"symbol": normalized})
        premium = binance_public_json("/fapi/v1/premiumIndex", {"symbol": normalized})
        oi_hist = binance_public_json(
            "/futures/data/openInterestHist",
            {"symbol": normalized, "period": "15m", "limit": 4},
        )
        top_position = binance_public_json(
            "/futures/data/topLongShortPositionRatio",
            {"symbol": normalized, "period": "15m", "limit": 1},
        )
        taker = binance_public_json(
            "/futures/data/takerlongshortRatio",
            {"symbol": normalized, "period": "15m", "limit": 1},
        )
    except Exception:
        return None, "Binance=暂无"

    price_24h = parse_float(ticker.get("priceChangePercent"))
    funding = parse_float(premium.get("lastFundingRate"))
    oi_change_pct: float | None = None
    if isinstance(oi_hist, list) and len(oi_hist) >= 2:
        first = parse_float((oi_hist[0] or {}).get("sumOpenInterestValue"))
        last = parse_float((oi_hist[-1] or {}).get("sumOpenInterestValue"))
        if first and last is not None and first > 0:
            oi_change_pct = (last - first) / first * 100.0
    top_long = None
    if isinstance(top_position, list) and top_position:
        top_long = parse_float((top_position[-1] or {}).get("longAccount"))
    taker_ratio = None
    if isinstance(taker, list) and taker:
        taker_ratio = parse_float((taker[-1] or {}).get("buySellRatio"))

    score = 0.0
    if oi_change_pct is not None and price_24h is not None and oi_change_pct >= 0.15:
        if price_24h > 0.1:
            score += 1.0
        elif price_24h < -0.1:
            score -= 1.0
    if top_long is not None:
        if top_long >= 0.54:
            score += 1.0
        elif top_long <= 0.46:
            score -= 1.0
    if taker_ratio is not None:
        if taker_ratio >= 1.05:
            score += 1.0
        elif taker_ratio <= 0.95:
            score -= 1.0
    if funding is not None:
        if funding >= 0.00005:
            score += 0.5
        elif funding <= -0.00005:
            score -= 0.5

    label = market_flow_label(score)
    text = (
        f"Binance：{label}｜24H={format_signed_pct_value(price_24h)}｜"
        f"OI15M={format_signed_pct_value(oi_change_pct)}｜"
        f"大户多={number((top_long or 0) * 100, 1) if top_long is not None else '--'}%｜"
        f"主动买卖={number(taker_ratio, 2)}｜资金费={format_funding_pct(funding)}"
    )
    return score, text


def parse_bridge_notes(notes: Any) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in str(notes or "").split("|"):
        item = part.strip()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def find_bridge_signal(bridge_name: str, symbol: str) -> dict[str, Any] | None:
    path = ROOT / "data/trading_system/skill_bridges" / f"{bridge_name}.json"
    payload = load_json(path)
    normalized = normalize_symbol(symbol)
    for item in payload.get("signals") or []:
        if normalize_symbol(str(item.get("symbol") or "")) == normalized:
            return item
    return None


def build_okx_market_flow(symbol: str) -> tuple[float | None, str]:
    item = find_bridge_signal("okx-dex-signal", symbol) or find_bridge_signal("okx-dex-market", symbol)
    if not item:
        return None, "OKX=暂无"
    notes = parse_bridge_notes(item.get("notes"))
    direction = str(item.get("direction") or notes.get("direction") or "").lower()
    score = market_flow_score_from_label(direction)
    label = market_flow_label(score) if score else "混合"
    text = (
        f"OKX：{label}｜24H={format_signed_pct_value(notes.get('24h'))}｜"
        f"15M={format_signed_pct_value(notes.get('15m8'))}｜"
        f"OI={compact_usd(notes.get('oiUsd'))}｜资金费={format_funding_pct(notes.get('funding'))}"
    )
    return score, text


def build_hyperdash_market_flow(symbol: str) -> tuple[float | None, str]:
    item = find_bridge_signal("hyperdash-hype-whale-distill", symbol)
    if not item:
        return None, "Hyper=暂无"
    metadata = item.get("metadata") or {}
    direction = str(item.get("direction") or "").lower()
    score = market_flow_score_from_label(direction)
    net = parse_float(metadata.get("net_usdt"))
    gross = parse_float(metadata.get("gross_usdt"))
    win = parse_float(metadata.get("weighted_win_rate"))
    drawdown = parse_float(metadata.get("max_drawdown_pct"))
    if net is not None and gross and abs(net) / gross >= 0.25:
        score = 1.0 if net > 0 else -1.0
    label = market_flow_label(score) if score else "混合"
    text = (
        f"Hyper：{label}｜净={compact_usd(net)}｜总={compact_usd(gross)}｜"
        f"胜={number((win or 0) * 100, 1) if win is not None else '--'}%｜"
        f"DD={number(drawdown, 1)}%"
    )
    return score, text


def build_market_flow_report_line(symbol: str) -> str:
    symbol = normalize_symbol(symbol)
    cached = MARKET_FLOW_REPORT_CACHE.get(symbol)
    if cached and time.time() - cached[0] < MARKET_FLOW_REPORT_CACHE_TTL_SECONDS:
        return cached[1]
    sources = [
        build_binance_market_flow(symbol),
        build_okx_market_flow(symbol),
        build_hyperdash_market_flow(symbol),
    ]
    votes = [score for score, _text in sources if score is not None]
    if not votes:
        overall = "暂无"
    else:
        overall = market_flow_label(sum(votes))
    lines = [f"💰 市场资金={overall}"]
    lines.extend(f"  • {text}" for _score, text in sources)
    output = "\n".join(lines)
    MARKET_FLOW_REPORT_CACHE[symbol] = (time.time(), output)
    return output


def smart_signal_is_fresh(value: Any) -> bool:
    if not value:
        return False
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        return False
    return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() <= max(60.0, SMART_SIGNAL_MAX_AGE_SECONDS)


def find_smart_signal(symbol: str) -> dict[str, Any] | None:
    normalized = normalize_symbol(symbol)
    payload = load_json(SMART_SIGNAL_FILE)
    generated_at = payload.get("generated_at")
    for item in payload.get("signals") or []:
        if not isinstance(item, dict):
            continue
        if normalize_symbol(str(item.get("symbol") or "")) != normalized:
            continue
        updated_at = item.get("updated_at") or generated_at
        if updated_at and not smart_signal_is_fresh(updated_at):
            continue
        return item
    return None


def smart_signal_ratio(numerator: Any, denominator: Any) -> float:
    left = parse_float(numerator) or 0.0
    right = parse_float(denominator) or 0.0
    if right <= 0:
        return 999.0 if left > 0 else 0.0
    return left / right


def smart_signal_auxiliary_state(item: dict[str, Any], snapshot_15m: dict[str, Any] | None = None) -> tuple[str, str]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    direction = str(item.get("direction") or "").lower()
    long_score = parse_float(item.get("long_score") or metadata.get("long_notional")) or 0.0
    short_score = parse_float(item.get("short_score") or metadata.get("short_notional")) or 0.0
    short_ratio = parse_float(metadata.get("short_to_long_notional"))
    if short_ratio is None:
        short_ratio = smart_signal_ratio(short_score, long_score)
    long_ratio = parse_float(metadata.get("long_to_short_notional"))
    if long_ratio is None:
        long_ratio = smart_signal_ratio(long_score, short_score)
    short_crowded = bool(metadata.get("short_crowded")) or short_ratio >= 3.0
    long_crowded = bool(metadata.get("long_crowded")) or long_ratio >= 3.0
    ma20 = snapshot_15m.get("ma20") if isinstance(snapshot_15m, dict) and isinstance(snapshot_15m.get("ma20"), dict) else {}
    ma20_distance = parse_float(ma20.get("distance_pct"))
    ma20_hint = ""
    if ma20_distance is not None:
        if ma20_distance >= 0:
            ma20_hint = "，15M在MA20上方"
        else:
            ma20_hint = "，15M在MA20下方"
    if short_crowded:
        status = "空单拥挤"
        if ma20_distance is not None and ma20_distance >= 0:
            detail = "逼空风险已激活：空单拥挤且价格站回15M MA20"
        else:
            detail = "逼空观察：空单拥挤，若站回15M MA20需防止继续拉升"
    elif long_crowded:
        status = "多单拥挤"
        if ma20_distance is not None and ma20_distance <= 0:
            detail = "踩踏风险已激活：多单拥挤且价格跌破15M MA20"
        else:
            detail = "踩踏观察：多单拥挤，若跌破15M MA20需防止回撤"
    elif direction == "long" and long_score > short_score:
        status = "偏多"
        detail = "聪明钱净买入偏多"
    elif direction == "short" and short_score > long_score:
        status = "偏空"
        detail = "聪明钱净卖出偏空"
    else:
        status = "混合"
        detail = "多空资金分歧"
    return status, f"{detail}{ma20_hint}"


def build_smart_signal_report_line(symbol: str, snapshot_15m: dict[str, Any] | None = None) -> str:
    symbol = normalize_symbol(symbol)
    cache_key = f"{symbol}:{round(time.time() // max(1.0, SMART_SIGNAL_REPORT_CACHE_TTL_SECONDS))}"
    cached = SMART_SIGNAL_REPORT_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < SMART_SIGNAL_REPORT_CACHE_TTL_SECONDS:
        return cached[1]
    item = find_smart_signal(symbol)
    if not item:
        output = "🧠 SmartSignal=暂无"
        SMART_SIGNAL_REPORT_CACHE[cache_key] = (time.time(), output)
        return output
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    status, detail = smart_signal_auxiliary_state(item, snapshot_15m)
    long_score = parse_float(item.get("long_score") or metadata.get("long_notional")) or 0.0
    short_score = parse_float(item.get("short_score") or metadata.get("short_notional")) or 0.0
    short_ratio = parse_float(metadata.get("short_to_long_notional"))
    long_ratio = parse_float(metadata.get("long_to_short_notional"))
    ratio_text = ""
    if short_ratio is not None and short_ratio >= 1.2:
        ratio_text = f"｜空/多={number(short_ratio, 2)}"
    elif long_ratio is not None and long_ratio >= 1.2:
        ratio_text = f"｜多/空={number(long_ratio, 2)}"
    output = (
        f"🧠 SmartSignal={status}｜多={compact_usd(long_score)}｜空={compact_usd(short_score)}"
        f"{ratio_text}｜辅助={detail}"
    )
    SMART_SIGNAL_REPORT_CACHE[cache_key] = (time.time(), output)
    return output

def compact_interval_plan(interval: str, bias: str, ma20_state: str, ma20: dict[str, Any]) -> tuple[str, str]:
    distance = float(ma20.get("distance_pct") or 0.0)
    slope = float(ma20.get("slope_pct") or 0.0)
    ma_label = f"{interval.upper()} MA20"
    if interval == "15m":
        if "2根不破" in ma20_state:
            return f"回踩不破{ma_label}可多", f"跌破{ma_label}再看空"
        if "2根不过" in ma20_state:
            return f"先等站回{ma_label}", f"反抽不过{ma_label}可空"
        if bias == "偏多":
            return f"回踩不破{ma_label}可多", f"冲高不过{ma_label}再看空"
        if bias == "偏空":
            return f"止跌站回{ma_label}再看", f"反抽不过{ma_label}可空"
        return f"等2根不破{ma_label}", f"等2根不过{ma_label}"
    if bias == "偏多" or (distance > 0 and slope >= 0):
        return f"回踩不破{ma_label}可多", f"跌回{ma_label}下方再看空"
    if bias == "偏空" or (distance < 0 and slope <= 0):
        return f"站回{ma_label}再看多", f"反抽不过{ma_label}可空"
    return f"等右侧站稳{ma_label}", f"等右侧跌破{ma_label}"


def compact_interval_snapshot(symbol: str, interval: str) -> dict[str, Any]:
    rows, context = build_bb_rows(symbol, interval)
    price = float(context["price"])
    closes = list(context.get("closes") or [])
    klines = list(context.get("klines") or [])
    ma20 = ma20_snapshot(closes, price)
    ma20_state = fifteen_m_ma20_hold_state(klines) if interval == "15m" else (
        "站上MA20" if ma20.get("distance_pct", 0.0) > 0 else "低于MA20" if ma20.get("ma20", 0.0) > 0 else "MA20缺失"
    )
    bias, _detail = classify_report_bias(interval, rows, price, ma20, ma20_state)
    long_plan, short_plan = compact_interval_plan(interval, bias, ma20_state, ma20)
    return {
        "symbol": symbol,
        "interval": interval,
        "bias": bias,
        "price": price,
        "rows": rows,
        "context": context,
        "ma20_state": ma20_state,
        "ma20": ma20,
        "long_plan": long_plan,
        "short_plan": short_plan,
    }


def format_compact_interval_snapshot(snapshot: dict[str, Any]) -> str:
    interval = str(snapshot.get("interval") or "").upper()
    ma20 = snapshot.get("ma20") if isinstance(snapshot.get("ma20"), dict) else {}
    ma20_parts = []
    if ma20.get("ma20"):
        ma20_parts.append(f"MA20={format_price(ma20.get('ma20'))}")
        ma20_parts.append(f"距MA20={format_pct_signed(ma20.get('distance_pct'))}")
        ma20_parts.append(f"斜率={format_pct_signed(ma20.get('slope_pct'))}")
    ma20_text = "｜".join(ma20_parts) if ma20_parts else "MA20=--"
    source = ""
    raw_context = snapshot.get("context")
    context = raw_context if isinstance(raw_context, dict) else {}
    if context.get("kline_source"):
        source = f"｜K线={context.get('kline_source')}"
    return (
        f"⏱️ {interval}方向={snapshot.get('bias') or '--'}｜{ma20_text}{source}"
    )


def build_compact_interval_report(symbol: str, interval: str) -> str:
    return format_compact_interval_snapshot(compact_interval_snapshot(symbol, interval))


def format_4h_zone_15m_selector_line(snapshot_15m: dict[str, Any]) -> str:
    bias = str(snapshot_15m.get("bias") or "")
    if bias == "偏多":
        choice = "15M偏多"
    elif bias == "偏空":
        choice = "15M偏空"
    elif "震荡" in bias:
        choice = "15M震荡"
    else:
        choice = f"15M={bias or '--'}"
    return f"15M选择={choice}"


def format_report_v6_context(symbol: str) -> str:
    signal = find_v6_signal(symbol)
    if not signal:
        return "V6=暂无最新分析"
    return (
        f"V6={zh_value(signal.get('side_bias'))} "
        f"多={number(float(signal.get('long_probability') or 0) * 100, 2)}% "
        f"空={number(float(signal.get('short_probability') or 0) * 100, 2)}% "
        f"等待={number(float(signal.get('wait_probability') or 0) * 100, 2)}%"
    )


def ma_value_at(closes: list[float], period: int, end_index: int | None = None) -> float:
    if end_index is None:
        end_index = len(closes) - 1
    start = end_index - period + 1
    if period <= 0 or start < 0 or end_index >= len(closes):
        return 0.0
    return sum(closes[start : end_index + 1]) / period


def row_by_period(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            output[int(row.get("period") or 0)] = row
        except Exception:
            continue
    return output


def build_btc_15m_conflict_confirmation(snapshot_15m: dict[str, Any]) -> str:
    price = float(snapshot_15m.get("price") or 0.0)
    rows = list(snapshot_15m.get("rows") or [])
    context = snapshot_15m.get("context") or {}
    klines = [row for row in context.get("klines") or [] if float(row.get("close") or 0) > 0]
    if price <= 0 or not rows or not klines:
        return "15M分歧确认=BTC：数据不足"
    last = klines[-1]
    high = float(last.get("high") or last.get("close") or 0.0)
    close = float(last.get("close") or 0.0)
    periods = row_by_period(rows)
    triggered: list[str] = []
    waiting: list[str] = []
    for period in (96, 105, 120):
        row = periods.get(period) or {}
        upper = float(row.get("upper") or 0.0)
        if upper <= 0:
            continue
        name = f"BB{period}上轨"
        if high >= upper * (1.0 - TOLERANCE_15M) and close <= upper:
            triggered.append(f"{name}压制")
        else:
            waiting.append(f"{name} {format_price(upper)}（{format_gap_from_price(price, upper)}）")
    if triggered:
        current_lines = ["当前=空确认：" + "、".join(triggered)]
    elif waiting:
        current_lines = ["当前=等待空确认", *[f"- {item}" for item in waiting[:3]]]
    else:
        current_lines = ["当前=等待空确认"]
    return "\n".join(
        [
            "🎯 15M压力数据：BB96/105/120上轨",
            *current_lines,
        ]
    )


def build_eth_15m_conflict_confirmation(snapshot_15m: dict[str, Any]) -> str:
    price = float(snapshot_15m.get("price") or 0.0)
    rows = list(snapshot_15m.get("rows") or [])
    context = snapshot_15m.get("context") or {}
    closes = [float(item) for item in context.get("closes") or [] if float(item or 0) > 0]
    klines = [row for row in context.get("klines") or [] if float(row.get("close") or 0) > 0]
    if price <= 0 or len(closes) < 121 or not rows or not klines:
        return "15M分歧确认=ETH：数据不足"
    last = klines[-1]
    prev_close = closes[-2] if len(closes) >= 2 else closes[-1]
    low = float(last.get("low") or last.get("close") or 0.0)
    close = float(last.get("close") or 0.0)
    periods = row_by_period(rows)
    triggered: list[str] = []
    waiting: list[str] = []
    for period in (96, 105, 120):
        row = periods.get(period) or {}
        lower = float(row.get("lower") or 0.0)
        if lower > 0:
            name = f"BB{period}下轨"
            if low <= lower * (1.0 + TOLERANCE_15M) and close >= lower:
                triggered.append(f"{name}反弹")
            else:
                waiting.append(f"{name} {format_price(lower)}（{format_gap_from_price(price, lower)}）")
        ma_now = ma_value_at(closes, period)
        ma_prev = ma_value_at(closes, period, len(closes) - 2)
        if ma_now <= 0:
            continue
        if (prev_close <= ma_prev and close > ma_now) or (low <= ma_now * (1.0 + TOLERANCE_15M) and close >= ma_now):
            triggered.append(f"MA{period}上穿/支撑")
    if triggered:
        current_lines = ["当前=多确认：" + "、".join(triggered[:6])]
    elif waiting:
        current_lines = ["当前=等待多确认", *[f"- {item}" for item in waiting[:3]]]
    else:
        current_lines = ["当前=等待多确认"]
    return "\n".join(
        [
            "🎯 15M支撑数据：BB96/105/120下轨 + MA96/105/120",
            *current_lines,
        ]
    )


def build_15m_conflict_confirmation(symbol: str, snapshot_15m: dict[str, Any]) -> str:
    normalized = normalize_symbol(symbol)
    if normalized == "BTCUSDT":
        return build_btc_15m_conflict_confirmation(snapshot_15m)
    if normalized == "ETHUSDT":
        return build_eth_15m_conflict_confirmation(snapshot_15m)
    return ""


def build_report_response(symbols: list[str] | None = None) -> str:
    targets = symbols or ["BTCUSDT", "ETHUSDT"]
    for symbol in targets:
        V6_DIRECTION_CACHE.pop(normalize_symbol(symbol), None)
    refresh_report_signals(targets)
    report_positions, _report_position_source, report_pnl_error_note = current_report_v6_positions()
    lines = ["📊 [Somnus宏观]"]
    for symbol in targets:
        signal = find_v6_signal(symbol)
        label = report_symbol_label(symbol)
        lines.extend(
            [
                "",
                f"🪙【{label}】",
                format_report_model_line(symbol),
                format_report_v6_position_line(symbol, report_positions, pnl_error_note=report_pnl_error_note),
            ]
        )
        try:
            lines.append(format_report_daily_checkpoint_line(symbol))
        except Exception:
            lines.append("📅 日线方向=数据不足")
        try:
            lines.append(format_report_short_4h_line(symbol))
        except Exception:
            lines.append("🕓 短周期4H=数据不足")
        snapshot_15m: dict[str, Any] | None = None
        try:
            snapshot_15m = compact_interval_snapshot(symbol, "15m")
        except Exception:
            snapshot_15m = None
        try:
            lines.append(build_market_flow_report_line(symbol))
        except Exception:
            lines.append("💰 市场资金=暂无")
        try:
            lines.append(build_smart_signal_report_line(symbol, snapshot_15m))
        except Exception:
            lines.append("🧠 SmartSignal=暂无")
        try:
            if snapshot_15m is None:
                snapshot_15m = compact_interval_snapshot(symbol, "15m")
            lines.append(format_compact_interval_snapshot(snapshot_15m))
            confirmation = build_15m_conflict_confirmation(symbol, snapshot_15m)
            if confirmation:
                lines.append(confirmation)
        except Exception:
            lines.append(format_report_15m_fallback_line(symbol))
        try:
            if symbol in V6_MACRO_4H_RULES:
                lines.append(signal_line(symbol, signal or {}))
            else:
                lines.append(build_compact_interval_report(symbol, "4h"))
        except Exception:
            lines.append("📈 4H/长周期信号=数据不足")
    return "\n".join(lines)


def parse_daily_schedule(raw: Any) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []
    for item in str(raw or "").replace("；", ",").replace("，", ",").split(","):
        text = item.strip()
        if not text:
            continue
        text = text.replace(".", ":")
        if ":" in text:
            hour_text, minute_text = text.split(":", 1)
        else:
            hour_text, minute_text = text, "00"
        try:
            hour = int(hour_text)
            minute = int(minute_text)
        except Exception:
            continue
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            item_tuple = (hour, minute)
            if item_tuple not in values:
                values.append(item_tuple)
    return sorted(values)


def scheduled_macro_report_due(
    state: dict[str, Any],
    schedule_times: str,
    *,
    now: datetime | None = None,
    window_minutes: int = 10,
) -> str:
    schedules = parse_daily_schedule(schedule_times)
    if not schedules:
        return ""
    local_now = now or datetime.now().astimezone()
    minute_of_day = local_now.hour * 60 + local_now.minute
    sent = state.setdefault("scheduled_macro_reports", {})
    if not isinstance(sent, dict):
        sent = {}
        state["scheduled_macro_reports"] = sent
    today = local_now.date().isoformat()
    for hour, minute in schedules:
        slot_minute = hour * 60 + minute
        if 0 <= minute_of_day - slot_minute < max(1, int(window_minutes)):
            key = f"{today} {hour:02d}:{minute:02d}"
            if sent.get(key):
                return ""
            sent[key] = time.time()
            for old_key in list(sent):
                if not str(old_key).startswith(today):
                    sent.pop(old_key, None)
            return key
    return ""


def scheduled_daily_news_due(
    state: dict[str, Any],
    schedule_times: str,
    *,
    now: datetime | None = None,
    window_minutes: int = 10,
) -> str:
    schedules = parse_daily_schedule(schedule_times)
    if not schedules:
        return ""
    local_now = now.astimezone(BEIJING_TZ) if now else datetime.now(BEIJING_TZ)
    minute_of_day = local_now.hour * 60 + local_now.minute
    sent = state.setdefault("scheduled_daily_news_reports", {})
    if not isinstance(sent, dict):
        sent = {}
        state["scheduled_daily_news_reports"] = sent
    today = local_now.date().isoformat()
    for hour, minute in schedules:
        slot_minute = hour * 60 + minute
        if 0 <= minute_of_day - slot_minute < max(1, int(window_minutes)):
            key = f"{today} {hour:02d}:{minute:02d}+08:00"
            if sent.get(key):
                return ""
            sent[key] = time.time()
            for old_key in list(sent):
                if not str(old_key).startswith(today):
                    sent.pop(old_key, None)
            return key
    return ""


DAILY_NEWS_TOPICS = [
    {
        "name": "AI新闻",
        "query": "OpenAI OR Anthropic OR Claude OR ChatGPT OR GPT OR DeepSeek OR Tencent Hunyuan OR Alibaba Qwen OR ByteDance Doubao OR large language model OR LLM",
        "twitter_query": "OpenAI OR Anthropic OR Claude OR ChatGPT OR DeepSeek OR Qwen OR Hunyuan OR Doubao OR LLM",
        "keywords": [
            "openai",
            "anthropic",
            "claude",
            "chatgpt",
            "gpt",
            "deepseek",
            "tencent",
            "hunyuan",
            "alibaba",
            "qwen",
            "byteDance",
            "doubao",
            "large language model",
            "llm",
            "model",
        ],
        "exclude_keywords": [],
        "impact": "重点看大模型能力迭代、推理成本、生态扩张，以及国内外模型竞争对科技股/风险偏好的影响。",
    },
    {
        "name": "科技新闻",
        "query": "Elon Musk OR Tesla OR SpaceX OR xAI OR Grok OR Neuralink OR Starlink OR biotech OR biotechnology OR gene therapy OR CRISPR OR FDA",
        "twitter_query": "Elon Musk OR Tesla OR SpaceX OR xAI OR Grok OR Neuralink OR biotech OR CRISPR",
        "keywords": [
            "elon",
            "musk",
            "tesla",
            "spacex",
            "starship",
            "starlink",
            "xai",
            "grok",
            "neuralink",
            "biotech",
            "biotechnology",
            "gene therapy",
            "crispr",
            "fda",
            "clinical trial",
        ],
        "exclude_keywords": [],
        "impact": "重点看马斯克相关公司进展、生物科技突破/监管节点，以及对科技风险偏好的边际影响。",
    },
    {
        "name": "地缘局势",
        "query": "geopolitical OR war OR sanctions OR tariff OR conflict OR Middle East OR Russia OR Ukraine OR G7",
        "twitter_query": "geopolitical crypto OR sanctions BTC OR war BTC OR tariff markets",
        "keywords": [
            "geopolitical",
            "war",
            "sanction",
            "tariff",
            "conflict",
            "middle east",
            "russia",
            "ukraine",
            "g7",
            "trade",
            "iran",
            "israel",
        ],
        "exclude_keywords": [
            "sports",
            "games",
            "tennis",
            "esports",
            "counter-strike",
            "counter strike",
            "league of legends",
            "libema open",
            "lol:",
            "bo3",
            "bo5",
        ],
        "impact": "重点看避险情绪、美元/能源/通胀预期，以及BTC/ETH风险资产压力。",
    },
    {
        "name": "BTC机构左侧买盘",
        "query": "IBIT OR Coinbase premium OR Coinbase Premium Index OR exchange USDC OR USDC supply OR stablecoin inflow OR Mubadala OR sovereign wealth fund OR family office OR MicroStrategy OR MSTR OR Michael Saylor OR Metaplanet OR MARA OR Semler Scientific OR Strive OR GameStop OR Bitcoin treasury OR BTC treasury",
        "twitter_query": "Coinbase premium OR USDC supply OR Mubadala IBIT OR MicroStrategy BTC OR MSTR bitcoin OR Metaplanet BTC OR MARA HODL BTC OR Semler BTC OR GameStop BTC",
        "keywords": [
            "ibit",
            "coinbase premium",
            "coinbase premium index",
            "usdc supply",
            "exchange usdc",
            "stablecoin inflow",
            "stablecoin supply",
            "mubadala",
            "sovereign wealth fund",
            "family office",
            "middle east",
            "abu dhabi",
            "microstrategy",
            "mstr",
            "michael saylor",
            "metaplanet",
            "mara",
            "marathon digital",
            "strive",
            "semler",
            "gamestop",
            "covered call",
            "bitcoin treasury",
            "btc treasury",
        ],
        "exclude_keywords": [],
        "quality_gate": "btc_institutional_left_buy",
        "impact": "只播确定性 BTC 机构买盘/资金流信号：Coinbase 溢价状态、交易所 USDC 流入流出、企业/主权/矿企明确买入或锁仓；单纯关键词命中不播。",
    },
]

NEWS_ZH_PHRASES = [
    ("Senior U.S. Administration Official", "美国高级政府官员"),
    ("U.S. Administration Official", "美国政府官员"),
    ("United States", "美国"),
    ("U.S.", "美国"),
    ("US ", "美国"),
    ("U.S. Trade Representative", "美国贸易代表"),
    ("Trade Representative", "贸易代表"),
    ("India", "印度"),
    ("China", "中国"),
    ("Japan", "日本"),
    ("Korea", "韩国"),
    ("Russia", "俄罗斯"),
    ("Ukraine", "乌克兰"),
    ("Iran", "伊朗"),
    ("Israel", "以色列"),
    ("UAE", "阿联酋"),
    ("Middle East", "中东"),
    ("Strait of Hormuz", "霍尔木兹海峡"),
    ("G7 Summit", "G7峰会"),
    ("G7", "G7"),
    ("Trump", "特朗普"),
    ("Netanyahu", "内塔尼亚胡"),
    ("bilateral meetings", "双边会谈"),
    ("Bilateral Meetings", "双边会谈"),
    ("sanctions", "制裁"),
    ("tariff", "关税"),
    ("trade deal", "贸易协议"),
    ("peace deal", "和平协议"),
    ("ceasefire", "停火"),
    ("war deal", "战争相关协议"),
    ("permanent peace deal", "永久和平协议"),
    ("demining efforts", "排雷工作"),
    ("de-escalation", "局势降温"),
    ("allies", "盟友"),
    ("confidence grows", "信心升温"),
    ("possible", "可能达成"),
    ("don’t expect", "预计不会"),
    ("do not expect", "预计不会"),
    ("to Travel to", "将前往"),
    ("Following G7 Summit", "在G7峰会之后"),
    ("Expect Trade Deal", "预计贸易协议"),
    ("Come Up During G7", "会在G7期间被讨论"),
    ("AI agents", "AI代理"),
    ("AI agent", "AI代理"),
    ("artificial intelligence", "人工智能"),
    ("Artificial Intelligence", "人工智能"),
    ("large language model", "大语言模型"),
    ("Large Language Model", "大语言模型"),
    ("LLM", "大模型"),
    ("OpenAI", "OpenAI"),
    ("Anthropic", "Anthropic"),
    ("Claude", "Claude"),
    ("ChatGPT", "ChatGPT"),
    ("DeepSeek", "DeepSeek"),
    ("Tencent", "腾讯"),
    ("Alibaba", "阿里巴巴"),
    ("ByteDance", "字节跳动"),
    ("Qwen", "通义千问"),
    ("Hunyuan", "混元"),
    ("Doubao", "豆包"),
    ("Elon Musk", "马斯克"),
    ("Musk", "马斯克"),
    ("Tesla", "特斯拉"),
    ("SpaceX", "SpaceX"),
    ("Starship", "星舰"),
    ("Starlink", "星链"),
    ("Neuralink", "Neuralink"),
    ("biotech", "生物科技"),
    ("biotechnology", "生物科技"),
    ("gene therapy", "基因疗法"),
    ("clinical trial", "临床试验"),
    ("AI subscriptions", "AI订阅"),
    ("semiconductor", "半导体"),
    ("chip", "芯片"),
    ("compute", "算力"),
    ("model", "模型"),
    ("stablecoins", "稳定币"),
    ("stablecoin", "稳定币"),
    ("regulated financial infrastructure", "受监管金融基础设施"),
    ("institutional market", "机构市场"),
    ("brokerages", "券商"),
    ("banks", "银行"),
    ("payment institutions", "支付机构"),
    ("regulation", "监管"),
    ("Digital Asset Basic Act", "数字资产基础法案"),
    ("IPO", "IPO"),
    ("tax", "税务"),
    ("Tax Breaks", "税收减免"),
    ("weekly highlights", "本周重点"),
    ("Weekly Highlights", "本周重点"),
    ("Buyback", "回购"),
    ("All-Time Volume", "历史总成交量"),
    ("Prediction markets", "预测市场"),
    ("crypto", "加密市场"),
    ("Coinbase Premium Index", "Coinbase溢价指数"),
    ("Coinbase premium", "Coinbase溢价"),
    ("USDC supply", "USDC供应量"),
    ("stablecoin inflow", "稳定币流入"),
    ("sovereign wealth fund", "主权财富基金"),
    ("family office", "家族办公室"),
    ("Mubadala", "穆巴达拉"),
    ("MicroStrategy", "MicroStrategy/Strategy"),
    ("Marathon Digital", "MARA/Marathon Digital"),
    ("Semler Scientific", "Semler Scientific"),
    ("GameStop", "GameStop"),
]


def post_json_with_bearer(url: str, token: str, payload: dict[str, Any], timeout: float = 25.0) -> dict[str, Any]:
    if not token:
        return {"success": False, "error": "missing_token", "data": []}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with open_url(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            http_status = int(getattr(response, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {
            "success": False,
            "error": "http_error",
            "http_status": int(getattr(exc, "code", 0) or 0),
            "raw": raw[:500],
            "data": [],
        }
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed.setdefault("http_status", http_status)
        return parsed
    except Exception:
        return {"success": False, "error": "invalid_json", "http_status": http_status, "raw": raw[:500], "data": []}


def normalize_token_pool(*values: Any) -> list[str]:
    tokens: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            token = str(candidate or "").strip()
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def env_token_pool(*names: str) -> list[str]:
    return normalize_token_pool(*(os.getenv(name) for name in names))


def is_rate_limited_response(result: dict[str, Any]) -> bool:
    status = int(result.get("http_status") or result.get("status") or 0)
    if status == 429:
        return True
    text = " ".join(
        str(result.get(key) or "")
        for key in ("error", "msg", "message", "raw")
    ).lower()
    limit_markers = (
        "rate limit",
        "rate_limit",
        "too many request",
        "too many requests",
        "quota",
        "limit exceeded",
        "limited",
        "限额",
        "额度",
        "频率",
        "请求过多",
        "次数",
    )
    return any(marker in text for marker in limit_markers)


def post_json_with_bearer_pool(
    url: str,
    tokens: list[str] | tuple[str, ...] | str,
    payload: dict[str, Any],
    timeout: float = 25.0,
) -> dict[str, Any]:
    token_pool = normalize_token_pool(tokens)
    if not token_pool:
        return {"success": False, "error": "missing_token", "data": []}
    last_result: dict[str, Any] = {"success": False, "error": "missing_token", "data": []}
    for index, token in enumerate(token_pool):
        result = post_json_with_bearer(url, token, payload, timeout=timeout)
        if isinstance(result, dict):
            result["token_pool_index"] = index
            result["token_pool_size"] = len(token_pool)
        last_result = result if isinstance(result, dict) else last_result
        if is_rate_limited_response(last_result) and index + 1 < len(token_pool):
            continue
        return last_result
    return last_result


def compact_news_text(value: Any, limit: int = 180) -> str:
    text = html.unescape(str(value or ""))
    for marker in ("<br/>", "<br>", "<br />", "</p>", "</div>", "</li>"):
        text = text.replace(marker, " ")
    text = text.replace("<p>", " ").replace("<div>", " ").replace("<li>", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = " ".join(text.replace("\n", " ").replace("\n", " ").split())
    return text[: max(20, limit)].rstrip() + ("..." if len(text) > limit else "")


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def apply_news_phrase_zh(text: str) -> str:
    output = text
    for src, dst in NEWS_ZH_PHRASES:
        output = output.replace(src, dst)
    output = re.sub(r"(以色列|特朗普|美国|中国|俄罗斯|伊朗|印度|日本|韩国)[’']s", r"\1的", output)
    output = re.sub(r"\bSell\b", "卖出", output)
    output = re.sub(r"\bBuy\b", "买入", output)
    output = re.sub(r"\bYes side\b", "Yes方向", output, flags=re.I)
    return output


def enough_chinese_news_text(text: str) -> bool:
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_word_chars = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    return cjk_count >= 8 and ascii_word_chars <= max(40, int(cjk_count * 1.8))


BTC_NEWS_QUALITY_GATE_FAIL_TEXT = "未通过质量门：仅关键词命中，未确认确定性买盘/资金流信号。"
BTC_CONTEXT_TERMS = (
    "btc", "bitcoin", "比特币", "ibit", "bitcoin etf", "spot bitcoin etf", "bitcoin treasury", "btc treasury"
)
BTC_BUY_TERMS = (
    "buy", "buys", "bought", "purchase", "purchases", "purchased", "acquire", "acquires", "acquired",
    "add", "adds", "added", "increase", "increases", "increased", "accumulate", "accumulates", "accumulated",
    "加仓", "购买", "买入", "增持", "囤币"
)
BTC_CONFIRMED_BUY_TERMS = (
    "buys", "bought", "purchase", "purchases", "purchased", "acquire", "acquires", "acquired",
    "adds", "added", "increase", "increases", "increased", "accumulate", "accumulates", "accumulated",
    "加仓", "购买", "买入", "增持", "囤币"
)
BTC_INTENT_TERMS = (
    "signal", "signals", "tease", "teases", "plan", "plans", "intend", "intends", "will buy", "to buy", "may buy", "another bitcoin buy", "预告", "计划", "暗示"
)
BTC_HOLD_TERMS = (
    "hodl", "hold", "holds", "held", "retain", "retains", "retained", "not sell", "no sell", "won't sell",
    "100%", "all bitcoin", "full hodl", "全量", "锁仓", "留存", "不卖", "拒绝抛售"
)
BTC_FLOW_TERMS = (
    "inflow", "inflows", "flow into", "flows into", "net inflow", "outflow", "outflows", "drain", "drains",
    "rebound", "rebounds", "recover", "recovers", "rise", "rises", "increase", "increases", "stop losing", "stops losing",
    "stabilize", "stabilizes", "turn positive", "turns positive", "turned positive", "flip positive", "flips positive",
    "negative", "positive", "discount", "premium widens", "premium narrows", "回流", "流入", "流出", "失血", "停止失血", "转正", "负值", "正值"
)


def _contains_any(lower: str, terms: tuple[str, ...]) -> bool:
    return any(term in lower for term in terms)


def btc_institutional_quality_gate(text: str) -> dict[str, Any]:
    """Quality gate for BTC institutional buy-flow news.

    Rejects mere keyword/entity mentions. A pass requires the watched entity/indicator
    plus an explicit state/action verb that makes the signal deterministic.
    """
    raw = str(text or "").strip()
    lower = raw.lower()
    if not lower:
        return {"passed": False, "reason": "empty"}

    coinbase_premium = "coinbase premium" in lower or "premium index" in lower or "coinbase溢价" in lower
    if coinbase_premium:
        if _contains_any(lower, BTC_FLOW_TERMS):
            if any(term in lower for term in ("turns positive", "turned positive", "turn positive", "flips positive", "positive", "转正", "正值")):
                return {
                    "passed": True,
                    "kind": "coinbase_premium_positive",
                    "summary": "确定性信号：Coinbase 溢价出现转正/走强描述，美区现货买盘有回归迹象。",
                    "detail": "Coinbase 溢价指数出现明确状态变化；若由负转正并持续，才算美区机构现货买盘回归的左侧确认。",
                }
            if any(term in lower for term in ("negative", "discount", "负值", "折价")):
                return {
                    "passed": True,
                    "kind": "coinbase_premium_negative",
                    "summary": "确定性信号：Coinbase 溢价仍偏负/折价，美区机构买盘尚未确认回归。",
                    "detail": "Coinbase 溢价仍显示折价或负值，这是左侧企稳未确认；需要转正并持续才可视为机构买盘回归。",
                }
            return {
                "passed": True,
                "kind": "coinbase_premium_state",
                "summary": "确定性信号：Coinbase 溢价有明确状态变化，需跟踪是否转正并持续。",
                "detail": "Coinbase 溢价指数出现方向/状态描述，可作为美区机构买盘温度计；重点看是否由负转正。",
            }
        return {"passed": False, "reason": "coinbase_premium_without_state"}

    stablecoin_context = "usdc" in lower or "stablecoin" in lower or "稳定币" in lower
    exchange_context = "exchange" in lower or "reserve" in lower or "supply" in lower or "交易所" in lower or "供应" in lower or "余额" in lower
    if stablecoin_context:
        if exchange_context and _contains_any(lower, BTC_FLOW_TERMS):
            if any(term in lower for term in ("inflow", "inflows", "net inflow", "flow into", "rebound", "recover", "rise", "increase", "回流", "流入", "停止失血")):
                return {
                    "passed": True,
                    "kind": "usdc_exchange_inflow",
                    "summary": "确定性信号：交易所 USDC/稳定币出现回流或供应回升，多头弹药开始上膛。",
                    "detail": "交易所 USDC/稳定币余额出现明确回流/回升描述；这是多头弹药恢复的左侧确认之一。",
                }
            if any(term in lower for term in ("outflow", "outflows", "drain", "drains", "流出", "失血")):
                return {
                    "passed": True,
                    "kind": "usdc_exchange_outflow",
                    "summary": "确定性信号：交易所 USDC/稳定币仍在流出，多头弹药未确认恢复。",
                    "detail": "交易所 USDC/稳定币余额仍有流出/失血描述，说明新一季度多头弹药尚未重新上膛。",
                }
            return {
                "passed": True,
                "kind": "usdc_exchange_flow_state",
                "summary": "确定性信号：交易所 USDC/稳定币供给出现明确流向变化。",
                "detail": "交易所 USDC/稳定币供给出现方向描述，可用于判断多头弹药是否恢复。",
            }
        return {"passed": False, "reason": "stablecoin_without_exchange_flow"}

    btc_context = _contains_any(lower, BTC_CONTEXT_TERMS)
    buy_context = _contains_any(lower, BTC_BUY_TERMS)
    confirmed_buy_context = _contains_any(lower, BTC_CONFIRMED_BUY_TERMS)
    intent_context = buy_context and _contains_any(lower, BTC_INTENT_TERMS)
    hold_context = _contains_any(lower, BTC_HOLD_TERMS)

    if any(term in lower for term in ("mubadala", "sovereign wealth fund", "family office", "abu dhabi", "主权财富", "家族办公室")):
        if btc_context and (buy_context or "holding" in lower or "holdings" in lower or "stake" in lower or "disclosed" in lower or "持仓" in lower):
            return {
                "passed": True,
                "kind": "sovereign_family_office_btc_buy",
                "summary": "确定性信号：中东主权/家族办公室出现 BTC/IBIT 持仓或加仓动作。",
                "detail": "中东主权基金/家族办公室有明确 BTC/IBIT 持仓、披露或加仓描述，属于离岸长线资金承接线索。",
            }
        return {"passed": False, "reason": "sovereign_mention_without_btc_buy"}

    strategy_entity = "microstrategy" in lower or "mstr" in lower or "saylor" in lower or ("strategy" in lower and btc_context)
    if strategy_entity:
        if btc_context and confirmed_buy_context:
            return {
                "passed": True,
                "kind": "strategy_btc_confirmed_buy",
                "summary": "确定性信号：Strategy/MicroStrategy 出现已买入/加仓 BTC 的确认描述。",
                "detail": "Strategy/MicroStrategy 有明确 BTC 已买入/加仓描述，计入企业财库买盘；同时观察融资成本和 MSTR 风险溢价。",
            }
        if btc_context and intent_context:
            return {
                "passed": True,
                "kind": "strategy_btc_buy_intent",
                "summary": "确定性信号：Strategy/MicroStrategy 出现下一笔 BTC 买入预告，但不是成交确认。",
                "detail": "Strategy/MicroStrategy/Saylor 释放下一笔 BTC 买入意向；这是财库买盘预告，不等于已成交加仓，需等待正式披露购买数量和均价。",
            }
        return {"passed": False, "reason": "strategy_mention_without_btc_buy"}

    if "metaplanet" in lower:
        if btc_context and (buy_context or "holding" in lower or "holdings" in lower or "持仓" in lower):
            return {
                "passed": True,
                "kind": "metaplanet_btc_buy",
                "summary": "确定性信号：Metaplanet 出现 BTC 持仓/买入/加仓动作。",
                "detail": "Metaplanet 有明确 BTC 持仓或加仓描述，属于亚洲企业财库化买盘线索。",
            }
        return {"passed": False, "reason": "metaplanet_mention_without_btc_buy"}

    if "mara" in lower or "marathon digital" in lower:
        if btc_context and (hold_context or buy_context or "mined" in lower or "production" in lower or "treasury" in lower):
            return {
                "passed": True,
                "kind": "mara_btc_hodl",
                "summary": "确定性信号：MARA 出现 BTC 留存/HODL/供给端减抛描述。",
                "detail": "MARA 有明确 BTC 留存、HODL 或矿工减抛描述，代表供给端抛压边际下降。",
            }
        return {"passed": False, "reason": "mara_mention_without_hodl"}

    if any(term in lower for term in ("strive", "semler", "gamestop")):
        if btc_context and (buy_context or hold_context or "treasury" in lower or "covered call" in lower or "备兑" in lower):
            return {
                "passed": True,
                "kind": "cross_company_btc_treasury",
                "summary": "确定性信号：Strive/Semler/GameStop 相关 BTC 财库或结构化持币动作出现。",
                "detail": "跨界公司有明确 BTC 财库、并购取得持仓或备兑策略锁仓描述，属于结构化持币线索。",
            }
        return {"passed": False, "reason": "cross_company_mention_without_btc_action"}

    return {"passed": False, "reason": "no_deterministic_btc_institutional_signal"}


def topic_quality_gate_result(item: dict[str, Any], topic: dict[str, Any]) -> dict[str, Any]:
    cached = item.get("_quality_gate") if isinstance(item.get("_quality_gate"), dict) else None
    if cached:
        return cached
    if str(topic.get("quality_gate") or "") == "btc_institutional_left_buy":
        return btc_institutional_quality_gate(news_item_text(item, 1000))
    return {"passed": True, "reason": "no_gate"}


def localize_news_text(item: dict[str, Any], topic: dict[str, Any], limit: int = 170) -> str:
    text = news_item_text(item, 520)
    if not text:
        return "暂无可读正文。"
    lower = text.lower()
    topic_name = str(topic.get("name") or "")

    if topic_name == "AI新闻":
        if "openai" in lower or "chatgpt" in lower or re.search(r"\bgpt\b", lower):
            return "大模型进展：OpenAI/ChatGPT/GPT 相关动态出现，重点关注模型能力、产品入口、推理成本和生态扩张。"
        if "anthropic" in lower or "claude" in lower:
            return "大模型进展：Anthropic/Claude 相关动态出现，重点关注长文本、代码能力、企业客户和安全路线。"
        if "deepseek" in lower:
            return "国产大模型进展：DeepSeek 相关动态出现，重点关注低成本推理、开源生态和国内模型竞争格局。"
        if "tencent" in lower or "hunyuan" in lower or "混元" in text:
            return "国产大模型进展：腾讯/混元相关动态出现，重点关注腾讯云、微信生态和企业应用落地。"
        if "alibaba" in lower or "qwen" in lower or "通义" in text:
            return "国产大模型进展：阿里巴巴/通义千问相关动态出现，重点关注开源模型、云业务和企业级应用。"
        if "bytedance" in lower or "doubao" in lower or "字节" in text or "豆包" in text:
            return "国产大模型进展：字节跳动/豆包相关动态出现，重点关注C端入口、推理成本和内容生态。"
        if "large language model" in lower or "llm" in lower:
            return "大模型行业动态：内容涉及 LLM 能力、成本或应用落地，重点观察是否影响科技股风险偏好。"
        if "talus" in lower and "ai agents" in lower:
            return "AI基础设施：Talus 表示正在建设让 AI代理 能够协商、交易和协同执行的基础设施，属于 AI代理+链上执行叙事。"
        if "weekly highlights" in lower and "aerodrome" in lower:
            return "项目周报：Aerodrome 将接入 Arc，历史成交量达到约400B，并出现 AI代理 合作/叙事；同时披露 AERO 回购等进展。"
        if "korea" in lower and "ai agents" in lower:
            return "韩国加密市场出现机构分化：券商对加密资产和AI代理金融更积极，银行/支付机构更谨慎，后续重点看监管和稳定币牌照。"
        if "tax breaks" in lower or "ai subscriptions" in lower:
            return "税务/科技新闻：市场讨论 AI 订阅、SpaceX IPO 和税务处理，对科技股与AI叙事有边际影响。"
        if "bittensor" in lower or "$tao" in lower:
            return "AI加密叙事：Bittensor/TAO 被讨论为较强 AI crypto 叙事，关注AI板块情绪是否扩散到主流风险资产。"
        if "deai" in lower:
            return "AI加密叙事：DeAI 被讨论为加密市场的重要方向，关注资金是否继续追逐AI相关资产。"
        if "solana" in lower and "ai" in lower and "crypto" in lower:
            return "AI链上叙事：市场讨论自动化AI与 Solana 链上资产结合，关注是否带动 AI+加密短线热度。"
        if "crypto" in lower and "wellness" in lower and "ai coach" in lower:
            return "AI应用叙事：市场讨论把加密、健康和 AI 教练结合的应用形态，属于AI消费级应用探索。"

    if topic_name == "科技新闻":
        if "elon" in lower or "musk" in lower:
            return "马斯克相关：新闻涉及马斯克本人或旗下公司，重点关注对科技风险偏好、特斯拉/SpaceX/xAI 叙事的影响。"
        if "tesla" in lower:
            return "科技公司进展：特斯拉相关动态出现，重点关注交付、自动驾驶、机器人和能源业务。"
        if "spacex" in lower or "starship" in lower:
            return "航天科技：SpaceX/星舰相关动态出现，重点关注发射节奏、商业航天和市场风险偏好。"
        if "starlink" in lower:
            return "航天通信：星链相关动态出现，重点关注卫星互联网扩张和SpaceX商业化能力。"
        if "xai" in lower or "grok" in lower:
            return "马斯克AI：xAI/Grok相关动态出现，重点关注大模型竞争、X生态接入和算力投入。"
        if "neuralink" in lower:
            return "脑机接口：Neuralink相关动态出现，重点关注临床进展、监管节点和生物科技风险偏好。"
        if "biotech" in lower or "biotechnology" in lower:
            return "生物科技：行业相关动态出现，重点关注创新药、平台技术和资本市场风险偏好。"
        if "gene therapy" in lower or "crispr" in lower:
            return "生物科技：基因疗法/CRISPR相关动态出现，重点关注临床数据、监管审批和赛道热度。"
        if "fda" in lower or "clinical trial" in lower:
            return "生物科技监管/临床：FDA或临床试验相关动态出现，重点关注审批结果和医药科技情绪。"

    if topic_name == "地缘局势":
        if "trump" in lower and "middle east" in lower and "ukraine" in lower and "g7" in lower:
            return "特朗普将在 G7 期间会见中东领导人，并参加乌克兰相关会议；市场关注中东与俄乌两条风险线是否同时影响避险情绪。"
        if "hormuz" in lower and "g7" in lower:
            return "美国方面称特朗普将在下周 G7 与盟友讨论霍尔木兹海峡排雷工作，市场关注伊朗相关协议和能源通道风险。"
        if "netanyahu" in lower and "g7" in lower:
            return "美国官员称，以色列总理内塔尼亚胡不会参加特朗普在 G7 期间与中东领导人的双边会谈，关注中东外交进展。"
        if "trade representative" in lower and "india" in lower and "g7" in lower:
            return "美国贸易代表 Greer 将在 G7 之后一周前往印度，美印贸易谈判可能继续推进。"
        if "trade deal" in lower and "india" in lower and "g7" in lower:
            return "美国官员预计，美印贸易协议会在 G7 期间被讨论，但短期内未必能最终敲定。"
        if "u.s.-india" in lower or "us-india" in lower or "u.s. india" in lower:
            return "美国官员称美印贸易协议存在达成可能，但预计不会在 G7 期间完成，影响贸易与风险偏好预期。"
        if "iran" in lower and "peace deal" in lower:
            return "伊朗相关和平协议预期成为市场关注点，若局势降温可能改善风险偏好；若失败则增加避险压力。"
        if "uae" in lower and "iran" in lower and ("de-escalation" in lower or "unlock" in lower):
            return "市场传出阿联酋可能推动释放伊朗相关资金，属于美伊谈判和中东局势降温信号，关注能源与避险情绪变化。"
        if "ukraine" in lower and "crimean" in lower:
            return "Polymarket 地缘盘口关注乌克兰是否能在2026年底前收复克里米亚，反映市场对俄乌战局走向的押注。"
        if "sanctions" in lower and "russia" in lower:
            return "制裁相关消息继续压制俄罗斯能源杠杆，可能影响能源价格、通胀预期和风险资产情绪。"
        if "war" in lower or "conflict" in lower or "ceasefire" in lower:
            return "地缘冲突/停火相关消息升温，重点观察避险情绪、能源价格和美元流动性对 BTC/ETH 的影响。"

    if topic_name == "BTC机构左侧买盘":
        gate = btc_institutional_quality_gate(text)
        if gate.get("passed"):
            return str(gate.get("summary") or "确定性 BTC 机构买盘/资金流信号。")
        return BTC_NEWS_QUALITY_GATE_FAIL_TEXT

    translated = apply_news_phrase_zh(text)
    if translated != text and enough_chinese_news_text(translated):
        return compact_news_text(translated, limit)
    if has_cjk(text):
        return compact_news_text(text, limit)
    if topic_name == "AI新闻":
        return "AI相关热议：内容没有形成明确大模型新闻要点。"
    if topic_name == "科技新闻":
        return "科技相关热议：内容没有形成明确科技新闻要点。"
    if topic_name == "地缘局势":
        return "地缘/政策相关热议：内容涉及国际关系、政策或冲突风险，主要用于观察避险情绪和风险偏好变化。"
    if topic_name == "BTC机构左侧买盘":
        return BTC_NEWS_QUALITY_GATE_FAIL_TEXT
    return "新闻热议：作为市场情绪参考，暂不作为单独交易信号。"


def is_generic_news_summary(text: str) -> bool:
    stripped = str(text or "").strip()
    return stripped in {
        "暂无可读正文。",
        "AI相关热议：内容没有形成明确大模型新闻要点。",
        "科技相关热议：内容没有形成明确科技新闻要点。",
        "地缘/政策相关热议：内容涉及国际关系、政策或冲突风险，主要用于观察避险情绪和风险偏好变化。",
        "地缘冲突/停火相关消息升温，重点观察避险情绪、能源价格和美元流动性对 BTC/ETH 的影响。",
        BTC_NEWS_QUALITY_GATE_FAIL_TEXT,
        "新闻热议：作为市场情绪参考，暂不作为单独交易信号。",
    }


def news_item_detail_text(item: dict[str, Any], topic: dict[str, Any], limit: int = 260) -> str:
    text = news_item_text(item, 900)
    if not text:
        return ""
    lower = text.lower()
    topic_name = str(topic.get("name") or "")
    if topic_name == "地缘局势":
        if "hormuz" in lower and "g7" in lower:
            return "美国 AP 消息称，特朗普将在下周 G7 与盟友讨论霍尔木兹海峡排雷工作；报道同时提到伊朗相关协议预期，核心风险在能源通道、原油价格和避险情绪。"
        if "trump" in lower and "middle east" in lower and "ukraine" in lower and "g7" in lower:
            return "美国官员称，特朗普将在 G7 期间会见中东领导人，并参加乌克兰相关会议；这意味着中东和俄乌两条风险线可能同时进入市场视野。"
        if "netanyahu" in lower and "g7" in lower:
            return "美国官员称，以色列总理内塔尼亚胡不会参加特朗普在 G7 期间与中东领导人的双边会谈；市场关注中东外交协调是否受影响。"
    if topic_name == "BTC机构左侧买盘":
        gate = btc_institutional_quality_gate(text)
        if gate.get("passed"):
            return str(gate.get("detail") or gate.get("summary") or "确定性 BTC 机构买盘/资金流信号。")
        return ""
    if topic_name == "科技新闻":
        if "spacex" in lower or "starship" in lower:
            return "SpaceX/星舰相关消息出现，重点观察发射节奏、商业航天进展以及马斯克科技叙事是否继续升温。"
        if "tesla" in lower:
            return "特斯拉相关消息出现，重点观察交付、自动驾驶、机器人或能源业务对科技股风险偏好的影响。"
        if "xai" in lower or "grok" in lower:
            return "xAI/Grok 相关消息出现，重点观察马斯克大模型业务、算力投入和 X 生态接入进展。"
        if "biotech" in lower or "crispr" in lower or "gene therapy" in lower or "fda" in lower:
            return "生物科技相关消息出现，重点观察临床数据、监管审批或平台技术突破是否带动成长股情绪。"
    translated = apply_news_phrase_zh(text)
    if has_cjk(text):
        return compact_news_text(text, limit)
    if enough_chinese_news_text(translated):
        return compact_news_text(translated, limit)
    return ""


def news_item_impact_text(item: dict[str, Any], topic: dict[str, Any]) -> str:
    text = news_item_text(item, 600)
    lower = text.lower()
    topic_name = str(topic.get("name") or "")
    if topic_name == "AI新闻":
        if "openai" in lower or "anthropic" in lower or "claude" in lower or "chatgpt" in lower or "deepseek" in lower:
            return "看模型能力/成本变化是否继续推高科技叙事和风险偏好。"
        if any(name in lower for name in ["tencent", "alibaba", "qwen", "bytedance", "doubao", "hunyuan"]):
            return "看国内大模型竞争是否推动云业务、应用入口和国产AI叙事。"
        return "只作AI板块情绪观察，不单独作为交易信号。"
    if topic_name == "科技新闻":
        if any(name in lower for name in ["musk", "tesla", "spacex", "xai", "grok", "neuralink", "starlink"]):
            return "看马斯克相关资产和科技风险偏好是否升温。"
        if any(name in lower for name in ["biotech", "crispr", "gene therapy", "fda", "clinical trial"]):
            return "看生物科技突破或监管节点是否带动成长股情绪。"
        return "只作科技风险偏好观察。"
    if topic_name == "地缘局势":
        if any(name in lower for name in ["hormuz", "iran", "middle east", "israel"]):
            return "重点看能源价格、避险情绪和美元流动性对BTC/ETH的压力。"
        if any(name in lower for name in ["ukraine", "russia", "sanctions"]):
            return "重点看制裁、能源和避险资金是否扰动风险资产。"
        return "只作宏观风险背景观察。"
    return "只作市场情绪观察。"


def news_item_text(item: dict[str, Any], limit: int = 220) -> str:
    return compact_news_text(
        item.get("title")
        or item.get("text")
        or item.get("description")
        or ((item.get("aiRating") or {}).get("summary"))
        or ((item.get("aiRating") or {}).get("enSummary"))
        or "",
        limit,
    )


def news_item_time(item: dict[str, Any]) -> str:
    dt = parse_dt(item.get("ts") or item.get("createdAt") or item.get("time"))
    if not dt:
        return "--"
    return dt.astimezone(BEIJING_TZ).strftime("%m-%d %H:%M")


def news_item_score(item: dict[str, Any]) -> str:
    rating = item.get("aiRating") if isinstance(item.get("aiRating"), dict) else {}
    raw = rating.get("score") or item.get("score")
    try:
        return f"{float(raw):.0f}"
    except Exception:
        return "--"


def news_match_score(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def fetch_opennews_topic(tokens: list[str] | tuple[str, ...] | str, topic: dict[str, Any], lookback_hours: int, limit: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, 4):
        payload = {
            "limit": limit,
            "page": page,
            "lookbackHours": lookback_hours,
            "q": topic["query"],
        }
        result = post_json_with_bearer_pool("https://ai.6551.io/open/news_search", tokens, payload)
        batch = result.get("data") if isinstance(result, dict) else []
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < limit:
            break
    keywords = [str(item) for item in topic.get("keywords") or []]
    exclude_keywords = [str(item).lower() for item in topic.get("exclude_keywords") or []]
    quality_gate_enabled = bool(topic.get("quality_gate"))
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        text = news_item_text(item, 1000)
        lowered = text.lower()
        if exclude_keywords and any(keyword in lowered for keyword in exclude_keywords):
            continue
        score = news_match_score(text, keywords)
        if quality_gate_enabled:
            quality = topic_quality_gate_result(item, topic)
            if not quality.get("passed"):
                continue
            item = dict(item)
            item["_quality_gate"] = quality
            score += 100
        ranked.append((score, -index, item))
    ranked.sort(key=lambda item: item[:2], reverse=True)
    selected = [item for score, _idx, item in ranked if score > 0][:8]
    if not selected and not quality_gate_enabled:
        selected = [item for _score, _idx, item in ranked[:8]]
    return selected


def fetch_opentwitter_topic(tokens: list[str] | tuple[str, ...] | str, topic: dict[str, Any], max_results: int = 6) -> list[dict[str, Any]]:
    payload = {
        "keywords": topic["twitter_query"],
        "maxResults": max_results,
        "product": "Latest",
        "excludeReplies": True,
        "excludeRetweets": True,
    }
    result = post_json_with_bearer_pool("https://ai.6551.io/open/twitter_search", tokens, payload)
    rows = result.get("data") if isinstance(result, dict) else []
    return [item for item in rows if isinstance(item, dict)][:max_results] if isinstance(rows, list) else []


def dedupe_news_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        text = news_item_text(item)
        normalized_text = "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace())
        normalized_text = " ".join(normalized_text.split())
        key = str(normalized_text[:120] or item.get("url") or item.get("id")).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def format_opennews_topic(topic: dict[str, Any], items: list[dict[str, Any]]) -> list[str]:
    topic_icons = {
        "AI新闻": "🧠",
        "科技新闻": "🚀",
        "地缘局势": "🌍",
        "BTC机构左侧买盘": "🟠",
    }
    icon = topic_icons.get(str(topic.get("name") or ""), "📰")
    lines = [f"{icon}【{topic['name']}】"]
    items = dedupe_news_items(items)
    if not items:
        return []
    visible_items: list[tuple[dict[str, Any], str]] = []
    seen_summaries: set[str] = set()
    for item in items:
        quality = topic_quality_gate_result(item, topic)
        if topic.get("quality_gate") and not quality.get("passed"):
            continue
        text = localize_news_text(item, topic, 180)
        if is_generic_news_summary(text):
            continue
        detail = news_item_detail_text(item, topic)
        if not detail:
            continue
        summary_key = " ".join(text.lower().split())
        if not summary_key or summary_key in seen_summaries:
            continue
        seen_summaries.add(summary_key)
        visible_items.append((item, detail))
        if len(visible_items) >= 6:
            break
    if not visible_items:
        return []
    for index, (item, detail) in enumerate(visible_items, 1):
        engine = str(item.get("engineType") or "news")
        source = str(item.get("source") or engine or "opennews")
        score = news_item_score(item)
        lines.append(f"🔹 {index}. {news_item_time(item)}｜{source}｜评分={score}")
        lines.append(f"   📌 {detail}")
    return lines


def format_twitter_topic(topic: dict[str, Any], tweets: list[dict[str, Any]]) -> str:
    if not tweets:
        return ""
    samples: list[str] = []
    seen_summaries: set[str] = set()
    keywords = [str(item) for item in topic.get("keywords") or []]
    exclude_keywords = [str(item).lower() for item in topic.get("exclude_keywords") or []]
    for tweet in tweets:
        raw_text = compact_news_text(tweet.get("text"), 400)
        lowered = raw_text.lower()
        if exclude_keywords and any(keyword in lowered for keyword in exclude_keywords):
            continue
        if keywords and news_match_score(raw_text, keywords) <= 0:
            continue
        tweet_item = {"text": tweet.get("text")}
        quality = topic_quality_gate_result(tweet_item, topic)
        if topic.get("quality_gate") and not quality.get("passed"):
            continue
        handle = str(tweet.get("userScreenName") or tweet.get("username") or "").strip()
        text = news_item_detail_text(tweet_item, topic, 120)
        if not text or is_generic_news_summary(text):
            continue
        summary_key = " ".join(text.lower().split())
        if not summary_key or summary_key in seen_summaries:
            continue
        seen_summaries.add(summary_key)
        prefix = f"@{handle}: " if handle else ""
        if text:
            samples.append(prefix + text)
        if len(samples) >= 2:
            break
    if samples:
        suffix = " / ".join(samples[:2])
        return f"▫️ {topic['name']}补充：{suffix}"
    return ""


def build_daily_news_report(args: argparse.Namespace | None = None) -> str:
    lookback_hours = int(getattr(args, "daily_news_lookback_hours", 24) if args else 24)
    opennews_tokens = env_token_pool("OPENNEWS_PRIMARY_TOKEN", "OPENNEWS_TOKEN", "OPEN_TOKEN")
    twitter_tokens = env_token_pool(
        "OPENTWITTER_PRIMARY_TOKEN",
        "TWITTER_PRIMARY_TOKEN",
        "TWITTER_TOKEN",
        "OPENTWITTER_TOKEN",
        "OPEN_TOKEN",
    )
    now_text = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")
    lines = [
        "🗞️ [Somnus新闻] 每日新闻简报",
        f"时间={now_text} 北京时间｜窗口={lookback_hours}小时",
    ]
    twitter_lines: list[str] = []
    for topic in DAILY_NEWS_TOPICS:
        try:
            news_items = fetch_opennews_topic(opennews_tokens, topic, lookback_hours)
        except Exception as exc:
            news_items = []
            lines.extend([f"【{topic['name']}】", f"opennews拉取失败：{type(exc).__name__}: {str(exc)[:160]}"])
        else:
            topic_lines = format_opennews_topic(topic, news_items)
            if topic_lines:
                lines.extend(["", *topic_lines])
        try:
            tweets = fetch_opentwitter_topic(twitter_tokens, topic)
        except Exception as exc:
            pass
        else:
            twitter_line = format_twitter_topic(topic, tweets)
            if twitter_line:
                twitter_lines.append(twitter_line)
    if twitter_lines:
        lines.extend(["", "💬【X/Twitter补充】", *twitter_lines])
    return "\n".join(lines)[:3900]


def parse_news_command_text(text: str) -> bool:
    parts, command, _has_slash = split_command_parts(text)
    return command in {"news", "新闻", "dailynews", "daily-news", "资讯", "早报"}


def translate_reason(reason: str) -> str:
    if reason in REASON_ZH_MAP:
        return REASON_ZH_MAP[reason]
    if reason.startswith("4H mid retest candidate:"):
        return "4H中轨回抽候选：" + reason.split(":", 1)[1].strip()
    return reason


def compact_v6_reasons(reasons: list[Any]) -> list[str]:
    compacted: list[str] = []
    skip_reasons = {
        "15M long-cycle structure is failed; use 4H as primary frame",
    }
    for raw in reasons:
        reason = str(raw)
        if reason in skip_reasons:
            continue
        translated = translate_reason(reason)
        if translated and translated not in compacted:
            compacted.append(translated)
    return compacted


def refresh_report_signals(symbols: list[str]) -> dict[str, dict[str, Any]]:
    global REPORT_SIGNAL_CACHE
    targets: list[str] = []
    for raw in symbols:
        symbol = normalize_symbol(raw)
        if symbol and symbol not in targets:
            targets.append(symbol)
    if not targets:
        return {}

    summary_age = file_age_seconds(V6_SUMMARY_FILE)
    if summary_age is not None and summary_age <= REPORT_SIGNAL_SUMMARY_MAX_AGE_SECONDS:
        summary = load_json(V6_SUMMARY_FILE)
        summary_signals: dict[str, dict[str, Any]] = {}
        for item in summary.get("signals") or []:
            if isinstance(item, dict):
                symbol = normalize_symbol(str(item.get("symbol") or ""))
                if symbol in targets:
                    summary_signals[symbol] = item
        if all(symbol in summary_signals for symbol in targets):
            REPORT_SIGNAL_CACHE.update(summary_signals)
            return summary_signals

    refreshed: dict[str, dict[str, Any]] = {}
    engine = V6ExecutionEngine(V6ExecutionConfig.from_env())
    try:
        for symbol in targets:
            try:
                refreshed[symbol] = engine.build_signal(symbol).to_dict()
            except Exception as exc:
                print(f"V6 report signal refresh failed for {symbol}: {type(exc).__name__}: {exc}", file=sys.stderr)
                REPORT_SIGNAL_CACHE.pop(symbol, None)
    finally:
        engine.close()
    if refreshed:
        REPORT_SIGNAL_CACHE.update(refreshed)
    return refreshed


def find_v6_signal(symbol: str) -> dict[str, Any] | None:
    symbol = normalize_symbol(symbol)
    cached = REPORT_SIGNAL_CACHE.get(symbol)
    if cached:
        return cached
    summary = load_json(V6_SUMMARY_FILE)
    for item in summary.get("signals") or []:
        if str(item.get("symbol") or "").upper() == symbol:
            return item
    return None


def format_v6_signal(symbol: str) -> list[str]:
    signal = find_v6_signal(symbol)
    if not signal:
        return [f"V6分析：暂无 {symbol} 最新分析（当前V6摘要只覆盖已配置交易对）。"]
    h4 = signal.get("h4") or {}
    fifteen_m = signal.get("fifteen_m") or {}
    mid_retest = signal.get("mid_retest") or {}
    short_zone = ((signal.get("levels") or {}).get("short_entry_retest_zone") or {})
    reasons = compact_v6_reasons(signal.get("reasons") or [])
    reason_text = "；".join(reasons[:4]) if reasons else "--"
    lines = [
        "V6分析：",
        (
            f"- 方向={zh_value(signal.get('side_bias'))} "
            f"多={number(float(signal.get('long_probability') or 0) * 100, 2)}% "
            f"空={number(float(signal.get('short_probability') or 0) * 100, 2)}% "
            f"等待={number(float(signal.get('wait_probability') or 0) * 100, 2)}% "
            f"质量分={number(signal.get('model_quality_score'), 4)}"
        ),
        (
            f"- 4H结构={zh_value(h4.get('structure_state'))} 位置={zh_value(h4.get('price_location'))} "
            f"MA20距离={format_pct_signed(h4.get('price_vs_ma20_pct'))} "
            f"MA20斜率={format_pct_signed(h4.get('ma20_slope_pct'))}"
        ),
        (
            f"- 15M长周期={'失效' if fifteen_m.get('long_cycle_failed') else '未失效'} "
            f"跌破中轨={','.join(fifteen_m.get('below_mid') or []) or '--'} "
            f"跌破下轨={','.join(fifteen_m.get('below_lower') or []) or '--'}"
        ),
        (
            f"- 回抽中轨={mid_retest.get('nearest_mid') or '--'} "
            f"距离={format_pct_signed(mid_retest.get('nearest_abs_distance_pct'))} "
            f"候选={','.join(mid_retest.get('retest_candidates') or []) or '--'}"
        ),
    ]
    if short_zone:
        lines.append(
            f"- 做空回抽区={short_zone.get('target_mid') or '--'} "
            f"{format_price(short_zone.get('zone_low'))}-{format_price(short_zone.get('zone_high'))}"
        )
    lines.append(f"- 理由={reason_text}")
    return lines


def infer_position_side(position: dict[str, Any]) -> str:
    raw = str(
        position.get("side")
        or position.get("position_side")
        or position.get("direction")
        or position.get("action")
        or ""
    ).lower()
    if "short" in raw or raw == "sell":
        return "short"
    if "long" in raw or raw == "buy":
        return "long"
    return "long"


def position_entry_price(position: dict[str, Any]) -> float:
    for key in ("entry_price", "entry_mark_price", "open_price", "avg_price", "price"):
        try:
            value = float(position.get(key) or 0)
        except Exception:
            value = 0.0
        if value > 0:
            return value
    return 0.0


def position_mark_price(position: dict[str, Any]) -> float:
    for key in ("current_price", "mark_price", "last_price", "entry_mark_price", "entry_price"):
        try:
            value = float(position.get(key) or 0)
        except Exception:
            value = 0.0
        if value > 0:
            return value
    return 0.0


def position_notional(position: dict[str, Any]) -> float:
    for key in ("notional_usdt", "quantity_hint_usdt", "notional", "original_notional_usdt"):
        try:
            value = float(position.get(key) or 0)
        except Exception:
            value = 0.0
        if value > 0:
            return value
    return 0.0


def position_quantity_text(position: dict[str, Any], entry_price: float) -> str:
    for key in ("quantity", "qty", "position_amt", "position_qty", "remaining_quantity", "executed_qty"):
        raw = position.get(key)
        try:
            value = abs(float(raw))
        except Exception:
            value = 0.0
        if value > 0:
            return format_price(value)
    notional = position_notional(position)
    if notional > 0 and entry_price > 0:
        return format_price(notional / entry_price)
    if notional > 0:
        return f"{format_price(notional)}U"
    return "--"


def position_stop_loss(position: dict[str, Any]) -> float:
    for key in ("stop_loss_price", "stop_price", "protective_stop_price"):
        try:
            value = float(position.get(key) or 0)
        except Exception:
            value = 0.0
        if value > 0:
            return value
    risk_plan = position.get("risk_plan") or {}
    if isinstance(risk_plan, dict):
        try:
            return float(risk_plan.get("stop_loss_price") or risk_plan.get("default_stop_price") or 0)
        except Exception:
            return 0.0
    return 0.0


def position_pnl(position: dict[str, Any], entry_price: float, mark_price: float) -> tuple[float | None, float | None]:
    for pnl_key in ("unrealized_pnl_usdt", "pnl_usdt", "unrealized_pnl"):
        if position.get(pnl_key) is None:
            continue
        try:
            pnl_usdt = float(position.get(pnl_key))
            notional = position_notional(position)
            margin = float(position.get("position_initial_margin_usdt") or 0.0)
            denominator = margin if margin > 0 else notional
            pnl_pct = (pnl_usdt / denominator * 100.0) if denominator > 0 else None
            return pnl_usdt, pnl_pct
        except Exception:
            continue
    notional = position_notional(position)
    if entry_price <= 0 or mark_price <= 0 or notional <= 0:
        return None, None
    side = infer_position_side(position)
    if side == "short":
        pnl_pct = (entry_price - mark_price) / entry_price * 100.0
    else:
        pnl_pct = (mark_price - entry_price) / entry_price * 100.0
    return notional * pnl_pct / 100.0, pnl_pct


def merge_exchange_position_with_local(row: dict[str, Any], local_by_symbol_side: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "").upper()
    amount = float(row.get("positionAmt") or 0.0)
    position_side = str(row.get("positionSide") or "").upper()
    if position_side == "LONG":
        side = "long"
    elif position_side == "SHORT":
        side = "short"
    elif amount < 0:
        side = "short"
    else:
        side = "long"
    local = local_by_symbol_side.get((symbol, side)) or {}
    entry_price = float(row.get("entryPrice") or local.get("entry_price") or 0.0)
    mark_price = float(row.get("markPrice") or local.get("current_price") or 0.0)
    notional = abs(amount) * mark_price if mark_price > 0 else float(local.get("notional_usdt") or 0.0)
    leverage = float(row.get("leverage") or local.get("leverage") or 0.0)
    position_initial_margin = notional / leverage if leverage > 0 else 0.0
    merged = {
        **local,
        "symbol": symbol,
        "side": side,
        "quantity": abs(amount),
        "entry_price": entry_price,
        "current_price": mark_price,
        "mark_price": mark_price,
        "notional_usdt": notional,
        "position_initial_margin_usdt": position_initial_margin,
        "unrealized_pnl_usdt": float(row.get("unRealizedProfit") or row.get("unrealizedProfit") or 0.0),
        "exchange_source": "binance_demo_positionRisk",
    }
    if row.get("leverage") is not None:
        merged["leverage"] = row.get("leverage")
    return merged


def fetch_v6_exchange_positions(local_positions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    local_by_symbol_side = {
        (str(position.get("symbol") or "").upper(), infer_position_side(position)): position
        for position in local_positions
    }
    client = V6BinanceClient(V6ExecutionConfig.from_env())
    try:
        rows = client.actual_position_rows()
    finally:
        client.close()
    positions = [merge_exchange_position_with_local(row, local_by_symbol_side) for row in rows]
    return positions, "binance_demo_positionRisk"


def v5_execution_config_from_env() -> V6ExecutionConfig:
    v5 = StrategyConfig()
    raw_mode = str(v5.execution_order_mode or "auto").strip().lower()
    if raw_mode in {"auto", ""}:
        raw_mode = "testnet_order" if v5.execution_test_order else "dry_run"
    mode = {
        "demo": "testnet_order",
        "testnet": "testnet_order",
        "testnet_order": "testnet_order",
        "live": "live_order",
        "live_order": "live_order",
        "dry": "dry_run",
        "dry_run": "dry_run",
        "test_order": "dry_run",
    }.get(raw_mode, raw_mode)
    config = V6ExecutionConfig.from_env()
    config.order_mode = canonical_order_mode(mode)
    if config.order_mode == "testnet_order":
        config.demo_base_url = str(v5.binance_testnet_base_url)
        config.demo_api_key = str(v5.binance_testnet_api_key or "")
        config.demo_api_secret = str(v5.binance_testnet_api_secret or "")
    elif config.order_mode == "live_order":
        config.live_base_url = str(v5.base_url)
        config.live_api_key = str(v5.binance_api_key or "")
        config.live_api_secret = str(v5.binance_api_secret or "")
    return config


def fetch_v5_exchange_positions(local_positions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    local_by_symbol_side = {
        (str(position.get("symbol") or "").upper(), infer_position_side(position)): position
        for position in local_positions
    }
    config = v5_execution_config_from_env()
    if config.order_mode == "dry_run":
        return local_positions, "本地V5状态"
    client = V6BinanceClient(config)
    try:
        rows = client.actual_position_rows()
    finally:
        client.close()
    positions = [merge_exchange_position_with_local(row, local_by_symbol_side) for row in rows]
    return positions, "binance_v5_positionRisk"


def format_position_line(position: dict[str, Any]) -> str:
    symbol = str(position.get("symbol") or "--").upper()
    side = infer_position_side(position)
    side_prefix = "🔴 空" if side == "short" else "🟢 多"
    entry_price = position_entry_price(position)
    mark_price = position_mark_price(position)
    stop_loss = position_stop_loss(position)
    pnl_usdt, pnl_pct = position_pnl(position, entry_price, mark_price)
    quantity = position_quantity_text(position, entry_price)
    if pnl_pct is None:
        return_text = "--"
    else:
        return_icon = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
        return_text = f"{return_icon} {pnl_pct:+.2f}%"
    return (
        f"{side_prefix} {symbol}-{quantity}-开仓={format_price(entry_price)}-标记={format_price(mark_price)}-"
        f"止损={format_price(stop_loss)}-盈亏={return_text}"
    )


def build_position_response(target: str, status: dict[str, Any]) -> str:
    if target == "v5":
        title = "[Somnus持仓] V5"
        local_positions = status.get("v5", {}).get("positions") or []
        try:
            positions, source = fetch_v5_exchange_positions(local_positions)
            source_note = f"来源={source}"
        except Exception as exc:
            positions = local_positions
            source_note = f"来源=本地V5状态（交易所读取失败：{zh_error(exc)}）"
    else:
        title = "[Somnus持仓] V6"
        local_positions = status.get("v6", {}).get("positions") or []
        try:
            positions, source = fetch_v6_exchange_positions(local_positions)
            source_note = f"来源={source}，回报率=按保证金"
        except Exception as exc:
            positions = local_positions
            source_note = f"来源=本地V6状态（demo读取失败：{zh_error(exc)}）"
    lines = [title]
    if not positions:
        lines.append("当前无持仓")
        return "\n".join(lines)
    lines.extend(format_position_line(position) for position in positions)
    return "\n".join(lines)


def compact_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if not text:
        return "--"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def zh_rationale(value: Any) -> str:
    text = str(value or "")
    replacements = [
        (
            "staged short take profit touched with current/delayed 15M volume confirmation:",
            "分批止盈触发（空，15M放量确认）：",
        ),
        (
            "staged long take profit touched with current/delayed 15M volume confirmation:",
            "分批止盈触发（多，15M放量确认）：",
        ),
        ("staged short take profit touched:", "分批止盈触发（空）："),
        ("staged long take profit touched:", "分批止盈触发（多）："),
        ("close_initial=", "初始减仓="),
        ("close_remaining=", "按剩余仓位减仓="),
        ("BB structure entry scan is mixed; keep conflict wait weight", "BB结构入场扫描冲突，保持等待权重"),
        ("BB structure long entry candidates near price:", "价格附近BB做多候选："),
        ("BB structure short entry candidates near price:", "价格附近BB做空候选："),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    parts = [part.strip() for part in text.split(";") if part.strip()]
    if parts:
        translated = [translate_reason(part) for part in parts]
        text = "；".join(translated)
    return text


def format_execution_intent(intent: dict[str, Any], index: int) -> str:
    risk_plan = intent.get("risk_plan") or {}
    stage = ((risk_plan.get("partial_take_profit") or {}).get("stage") or {}) if isinstance(risk_plan, dict) else {}
    stage_text = ""
    if stage:
        stage_text = (
            f" 触发={stage.get('name', '--')} "
            f"目标价={format_price(stage.get('price'))} "
            f"减仓={number(float(stage.get('fraction_of_initial') or 0) * 100, 2)}%"
        )
    return (
        f"意图{index}: {str(intent.get('symbol') or '--').upper()} {zh_action(intent.get('action'))} "
        f"方向={zh_value(intent.get('side'))} 数量={format_price(intent.get('quantity_hint_usdt'))}U "
        f"杠杆={intent.get('leverage', '--')}x reduceOnly={zh_value(str(bool(intent.get('reduce_only'))).lower())}\n"
        f"  开仓={format_price(intent.get('entry_price'))} 止损={format_price(intent.get('stop_loss_price'))} "
        f"止盈={format_price(intent.get('take_profit_price'))}{stage_text}\n"
        f"  理由={compact_text(zh_rationale(intent.get('rationale')))}"
    )


def zh_close_mode(decision: dict[str, Any]) -> str:
    mode = str(decision.get("mode") or "")
    reason = str(decision.get("reason") or "")
    if mode == "full_flatten":
        return "粉尘全平" if "dust" in reason else "全平"
    if mode == "partial":
        return "部分平仓"
    if mode == "skip_zero":
        return "跳过(数量为零)"
    if mode == "skip_missing_exchange_position":
        return "跳过(交易所无持仓)"
    return ""


def close_decision_from_intent(intent: dict[str, Any]) -> dict[str, Any]:
    risk_plan = intent.get("risk_plan")
    if not isinstance(risk_plan, dict):
        return {}
    decision = risk_plan.get("close_decision")
    return decision if isinstance(decision, dict) else {}


def format_compact_execution_intent(intent: dict[str, Any]) -> str:
    symbol = str(intent.get("symbol") or "--").upper()
    action_key = str(intent.get("action") or "").lower()
    action = zh_action(action_key)
    icon = action_icon(action_key)
    quantity = format_price(intent.get("quantity_hint_usdt"))
    rationale = compact_text(zh_rationale(intent.get("rationale")), limit=150)
    if action_key in {"open_long", "open_short", "add_long", "add_short", "pyramid_long", "pyramid_short"}:
        return (
            f"{icon} {symbol} {action} "
            f"开仓={format_price(intent.get('entry_price'))} "
            f"止损={format_price(intent.get('stop_loss_price'))} "
            f"止盈={format_price(intent.get('take_profit_price'))} "
            f"理由={rationale}"
        )
    if action_key in {"close_long", "close_short"}:
        mode_text = zh_close_mode(close_decision_from_intent(intent))
        mode_part = f" 模式={mode_text}" if mode_text else ""
        return f"{icon} {symbol} {action}{mode_part} 数量={quantity}U 理由={rationale}"
    return f"{icon} {symbol} {action} 数量={quantity}U 理由={rationale}"


def format_execution_result(result: dict[str, Any], index: int) -> str:
    status = result.get("status") or result.get("submit_status") or "--"
    symbol = str(result.get("symbol") or result.get("orig_symbol") or "--").upper()
    action = zh_action(result.get("action"))
    order_id = result.get("order_id") or result.get("client_order_id") or result.get("protective_stop_order_id") or "--"
    executed_qty = result.get("executed_qty") or result.get("origQty") or result.get("quantity") or "--"
    avg_price = result.get("avg_price") or result.get("avgPrice") or result.get("price") or "--"
    error = result.get("error") or result.get("message") or result.get("msg") or ""
    base = (
        f"结果{index}: {symbol} {action} 状态={zh_value(status)} "
        f"订单={order_id} 成交数量={executed_qty} 均价={format_price(avg_price)}"
    )
    if error:
        base += f"\n  错误={zh_error(error)}"
    return base


def format_compact_execution_result(result: dict[str, Any]) -> str:
    status = result.get("status") or result.get("submit_status") or "--"
    order_id = result.get("order_id") or result.get("client_order_id") or result.get("protective_stop_order_id") or ""
    executed_qty = result.get("executed_qty") or result.get("origQty") or result.get("quantity") or ""
    avg_price = result.get("avg_price") or result.get("avgPrice") or result.get("price") or ""
    error = result.get("error") or result.get("message") or result.get("msg") or ""
    parts: list[str] = []
    fill = compact_fill_text(executed_qty, avg_price)
    if fill:
        parts.append(f"成交={fill}")
    if order_id not in {"", None, "--"}:
        parts.append(f"订单={order_id}")
    if error:
        parts.append(f"错误={compact_text(zh_error(error), limit=180)}")
    parts.append(f"执行={execution_status_text(status)}")
    return " ".join(parts)


def pair_execution_intents_results(
    intents: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    result_queues: dict[tuple[str, str], list[dict[str, Any]]] = {}
    result_only: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        symbol = str(result.get("symbol") or "").upper()
        action = str(result.get("action") or "").lower()
        if symbol and action:
            result_queues.setdefault((symbol, action), []).append(result)
        else:
            result_only.append(result)

    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        key = (str(intent.get("symbol") or "").upper(), str(intent.get("action") or "").lower())
        matched = result_queues.get(key, []).pop(0) if result_queues.get(key) else None
        pairs.append((intent, matched))

    for queue in result_queues.values():
        for result in queue:
            pairs.append((None, result))
    for result in result_only:
        pairs.append((None, result))
    return pairs


def format_v6_activity_details(v6: dict[str, Any]) -> str:
    intents = v6.get("intents") or []
    results = v6.get("results") or []
    lines = ["☎️ [Somnus执行] V6"]
    if intents:
        for idx, intent in enumerate(intents[:5]):
            line = format_compact_execution_intent(intent)
            if idx < len(results):
                line += f"\n{format_compact_execution_result(results[idx])}"
            lines.append(line)
    else:
        lines.append("下单意图=无")
        if results:
            lines.extend(format_compact_execution_result(result) for result in results[:5])
    return "\n".join(lines)


def format_v5_trade_request(request: dict[str, Any]) -> str:
    symbol = str(request.get("symbol") or "--").upper()
    action_key = str(request.get("action") or "").lower()
    action = zh_action(action_key)
    status = execution_status_text(request.get("status") or request.get("submit_status"))
    quantity = format_price(request.get("quantity_hint_usdt"))
    rationale = compact_text(zh_rationale(request.get("rationale")), limit=150)
    parts = [f"{action_icon(action_key)} {symbol} {action}"]
    if action_key in {"open_long", "open_short", "add_long", "add_short", "pyramid_long", "pyramid_short"}:
        entry = request.get("entry_price") or request.get("price") or request.get("avg_price")
        stop = request.get("stop_loss_price")
        take = request.get("take_profit_price")
        if entry not in {None, "", "--"}:
            parts.append(f"开仓={format_price(entry)}")
        if stop not in {None, "", "--"}:
            parts.append(f"止损={format_price(stop)}")
        if take not in {None, "", "--"}:
            parts.append(f"止盈={format_price(take)}")
    else:
        parts.append(f"数量={quantity}U")
    if rationale and rationale != "--":
        parts.append(f"理由={rationale}")
    parts.append(f"执行={status}")
    return " ".join(parts)


def format_v5_paper_journal_event(event: dict[str, Any]) -> str:
    symbol = str(event.get("symbol") or "--").upper()
    action_key = str(event.get("action") or "").lower()
    action = zh_action(action_key)
    reason = compact_text(zh_rationale(event.get("reason")), limit=120)
    parts = [f"{action_icon(action_key)} {symbol} {action}"]
    entry = event.get("entry_price")
    exit_price = event.get("exit_price")
    mark = event.get("current_price") or event.get("entry_mark_price")
    notional = event.get("notional_usdt")
    pnl_pct = event.get("pnl_pct")
    pnl_usdt = event.get("pnl_usdt")
    if action_key in {"open_long", "open_short"}:
        if entry not in {None, "", "--"}:
            parts.append(f"开仓={format_price(entry)}")
        if notional not in {None, "", "--"}:
            parts.append(f"名义={format_price(notional)}U")
    else:
        if entry not in {None, "", "--"}:
            parts.append(f"入场={format_price(entry)}")
        if exit_price not in {None, "", "--"}:
            parts.append(f"平仓={format_price(exit_price)}")
        elif mark not in {None, "", "--"}:
            parts.append(f"标记={format_price(mark)}")
        if pnl_pct not in {None, "", "--"}:
            try:
                parts.append(f"收益={number(float(pnl_pct) * 100, 2)}%")
            except Exception:
                parts.append(f"收益={pnl_pct}")
        if pnl_usdt not in {None, "", "--"}:
            parts.append(f"盈亏={format_price(pnl_usdt)}U")
        elif notional not in {None, "", "--"} and pnl_pct not in {None, "", "--"}:
            try:
                parts.append(f"盈亏={format_price(float(notional) * float(pnl_pct))}U")
            except Exception:
                pass
    if reason and reason != "--":
        parts.append(f"理由={reason}")
    parts.append("执行=✅")
    return " ".join(parts)


def format_v5_trade_activity_details(v5: dict[str, Any]) -> str:
    requests = v5.get("trade_requests") or []
    journal_events = v5.get("paper_journal_events") or []
    intents = v5.get("intents") or []
    results = v5.get("results") or []
    body: list[str] = []
    if intents or results:
        for intent, result in pair_execution_intents_results(intents, results)[:5]:
            if intent:
                line = format_compact_execution_intent(intent)
                if result:
                    line += f"\n{format_compact_execution_result(result)}"
                body.append(line)
            elif result:
                body.append(format_compact_execution_result(result))
    else:
        for request in requests[:5]:
            body.append(format_v5_trade_request(request))
        for event in journal_events[-5:]:
            body.append(format_v5_paper_journal_event(event))
    if not body:
        return ""
    return "\n".join(["☎️ [Somnus执行] V5", *body])


def build_chart_image(symbol: str, interval: str, focus_period: int | None = None) -> dict[str, Any]:
    if not CHART_RENDER_SCRIPT.exists():
        raise RuntimeError(f"图表渲染脚本不存在：{CHART_RENDER_SCRIPT}")
    focus = focus_period if focus_period and focus_period > 0 else None
    suffix = f"bb{focus}" if focus else "kline"
    output = CHART_DIR / f"{symbol.lower()}_{interval}_{suffix}_{int(time.time())}.png"
    command = [
        sys.executable,
        str(CHART_RENDER_SCRIPT),
        "--symbol",
        symbol,
        "--interval",
        interval,
        "--output",
        str(output),
    ]
    if focus:
        command.extend(["--focus-period", str(focus)])
    result = subprocess.run(
        command,
        cwd=str(ROOT),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"图表渲染失败：{detail[:800]}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("图表渲染成功但没有返回元数据")
    try:
        payload = json.loads(lines[-1])
    except Exception as exc:
        raise RuntimeError(f"图表元数据无法解析：{lines[-1][:300]}") from exc
    output_path = Path(payload.get("output") or output)
    if not output_path.exists():
        raise RuntimeError(f"图表文件未生成：{output_path}")
    payload["output"] = str(output_path)
    return payload


def format_chart_caption(payload: dict[str, Any]) -> str:
    symbol = str(payload.get("symbol") or "").upper()
    interval = str(payload.get("interval") or "").upper()
    focus_period = payload.get("focus_period")
    title = f"{symbol} {interval} BB{focus_period}" if focus_period else f"{symbol} {interval}"
    lines = [
        f"[Somnus图表] {title}",
        f"价格={format_price(payload.get('price'))} K线数={payload.get('bars', '--')}｜K线来源={payload.get('kline_source') or 'Binance实时刷新'}",
    ]
    signal = payload.get("v6") or {}
    if signal:
        lines.append(
            f"V6={zh_value(signal.get('side_bias'))} "
            f"多={number(float(signal.get('long_probability') or 0) * 100, 2)}% "
            f"空={number(float(signal.get('short_probability') or 0) * 100, 2)}% "
            f"等待={number(float(signal.get('wait_probability') or 0) * 100, 2)}%"
        )
    else:
        lines.append("V6=暂无该币最新分析")
    return "\n".join(lines)


def build_command_response(text: str) -> str:
    parsed = parse_command_text(text)
    if not parsed or parsed[0] == "HELP":
        return command_help()
    symbol, interval = parsed
    rows, context = build_bb_rows(symbol, interval)
    price = float(context["price"])
    interval_label = interval.upper()
    lines = [
        f"[Somnus命令] {symbol} {interval_label} BB指标",
        f"价格={format_price(price)} K线数={context['kline_count']} 数据源={context['api_symbol']}｜K线来源={context.get('kline_source', 'binance实时刷新')}",
        *format_nearest_levels(rows, price),
        "",
        "BB轨道：",
    ]
    for row in rows:
        period = int(row["period"])
        if not row.get("ready"):
            lines.append(f"- BB{period}: 数据不足 {row.get('available', 0)}/{period}")
            continue
        lines.append(
            f"- BB{period}: 上={format_price(row['upper'])} 中={format_price(row['mid'])} 下={format_price(row['lower'])} "
            f"位置={number(row['position'], 3)} 距中={format_pct_signed(row['dist_mid'])} 距下={format_pct_signed(row['dist_lower'])}"
        )
    lines.extend(["", *format_v6_signal(symbol)])
    return "\n".join(lines)


def is_manual_command_text(text: str) -> bool:
    clean = strip_bot_mentions(str(text or ""))
    if not clean:
        return False
    if parse_position_command_text(clean) is not None:
        return True
    if parse_news_command_text(clean):
        return True
    if parse_report_command_text(clean) is not None:
        return True
    if parse_chart_command_text(clean) is not None:
        return True
    return parse_command_text(clean) is not None


def handle_telegram_commands(args: argparse.Namespace, client: TelegramClient, state: dict[str, Any]) -> int:
    if args.disable_commands:
        return 0
    if args.dry_run:
        return 0
    bot_username = ""
    if args.require_command_mention:
        bot_username = normalize_bot_username(os.getenv("TS_TELEGRAM_BOT_USERNAME") or client.username())
        if not bot_username:
            return 0
    offset = state.get("telegram_update_offset")
    updates = client.get_updates(offset=int(offset) if offset is not None else None)
    if not updates:
        return 0

    max_update_id = max(int(item.get("update_id") or 0) for item in updates)
    if offset is None:
        state["telegram_update_offset"] = max_update_id + 1
        return 0

    sent = 0
    configured_chat = str(client.chat_id)
    for update in updates:
        state["telegram_update_offset"] = max(int(update.get("update_id") or 0) + 1, int(state.get("telegram_update_offset") or 0))
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if str(chat_id) != configured_chat:
            continue
        text = str(message.get("text") or "")
        current_scope = str(args.scope or "").lower()
        mentions_current_bot = message_mentions_bot(message, bot_username) if bot_username else False
        mentions_any_bot = text_mentions_any_bot(text)
        mentioned_command_text = strip_bot_mentions(text) if mentions_current_bot else ""
        original_command_text = strip_bot_mentions(text) if mentions_current_bot else text

        if getattr(args, "execution_only_commands", True):
            command_text = mentioned_command_text if mentions_current_bot else original_command_text
            if mentions_current_bot and is_manual_command_text(command_text):
                client.send(command_help(), chat_id=chat_id)
                sent += 1
            continue

        position_text = mentioned_command_text if mentions_current_bot else ""
        position_target = parse_position_command_text(position_text)
        if position_target:
            if position_target == "HELP":
                if current_scope in {"v5", "v6"}:
                    client.send(f"用法：@{bot_username} p", chat_id=chat_id)
                else:
                    client.send(f"用法：@{bot_username} p", chat_id=chat_id)
            else:
                if position_target == "SELF":
                    if current_scope not in {"v5", "v6"}:
                        continue
                    position_target = current_scope
                if not position_target_matches_scope(args.scope, position_target):
                    continue
                try:
                    response = build_position_response(position_target, collect_status(args))
                except Exception as exc:
                    response = f"[Somnus持仓] 查询失败\n命令={position_text}\n错误={exc}"
                client.send(response, chat_id=chat_id)
            sent += 1
            continue

        if parse_position_command_text(text):
            continue

        if parse_news_command_text(mentioned_command_text):
            if args.disable_intel_commands:
                continue
            try:
                response = build_daily_news_report(args)
            except Exception as exc:
                response = f"[Somnus新闻] 生成失败\n命令={mentioned_command_text}\n错误={exc}"
            client.send(response, chat_id=chat_id)
            sent += 1
            continue

        if current_scope != "v6":
            continue
        if mentions_any_bot and not mentions_current_bot:
            continue

        command_text = original_command_text
        report_symbols = parse_report_command_text(command_text)
        if report_symbols is not None:
            if args.disable_intel_commands:
                continue
            try:
                response = build_report_response(report_symbols)
            except Exception as exc:
                response = f"[Somnus报告] 生成失败\n命令={command_text}\n错误={exc}"
            client.send(response, chat_id=chat_id)
            sent += 1
            continue
        chart_command = parse_chart_command_text(command_text)
        if chart_command:
            if chart_command[0] == "HELP":
                client.send(command_help(), chat_id=chat_id)
                sent += 1
                continue
            symbol, interval, focus_period = chart_command
            try:
                chart_payload = build_chart_image(symbol, interval, focus_period)
                client.send_photo(
                    Path(chart_payload["output"]),
                    caption=format_chart_caption(chart_payload),
                    chat_id=chat_id,
                )
            except Exception as exc:
                client.send(f"[Somnus图表] 生成失败\n命令={command_text}\n错误={exc}", chat_id=chat_id)
            sent += 1
            continue
        parsed = parse_command_text(command_text)
        if not parsed:
            continue
        try:
            response = build_command_response(command_text)
        except Exception as exc:
            response = f"[Somnus命令] 查询失败\n命令={command_text}\n错误={exc}"
        client.send(response, chat_id=chat_id)
        sent += 1
    return sent


def scope_enabled(scope: str, target: str) -> bool:
    return str(scope or "all").lower() in {"all", target}


def build_summary(status: dict[str, Any], scope: str = "all") -> str:
    v5 = status["v5"]
    v6 = status["v6"]
    macro = v5.get("macro") or {}
    perf = v5.get("performance") or {}
    v6_metrics = v6.get("metrics") or {}
    signals = v6.get("signals") or {}

    lines = [
        "[Somnus监控] V5/V6状态" if scope_enabled(scope, "v5") and scope_enabled(scope, "v6")
        else "[Somnus监控] V5状态" if scope_enabled(scope, "v5")
        else "[Somnus监控] V6状态",
        f"时间={status['generated_at']}",
    ]
    if scope_enabled(scope, "v5"):
        lines.extend([
            "",
            "V5",
            f"- 进程={'正常' if v5['running'] else '停止'} PID={v5.get('pid') or '--'} 报告延迟={format_age(v5.get('report_age'))} 活动延迟={format_age(v5.get('activity_age'))}",
            f"- 宏观方向={zh_value(macro.get('macro_direction'))} 置信度={zh_value(macro.get('macro_confidence'))} 市场状态={zh_value(v5.get('market_regime'))}",
            (
                f"- BTC 趋势={zh_value(macro.get('btc_trend'))} 价格={number(macro.get('btc_price'), 2)} "
                f"MA7距离={pct(macro.get('btc_distance_pct'))} MA20距离={pct(macro.get('btc_distance_ma20_pct'))}"
            ),
            (
                f"- ETH 趋势={zh_value(macro.get('eth_trend'))} 价格={number(macro.get('eth_price'), 2)} "
                f"MA7距离={pct(macro.get('eth_distance_pct'))} MA20距离={pct(macro.get('eth_distance_ma20_pct'))}"
            ),
            f"- 持仓数={perf.get('open_positions', '--')}",
            f"- 下单意图={v5.get('intent_count', 0)} 最新错误={len(v5.get('latest_errors') or [])} 问题数={v5.get('issue_count', 0)}",
        ])
    if scope_enabled(scope, "v6"):
        distill_resources = systemd_service_resources("v6-distill.service")
        swap_text = format_percent_of_total(
            distill_resources.get("MemorySwapCurrent"),
            system_swap_total_bytes(),
        )
        v6_local_positions = v6.get("positions") or []
        try:
            v6_exchange_positions, _v6_position_source = fetch_v6_exchange_positions(v6_local_positions)
            v6_position_count_text = str(len(v6_exchange_positions))
        except Exception:
            v6_position_count_text = f"{len(v6_local_positions)}（本地）"
        train_line = (
            f"- 训练={'正常' if v6.get('distill_running') else '停止'} "
            f"延迟={format_age(v6.get('distill_activity_age'))} "
            f"样本={v6.get('live_ops_total')} "
            f"swap={swap_text}"
        )
        lines.extend([
            "",
            "V6",
            (
                f"- 执行层={'正常' if v6['running'] else '停止'} PID={v6.get('pid') or '--'} "
                f"来源={v6.get('running_source') or '--'} 执行摘要延迟={format_age(v6.get('summary_age'))}"
            ),
            train_line,
            f"- {signal_line('BTCUSDT', signals.get('BTCUSDT') or {})}",
            f"- {signal_line('ETHUSDT', signals.get('ETHUSDT') or {})}",
            f"- 持仓数={v6_position_count_text}",
        ])
    return "\n".join(lines)


def normalize_summary_for_dedupe(message: str) -> str:
    lines: list[str] = []
    for raw_line in str(message or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("时间="):
            continue
        line = re.sub(r"(报告延迟=)[^ ]+", r"\1<age>", line)
        line = re.sub(r"(活动延迟=)[^ ]+", r"\1<age>", line)
        line = re.sub(r"(执行摘要延迟=)[^ ]+", r"\1<age>", line)
        line = re.sub(r"( 延迟=)[^ ]+", r"\1<age>", line)
        lines.append(line)
    return "\n".join(lines)


def summary_dedupe_digest(message: str, scope: str) -> str:
    return short_hash({
        "scope": scope,
        "message": normalize_summary_for_dedupe(message),
    })


def extract_error_code(error_text: Any) -> str:
    match = re.search(r"-\d{3,5}", str(error_text or ""))
    return match.group(0) if match else ""


def close_failure_dedupe_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Stable dedup fields for an unresolved close failure.

    Key = symbol + side + action + reason + target_name + error_code, so the
    same failing close does not rebroadcast every loop just because
    timestamps in the raw result changed.
    """
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    decision = close_decision_from_intent(intent)
    return {
        "kind": "close_failure",
        "symbol": str(intent.get("symbol") or result.get("symbol") or "").upper(),
        "side": str(intent.get("side") or ""),
        "action": str(intent.get("action") or result.get("action") or ""),
        "reason": str(decision.get("reason") or ""),
        "target_name": str(intent.get("target_name") or ""),
        "error_code": extract_error_code(result.get("error") or result.get("original_error")),
    }


def is_close_failure_result(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    action = str(intent.get("action") or result.get("action") or "").lower()
    if action not in {"close_long", "close_short"}:
        return False
    status = str(result.get("status") or "")
    return status == "failed" or bool(result.get("error"))


def stable_v6_activity_fingerprint(v6: dict[str, Any]) -> dict[str, Any]:
    results_fp: list[Any] = []
    for result in v6.get("results") or []:
        if is_close_failure_result(result):
            results_fp.append(close_failure_dedupe_fields(result))
        else:
            results_fp.append(result)
    return {"intents": v6.get("intents"), "results": results_fp}


def build_alerts(status: dict[str, Any], scope: str = "all") -> list[tuple[str, str]]:
    alerts: list[tuple[str, str]] = []
    v5 = status["v5"]
    v6 = status["v6"]
    thresholds = status["thresholds"]

    if scope_enabled(scope, "v5"):
        if not v5["running"]:
            alerts.append(("v5_down", "[Somnus告警] V5流水线已停止"))
        v5_activity_age = v5.get("activity_age")
        if (v5.get("report_age") or 0) > thresholds["v5_stale_seconds"] and (
            v5_activity_age is None or v5_activity_age > thresholds["v5_stale_seconds"]
        ):
            alerts.append((
                "v5_stale",
                f"[Somnus告警] V5报告/心跳已过期：报告延迟={format_age(v5.get('report_age'))} 活动延迟={format_age(v5_activity_age)}",
            ))

        latest_errors = v5.get("latest_errors") or []
        if latest_errors:
            err = latest_errors[0]
            if not is_silent_execution_error(err):
                key = "v5_error_" + short_hash(err)
                alerts.append(
                    (
                        key,
                        "[Somnus告警] V5最新执行错误\n"
                        f"交易对={err.get('symbol', '--')} 动作={zh_action(err.get('action'))}\n"
                        f"错误={zh_error(err.get('error'))}",
                    )
                )

    if scope_enabled(scope, "v6"):
        if not v6["running"]:
            alerts.append(("v6_down", "[Somnus告警] V6执行循环已停止"))
        v6_distill_activity_age = v6.get("distill_activity_age")
        if not v6.get("distill_running") and (
            v6_distill_activity_age is None or v6_distill_activity_age > thresholds["v6_distill_stale_seconds"]
        ):
            alerts.append(("v6_distill_down", "[Somnus告警] V6蒸馏/训练循环已停止"))
        if (v6.get("summary_age") or 0) > thresholds["v6_stale_seconds"]:
            alerts.append(("v6_stale", f"[Somnus告警] V6执行摘要已过期：延迟={format_age(v6.get('summary_age'))}"))
        if (v6.get("distill_age") or 0) > thresholds["v6_distill_stale_seconds"] and (
            v6_distill_activity_age is None or v6_distill_activity_age > thresholds["v6_distill_stale_seconds"]
        ):
            alerts.append(("v6_distill_stale", f"[Somnus告警] V6蒸馏报告/心跳已过期：报告延迟={format_age(v6.get('distill_age'))} 活动延迟={format_age(v6_distill_activity_age)}"))

        if v6.get("intent_count", 0) > 0 or v6.get("result_count", 0) > 0:
            key = "v6_activity_" + short_hash(stable_v6_activity_fingerprint(v6))
            alerts.append((key, format_v6_activity_details(v6)))

    return alerts


def should_send_alert(state: dict[str, Any], key: str, repeat_seconds: int) -> bool:
    now_ts = time.time()
    last = float((state.get("alerts") or {}).get(key, 0) or 0)
    return now_ts - last >= repeat_seconds


def mark_alert_sent(state: dict[str, Any], key: str) -> None:
    state.setdefault("alerts", {})[key] = time.time()


def prune_alert_state(state: dict[str, Any], now_ts: float) -> None:
    alerts = state.get("alerts")
    if not isinstance(alerts, dict):
        state["alerts"] = {}
        return
    pruned: dict[str, Any] = {}
    for key, value in alerts.items():
        try:
            ts = float(value)
        except Exception:
            continue
        ttl = V6_ACTIVITY_ALERT_STATE_TTL_SECONDS if str(key).startswith("v6_activity_") else DEFAULT_ALERT_STATE_TTL_SECONDS
        if now_ts - ts <= ttl:
            pruned[str(key)] = value
    state["alerts"] = pruned


def v5_trade_request_key(request: dict[str, Any]) -> str:
    return short_hash(
        {
            "symbol": str(request.get("symbol") or "").upper(),
            "action": str(request.get("action") or "").lower(),
            "status": str(request.get("status") or request.get("submit_status") or "").lower(),
            "request_mode": request.get("request_mode"),
            "side": request.get("side"),
            "quantity_hint_usdt": request.get("quantity_hint_usdt"),
            "leverage": request.get("leverage"),
            "rationale": request.get("rationale"),
            "entry_price": request.get("entry_price") or request.get("price") or request.get("avg_price"),
            "stop_loss_price": request.get("stop_loss_price"),
            "take_profit_price": request.get("take_profit_price"),
        }
    )


def v5_paper_journal_event_key(event: dict[str, Any]) -> str:
    return "journal_" + short_hash(
        {
            "timestamp": event.get("timestamp"),
            "symbol": str(event.get("symbol") or "").upper(),
            "action": str(event.get("action") or "").lower(),
            "reason": event.get("reason"),
            "entry_price": event.get("entry_price"),
            "exit_price": event.get("exit_price"),
            "notional_usdt": event.get("notional_usdt"),
            "pnl_pct": event.get("pnl_pct"),
        }
    )


def v5_overlay_activity_key(intent: dict[str, Any] | None, result: dict[str, Any] | None, index: int) -> str:
    intent = intent or {}
    result = result or {}
    return "overlay_" + short_hash(
        {
            "index": index,
            "generated_at": intent.get("generated_at") or result.get("submitted_at"),
            "symbol": str(intent.get("symbol") or result.get("symbol") or "").upper(),
            "action": str(intent.get("action") or result.get("action") or "").lower(),
            "status": str(result.get("status") or result.get("submit_status") or "").lower(),
            "entry_price": intent.get("entry_price"),
            "stop_loss_price": intent.get("stop_loss_price"),
            "take_profit_price": intent.get("take_profit_price"),
            "quantity_hint_usdt": intent.get("quantity_hint_usdt"),
            "error": result.get("error"),
            "order_id": result.get("order_id") or result.get("client_order_id"),
        }
    )


def prune_v5_trade_activity_state(state: dict[str, Any], now_ts: float) -> None:
    sent = state.get("v5_trade_activity_sent")
    if not isinstance(sent, dict):
        state["v5_trade_activity_sent"] = {}
        return
    ttl_seconds = 7 * 24 * 3600
    pruned: dict[str, Any] = {}
    for key, value in sent.items():
        try:
            ts = float(value)
        except Exception:
            continue
        if now_ts - ts <= ttl_seconds:
            pruned[str(key)] = value
    state["v5_trade_activity_sent"] = pruned


def maybe_send_v5_trade_activity(
    args: argparse.Namespace,
    client: TelegramClient,
    state: dict[str, Any],
    status: dict[str, Any],
    now_ts: float,
) -> bool:
    if not args.v5_trade_activity or not scope_enabled(args.scope, "v5"):
        return False
    requests = status.get("v5", {}).get("trade_requests") or []
    journal_events = status.get("v5", {}).get("paper_journal_events") or []
    overlay_intents = status.get("v5", {}).get("intents") or []
    overlay_results = status.get("v5", {}).get("results") or []
    if not requests and not journal_events and not overlay_intents and not overlay_results:
        state["v5_trade_activity_initialized"] = True
        return False

    sent = state.setdefault("v5_trade_activity_sent", {})
    if not isinstance(sent, dict):
        sent = {}
        state["v5_trade_activity_sent"] = sent

    keyed_requests = [(v5_trade_request_key(request), request) for request in requests if isinstance(request, dict)]
    keyed_journal_events = [
        (v5_paper_journal_event_key(event), event)
        for event in journal_events
        if isinstance(event, dict)
    ]
    keyed_overlay: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]] = []
    for idx, (intent, result) in enumerate(pair_execution_intents_results(overlay_intents, overlay_results)):
        keyed_overlay.append((v5_overlay_activity_key(intent, result, idx), intent, result))
    if not state.get("v5_trade_activity_initialized") and not args.force_v5_trade_activity:
        for key, _request in keyed_requests:
            sent[key] = now_ts
        for key, _event in keyed_journal_events:
            sent[key] = now_ts
        for key, _intent, _result in keyed_overlay:
            sent[key] = now_ts
        state["v5_trade_activity_initialized"] = True
        state["last_v5_trade_activity_status"] = "primed"
        return False

    new_requests = [request for key, request in keyed_requests if key not in sent]
    new_journal_events = [event for key, event in keyed_journal_events if key not in sent]
    new_overlay = [(intent, result) for key, intent, result in keyed_overlay if key not in sent]
    if not new_requests and not new_journal_events and not new_overlay and not args.force_v5_trade_activity:
        state["last_v5_trade_activity_status"] = "skipped_duplicate"
        return False

    new_overlay_intents = [intent for intent, _result in new_overlay if intent is not None]
    new_overlay_results = [result for _intent, result in new_overlay if result is not None]
    message = format_v5_trade_activity_details(
        {
            "trade_requests": new_requests or ([] if new_journal_events else requests),
            "paper_journal_events": new_journal_events or ([] if new_requests else journal_events),
            "intents": new_overlay_intents,
            "results": new_overlay_results,
        }
    )
    client.send(message)
    for key, _request in keyed_requests:
        sent[key] = now_ts
    for key, _event in keyed_journal_events:
        sent[key] = now_ts
    for key, _intent, _result in keyed_overlay:
        sent[key] = now_ts
    state["v5_trade_activity_initialized"] = True
    state["last_v5_trade_activity_status"] = "sent"
    state["last_v5_trade_activity_sent_at"] = iso_now()
    return True


def run_once(args: argparse.Namespace, client: TelegramClient, state: dict[str, Any]) -> bool:
    handle_telegram_commands(args, client, state)
    status = collect_status(args)
    sent = False
    now_ts = time.time()
    prune_alert_state(state, now_ts)
    prune_v5_trade_activity_state(state, now_ts)
    last_summary = float(state.get("last_summary_sent_at", 0) or 0)

    if maybe_send_v5_trade_activity(args, client, state, status, now_ts):
        sent = True

    if args.scope == "v5":
        args.disable_summary = True

    if not args.disable_summary and (args.force_summary or now_ts - last_summary >= args.summary_seconds):
        summary_message = build_summary(status, args.scope)
        summary_digest = summary_dedupe_digest(summary_message, args.scope)
        summary_digest_key = f"last_summary_content_digest_{args.scope}"
        if args.force_summary or state.get(summary_digest_key) != summary_digest:
            client.send(summary_message)
            state[summary_digest_key] = summary_digest
            state["last_summary_dedupe_status"] = "sent"
            sent = True
        else:
            state["last_summary_dedupe_status"] = "skipped_duplicate"
            state["last_summary_skipped_duplicate_at"] = iso_now()
        state["last_summary_sent_at"] = now_ts

    if scope_enabled(args.scope, "v6") and not args.disable_macro_report:
        macro_slot = scheduled_macro_report_due(state, args.macro_report_times)
        if macro_slot:
            client.send(build_report_response(["BTCUSDT", "ETHUSDT"]))
            state["last_macro_report_sent_at"] = now_ts
            state["last_macro_report_slot"] = macro_slot
            sent = True

    if scope_enabled(args.scope, "v5") and not args.disable_daily_news:
        news_slot = scheduled_daily_news_due(state, args.daily_news_times)
        if news_slot:
            client.send(build_daily_news_report(args))
            state["last_daily_news_sent_at"] = now_ts
            state["last_daily_news_slot"] = news_slot
            sent = True

    if not args.disable_alerts:
        for key, message in build_alerts(status, args.scope):
            if should_send_alert(state, key, args.alert_repeat_seconds):
                client.send(message)
                mark_alert_sent(state, key)
                sent = True

    state["last_checked_at"] = iso_now()
    state["last_status_digest"] = short_hash(status)
    return sent


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram monitor for Somnus V5 pipeline and V6 execution.")
    parser.add_argument("--loop", action="store_true", help="Run continuously.")
    parser.add_argument("--once", action="store_true", help="Run one check and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print Telegram messages instead of sending.")
    parser.add_argument("--force-summary", action="store_true", help="Send summary regardless of summary interval.")
    parser.add_argument("--send-test", action="store_true", help="Send a test message and exit.")
    parser.add_argument("--send-news-report", action="store_true", help="Send one daily news report and exit.")
    parser.add_argument(
        "--send-chart-test",
        nargs="*",
        help="Render and send one chart. Usage: --send-chart-test HYPEUSDT 15m 672",
    )
    parser.add_argument("--simulate-command", help="Build one command response without reading Telegram updates.")
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("TS_TELEGRAM_MONITOR_INTERVAL_SECONDS", "10")))
    parser.add_argument("--summary-seconds", type=int, default=int(os.getenv("TS_TELEGRAM_MONITOR_SUMMARY_SECONDS", "7200")))
    parser.add_argument("--alert-repeat-seconds", type=int, default=int(os.getenv("TS_TELEGRAM_MONITOR_ALERT_REPEAT_SECONDS", "3600")))
    parser.add_argument("--v5-stale-seconds", type=int, default=int(os.getenv("TS_TELEGRAM_MONITOR_V5_STALE_SECONDS", "1800")))
    parser.add_argument("--v6-stale-seconds", type=int, default=int(os.getenv("TS_TELEGRAM_MONITOR_V6_STALE_SECONDS", "180")))
    parser.add_argument("--v6-distill-stale-seconds", type=int, default=int(os.getenv("TS_TELEGRAM_MONITOR_V6_DISTILL_STALE_SECONDS", "7200")))
    parser.add_argument("--macro-report-times", default=os.getenv("TS_TELEGRAM_MONITOR_MACRO_REPORT_TIMES", "07:30,22:00"))
    parser.add_argument("--daily-news-times", default=os.getenv("TS_TELEGRAM_MONITOR_DAILY_NEWS_TIMES", "09:00"))
    parser.add_argument("--daily-news-lookback-hours", type=int, default=int(os.getenv("TS_TELEGRAM_MONITOR_DAILY_NEWS_LOOKBACK_HOURS", "24")))
    parser.add_argument(
        "--disable-summary",
        action="store_true",
        default=str(os.getenv("TS_TELEGRAM_MONITOR_DISABLE_SUMMARY", "")).lower() in {"1", "true", "yes", "on"},
        help="Disable automatic V5/V6 status summary broadcasts; commands and alerts still work.",
    )
    parser.add_argument(
        "--disable-macro-report",
        action="store_true",
        default=str(os.getenv("TS_TELEGRAM_MONITOR_DISABLE_MACRO_REPORT", "")).lower() in {"1", "true", "yes", "on"},
        help="Disable scheduled BTC/ETH macro report broadcasts; manual report commands still work.",
    )
    parser.add_argument(
        "--disable-daily-news",
        action="store_true",
        default=str(os.getenv("TS_TELEGRAM_MONITOR_DISABLE_DAILY_NEWS", "")).lower() in {"1", "true", "yes", "on"},
        help="Disable scheduled daily opennews/opentwitter broadcasts; manual news commands still work.",
    )
    parser.add_argument(
        "--disable-alerts",
        action="store_true",
        default=str(os.getenv("TS_TELEGRAM_MONITOR_DISABLE_ALERTS", "")).lower() in {"1", "true", "yes", "on"},
        help="Disable stale-service and execution alert broadcasts.",
    )
    parser.add_argument(
        "--disable-commands",
        action="store_true",
        default=str(os.getenv("TS_TELEGRAM_MONITOR_DISABLE_COMMANDS", "")).lower() in {"1", "true", "yes", "on"},
        help="Disable manual Telegram command responses while keeping scheduled alerts/activity broadcasts.",
    )
    parser.add_argument(
        "--execution-only-commands",
        action="store_true",
        default=env_flag("TS_TELEGRAM_MONITOR_EXECUTION_ONLY_COMMANDS", True),
        help="Keep V5/V6 bots execution-only; manual p/fc/report/news commands are handed off to Xuejian.",
    )
    parser.add_argument(
        "--enable-manual-commands",
        dest="execution_only_commands",
        action="store_false",
        help="Allow this bot to answer manual p/fc/report/news commands again.",
    )
    parser.add_argument(
        "--disable-intel-commands",
        action="store_true",
        default=str(os.getenv("TS_TELEGRAM_MONITOR_DISABLE_INTEL_COMMANDS", "")).lower() in {"1", "true", "yes", "on"},
        help="Disable manual macro/report/news command responses; trade, position, and chart commands still work.",
    )
    parser.add_argument(
        "--require-command-mention",
        action="store_true",
        default=str(os.getenv("TS_TELEGRAM_MONITOR_REQUIRE_COMMAND_MENTION", "true")).lower() not in {"0", "false", "no", "off"},
        help="Only respond to manual commands that explicitly @mention this bot.",
    )
    parser.add_argument(
        "--no-require-command-mention",
        dest="require_command_mention",
        action="store_false",
        help="Allow manual commands without an explicit @mention.",
    )
    parser.add_argument(
        "--v5-trade-activity",
        action="store_true",
        default=str(os.getenv("TS_TELEGRAM_MONITOR_V5_TRADE_ACTIVITY", "")).lower() in {"1", "true", "yes", "on"},
        help="Broadcast V5 trade activity only when new open/add/reduce/close requests appear.",
    )
    parser.add_argument(
        "--force-v5-trade-activity",
        action="store_true",
        default=str(os.getenv("TS_TELEGRAM_MONITOR_FORCE_V5_TRADE_ACTIVITY", "")).lower() in {"1", "true", "yes", "on"},
        help="Send current V5 trade activity once even if it was already seen.",
    )
    parser.add_argument("--scope", choices=("all", "v5", "v6"), default=os.getenv("TS_TELEGRAM_MONITOR_SCOPE", "all"))
    parser.add_argument("--state-file", type=Path, default=Path(os.getenv("TS_TELEGRAM_MONITOR_STATE", str(DEFAULT_STATE_FILE))))
    parser.add_argument("--env-file", type=Path, default=Path(os.getenv("TS_TELEGRAM_ENV_FILE", str(ROOT / ".env.local"))))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    load_dotenv(args.env_file)

    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_BOT_TOKEN") or ""
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TG_CHAT_ID") or ""
    client = TelegramClient(token, chat_id, dry_run=args.dry_run)

    if args.send_test:
        client.send(f"[Somnus监控] Telegram测试成功\n项目目录={ROOT}\n时间={iso_now()}")
        return 0

    if args.send_news_report:
        client.send(build_daily_news_report(args))
        return 0

    if args.send_chart_test is not None:
        values = args.send_chart_test or []
        symbol = normalize_symbol(values[0] if len(values) >= 1 else "HYPEUSDT")
        interval = normalize_interval(values[1] if len(values) >= 2 else "15m") or "15m"
        focus_period = parse_focus_period(values[2]) if len(values) >= 3 else None
        chart_payload = build_chart_image(symbol, interval, focus_period)
        client.send_photo(Path(chart_payload["output"]), caption=format_chart_caption(chart_payload))
        return 0

    if args.simulate_command:
        if args.disable_commands:
            return 0
        if getattr(args, "execution_only_commands", True) and is_manual_command_text(args.simulate_command):
            client.send(command_help())
            return 0
        if parse_news_command_text(args.simulate_command):
            if args.disable_intel_commands:
                return 0
            client.send(build_daily_news_report(args))
            return 0
        report_symbols = parse_report_command_text(args.simulate_command)
        if report_symbols is not None:
            if args.disable_intel_commands:
                return 0
            client.send(build_report_response(report_symbols))
            return 0
        position_target = parse_position_command_text(args.simulate_command)
        if position_target:
            if position_target == "HELP":
                client.send("用法：p v5 或 p v6")
            else:
                client.send(build_position_response(position_target, collect_status(args)))
            return 0
        chart_command = parse_chart_command_text(args.simulate_command)
        if chart_command and chart_command[0] != "HELP":
            symbol, interval, focus_period = chart_command
            chart_payload = build_chart_image(symbol, interval, focus_period)
            client.send_photo(Path(chart_payload["output"]), caption=format_chart_caption(chart_payload))
        else:
            client.send(build_command_response(args.simulate_command))
        return 0

    if not args.dry_run and (not token or not chat_id):
        print("缺少 TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID；请写入 .env.local 或环境变量", file=sys.stderr)
        return 2

    state = load_json(args.state_file)
    if args.once or not args.loop:
        run_once(args, client, state)
        if not args.dry_run:
            save_json(args.state_file, state)
        return 0

    while True:
        try:
            run_once(args, client, state)
            if not args.dry_run:
                save_json(args.state_file, state)
        except Exception as exc:
            print(f"[{iso_now()}] 监控异常：{exc}", file=sys.stderr)
        time.sleep(max(10, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
