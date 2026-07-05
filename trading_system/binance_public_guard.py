from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_BAN_FILE = Path("data/trading_system/binance_public_ban.json")
BAN_UNTIL_RE = re.compile(r"banned until\s+(\d{12,})", re.IGNORECASE)


class BinancePublicRateLimitError(RuntimeError):
    """Raised locally when Binance public endpoints are cooling down after HTTP 418."""


def ban_file_path() -> Path:
    return Path(os.getenv("TS_BINANCE_PUBLIC_BAN_FILE", str(DEFAULT_BAN_FILE)))


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_ban_until_ms(text: Any) -> int:
    match = BAN_UNTIL_RE.search(str(text or ""))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def read_binance_public_ban() -> dict[str, Any]:
    path = ban_file_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_binance_public_ban(payload: dict[str, Any]) -> None:
    path = ban_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    try:
        tmp.replace(path)
        path.chmod(0o600)
    finally:
        tmp.unlink(missing_ok=True)


def active_binance_public_ban(*, grace_seconds: float = 5.0) -> dict[str, Any]:
    payload = read_binance_public_ban()
    try:
        until_ms = int(payload.get("until_ms") or 0)
    except Exception:
        until_ms = 0
    if until_ms <= now_ms() + int(grace_seconds * 1000):
        return {}
    return payload


def binance_public_ban_remaining_seconds() -> float:
    payload = active_binance_public_ban(grace_seconds=0.0)
    if not payload:
        return 0.0
    return max(0.0, (int(payload.get("until_ms") or 0) - now_ms()) / 1000.0)


def assert_binance_public_not_banned(context: str = "") -> None:
    payload = active_binance_public_ban()
    if not payload:
        return
    remaining = binance_public_ban_remaining_seconds()
    until_text = str(payload.get("until_iso") or payload.get("until_ms") or "--")
    detail = f" context={context}" if context else ""
    raise BinancePublicRateLimitError(
        f"Binance public API cooling down after HTTP 418; remaining={remaining:.0f}s until={until_text}{detail}"
    )


def throttle_binance_public_request(min_interval_seconds: float | None = None) -> None:
    if os.name == "nt":
        return
    interval = float(
        min_interval_seconds
        if min_interval_seconds is not None
        else os.getenv("TS_BINANCE_PUBLIC_MIN_INTERVAL_SECONDS", "0.08")
    )
    if interval <= 0:
        return
    try:
        import fcntl
    except Exception:
        return

    path = ban_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    lock_path = path.with_name("binance_public_rate.lock")
    state_path = path.with_name("binance_public_rate.json")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        lock_path.chmod(0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        last_at = 0.0
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            last_at = float(payload.get("last_at_epoch") or 0.0)
        except Exception:
            last_at = 0.0
        elapsed = time.time() - last_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        tmp = state_path.with_name(f"{state_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        tmp.write_text(json.dumps({"last_at_epoch": time.time()}, separators=(",", ":")), encoding="utf-8")
        tmp.chmod(0o600)
        try:
            tmp.replace(state_path)
            state_path.chmod(0o600)
        finally:
            tmp.unlink(missing_ok=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def note_binance_public_response(status_code: int, text: Any, context: str = "") -> bool:
    until_ms = parse_ban_until_ms(text)
    if status_code == 418 and until_ms <= 0:
        until_ms = now_ms() + 10 * 60 * 1000
    if until_ms <= 0:
        return False

    current = read_binance_public_ban()
    try:
        current_until = int(current.get("until_ms") or 0)
    except Exception:
        current_until = 0
    until_ms = max(until_ms, current_until)
    payload = {
        "until_ms": until_ms,
        "until_epoch": until_ms / 1000.0,
        "until_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(until_ms / 1000.0)),
        "updated_at_epoch": time.time(),
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status_code": int(status_code),
        "context": context,
        "message": str(text or "")[:1000],
    }
    write_binance_public_ban(payload)
    return True
