#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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


if os.getenv("SOMNUS_SKIP_LOCAL_ENV") != "1":
    load_local_env(ROOT / ".env.local")

from trading_system.config import StrategyConfig  # noqa: E402
from trading_system.persistence import atomic_write_json, load_json  # noqa: E402
from trading_system.v6_execution import V6BinanceClient, V6ExecutionConfig, V6ExecutionEngine, canonical_order_mode  # noqa: E402


DEFAULT_REPORT = Path("data/trading_system/pipeline_run_report.json")
DEFAULT_STATE_DIR = Path("data/trading_system/v5_v6_execution")
DEFAULT_AI_SELECT_FILE = Path(os.getenv("TS_BINANCE_AI_SELECT_FILE", "data/trading_system/skill_bridges/binance-ai-select.json"))
INVALID_SYMBOL_COOLDOWN_HOURS = float(os.getenv("TS_V5_INVALID_SYMBOL_COOLDOWN_HOURS", "24"))
ENTRY_WATCHABLE_BLOCK_REASONS = {
    "v5_long_wait_below_15m_resistance",
    "v5_short_wait_above_15m_support",
    "v5_long_upper_band_breakout_wait_retest",
    "v5_short_lower_band_breakdown_wait_retest",
    "v5_long_wait_15m_ma20_pullback",
    "v5_short_wait_15m_ma20_retest",
    "v5_ma7_15m_single_entry_blocked",
    "v5_mixed_chase_low_quality_blocked",
    "technical_entry_only_unconfirmed_15m_trigger",
    "long_high_chase_without_15m_reclaim",
    "short_low_chase_without_15m_rejection",
    "long_resistance_confluence_hold",
    "short_support_confluence_hold",
    "four_h_trend_waiting_for_15m_selector",
    "short_15m_requires_three_line_or_4h_pressure",
    "generic_indicator_1m_ma105_ma300_not_confirmed",
    "short_blocked_by_15m_bb96_210_lower_second_test_support",
    "long_blocked_by_15m_bb96_210_upper_second_test_pressure",
}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def v5_execution_lock(state_dir: Path, timeout_seconds: float = 240.0):
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "v5_execution.lock"
    handle = lock_path.open("w", encoding="utf-8")
    locked = False
    try:
        if os.name == "posix":
            import fcntl

            deadline = time.time() + max(1.0, float(timeout_seconds))
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except BlockingIOError:
                    if time.time() >= deadline:
                        raise TimeoutError(f"Timed out waiting for V5 execution lock: {lock_path}")
                    time.sleep(0.25)
        yield
    finally:
        if os.name == "posix" and locked:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


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


def candidate_metadata_by_symbol(report: dict[str, Any], limit: int) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    count = 0
    for row in report.get("candidate_basket") or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol or symbol in metadata:
            continue
        metadata[symbol] = {
            "symbol": symbol,
            "score": row.get("score"),
            "sentiment_score": row.get("sentiment_score"),
            "sentiment_mentions": row.get("sentiment_mentions"),
            "sentiment_source": row.get("sentiment_source") or row.get("source") or "",
            "rationale": list(row.get("rationale") or row.get("reasons") or [])[:12]
            if isinstance(row.get("rationale") or row.get("reasons"), list)
            else [],
            "notes": str(row.get("notes") or "")[:300],
        }
        count += 1
        if limit > 0 and count >= limit:
            break
    return metadata


def open_position_symbols(state_file: Path) -> list[str]:
    if not state_file.exists():
        return []
    try:
        state = load_json(state_file)
    except Exception:
        return []
    symbols: list[str] = []
    for row in state.get("positions") or []:
        symbol = str(row.get("symbol") or "").upper().strip()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def v5_entry_watchlist_enabled() -> bool:
    return env_bool("TS_V5_ENTRY_WATCHLIST_ENABLED", True)


def v5_entry_watchlist_path(state_dir: Path) -> Path:
    return Path(os.getenv("TS_V5_ENTRY_WATCHLIST_FILE", str(state_dir / "entry_watchlist.json")))


def load_v5_entry_watchlist(path: Path, *, include_expired: bool = False) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except Exception:
        return {}
    rows = payload.get("symbols") if isinstance(payload, dict) else {}
    if not isinstance(rows, dict):
        return {}
    now = datetime.now(timezone.utc)
    kept: dict[str, dict[str, Any]] = {}
    for key, value in rows.items():
        symbol = str(key or "").upper().strip()
        if not symbol or not isinstance(value, dict):
            continue
        expires_at = parse_iso_datetime(value.get("expires_at"))
        if not include_expired and expires_at and expires_at <= now:
            continue
        row = dict(value)
        row["symbol"] = symbol
        kept[symbol] = row
    return kept


def v5_entry_watch_symbols(path: Path, *, limit: int) -> list[str]:
    rows = load_v5_entry_watchlist(path)
    ordered = sorted(
        rows.values(),
        key=lambda item: str(item.get("updated_at") or item.get("first_seen_at") or ""),
        reverse=True,
    )
    symbols: list[str] = []
    for row in ordered:
        symbol = str(row.get("symbol") or "").upper().strip()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
        if limit > 0 and len(symbols) >= limit:
            break
    return symbols


def _signal_entry_block(signal: dict[str, Any]) -> dict[str, Any]:
    levels = signal.get("levels") if isinstance(signal.get("levels"), dict) else {}
    block = levels.get("entry_structure_filter") if isinstance(levels.get("entry_structure_filter"), dict) else {}
    return dict(block) if block else {}


def _watchable_entry_block(block: dict[str, Any]) -> bool:
    if not block.get("blocked"):
        return False
    if block.get("watchable"):
        return True
    return str(block.get("reason") or "") in ENTRY_WATCHABLE_BLOCK_REASONS


def update_v5_entry_watchlist(
    path: Path,
    summary: dict[str, Any],
    *,
    source_symbols: list[str],
    max_age_hours: float,
    max_symbols: int,
) -> dict[str, Any]:
    if not v5_entry_watchlist_enabled():
        return {"enabled": False, "symbols": [], "count": 0}
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=max(0.1, float(max_age_hours)))
    rows = load_v5_entry_watchlist(path)
    source_set = {str(symbol or "").upper().strip() for symbol in source_symbols if str(symbol or "").strip()}
    position_symbols = {
        str(item.get("symbol") or "").upper().strip()
        for item in summary.get("positions") or []
        if isinstance(item, dict) and str(item.get("symbol") or "").strip()
    }
    opened_symbols = {
        str((item.get("intent") or {}).get("symbol") or item.get("symbol") or "").upper().strip()
        for item in summary.get("results") or []
        if isinstance(item, dict)
        and str(item.get("status") or "") == "submitted"
        and isinstance(item.get("intent"), dict)
        and str((item.get("intent") or {}).get("action") or "") in {"open_long", "open_short"}
    }
    removed: list[str] = []
    for symbol in sorted(set(rows) & (position_symbols | opened_symbols)):
        rows.pop(symbol, None)
        removed.append(symbol)

    added_or_updated: list[str] = []
    for signal in summary.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        symbol = str(signal.get("symbol") or "").upper().strip()
        if not symbol or (source_set and symbol not in source_set) or symbol in position_symbols:
            continue
        block = _signal_entry_block(signal)
        if not _watchable_entry_block(block):
            continue
        old = rows.get(symbol, {})
        entry_context = block.get("entry_context") if isinstance(block.get("entry_context"), dict) else {}
        rows[symbol] = {
            "symbol": symbol,
            "first_seen_at": old.get("first_seen_at") or now.isoformat(),
            "updated_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "price": signal.get("price"),
            "side_bias": signal.get("side_bias"),
            "action": block.get("action"),
            "side": block.get("side"),
            "reason": block.get("reason"),
            "message": block.get("message"),
            "selected_entry_level": block.get("selected_entry_level") or entry_context.get("selected_entry_level"),
            "boundary": block.get("boundary"),
            "watch_source": "v5_strategy_entry_location",
        }
        added_or_updated.append(symbol)

    rows = {
        symbol: row
        for symbol, row in rows.items()
        if not ((parse_iso_datetime(row.get("expires_at")) or now) <= now)
    }
    ordered = sorted(
        rows.items(),
        key=lambda item: str(item[1].get("updated_at") or item[1].get("first_seen_at") or ""),
        reverse=True,
    )
    if max_symbols > 0:
        ordered = ordered[:max_symbols]
    final_rows = {symbol: row for symbol, row in ordered}
    payload = {
        "generated_at": now.isoformat(),
        "ttl_hours": max_age_hours,
        "symbols": final_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return {
        "enabled": True,
        "path": str(path),
        "count": len(final_rows),
        "symbols": list(final_rows),
        "added_or_updated": added_or_updated,
        "removed": removed,
    }


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def invalid_symbol_error_text(value: Any) -> bool:
    text = str(value or "").lower()
    return (
        '"code":-1121' in text
        or '"code": -1121' in text
        or "invalid symbol" in text
        or "无效交易对" in text
    )


def load_invalid_symbol_denylist(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except Exception:
        return {}
    rows = payload.get("symbols") if isinstance(payload, dict) else {}
    return {str(key).upper(): dict(value) for key, value in rows.items() if isinstance(value, dict)}


def active_invalid_symbols(path: Path) -> dict[str, dict[str, Any]]:
    now = datetime.now(timezone.utc)
    active: dict[str, dict[str, Any]] = {}
    for symbol, row in load_invalid_symbol_denylist(path).items():
        until = parse_iso_datetime(row.get("until"))
        if until and until > now:
            active[symbol] = row
    return active


def filter_invalid_symbol_cooldowns(
    symbols: list[str],
    open_symbols: list[str],
    denylist: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    open_set = {symbol.upper() for symbol in open_symbols}
    kept: list[str] = []
    skipped: list[str] = []
    for symbol in symbols:
        upper = symbol.upper()
        if upper in denylist and upper not in open_set:
            skipped.append(symbol)
        else:
            kept.append(symbol)
    return kept, skipped


def ai_select_filter_enabled() -> bool:
    return str(os.getenv("TS_V5_AI_SELECT_FINAL_FILTER", "0")).strip().lower() not in {"0", "false", "no", "off"}


def normalize_ai_select_symbol(value: Any) -> str:
    raw = str(value or "").upper().strip()
    if not raw:
        return ""
    raw = raw.replace("-", "").replace("_", "").replace("/", "")
    if raw.endswith("PERP"):
        raw = raw[:-4]
    if raw.endswith("USDT"):
        return raw
    if raw.endswith("USD"):
        return f"{raw[:-3]}USDT"
    return f"{raw}USDT"


def ai_select_text_state(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    bullish_markers = ("强烈看多", "看多", "看涨", "bullish", "strong_bullish", "positive", "buy", "up")
    bearish_markers = ("强烈看空", "看空", "看跌", "bearish", "strong_bearish", "negative", "sell", "down")
    neutral_markers = ("一般", "中性", "neutral", "mixed", "normal")
    if any(marker in text for marker in bullish_markers):
        return "bullish"
    if any(marker in text for marker in bearish_markers):
        return "bearish"
    if any(marker in text for marker in neutral_markers):
        return "neutral"
    return ""


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def ai_select_score(value: Any) -> float:
    text = str(value or "").strip().lower()
    if not text:
        return 0.0
    if any(marker in text for marker in ("强劲看涨", "强烈看多", "strong bullish", "strong_bullish")):
        return 100.0
    if any(marker in text for marker in ("金叉", "golden cross")):
        return 90.0
    if any(marker in text for marker in ("轻度看涨", "中性偏看涨", "slightly bullish", "mild bullish")):
        return 60.0
    if any(marker in text for marker in ("看涨", "看多", "bullish", "positive")):
        return 75.0
    if any(marker in text for marker in ("超买", "overbought")):
        return 80.0
    if any(marker in text for marker in ("强趋势", "strong trend")):
        return 100.0
    if any(marker in text for marker in ("中等趋势", "medium trend", "moderate trend")):
        return 70.0
    if any(marker in text for marker in ("初现趋势", "early trend", "weak trend")):
        return 55.0
    if any(marker in text for marker in ("中性", "一般", "neutral", "mixed", "normal")):
        return 50.0
    if any(marker in text for marker in ("强劲看跌", "强烈看空", "strong bearish", "strong_bearish")):
        return 0.0
    if any(marker in text for marker in ("轻度看跌", "中性偏看跌", "slightly bearish", "mild bearish")):
        return 40.0
    if any(marker in text for marker in ("看跌", "看空", "bearish", "negative")):
        return 25.0
    if any(marker in text for marker in ("超卖", "oversold")):
        return 35.0
    if any(marker in text for marker in ("无强势信号", "无趋势", "no trend", "no strong signal")):
        return 35.0
    return 0.0


def iter_ai_select_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("signals", "rows", "items", "list", "data", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = iter_ai_select_rows(value)
            if nested:
                return nested
    return []


def load_ai_select_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return {}, {"status": "missing", "path": str(path)}
    try:
        payload = load_json(path)
    except Exception as exc:
        return {}, {"status": "invalid_json", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    rows = iter_ai_select_rows(payload)
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = normalize_ai_select_symbol(
            first_present(row, ("symbol", "name", "baseAsset", "asset", "token", "coin", "币种", "名称"))
        )
        if symbol:
            by_symbol[symbol] = row
    return by_symbol, {
        "status": "ok" if by_symbol else "empty",
        "path": str(path),
        "row_count": len(rows),
        "symbol_count": len(by_symbol),
    }


def ai_select_confirmation(row: dict[str, Any]) -> dict[str, Any]:
    ai_rating = first_present(row, ("ai_rating", "aiRating", "ai_score", "aiScore", "rating", "AI评分"))
    community = first_present(row, ("community", "community_sentiment", "social", "social_sentiment", "socialSignal", "社群"))
    news = first_present(row, ("news", "news_sentiment", "newsSignal", "新闻"))
    kol = first_present(row, ("kol", "kol_sentiment", "kolSignal", "KOL"))
    social_vol = first_present(row, ("social_vol_24h", "socialVol24h", "24hSocialVol", "24H Social Vol", "24h Social Vol"))
    ma50_ma200 = first_present(row, ("ma50_ma200", "ma50Ma200", "ma50/ma200", "ma_50_200", "maSignal", "MA50/MA200", "MA50MA200"))
    adx = first_present(row, ("adx", "ADX"))
    macd = first_present(row, ("macd", "MACD"))
    rsi = first_present(row, ("rsi", "RSI"))
    states = {
        "ai_rating": ai_select_text_state(ai_rating),
        "community": ai_select_text_state(community),
        "news": ai_select_text_state(news),
        "kol": ai_select_text_state(kol),
    }
    sentiment_states = [states["community"], states["news"], states["kol"]]
    bullish_count = sum(1 for state in sentiment_states if state == "bullish")
    bearish_count = sum(1 for state in sentiment_states if state == "bearish")
    has_sentiment_vote = any(sentiment_states)
    min_bullish = max(1, int(float(os.getenv("TS_V5_AI_SELECT_MIN_BULLISH_SIGNALS", "1"))))
    allow_ai_rating_fallback = str(os.getenv("TS_V5_AI_SELECT_ALLOW_AI_RATING_FALLBACK", "1")).strip().lower() not in {"0", "false", "no", "off"}
    sentiment_confirmed = bullish_count >= min_bullish or (
        allow_ai_rating_fallback and not has_sentiment_vote and states["ai_rating"] == "bullish"
    )
    technical_values = {
        "ai_rating": ai_rating,
        "ma50_ma200": ma50_ma200,
        "adx": adx,
        "macd": macd,
        "rsi": rsi,
    }
    technical_scores = {key: ai_select_score(value) for key, value in technical_values.items()}
    technical_available = any(value not in (None, "") for value in (ma50_ma200, adx, macd, rsi))
    min_scores = {
        "ai_rating": float(os.getenv("TS_V5_AI_SELECT_MIN_AI_RATING_SCORE", "75")),
        "ma50_ma200": float(os.getenv("TS_V5_AI_SELECT_MIN_MA50_MA200_SCORE", "90")),
        "adx": float(os.getenv("TS_V5_AI_SELECT_MIN_ADX_SCORE", "90")),
        "macd": float(os.getenv("TS_V5_AI_SELECT_MIN_MACD_SCORE", "90")),
        "rsi": float(os.getenv("TS_V5_AI_SELECT_MIN_RSI_SCORE", "50")),
    }
    technical_confirmed = technical_available and all(
        technical_scores[key] >= min_scores[key]
        for key in min_scores
    )
    if technical_available and has_sentiment_vote:
        confirmed = technical_confirmed and sentiment_confirmed
        confirmation_mode = "technical_and_sentiment"
    elif technical_available:
        confirmed = technical_confirmed
        confirmation_mode = "technical_only"
    else:
        confirmed = sentiment_confirmed
        confirmation_mode = "sentiment_only"
    return {
        "confirmed": bool(confirmed),
        "confirmation_mode": confirmation_mode,
        "sentiment_confirmed": bool(sentiment_confirmed),
        "technical_confirmed": bool(technical_confirmed),
        "technical_available": bool(technical_available),
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "min_bullish": min_bullish,
        "states": states,
        "technical_scores": technical_scores,
        "technical_min_scores": min_scores,
        "raw": {
            "ai_rating": ai_rating,
            "community": community,
            "news": news,
            "kol": kol,
            "social_vol_24h": social_vol,
            "ma50_ma200": ma50_ma200,
            "adx": adx,
            "macd": macd,
            "rsi": rsi,
        },
    }


def filter_ai_select_final(
    symbols: list[str],
    open_symbols: list[str],
    path: Path,
) -> tuple[list[str], list[str], dict[str, Any]]:
    if not ai_select_filter_enabled():
        return symbols, [], {"enabled": False, "status": "disabled"}
    rows, meta = load_ai_select_rows(path)
    meta["enabled"] = True
    if meta.get("status") != "ok":
        meta["mode"] = "fail_open"
        meta["reason"] = "AI Select data unavailable; keep V5 running"
        return symbols, [], meta
    open_set = {symbol.upper() for symbol in open_symbols}
    kept: list[str] = []
    skipped: list[str] = []
    details: list[dict[str, Any]] = []
    for symbol in symbols:
        upper = symbol.upper()
        if upper in open_set:
            kept.append(symbol)
            continue
        row = rows.get(upper)
        if not row:
            skipped.append(symbol)
            details.append({"symbol": upper, "reason": "missing_from_ai_select"})
            continue
        confirmation = ai_select_confirmation(row)
        if confirmation["confirmed"]:
            kept.append(symbol)
            details.append({"symbol": upper, "reason": "confirmed", **confirmation})
        else:
            skipped.append(symbol)
            details.append({"symbol": upper, "reason": "not_bullish_confirmed", **confirmation})
    meta.update(
        {
            "mode": "final_candidate_filter",
            "input_count": len(symbols),
            "kept_count": len(kept),
            "skipped_count": len(skipped),
            "skipped_symbols": skipped,
            "details": details[:50],
        }
    )
    return kept, skipped, meta


def update_invalid_symbol_denylist(path: Path, symbols: list[str], *, reason: str) -> None:
    if not symbols:
        return
    rows = active_invalid_symbols(path)
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=INVALID_SYMBOL_COOLDOWN_HOURS)
    for symbol in symbols:
        upper = str(symbol or "").upper().strip()
        if not upper:
            continue
        rows[upper] = {
            "symbol": upper,
            "reason": reason,
            "created_at": now.isoformat(),
            "until": until.isoformat(),
            "cooldown_hours": INVALID_SYMBOL_COOLDOWN_HOURS,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"updated_at": now.isoformat(), "symbols": rows})


def collect_invalid_symbols_from_results(results: list[dict[str, Any]]) -> list[str]:
    symbols: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if not invalid_symbol_error_text(item.get("error") or item.get("original_error")):
            continue
        symbol = str(item.get("symbol") or ((item.get("intent") or {}).get("symbol") if isinstance(item.get("intent"), dict) else "")).upper().strip()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def seed_invalid_symbol_denylist_from_previous_report(path: Path, report_path: Path) -> list[str]:
    if not report_path.exists():
        return []
    try:
        payload = load_json(report_path)
    except Exception:
        return []
    symbols = collect_invalid_symbols_from_results([item for item in payload.get("results") or [] if isinstance(item, dict)])
    update_invalid_symbol_denylist(path, symbols, reason="previous_v5_overlay_invalid_symbol")
    return symbols


def archive_pending_v5_requests(config: StrategyConfig, archive_dir: Path) -> list[str]:
    request_dir = config.execution_request_dir
    request_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for path in sorted(request_dir.glob("execution_request_*.json"), key=lambda item: item.stat().st_mtime):
        target = archive_dir / path.name
        if target.exists():
            stem = path.stem
            target = archive_dir / f"{stem}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        shutil.move(str(path), str(target))
        moved.append(str(target))
    return moved


def v5_order_mode(config: StrategyConfig) -> str:
    raw = str(config.execution_order_mode or "auto").strip().lower()
    if raw == "auto":
        return "test_order" if config.execution_test_order else "live_order"
    aliases = {
        "test": "test_order",
        "test_order": "test_order",
        "dry_run": "dry_run",
        "demo": "testnet_order",
        "testnet": "testnet_order",
        "testnet_order": "testnet_order",
        "live": "live_order",
        "live_order": "live_order",
    }
    return aliases.get(raw, "test_order")


def configure_v6_with_v5_keys(
    v5: StrategyConfig,
    state_dir: Path,
    symbols: list[str],
    candidate_metadata: dict[str, dict[str, Any]] | None = None,
) -> V6ExecutionConfig:
    mode = v5_order_mode(v5)
    v6_mode = "dry_run" if mode == "test_order" else canonical_order_mode(mode)
    config = V6ExecutionConfig.from_env()
    config.order_mode = v6_mode
    config.position_mode = str(v5.binance_position_mode or "auto")
    config.symbols = symbols
    config.state_dir = state_dir
    config.state_file = state_dir / "state.json"
    config.signals_file = state_dir / "signals.jsonl"
    config.orders_file = state_dir / "orders.jsonl"
    config.wal_file = state_dir / "execution_wal.jsonl"
    config.latest_summary_file = state_dir / "latest_run_summary.json"
    config.closed_trade_review_queue_file = state_dir / "closed_trade_review_queue.jsonl"
    config.post_trade_training_samples_file = state_dir / "post_trade_training_samples.jsonl"
    config.technical_entry_only_symbols = tuple(symbols)
    config.v5_candidate_metadata_by_symbol = dict(candidate_metadata or {})
    config.kline_4h_supplement_enabled = env_bool("TS_V5_KLINE_4H_SUPPLEMENT_ENABLED", False)
    config.kline_4h_supplement_min_limit = int(os.getenv("TS_V5_KLINE_4H_SUPPLEMENT_MIN_LIMIT", "2880"))
    config.v5_no_news_momentum_tp_enabled = env_bool("TS_V5_NO_NEWS_MOMENTUM_TP_ENABLED", True)
    config.v5_momentum_strength_tp_enabled = env_bool("TS_V5_MOMENTUM_STRENGTH_TP_ENABLED", True)
    config.v5_momentum_guard_enabled = env_bool("TS_V5_MOMENTUM_GUARD_ENABLED", True)
    config.v5_momentum_guard_first_check_minutes = float(os.getenv("TS_V5_MOMENTUM_GUARD_FIRST_CHECK_MINUTES", "30"))
    config.v5_momentum_guard_first_min_roi_pct = float(os.getenv("TS_V5_MOMENTUM_GUARD_FIRST_MIN_ROI_PCT", "5"))
    config.v5_momentum_guard_first_fraction = float(os.getenv("TS_V5_MOMENTUM_GUARD_FIRST_FRACTION", "0.50"))
    config.v5_momentum_guard_mid_check_minutes = float(os.getenv("TS_V5_MOMENTUM_GUARD_MID_CHECK_MINUTES", "60"))
    config.v5_momentum_guard_mid_min_roi_pct = float(os.getenv("TS_V5_MOMENTUM_GUARD_MID_MIN_ROI_PCT", "10"))
    config.v5_momentum_guard_final_check_minutes = float(os.getenv("TS_V5_MOMENTUM_GUARD_FINAL_CHECK_MINUTES", "120"))
    config.v5_momentum_guard_final_min_roi_pct = float(os.getenv("TS_V5_MOMENTUM_GUARD_FINAL_MIN_ROI_PCT", "18"))
    config.v5_momentum_guard_oi_expand_pct = float(os.getenv("TS_V5_MOMENTUM_GUARD_OI_EXPAND_PCT", "1"))
    config.v5_momentum_guard_oi_refresh_seconds = float(os.getenv("TS_V5_MOMENTUM_GUARD_OI_REFRESH_SECONDS", "60"))
    config.v5_momentum_guard_smart_money_ratio = float(os.getenv("TS_V5_MOMENTUM_GUARD_SMART_MONEY_RATIO", "1.20"))
    config.v5_momentum_guard_snapshot_max_age_hours = float(os.getenv("TS_V5_MOMENTUM_GUARD_SNAPSHOT_MAX_AGE_HOURS", "0.25"))
    config.v5_momentum_guard_use_hyper_panel = env_bool("TS_V5_MOMENTUM_GUARD_USE_HYPER_PANEL", False)
    v5_forced_stop_loss_pct = float(os.getenv("TS_V5_FORCED_STOP_LOSS_PCT", os.getenv("TS_V5_STOP_LOSS_PCT", "0.00625")))
    config.stop_loss_pct = v5_forced_stop_loss_pct
    config.v5_forced_stop_loss_pct = v5_forced_stop_loss_pct
    config.v5_stop_loss_cooldown_minutes = float(os.getenv("TS_V5_STOP_LOSS_COOLDOWN_MINUTES", "100"))
    config.v5_profit_reentry_guard_enabled = env_bool("TS_V5_PROFIT_REENTRY_GUARD_ENABLED", True)
    config.v5_staged_entry_enabled = env_bool("TS_V5_STAGED_ENTRY_ENABLED", True)
    config.v5_initial_entry_notional_fraction = float(os.getenv("TS_V5_INITIAL_ENTRY_NOTIONAL_FRACTION", "0.30"))
    config.v5_scale_in_min_hold_minutes = float(os.getenv("TS_V5_SCALE_IN_MIN_HOLD_MINUTES", "0"))
    config.v5_scale_in_require_15m_ma20_retest = env_bool("TS_V5_SCALE_IN_REQUIRE_15M_MA20_RETEST", True)
    config.v5_scale_in_15m_ma20_retest_near_pct = float(os.getenv("TS_V5_SCALE_IN_15M_MA20_RETEST_NEAR_PCT", "0.6"))
    config.v5_scale_in_15m_ma20_max_extension_pct = float(os.getenv("TS_V5_SCALE_IN_15M_MA20_MAX_EXTENSION_PCT", "1.2"))
    config.v5_block_mixed_structure_initial_entry = env_bool("TS_V5_BLOCK_MIXED_STRUCTURE_INITIAL_ENTRY", True)
    config.v5_block_unclean_15m_ma20_initial_entry = env_bool("TS_V5_BLOCK_UNCLEAN_15M_MA20_INITIAL_ENTRY", True)
    config.max_scale_ins_per_position = int(os.getenv("TS_V5_MAX_SCALE_INS_PER_POSITION", "2"))
    config.scale_in_notional_fraction = float(os.getenv("TS_V5_SCALE_IN_NOTIONAL_FRACTION", "0.35"))
    config.scale_in_min_adverse_move_pct = float(os.getenv("TS_V5_SCALE_IN_MIN_ADVERSE_MOVE_PCT", "1.0"))
    config.scale_in_max_total_notional_multiplier = float(os.getenv("TS_V5_SCALE_IN_MAX_TOTAL_NOTIONAL_MULTIPLIER", "1.0"))
    config.v5_overlay_rules_enabled = env_bool("TS_V5_OVERLAY_RULES_ENABLED", True)
    config.v5_block_ma7_15m_single_entry = env_bool("TS_V5_BLOCK_MA7_15M_SINGLE_ENTRY", True)
    config.v5_block_mixed_avoid_low_quality_entry = env_bool("TS_V5_BLOCK_MIXED_AVOID_LOW_QUALITY_ENTRY", True)
    config.v5_entry_location_guard_enabled = env_bool("TS_V5_ENTRY_LOCATION_GUARD_ENABLED", True)
    config.v5_entry_location_guard_near_pct = float(os.getenv("TS_V5_ENTRY_LOCATION_GUARD_NEAR_PCT", "1.2"))
    config.v5_entry_15m_ma20_extension_pct = float(os.getenv("TS_V5_ENTRY_15M_MA20_EXTENSION_PCT", "4.0"))
    config.v5_disaster_stop_multiplier = float(os.getenv("TS_V5_DISASTER_STOP_MULTIPLIER", "2.0"))
    config.v5_reentry_wait_15m_candles = int(os.getenv("TS_V5_REENTRY_WAIT_15M_CANDLES", "4"))
    config.percent_price_retry_enabled = env_bool("TS_V5_PERCENT_PRICE_RETRY_ENABLED", True)
    config.percent_price_retry_wait_seconds = float(os.getenv("TS_V5_PERCENT_PRICE_RETRY_WAIT_SECONDS", "60"))
    config.percent_price_retry_max_deviation_pct = float(os.getenv("TS_V5_PERCENT_PRICE_RETRY_MAX_DEVIATION_PCT", "2.0"))
    config.percent_price_retry_cooldown_hours = float(os.getenv("TS_V5_PERCENT_PRICE_RETRY_COOLDOWN_HOURS", "0.5"))
    if v6_mode == "testnet_order":
        config.demo_base_url = str(v5.binance_testnet_base_url)
        config.demo_api_key = str(v5.binance_testnet_api_key or "")
        config.demo_api_secret = str(v5.binance_testnet_api_secret or "")
    elif v6_mode == "live_order":
        config.live_base_url = str(v5.base_url)
        config.live_api_key = str(v5.binance_api_key or "")
        config.live_api_secret = str(v5.binance_api_secret or "")
    return config


def filter_exchange_supported_symbols(config: V6ExecutionConfig, symbols: list[str]) -> tuple[list[str], list[str], str]:
    if not symbols:
        return [], [], ""
    client = V6BinanceClient(config)
    try:
        info = client.exchange_info()
    except Exception as exc:
        return symbols, [], f"{type(exc).__name__}: {exc}"
    finally:
        client.close()
    supported = {
        str(item.get("symbol") or "").upper()
        for item in info.get("symbols", [])
        if (
            isinstance(item, dict)
            and str(item.get("symbol") or "").strip()
            and str(item.get("status") or "").upper() == "TRADING"
        )
    }
    if not supported:
        return symbols, [], "exchangeInfo returned empty TRADING symbol list"
    kept: list[str] = []
    skipped: list[str] = []
    for symbol in symbols:
        if symbol.upper() in supported:
            kept.append(symbol)
        else:
            skipped.append(symbol)
    return kept, skipped, ""


def compact(
    summary: dict[str, Any],
    archived_requests: list[str],
    source_symbols: list[str],
    skipped_symbols: list[str] | None = None,
    skipped_invalid_symbols: list[str] | None = None,
    skipped_ai_select_symbols: list[str] | None = None,
    new_invalid_symbols: list[str] | None = None,
    symbol_filter_error: str = "",
    ai_select_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": summary.get("generated_at") or utc_now(),
        "mode": "v5_key_v6_trading_system",
        "order_mode": summary.get("order_mode"),
        "submit": summary.get("submit"),
        "source_symbols": source_symbols,
        "skipped_unsupported_symbols": skipped_symbols or [],
        "skipped_invalid_symbols": skipped_invalid_symbols or [],
        "skipped_ai_select_symbols": skipped_ai_select_symbols or [],
        "new_invalid_symbols": new_invalid_symbols or [],
        "symbol_filter_error": symbol_filter_error,
        "ai_select_filter": ai_select_filter or {},
        "symbols": summary.get("symbols"),
        "signal_count": len(summary.get("signals") or []),
        "intent_count": len(summary.get("intents") or []),
        "result_count": len(summary.get("results") or []),
        "archived_v5_request_count": len(archived_requests),
        "archived_v5_requests": archived_requests[-10:],
        "intents": [
            {
                "symbol": item.get("symbol"),
                "action": item.get("action"),
                "side": item.get("side"),
                "quantity_hint_usdt": item.get("quantity_hint_usdt"),
                "leverage": item.get("leverage"),
                "entry_price": item.get("entry_price"),
                "stop_loss_price": item.get("stop_loss_price"),
                "take_profit_price": item.get("take_profit_price"),
                "rationale": str(item.get("rationale") or "")[:500],
            }
            for item in summary.get("intents") or []
        ],
        "results": [
            {
                "symbol": item.get("symbol") or ((item.get("intent") or {}).get("symbol") if isinstance(item.get("intent"), dict) else ""),
                "status": item.get("status"),
                "action": ((item.get("intent") or {}).get("action") if isinstance(item.get("intent"), dict) else ""),
                "error": item.get("error") or item.get("original_error") or "",
                "reconcile_warning": item.get("reconcile_warning") or "",
                "position_reconcile": item.get("position_reconcile") if isinstance(item.get("position_reconcile"), dict) else {},
                "order_id": ((item.get("response") or {}).get("orderId") if isinstance(item.get("response"), dict) else item.get("order_id")),
                "executed_qty": ((item.get("response") or {}).get("executedQty") if isinstance(item.get("response"), dict) else item.get("executed_qty")),
            }
            for item in summary.get("results") or []
        ],
        "positions": summary.get("positions") or [],
    }


def run_locked(args: argparse.Namespace, state_dir: Path) -> dict[str, Any]:
    report = {"generated_at": utc_now(), "candidate_basket": []} if args.positions_only else json.loads(Path(args.report).read_text(encoding="utf-8"))
    v5_config = StrategyConfig()
    state_file = state_dir / "state.json"
    invalid_symbols_file = state_dir / "invalid_symbols.json"
    entry_watchlist_file = v5_entry_watchlist_path(state_dir)
    output_report = Path(args.output_report) if args.output_report else state_dir / "latest_v5_v6_execution_report.json"
    seed_invalid_symbol_denylist_from_previous_report(
        invalid_symbols_file,
        state_dir / "latest_v5_v6_execution_report.json",
    )
    candidate_metadata = candidate_metadata_by_symbol(report, args.limit) if not args.positions_only else {}
    fresh_candidate_symbols = candidate_symbols(report, args.limit)
    symbols = list(fresh_candidate_symbols)
    open_symbols = open_position_symbols(state_file)
    for symbol in open_symbols:
        if symbol not in symbols:
            symbols.append(symbol)
    watch_limit = int(os.getenv("TS_V5_ENTRY_WATCHLIST_MAX_SYMBOLS", "30"))
    if args.positions_only:
        include_entry_watchlist = False
        existing_watch_symbols: list[str] = []
    else:
        include_entry_watchlist = True
        existing_watch_symbols = v5_entry_watch_symbols(entry_watchlist_file, limit=watch_limit)
    watch_symbols = existing_watch_symbols if include_entry_watchlist else []
    for symbol in watch_symbols:
        if symbol not in symbols:
            symbols.append(symbol)
    source_symbols = list(symbols)
    symbols, skipped_invalid_symbols = filter_invalid_symbol_cooldowns(
        symbols,
        open_symbols,
        active_invalid_symbols(invalid_symbols_file),
    )
    symbols, skipped_ai_select_symbols, ai_select_filter = filter_ai_select_final(
        symbols,
        open_symbols,
        DEFAULT_AI_SELECT_FILE,
    )
    archived = archive_pending_v5_requests(v5_config, state_dir / "archived_v5_requests") if args.archive_v5_requests else []
    if not symbols:
        payload = {
            "generated_at": utc_now(),
            "mode": "v5_key_v6_trading_system",
            "run_scope": "positions_only" if args.positions_only else "scan_candidates",
            "order_mode": v5_order_mode(v5_config),
            "submit": bool(args.submit),
            "source_symbols": source_symbols,
            "skipped_unsupported_symbols": [],
            "skipped_invalid_symbols": skipped_invalid_symbols,
            "skipped_ai_select_symbols": skipped_ai_select_symbols,
            "new_invalid_symbols": [],
            "ai_select_filter": ai_select_filter,
            "symbols": [],
            "signal_count": 0,
            "intent_count": 0,
            "result_count": 0,
            "archived_v5_request_count": len(archived),
            "archived_v5_requests": archived[-10:],
            "intents": [],
            "results": [],
            "positions": [],
            "entry_watchlist": {
                "enabled": v5_entry_watchlist_enabled(),
                "path": str(entry_watchlist_file),
                "count": len(existing_watch_symbols),
                "symbols": existing_watch_symbols,
                "used_for_scan": include_entry_watchlist,
                "updated": False,
            },
        }
        state_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_report, payload)
        return payload
    config = configure_v6_with_v5_keys(v5_config, state_dir, symbols, candidate_metadata)
    filtered_symbols, skipped_symbols, symbol_filter_error = filter_exchange_supported_symbols(config, symbols)
    if filtered_symbols != symbols:
        config = configure_v6_with_v5_keys(v5_config, state_dir, filtered_symbols, candidate_metadata)
    symbols = filtered_symbols
    if not symbols:
        payload = {
            "generated_at": utc_now(),
            "mode": "v5_key_v6_trading_system",
            "run_scope": "positions_only" if args.positions_only else "scan_candidates",
            "order_mode": v5_order_mode(v5_config),
            "submit": bool(args.submit),
            "source_symbols": source_symbols,
            "skipped_unsupported_symbols": skipped_symbols,
            "skipped_invalid_symbols": skipped_invalid_symbols,
            "skipped_ai_select_symbols": skipped_ai_select_symbols,
            "new_invalid_symbols": [],
            "symbol_filter_error": symbol_filter_error,
            "ai_select_filter": ai_select_filter,
            "symbols": [],
            "signal_count": 0,
            "intent_count": 0,
            "result_count": 0,
            "archived_v5_request_count": len(archived),
            "archived_v5_requests": archived[-10:],
            "intents": [],
            "results": [],
            "positions": [],
            "entry_watchlist": {
                "enabled": v5_entry_watchlist_enabled(),
                "path": str(entry_watchlist_file),
                "count": len(existing_watch_symbols),
                "symbols": existing_watch_symbols,
                "used_for_scan": include_entry_watchlist,
                "updated": False,
            },
        }
        state_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_report, payload)
        return payload
    engine = V6ExecutionEngine(config)
    try:
        summary = engine.run_once(submit=bool(args.submit), commit_dry_run_state=False)
    finally:
        engine.close()
    new_invalid_symbols = collect_invalid_symbols_from_results([item for item in summary.get("results") or [] if isinstance(item, dict)])
    update_invalid_symbol_denylist(invalid_symbols_file, new_invalid_symbols, reason="v5_overlay_invalid_symbol")
    payload = compact(
        summary,
        archived,
        source_symbols,
        skipped_symbols,
        skipped_invalid_symbols,
        skipped_ai_select_symbols,
        new_invalid_symbols,
        symbol_filter_error,
        ai_select_filter,
    )
    payload["run_scope"] = "positions_only" if args.positions_only else "scan_candidates"
    payload["fresh_candidate_symbols"] = fresh_candidate_symbols
    if args.positions_only:
        payload["entry_watchlist"] = {
            "enabled": v5_entry_watchlist_enabled(),
            "path": str(entry_watchlist_file),
            "count": len(existing_watch_symbols),
            "symbols": existing_watch_symbols,
            "used_for_scan": include_entry_watchlist,
            "updated": False,
            "reason": "positions_only_does_not_refresh_entry_watchlist",
        }
    else:
        payload["entry_watchlist"] = update_v5_entry_watchlist(
            entry_watchlist_file,
            summary,
            source_symbols=fresh_candidate_symbols,
            max_age_hours=float(os.getenv("TS_V5_ENTRY_WATCHLIST_TTL_HOURS", "2")),
            max_symbols=watch_limit,
        )
    atomic_write_json(output_report, payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    state_dir = Path(args.state_dir)
    with v5_execution_lock(state_dir, float(args.lock_timeout_seconds)):
        return run_locked(args, state_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V6 trading system on V5 candidate basket using V5 Binance credentials.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("TS_V5_EXECUTION_CANDIDATE_LIMIT", os.getenv("TS_CANDIDATE_BASKET_LIMIT", "50"))),
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--archive-v5-requests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--positions-only", action="store_true", help="Manage existing V5 positions only; do not read candidate basket for new entries.")
    parser.add_argument("--output-report", default="", help="Write compact execution report to this path instead of the default V5 overlay report.")
    parser.add_argument("--lock-timeout-seconds", type=float, default=float(os.getenv("TS_V5_EXECUTION_LOCK_TIMEOUT_SECONDS", "240")))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({k: payload.get(k) for k in ["mode", "order_mode", "submit", "signal_count", "intent_count", "result_count", "archived_v5_request_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
