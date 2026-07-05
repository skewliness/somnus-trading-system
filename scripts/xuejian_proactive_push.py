#!/usr/bin/env python3
"""Xuejian proactive Somnus intelligence push service.

Responsibilities:
- Push major geopolitical events from OpenNews WSS bridge.
- Push BTC/ETH macro report once per day at 09:00 Beijing time.

This script only reads Somnus intelligence files and sends Telegram messages
through Xuejian's bot token. It does not submit orders or mutate V5/V6 ledgers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

os.umask(0o077)

ROOT = Path(__file__).resolve().parents[1]
HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/Users/ma7/.hermes"))
BRIDGE_FILE = ROOT / "data/trading_system/skill_bridges/opennews_wss.json"
STATE_FILE = ROOT / "data/trading_system/hermes_intel/xuejian_proactive_push_state.json"
LOG_FILE = ROOT / "data/trading_system/logs/xuejian_proactive_push.log"
BEIJING_TZ = ZoneInfo("Asia/Shanghai")
GEO_EVENT_FINGERPRINT_TTL_SECONDS = int(os.environ.get("XUEJIAN_GEO_DEDUPE_TTL_SECONDS", str(6 * 3600)))

MAJOR_GEO_KEYWORDS = [
    "geopolitical", "war", "military", "missile", "drone", "strike", "airstrike",
    "attack", "ceasefire", "truce", "invasion", "sanction", "nuclear", "embargo",
    "strait", "hormuz", "persian gulf", "red sea", "houthi", "vessel", "safe passage",
    "iran", "israel", "gaza", "palestine", "lebanon", "hezbollah", "syria",
    "russia", "ukraine", "nato", "taiwan", "south china sea", "north korea",
    "地缘", "战争", "军事", "导弹", "无人机", "袭击", "空袭", "停火", "制裁", "核",
    "海峡", "霍尔木兹", "波斯湾", "红海", "胡塞", "船只", "安全通行",
    "伊朗", "以色列", "加沙", "巴勒斯坦", "黎巴嫩", "叙利亚", "俄罗斯", "乌克兰", "北约", "台湾", "南海", "朝鲜",
]
MAJOR_GEO_SOURCES = [
    "reuters", "bloomberg", "jin10", "金十", "6551news", "news aggregator", "新闻聚合",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now().isoformat()} {message}\n"
    LOG_FILE.open("a", encoding="utf-8").write(line)
    print(line, end="")


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def load_config_home_channel(path: Path) -> str:
    if not path.exists():
        return ""
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw.strip()
        if stripped.startswith("TELEGRAM_HOME_CHANNEL:"):
            value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            return value
    return ""


def load_telegram_config() -> tuple[str, str]:
    env = {}
    env.update(load_env_file(HERMES_HOME / ".env"))
    # xuejian-push is launched through macos_run_somnus_service.sh, which first
    # sources the trading-system .env.local. Keep proactive news on the bot used
    # by that runtime environment (currently @keke402bot) instead of forcing the
    # Hermes/snow bot from ~/.hermes/.env. XUEJIAN_* remains an explicit override.
    token = (
        os.environ.get("XUEJIAN_TELEGRAM_BOT_TOKEN")
        or os.environ.get("TELEGRAM_BOT_TOKEN")
        or env.get("XUEJIAN_TELEGRAM_BOT_TOKEN")
        or env.get("TELEGRAM_BOT_TOKEN")
        or ""
    )
    chat_id = (
        os.environ.get("XUEJIAN_TELEGRAM_CHAT_ID")
        or env.get("XUEJIAN_TELEGRAM_CHAT_ID")
        or os.environ.get("TELEGRAM_HOME_CHANNEL")
        or env.get("TELEGRAM_HOME_CHANNEL")
        or load_config_home_channel(HERMES_HOME / "config.yaml")
    )
    if not token:
        raise RuntimeError("missing TELEGRAM_BOT_TOKEN for Xuejian")
    if not chat_id:
        raise RuntimeError("missing TELEGRAM_HOME_CHANNEL for Xuejian")
    return token, chat_id


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {
            "created_at": utc_now().isoformat(),
            "sent_geo_event_keys": [],
            "sent_geo_event_fingerprints": [],
            "last_macro_beijing_date": "",
        }
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {
            "created_at": utc_now().isoformat(),
            "sent_geo_event_keys": [],
            "sent_geo_event_fingerprints": [],
            "last_macro_beijing_date": "",
        }


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sent = list(dict.fromkeys(state.get("sent_geo_event_keys") or []))[-500:]
    state["sent_geo_event_keys"] = sent
    state["sent_geo_event_fingerprints"] = prune_geo_fingerprints(state.get("sent_geo_event_fingerprints") or [])
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def prune_geo_fingerprints(raw_items: list[Any]) -> list[dict[str, str]]:
    cutoff = utc_now() - timedelta(seconds=GEO_EVENT_FINGERPRINT_TTL_SECONDS)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_items:
        if isinstance(raw, str):
            key = raw
            ts = utc_now()
        elif isinstance(raw, dict):
            key = str(raw.get("key") or "")
            ts = parse_dt(raw.get("sent_at")) or utc_now()
        else:
            continue
        if not key or key in seen or ts < cutoff:
            continue
        seen.add(key)
        out.append({"key": key, "sent_at": ts.astimezone(timezone.utc).isoformat()})
    return out[-500:]


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def compact(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def wrap_push_text(text: str, *, width: int = 48) -> list[str]:
    text = compact(text, 760)
    if not text:
        return ["--"]
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [text]


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def clean_news_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"(?i)^\s*[a-z_]+/[a-z_]+:\s*", "", text)
    text = re.sub(r"(?i)^\s*info_stream/news:\s*", "", text)
    text = re.sub(r"(?i),?\s*signal=event\b.*$", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+-\s+Post on X\s*$", "", text, flags=re.I)
    return compact(text, 760)


def source_label_zh(source: Any) -> str:
    text = str(source or "").strip()
    key = text.lower()
    labels = {
        "reuters": "路透",
        "bloomberg": "彭博",
        "jin10": "金十",
        "金十": "金十",
        "6551news": "6551新闻",
        "6551 news": "6551新闻",
        "formula": "方程式新闻",
    }
    return labels.get(key, text or "--")


def geo_news_to_chinese(text: str) -> str:
    news = clean_news_text(text)
    if not news:
        return "暂无可读新闻内容。"
    if has_chinese(news):
        return news

    lowered = news.lower()
    if (
        "iran" in lowered
        and "deputy foreign minister" in lowered
        and "strait of hormuz" in lowered
        and "sovereignty" in lowered
        and "not guaranteed" in lowered
    ):
        return "伊朗副外长表示，如果不考虑伊朗主权，霍尔木兹海峡的安全通行无法得到保障。"
    if (
        "iran" in lowered
        and ("fm deputy" in lowered or "foreign minister" in lowered)
        and "framework" in lowered
        and "passage through strait" in lowered
        and "coordination with iran" in lowered
    ):
        return "伊朗副外长表示，任何通过海峡的通行框架都必须与伊朗协调，否则指定航线可能暂停。"
    if "jet fuel refinery" in lowered and "iran war oil shock" in lowered:
        return "伊朗战争引发油价冲击后，航空燃油炼厂正紧急应对供应、成本和运输压力。"

    themes: list[str] = []
    if "strait of hormuz" in lowered or "hormuz" in lowered:
        themes.append("霍尔木兹海峡通行")
    if "iran" in lowered:
        themes.append("伊朗局势")
    if "israel" in lowered:
        themes.append("以色列局势")
    if "oil" in lowered or "crude" in lowered or "fuel" in lowered:
        themes.append("能源与油价")
    if "war" in lowered or "conflict" in lowered or "missile" in lowered or "attack" in lowered:
        themes.append("军事冲突风险")
    if "sanction" in lowered:
        themes.append("制裁风险")
    if "ceasefire" in lowered:
        themes.append("停火进展")
    if "tariff" in lowered or "trade" in lowered:
        themes.append("贸易摩擦")
    if "fed" in lowered or "yield" in lowered or "dollar" in lowered:
        themes.append("美元与美债")
    subject = "、".join(dict.fromkeys(themes[:4])) or "外媒突发消息"
    return f"外媒消息指向{subject}，需要关注事件后续对避险情绪、油价、美元和加密市场波动的影响。"


def event_key(item: dict[str, Any]) -> str:
    meta = item.get("metadata") or {}
    raw = meta.get("event_key") or "|".join([
        str(item.get("symbol") or ""),
        str(item.get("updated_at") or ""),
        str(item.get("notes") or meta.get("text_original") or ""),
    ])
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def event_fingerprint(item: dict[str, Any]) -> str:
    meta = item.get("metadata") or {}
    raw = str(item.get("notes") or meta.get("text_original") or "")
    raw = re.sub(r"(?i)info_stream/news:", "", raw)
    raw = re.sub(r"(?i)signal=event.*$", "", raw)
    raw = re.sub(r"(?i)\b(decrypt|the street|reuters|bloomberg|jin10|6551news):", "", raw)
    raw = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", raw.lower()).strip()
    return hashlib.sha1(raw[:240].encode("utf-8", errors="ignore")).hexdigest()


def event_theme_fingerprint(item: dict[str, Any]) -> str:
    meta = item.get("metadata") or {}
    raw = " ".join([
        str(item.get("notes") or ""),
        str(meta.get("text_original") or ""),
        str(meta.get("matched_source") or ""),
        str(meta.get("source_text") or ""),
    ]).lower()
    raw = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", raw)
    groups = {
        "iran_hormuz": ["iran", "hormuz", "persian gulf", "strait", "伊朗", "霍尔木兹", "波斯湾", "海峡"],
        "israel_gaza": ["israel", "gaza", "palestine", "lebanon", "hezbollah", "以色列", "加沙", "巴勒斯坦", "黎巴嫩"],
        "russia_ukraine": ["russia", "ukraine", "nato", "俄罗斯", "乌克兰", "北约"],
        "asia_geopolitics": ["taiwan", "south china sea", "north korea", "台湾", "南海", "朝鲜"],
        "energy_oil": ["oil", "crude", "fuel", "energy", "能源", "油价", "原油", "燃油"],
        "military": ["war", "military", "missile", "drone", "strike", "airstrike", "attack", "战争", "军事", "导弹", "袭击", "空袭"],
        "sanction": ["sanction", "embargo", "制裁", "禁运"],
        "ceasefire": ["ceasefire", "truce", "停火"],
        "safe_passage": ["safe passage", "passage", "route", "通行", "航线"],
        "usd_rates": ["fed", "yield", "dollar", "美元", "美债", "收益率"],
    }
    matched = [name for name, terms in groups.items() if any(term in raw for term in terms)]
    if not matched:
        return event_fingerprint(item)
    basis = "|".join(sorted(set(matched)))
    return hashlib.sha1(basis.encode("utf-8", errors="ignore")).hexdigest()


def is_major_geopolitical(item: dict[str, Any], *, max_age_minutes: int) -> bool:
    meta = item.get("metadata") or {}
    category = str(meta.get("category") or "").lower()
    if category != "info_stream":
        return False
    updated = parse_dt(item.get("updated_at"))
    if not updated or utc_now() - updated > timedelta(minutes=max_age_minutes):
        return False
    blob = " ".join([
        str(item.get("notes") or ""),
        str(meta.get("text_original") or ""),
        str(meta.get("matched_source") or ""),
        str(meta.get("source_text") or ""),
    ]).lower()
    source_ok = any(src.lower() in blob for src in MAJOR_GEO_SOURCES)
    keyword_ok = any(keyword.lower() in blob for keyword in MAJOR_GEO_KEYWORDS)
    return source_ok and keyword_ok


def load_wss_signals() -> list[dict[str, Any]]:
    if not BRIDGE_FILE.exists():
        return []
    try:
        payload = json.loads(BRIDGE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"wss_bridge_read_error={type(exc).__name__}: {str(exc)[:160]}")
        return []
    signals = payload.get("signals") or []
    if not isinstance(signals, list):
        return []
    return [item for item in signals if isinstance(item, dict)]


def send_telegram(text: str, *, dry_run: bool = False) -> None:
    token, chat_id = load_telegram_config()
    if dry_run:
        print("---DRY RUN TELEGRAM---")
        print(text)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = split_message(text, 3900)
    for chunk in chunks:
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": "true"},
            timeout=20,
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"telegram send failed {resp.status_code}: {resp.text[:200]}")


def split_message(text: str, limit: int) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.splitlines():
        add = len(line) + 1
        if current and length + add > limit:
            parts.append("\n".join(current))
            current = [line]
            length = add
        else:
            current.append(line)
            length += add
    if current:
        parts.append("\n".join(current))
    return parts


def format_geo_push(item: dict[str, Any]) -> str:
    return format_geo_push_batch([item])


def format_geo_push_batch(items: list[dict[str, Any]]) -> str:
    lines = ["🌐【重大地缘快讯】", ""]
    for idx, item in enumerate(items, 1):
        meta = item.get("metadata") or {}
        symbol = item.get("symbol") or "--"
        source = meta.get("matched_source") or meta.get("source_text") or item.get("source") or "--"
        raw_text = str(meta.get("text_original") or item.get("notes") or "")
        news_zh = geo_news_to_chinese(raw_text)
        lines.append(f"{idx}. 🧭 {symbol}｜来源={source_label_zh(source)}｜热度={item.get('hotness_score')}")
        lines.append("📰 中文快讯：")
        lines.extend(f"   {line}" for line in wrap_push_text(news_zh))
        lines.append("")
    lines.append("判断：")
    lines.extend(
        f"   {line}"
        for line in wrap_push_text("关注避险情绪、美元/油价/美债波动，以及 BTC/ETH 的短线波动放大。")
    )
    return "\n".join(lines)


def process_geo_events(state: dict[str, Any], *, dry_run: bool = False, max_age_minutes: int = 30, max_per_cycle: int = 3) -> int:
    sent = set(state.get("sent_geo_event_keys") or [])
    recent_fingerprints = prune_geo_fingerprints(state.get("sent_geo_event_fingerprints") or [])
    sent_fingerprints = {str(item.get("key") or "") for item in recent_fingerprints if isinstance(item, dict)}
    candidates: list[dict[str, Any]] = []
    for item in load_wss_signals():
        key = event_key(item)
        if key in sent:
            sent_fingerprints.add(event_theme_fingerprint(item))
            continue
        if event_theme_fingerprint(item) in sent_fingerprints:
            continue
        if is_major_geopolitical(item, max_age_minutes=max_age_minutes):
            candidates.append(item)
    candidates.sort(key=lambda x: str(x.get("updated_at") or ""))
    selected: list[dict[str, Any]] = []
    cycle_fingerprints: set[str] = set()
    for item in candidates[-max_per_cycle * 3:]:
        fp = event_theme_fingerprint(item)
        if fp in cycle_fingerprints:
            continue
        cycle_fingerprints.add(fp)
        selected.append(item)
        if len(selected) >= max_per_cycle:
            break
    if not selected:
        return 0
    send_telegram(format_geo_push_batch(selected), dry_run=dry_run)
    for item in selected:
        key = event_key(item)
        fp = event_theme_fingerprint(item)
        if not dry_run:
            sent.add(key)
            recent_fingerprints.append({"key": fp, "sent_at": utc_now().isoformat()})
        prefix = "dry_run_geo_event" if dry_run else "sent_geo_event"
        log(f"{prefix} symbol={item.get('symbol')} key={key[:12]}")
    if not dry_run:
        state["sent_geo_event_keys"] = list(sent)
        state["sent_geo_event_fingerprints"] = recent_fingerprints
    return len(selected)


def build_macro_report() -> str:
    cmd = [str(ROOT / "venv/bin/python"), str(ROOT / "scripts/hermes_somnus_command.py"), "report"]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
    env.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
    env.setdefault("ALL_PROXY", "http://127.0.0.1:7897")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"report failed rc={proc.returncode}: {proc.stderr[:500]}")
    report = proc.stdout.strip()
    return "\n".join(["📊【每日9点宏观播报】", report])


def maybe_send_daily_macro(state: dict[str, Any], *, dry_run: bool = False, force: bool = False) -> bool:
    now_bj = datetime.now(BEIJING_TZ)
    today = now_bj.strftime("%Y-%m-%d")
    due = force or (now_bj.hour >= 9 and state.get("last_macro_beijing_date") != today)
    if not due:
        return False
    text = build_macro_report()
    send_telegram(text, dry_run=dry_run)
    if not dry_run:
        state["last_macro_beijing_date"] = today
    log(f"sent_daily_macro date={today} force={force}")
    return True


def run_once(args: argparse.Namespace) -> int:
    state = load_state()
    geo_count = process_geo_events(state, dry_run=args.dry_run, max_age_minutes=args.geo_max_age_minutes)
    macro_sent = maybe_send_daily_macro(state, dry_run=args.dry_run, force=args.force_macro)
    if not args.dry_run:
        save_state(state)
    if args.dry_run:
        print(json.dumps({"geo_sent": geo_count, "macro_sent": macro_sent}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Xuejian proactive Somnus push service.")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--geo-max-age-minutes", type=int, default=30)
    parser.add_argument("--force-macro", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.loop:
        return run_once(args)
    while True:
        try:
            run_once(args)
        except Exception as exc:
            log(f"loop_error={type(exc).__name__}: {str(exc)[:300]}")
        time.sleep(max(10, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
