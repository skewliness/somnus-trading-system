"""
OpenNews WSS 实时情报监听器
============================

连接 6551 WebSocket 推送端点，按 Somnus 情报层规则过滤信息流：
- 信息流：6551news / Reuters / 金十数据 / 方程式新闻 / Bloomberg / A16Z / Binance 官方 / 新闻聚合
- 上币公告：binance / hyperliquid / bithumb / upbit / robinhood / bybit / okx / coinbase
- 链上：Hyper 巨鲸交易 / Hyper 大额建仓
- 行情异动：涨跌幅 / 资金费率 / 费率差价 / 大额清算 / OI 异动 / 主力持仓异动

输出继续写入 skill_bridges/opennews_wss.json，供 V5/V6 信号层和雪见读取。
内容保留原文，雪见负责中文解释。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import ssl
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WSS_URL_TEMPLATE = "wss://ai.6551.io/open/news_wss?token={token}"
MIN_SCORE_THRESHOLD = 60
URGENT_LISTING_SCORE = 80
URGENT_MARKET_SCORE = 85
MAX_BUFFER_SIZE = 80
WSS_CACHE_RETENTION_HOURS = 24
WSS_CACHE_RETENTION_SECONDS = WSS_CACHE_RETENTION_HOURS * 60 * 60
CACHE_PRUNE_INTERVAL_SECONDS = 300
HEALTH_INTERVAL_SECONDS = 300
RECONNECT_BASE_SECONDS = 5
RECONNECT_MAX_SECONDS = 60

DEFAULT_FILTER_CONFIG: dict[str, Any] = {
    "version": 1,
    "description": "Somnus/Xuejian OpenNews WSS filter. Binance/OKX native feeds are primary; 6551 WSS supplements missing intelligence.",
    "display_preferences": {
        "language": "original",
        "xuejian_translate_to": "zh-CN",
        "content_density": "professional",
        "dedupe_repeated_content": True,
    },
    "primary_source_policy": {
        "prefer": ["binance", "okx"],
        "fallback": "6551_opennews_wss",
        "note": "If a Binance/OKX native bridge already provides the same event, prefer native bridge semantics; WSS still keeps non-duplicate supplemental intelligence.",
    },
    "min_score": {
        "info_stream": 60,
        "listing": 70,
        "onchain": 68,
        "market": 68,
    },
    "dedupe_window_seconds": 1800,
    "cache_retention_hours": WSS_CACHE_RETENTION_HOURS,
    "info_stream_sources": [
        "6551news",
        "6551 news",
        "reuters",
        "路透",
        "jin10",
        "金十",
        "formula",
        "方程式",
        "bloomberg",
        "彭博",
        "a16z",
        "andreessen",
        "binance official",
        "binance公告",
        "币安官方",
        "news aggregator",
        "新闻聚合",
        "aggregator",
        "coindesk",
        "cointelegraph",
        "the block",
        "blockworks",
        "decrypt",
        "cnbc",
        "coinbase",
    ],
    "listing_sources": [
        "binance",
        "hyperliquid",
        "hyper liquid",
        "bithumb",
        "upbit",
        "robinhood",
        "bybit",
        "okx",
        "coinbase",
    ],
    "onchain_sources": [
        "hyper",
        "hyperliquid",
        "hyperdash",
    ],
    "onchain_keywords": [
        "whale",
        "巨鲸",
        "大额",
        "large position",
        "large build",
        "建仓",
        "大额建仓",
        "增仓",
        "开仓",
        "position increase",
    ],
    "market_keywords": [
        "涨跌幅",
        "price change",
        "price move",
        "funding",
        "资金费率",
        "费率差价",
        "funding spread",
        "liquidation",
        "清算",
        "大额清算",
        "open interest",
        "oi",
        "oi异动",
        "持仓异动",
        "主力持仓",
        "long short",
        "多空比",
        "coinbase premium",
        "coinbase premium index",
        "usdc supply",
        "exchange usdc",
        "stablecoin inflow",
        "stablecoin supply",
    ],
    "priority_watch_keywords": [
        "ibit",
        "coinbase premium",
        "coinbase premium index",
        "usdc supply",
        "exchange usdc",
        "stablecoin inflow",
        "mubadala",
        "sovereign wealth fund",
        "family office",
        "microstrategy",
        "mstr",
        "michael saylor",
        "metaplanet",
        "mara",
        "marathon digital",
        "strive",
        "semler",
        "gamestop",
        "bitcoin treasury",
        "btc treasury",
    ],
    "ignored_engine_types": ["meme"],
}


def run_news_wss(config: Any) -> None:
    token = (
        os.environ.get("OPENNEWS_PRIMARY_TOKEN")
        or os.environ.get("OPENNEWS_TOKEN")
        or os.environ.get("OPEN_TOKEN")
        or getattr(config, "opennews_token", "")
        or ""
    )
    if not token:
        logger.error("OPENNEWS token not set, cannot start WSS listener")
        return
    data_dir = Path(getattr(config, "data_dir", "data/trading_system"))
    output_file = data_dir / "skill_bridges" / "opennews_wss.json"
    priority_hit_file = data_dir / "skill_bridges" / "opennews_wss_priority_hits.jsonl"
    trigger_file = data_dir / "news_urgent_trigger"
    health_file = data_dir / "news_wss_health.json"
    filter_config_file = data_dir / "hermes_intel" / "opennews_wss_filter_config.json"
    filter_config = _load_filter_config(filter_config_file)
    listener = NewsWSSListener(
        token=token,
        output_file=output_file,
        priority_hit_file=priority_hit_file,
        trigger_file=trigger_file,
        health_file=health_file,
        filter_config=filter_config,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("starting news WSS listener, output=%s config=%s", output_file, filter_config_file)
    asyncio.run(listener.run())


class NewsWSSListener:
    def __init__(
        self,
        token: str,
        output_file: Path,
        priority_hit_file: Path,
        trigger_file: Path,
        health_file: Path,
        filter_config: dict[str, Any] | None = None,
        max_buffer: int = MAX_BUFFER_SIZE,
    ) -> None:
        self.token = token
        self.output_file = output_file
        self.priority_hit_file = priority_hit_file
        self.trigger_file = trigger_file
        self.health_file = health_file
        self.filter_config = json.loads(json.dumps(DEFAULT_FILTER_CONFIG))
        if filter_config:
            _deep_update(self.filter_config, filter_config)
        self.max_buffer = int(self.filter_config.get("max_buffer_size") or max_buffer)
        self.cache_retention_seconds = _cache_retention_seconds(self.filter_config)
        self.buffer: list[dict[str, Any]] = self._load_existing_buffer()
        self.recent_event_keys: dict[str, float] = {}
        self._last_health_write = 0.0
        self._last_cache_prune = 0.0
        self._message_count = 0
        self._accepted_count = 0
        self._filtered_count = 0
        self._duplicate_count = 0
        self._connect_count = 0
        self._last_filtered_reason = ""

    async def run(self) -> None:
        self._maybe_prune_disk_caches(force=True)
        self._write_signal_file(datetime.now(timezone.utc).isoformat())
        backoff = RECONNECT_BASE_SECONDS
        while True:
            try:
                await self._connect_and_listen()
                backoff = RECONNECT_BASE_SECONDS
            except Exception as exc:
                self._write_health(status="disconnected", error=str(exc))
                logger.warning("WSS disconnected: %s, reconnecting in %ds", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX_SECONDS)

    async def _connect_and_listen(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed: pip install websockets")
            raise SystemExit(1)

        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        uri = WSS_URL_TEMPLATE.format(token=self.token)
        self._connect_count += 1
        logger.info("connecting to WSS (attempt #%d)", self._connect_count)

        async with websockets.connect(uri, ssl=ssl_ctx, open_timeout=15, close_timeout=5) as ws:
            subscribe_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "news.subscribe",
                "params": {
                    "engineTypes": {
                        "news": [],
                        "listing": [],
                        "onchain": [],
                        "market": [],
                    },
                    "hasCoin": True,
                },
            }
            await ws.send(json.dumps(subscribe_msg))
            self._write_health(status="connected")
            logger.info("subscribed to filtered engine types: news/listing/onchain/market")
            while True:
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=HEALTH_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    self._maybe_prune_disk_caches(force=True)
                    self._write_signal_file(datetime.now(timezone.utc).isoformat())
                    self._write_health(status="connected")
                    continue
                try:
                    data = json.loads(raw_msg)
                except (json.JSONDecodeError, TypeError):
                    continue
                self._handle_message(data)
                self._maybe_write_health()

    def _handle_message(self, data: dict[str, Any]) -> None:
        method = data.get("method", "")
        if method == "news.ai_update":
            self._process_event(data.get("params") or {}, is_ai=True)
        elif method == "news.update":
            self._process_event(data.get("params") or {}, is_ai=False)

    def _process_event(self, params: dict[str, Any], *, is_ai: bool) -> None:
        self._message_count += 1
        engine_type = str(params.get("engineType") or "news").lower()
        if engine_type in {str(x).lower() for x in self.filter_config.get("ignored_engine_types", [])}:
            self._filtered("ignored_engine")
            return

        classification = _classify_event(params, self.filter_config)
        if classification is None:
            self._filtered("source_or_type_not_in_allowlist")
            return

        category = classification["category"]
        score, signal = _event_score_and_signal(params, is_ai=is_ai, category=category)
        min_score = float((self.filter_config.get("min_score") or {}).get(category, MIN_SCORE_THRESHOLD))
        if score < min_score:
            self._filtered(f"score_below_{category}_threshold")
            return

        coins = _coins_from_params(params)
        if not coins:
            self._filtered("no_coin")
            return

        text = _event_text(params, limit=900)
        event_key = _event_dedupe_key(category, params, coins, text)
        if self._is_duplicate(event_key):
            self._duplicate_count += 1
            return

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        source_text = _source_text(params)
        entries: list[dict[str, Any]] = []
        for coin_info in coins:
            symbol_base = str(coin_info.get("symbol") or coin_info.get("coin") or "").strip().upper()
            if not symbol_base or len(symbol_base) > 12 or symbol_base in _NON_CRYPTO_SYMBOLS:
                continue
            symbol = symbol_base if symbol_base.endswith("USDT") else f"{symbol_base}USDT"
            entries.append(
                {
                    "symbol": symbol,
                    "hotness_score": round(score, 2),
                    "mention_count": 1,
                    "source": "opennews_wss",
                    "updated_at": now_iso,
                    "notes": f"{category}/{engine_type}: {_compact(text, 220)}, signal={signal}, source={classification.get('matched_source') or source_text or 'unknown'}",
                    "metadata": {
                        "category": category,
                        "engine_type": engine_type,
                        "news_type": str(params.get("newsType") or ""),
                        "signal": signal,
                        "matched_source": classification.get("matched_source"),
                        "matched_keyword": classification.get("matched_keyword"),
                        "quality_gate": classification.get("quality_gate"),
                        "quality_signal": classification.get("quality_signal"),
                        "source_text": source_text,
                        "event_key": event_key,
                        "language": (self.filter_config.get("display_preferences") or {}).get("language", "original"),
                        "content_density": (self.filter_config.get("display_preferences") or {}).get("content_density", "professional"),
                        "primary_source_policy": self.filter_config.get("primary_source_policy"),
                        "text_original": text,
                    },
                }
            )

        if not entries:
            self._filtered("no_valid_symbol")
            return

        self.buffer.extend(entries)
        self._prune_buffer()
        if classification.get("quality_gate") == "btc_institutional_left_buy":
            self._append_priority_hit(
                updated_at=now_iso,
                entries=entries,
                classification=classification,
                params=params,
                score=score,
                signal=signal,
                text=text,
                source_text=source_text,
            )
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer:]
        self._accepted_count += len(entries)
        self._write_signal_file(now_iso)
        self._check_urgent_trigger(score, category)
        logger.info(
            "accepted opennews_wss: category=%s score=%d coins=%s signal=%s source=%s",
            category,
            int(score),
            [entry["symbol"] for entry in entries[:8]],
            signal,
            classification.get("matched_source") or source_text or "unknown",
        )

    def _append_priority_hit(
        self,
        *,
        updated_at: str,
        entries: list[dict[str, Any]],
        classification: dict[str, str],
        params: dict[str, Any],
        score: float,
        signal: str,
        text: str,
        source_text: str,
    ) -> None:
        """Persist deterministic BTC institutional WSS hits within the 24h cache window."""
        self._prune_priority_hit_file()
        symbols = sorted({str(entry.get("symbol") or "") for entry in entries if entry.get("symbol")})
        if not symbols:
            return
        metadata = entries[0].get("metadata") or {}
        record = {
            "updated_at": updated_at,
            "symbols": symbols,
            "category": classification.get("category"),
            "quality_gate": classification.get("quality_gate"),
            "quality_signal": classification.get("quality_signal"),
            "matched_keyword": classification.get("matched_keyword"),
            "matched_source": classification.get("matched_source") or source_text,
            "score": round(score, 2),
            "signal": signal,
            "source_text": source_text,
            "event_key": str(metadata.get("event_key") or ""),
            "news_type": str(params.get("newsType") or ""),
            "text_original": _compact(text, 900),
        }
        self.priority_hit_file.parent.mkdir(parents=True, exist_ok=True)
        with self.priority_hit_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _load_existing_buffer(self) -> list[dict[str, Any]]:
        if not self.output_file.exists():
            return []
        try:
            payload = json.loads(self.output_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("failed to load WSS cache %s: %s", self.output_file, exc)
            return []
        signals = payload.get("signals") if isinstance(payload, dict) else []
        if not isinstance(signals, list):
            return []
        return self._records_within_retention(signals)[-self.max_buffer:]

    def _records_within_retention(
        self,
        records: list[Any],
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = now or datetime.now(timezone.utc)
        cutoff = current - timedelta(seconds=self.cache_retention_seconds)
        fresh: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            ts = _record_updated_at(record)
            if ts is None:
                continue
            if ts >= cutoff:
                fresh.append(record)
        return fresh

    def _prune_buffer(self, now: datetime | None = None) -> None:
        self.buffer = self._records_within_retention(self.buffer, now)[-self.max_buffer:]

    def _maybe_prune_disk_caches(self, *, force: bool = False, now: datetime | None = None) -> None:
        monotonic_now = time.time()
        if not force and monotonic_now - self._last_cache_prune < CACHE_PRUNE_INTERVAL_SECONDS:
            return
        self._last_cache_prune = monotonic_now
        current = now or datetime.now(timezone.utc)
        self._prune_buffer(current)
        self._prune_priority_hit_file(current)
        self._prune_trigger_file(current)

    def _prune_priority_hit_file(self, now: datetime | None = None) -> None:
        if not self.priority_hit_file.exists():
            return
        current = now or datetime.now(timezone.utc)
        kept: list[dict[str, Any]] = []
        changed = False
        try:
            lines = self.priority_hit_file.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            logger.warning("failed to read WSS priority cache %s: %s", self.priority_hit_file, exc)
            return
        for line in lines:
            raw = line.strip()
            if not raw:
                changed = True
                continue
            try:
                record = json.loads(raw)
            except Exception:
                changed = True
                continue
            if isinstance(record, dict) and self._records_within_retention([record], current):
                kept.append(record)
            else:
                changed = True
        if not changed and len(kept) == len(lines):
            return
        self.priority_hit_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.priority_hit_file.with_name(self.priority_hit_file.name + ".tmp")
        content = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in kept)
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.priority_hit_file)

    def _prune_trigger_file(self, now: datetime | None = None) -> None:
        if not self.trigger_file.exists():
            return
        current = now or datetime.now(timezone.utc)
        ts: datetime | None = None
        try:
            ts = _parse_wss_timestamp(self.trigger_file.read_text(encoding="utf-8").strip())
        except Exception:
            ts = None
        if ts is None:
            try:
                ts = datetime.fromtimestamp(self.trigger_file.stat().st_mtime, timezone.utc)
            except OSError:
                return
        if current - ts > timedelta(seconds=self.cache_retention_seconds):
            try:
                self.trigger_file.unlink()
            except FileNotFoundError:
                pass

    def _filtered(self, reason: str) -> None:
        self._filtered_count += 1
        self._last_filtered_reason = reason

    def _is_duplicate(self, event_key: str) -> bool:
        if not (self.filter_config.get("display_preferences") or {}).get("dedupe_repeated_content", True):
            return False
        now = time.time()
        window = float(self.filter_config.get("dedupe_window_seconds") or 1800)
        self.recent_event_keys = {key: ts for key, ts in self.recent_event_keys.items() if now - ts <= window}
        if event_key in self.recent_event_keys:
            return True
        self.recent_event_keys[event_key] = now
        return False

    def _write_signal_file(self, updated_at: str) -> None:
        self._prune_buffer()
        payload = {
            "bridge_name": "opennews_wss",
            "source": "opennews_wss",
            "updated_at": updated_at,
            "capability_category": "wss_realtime_intelligence",
            "display_preferences": self.filter_config.get("display_preferences", {}),
            "primary_source_policy": self.filter_config.get("primary_source_policy", {}),
            "filter_summary": {
                "accepted_count": self._accepted_count,
                "filtered_count": self._filtered_count,
                "duplicate_count": self._duplicate_count,
                "last_filtered_reason": self._last_filtered_reason,
                "allowed_categories": ["info_stream", "listing", "onchain", "market"],
                "cache_retention_hours": round(self.cache_retention_seconds / 3600, 4),
            },
            "signals": self.buffer[-self.max_buffer:],
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.output_file)

    def _check_urgent_trigger(self, score: float, category: str) -> None:
        urgent = False
        if category == "listing" and score >= URGENT_LISTING_SCORE:
            urgent = True
        elif category in {"market", "onchain"} and score >= URGENT_MARKET_SCORE:
            urgent = True
        if urgent:
            self.trigger_file.parent.mkdir(parents=True, exist_ok=True)
            self.trigger_file.write_text(
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                encoding="utf-8",
            )
            logger.warning("URGENT trigger written: category=%s score=%d", category, int(score))

    def _maybe_write_health(self) -> None:
        now = time.time()
        if now - self._last_health_write < HEALTH_INTERVAL_SECONDS:
            return
        self._maybe_prune_disk_caches()
        self._write_signal_file(datetime.now(timezone.utc).isoformat())
        self._write_health(status="connected")

    def _write_health(self, *, status: str, error: str = "") -> None:
        self._last_health_write = time.time()
        health = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "message_count": self._message_count,
            "accepted_count": self._accepted_count,
            "filtered_count": self._filtered_count,
            "duplicate_count": self._duplicate_count,
            "buffer_size": len(self.buffer),
            "cache_retention_hours": round(self.cache_retention_seconds / 3600, 4),
            "connect_count": self._connect_count,
            "last_filtered_reason": self._last_filtered_reason,
            "error": error,
            "filter_config_version": self.filter_config.get("version"),
        }
        self.health_file.parent.mkdir(parents=True, exist_ok=True)
        self.health_file.write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_retention_seconds(config: dict[str, Any]) -> float:
    raw_hours = config.get("cache_retention_hours", WSS_CACHE_RETENTION_HOURS)
    try:
        hours = float(raw_hours)
    except (TypeError, ValueError):
        hours = float(WSS_CACHE_RETENTION_HOURS)
    if hours <= 0:
        hours = float(WSS_CACHE_RETENTION_HOURS)
    return hours * 60 * 60


def _parse_wss_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        try:
            return datetime.fromtimestamp(seconds, timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return _parse_wss_timestamp(int(text))
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                parsed = None  # type: ignore[assignment]
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _record_updated_at(record: dict[str, Any]) -> datetime | None:
    metadata_obj = record.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
    for key in ("updated_at", "received_at", "created_at", "timestamp", "ts", "time"):
        parsed = _parse_wss_timestamp(record.get(key))
        if parsed is not None:
            return parsed
        parsed = _parse_wss_timestamp(metadata.get(key))
        if parsed is not None:
            return parsed
    return None


def _load_filter_config(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_FILTER_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        return json.loads(json.dumps(DEFAULT_FILTER_CONFIG))
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("failed to read filter config %s: %s; using defaults", path, exc)
        return json.loads(json.dumps(DEFAULT_FILTER_CONFIG))
    merged = json.loads(json.dumps(DEFAULT_FILTER_CONFIG))
    _deep_update(merged, loaded)
    return merged


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


_BTC_CONTEXT_TERMS = (
    "btc", "bitcoin", "比特币", "ibit", "bitcoin etf", "spot bitcoin etf", "bitcoin treasury", "btc treasury"
)
_BTC_BUY_TERMS = (
    "buy", "buys", "bought", "purchase", "purchases", "purchased", "acquire", "acquires", "acquired",
    "add", "adds", "added", "increase", "increases", "increased", "accumulate", "accumulates", "accumulated",
    "加仓", "购买", "买入", "增持", "囤币"
)
_BTC_CONFIRMED_BUY_TERMS = (
    "buys", "bought", "purchase", "purchases", "purchased", "acquire", "acquires", "acquired",
    "adds", "added", "increase", "increases", "increased", "accumulate", "accumulates", "accumulated",
    "加仓", "购买", "买入", "增持", "囤币"
)
_BTC_INTENT_TERMS = (
    "signal", "signals", "tease", "teases", "plan", "plans", "intend", "intends", "will buy", "to buy", "may buy", "another bitcoin buy", "预告", "计划", "暗示"
)
_BTC_HOLD_TERMS = (
    "hodl", "hold", "holds", "held", "retain", "retains", "retained", "not sell", "no sell", "won't sell",
    "100%", "all bitcoin", "full hodl", "全量", "锁仓", "留存", "不卖", "拒绝抛售"
)
_BTC_FLOW_TERMS = (
    "inflow", "inflows", "flow into", "flows into", "net inflow", "outflow", "outflows", "drain", "drains",
    "rebound", "rebounds", "recover", "recovers", "rise", "rises", "increase", "increases", "stop losing", "stops losing",
    "stabilize", "stabilizes", "turn positive", "turns positive", "turned positive", "flip positive", "flips positive",
    "negative", "positive", "discount", "premium widens", "premium narrows", "回流", "流入", "流出", "失血", "停止失血", "转正", "负值", "正值"
)
_PRIORITY_MARKET_KEYWORDS = {
    "coinbase premium",
    "coinbase premium index",
    "usdc supply",
    "exchange usdc",
    "stablecoin inflow",
    "stablecoin supply",
}


def _has_any(lower: str, terms: tuple[str, ...]) -> bool:
    return any(term in lower for term in terms)


def _btc_institutional_quality_signal(text: str) -> str:
    """Return deterministic BTC institutional/flow signal kind, or empty for keyword-only noise."""
    lower = _norm(text)
    if not lower:
        return ""

    if "coinbase premium" in lower or "premium index" in lower or "coinbase溢价" in lower:
        return "coinbase_premium_state" if _has_any(lower, _BTC_FLOW_TERMS) else ""

    stablecoin_context = "usdc" in lower or "stablecoin" in lower or "稳定币" in lower
    exchange_context = "exchange" in lower or "reserve" in lower or "supply" in lower or "交易所" in lower or "供应" in lower or "余额" in lower
    if stablecoin_context:
        return "usdc_exchange_flow_state" if exchange_context and _has_any(lower, _BTC_FLOW_TERMS) else ""

    btc_context = _has_any(lower, _BTC_CONTEXT_TERMS)
    buy_context = _has_any(lower, _BTC_BUY_TERMS)
    confirmed_buy_context = _has_any(lower, _BTC_CONFIRMED_BUY_TERMS)
    intent_context = buy_context and _has_any(lower, _BTC_INTENT_TERMS)
    hold_context = _has_any(lower, _BTC_HOLD_TERMS)

    if any(term in lower for term in ("mubadala", "sovereign wealth fund", "family office", "abu dhabi", "主权财富", "家族办公室")):
        if btc_context and (buy_context or "holding" in lower or "holdings" in lower or "stake" in lower or "disclosed" in lower or "持仓" in lower):
            return "sovereign_family_office_btc_buy"
        return ""

    strategy_entity = "microstrategy" in lower or "mstr" in lower or "saylor" in lower or ("strategy" in lower and btc_context)
    if strategy_entity:
        if btc_context and confirmed_buy_context:
            return "strategy_btc_confirmed_buy"
        if btc_context and intent_context:
            return "strategy_btc_buy_intent"
        return ""

    if "metaplanet" in lower:
        return "metaplanet_btc_buy" if btc_context and (buy_context or "holding" in lower or "holdings" in lower or "持仓" in lower) else ""

    if "mara" in lower or "marathon digital" in lower:
        return "mara_btc_hodl" if btc_context and (hold_context or buy_context or "mined" in lower or "production" in lower or "treasury" in lower) else ""

    if any(term in lower for term in ("strive", "semler", "gamestop")):
        if btc_context and (buy_context or hold_context or "treasury" in lower or "covered call" in lower or "备兑" in lower):
            return "cross_company_btc_treasury"
        return ""

    return ""


def _classify_event(params: dict[str, Any], config: dict[str, Any]) -> dict[str, str] | None:
    engine_type = str(params.get("engineType") or "news").lower()
    blob = _norm(" ".join([_source_text(params), _event_text(params, limit=1400), str(params.get("newsType") or "")]))

    if engine_type == "listing" or "listing" in blob or "上币" in blob or "上线" in blob:
        matched = _first_match(blob, config.get("listing_sources", []))
        if matched:
            return {"category": "listing", "matched_source": matched, "matched_keyword": matched}
        return None

    if engine_type == "onchain":
        source = _first_match(blob, config.get("onchain_sources", []))
        keyword = _first_match(blob, config.get("onchain_keywords", []))
        if source and keyword:
            return {"category": "onchain", "matched_source": source, "matched_keyword": keyword}
        return None

    if engine_type == "market":
        keyword = _first_match(blob, config.get("market_keywords", []))
        if keyword:
            if _norm(keyword) in _PRIORITY_MARKET_KEYWORDS:
                quality_signal = _btc_institutional_quality_signal(blob)
                if not quality_signal:
                    return None
                return {
                    "category": "market",
                    "matched_source": _source_text(params),
                    "matched_keyword": keyword,
                    "quality_gate": "btc_institutional_left_buy",
                    "quality_signal": quality_signal,
                }
            return {"category": "market", "matched_source": _source_text(params), "matched_keyword": keyword}
        return None

    if engine_type == "news":
        priority_keyword = _first_match(blob, config.get("priority_watch_keywords", []))
        if priority_keyword:
            quality_signal = _btc_institutional_quality_signal(blob)
            if not quality_signal:
                return None
            return {
                "category": "info_stream",
                "matched_source": _source_text(params) or "priority_watch",
                "matched_keyword": priority_keyword,
                "quality_gate": "btc_institutional_left_buy",
                "quality_signal": quality_signal,
            }
        matched = _first_match(blob, config.get("info_stream_sources", []))
        if matched:
            return {"category": "info_stream", "matched_source": matched, "matched_keyword": matched}
        return None

    return None


def _event_score_and_signal(params: dict[str, Any], *, is_ai: bool, category: str) -> tuple[float, str]:
    ai_rating = params.get("aiRating") or {}
    if is_ai:
        score = float(ai_rating.get("score") or _engine_implied_score(category))
        signal = str(ai_rating.get("signal") or ai_rating.get("rating") or "neutral")
    else:
        score = _engine_implied_score(category)
        signal = "event"
    return score, signal


def _engine_implied_score(category: str) -> float:
    return float({
        "listing": 85,
        "onchain": 78,
        "market": 75,
        "info_stream": 62,
    }.get(category, 50))


def _coins_from_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    coins = params.get("coins") or params.get("symbols") or []
    if isinstance(coins, list):
        out: list[dict[str, Any]] = []
        for item in coins:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                out.append({"symbol": item})
        return out
    if isinstance(coins, str):
        return [{"symbol": part.strip()} for part in re.split(r"[,\s]+", coins) if part.strip()]
    return []


def _source_text(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "source",
        "sourceName",
        "publisher",
        "provider",
        "origin",
        "channel",
        "platform",
        "author",
        "site",
        "engineType",
        "newsType",
        "url",
        "link",
    ):
        value = params.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _event_text(params: dict[str, Any], *, limit: int) -> str:
    raw_parts: list[str] = []
    for key in ("title", "headline", "text", "content", "summary", "body", "description"):
        value = params.get(key)
        if value:
            raw_parts.append(str(value))
    text = _strip_html(" ".join(raw_parts))
    return _compact(text, limit)


def _event_dedupe_key(category: str, params: dict[str, Any], coins: list[dict[str, Any]], text: str) -> str:
    event_id = str(params.get("id") or params.get("eventId") or params.get("newsId") or "")
    symbols = ",".join(sorted(str(c.get("symbol") or c.get("coin") or "").upper() for c in coins))
    basis = "|".join([category, event_id, symbols, _source_text(params), _compact(text, 300)])
    return hashlib.sha1(basis.encode("utf-8", errors="ignore")).hexdigest()


def _first_match(blob: str, candidates: list[Any]) -> str:
    for item in candidates or []:
        candidate = _norm(str(item).strip())
        if not candidate:
            continue
        if re.fullmatch(r"[a-z0-9]{1,3}", candidate):
            if re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", blob):
                return str(item).strip()
            continue
        if candidate in blob:
            return str(item).strip()
    return ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _compact(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


_NON_CRYPTO_SYMBOLS = {"CL", "MU", "XYZ-CL", "XYZ-MU", "GC", "SI", "ES", "NQ", "SPX", "DXY"}
