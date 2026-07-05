from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import requests

from trading_system.models import SentimentSignal


STABLECOIN_BASES = {"USD", "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDE", "DAI"}
OPENNEWS_ENGINE_WEIGHTS = {
    "listing": 1.0,
    "onchain": 0.9,
    "market": 0.85,
    "meme": 0.85,
    "news": 0.75,
    "prediction": 0.7,
}


@dataclass(slots=True)
class BridgeLoadResult:
    bridge_name: str
    records: list[SentimentSignal]
    metadata: dict


def _normalize_token_pool(tokens: str | list[str] | tuple[str, ...]) -> list[str]:
    raw_tokens = tokens if isinstance(tokens, (list, tuple)) else [tokens]
    output: list[str] = []
    for raw in raw_tokens:
        token = str(raw or "").strip()
        if token and token not in output:
            output.append(token)
    return output


def _is_rate_limited_payload(status_code: int, payload: dict | None = None, raw_text: str = "") -> bool:
    if status_code == 429:
        return True
    payload = payload or {}
    text = " ".join(
        str(payload.get(key) or "")
        for key in ("error", "msg", "message", "raw")
    )
    text = f"{text} {raw_text}".lower()
    markers = (
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
    return any(marker in text for marker in markers)


def _post_json_with_token_pool(
    url: str,
    tokens: str | list[str] | tuple[str, ...],
    body: dict,
    *,
    timeout: float = 25.0,
) -> tuple[dict, dict]:
    token_pool = _normalize_token_pool(tokens)
    if not token_pool:
        return {}, {"status": "missing_token", "token_pool_size": 0}
    last_meta: dict = {"status": "error", "token_pool_size": len(token_pool)}
    for index, token in enumerate(token_pool):
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
            try:
                raw_text = str(getattr(response, "text", "") or "")[:500]
            except Exception:
                raw_text = ""
            try:
                payload = response.json()
            except Exception:
                payload = {}
            status_code = int(getattr(response, "status_code", 200) or 200)
            rate_limited = _is_rate_limited_payload(status_code, payload, raw_text)
            if rate_limited and index + 1 < len(token_pool):
                last_meta = {
                    "status": "rate_limited_retry",
                    "http_status": status_code,
                    "token_pool_index": index,
                    "token_pool_size": len(token_pool),
                }
                continue
            if status_code >= 400:
                return {}, {
                    "status": "rate_limited" if rate_limited else "error",
                    "http_status": status_code,
                    "error": raw_text,
                    "token_pool_index": index,
                    "token_pool_size": len(token_pool),
                }
            if rate_limited:
                return {}, {
                    "status": "rate_limited",
                    "http_status": status_code,
                    "error": raw_text,
                    "token_pool_index": index,
                    "token_pool_size": len(token_pool),
                }
            return payload, {
                "status": "ok",
                "http_status": status_code,
                "token_pool_index": index,
                "token_pool_size": len(token_pool),
            }
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            status_code = int(getattr(response, "status_code", 0) or 0)
            raw_text = str(exc)
            rate_limited = _is_rate_limited_payload(status_code, None, raw_text)
            if rate_limited and index + 1 < len(token_pool):
                last_meta = {
                    "status": "rate_limited_retry",
                    "http_status": status_code,
                    "token_pool_index": index,
                    "token_pool_size": len(token_pool),
                }
                continue
            return {}, {
                "status": "rate_limited" if rate_limited else "error",
                "http_status": status_code,
                "error": raw_text,
                "token_pool_index": index,
                "token_pool_size": len(token_pool),
            }
    return {}, last_meta


class ManualSentimentBridge:
    name = "manual_sentiment"

    def load(self, path: Path) -> BridgeLoadResult:
        records: list[SentimentSignal] = []
        if not path.exists():
            return BridgeLoadResult(self.name, [], {"status": "missing", "path": str(path)})
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                symbol = (row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                records.append(
                    SentimentSignal(
                        symbol=symbol,
                        hotness_score=float(row.get("hotness_score") or 0.0),
                        mention_count=int(float(row.get("mention_count") or 0)),
                        source=(row.get("source") or self.name).strip() or self.name,
                        updated_at=_parse_dt(row.get("updated_at") or ""),
                        notes=(row.get("notes") or "").strip(),
                        earliest_strong_source=(row.get("earliest_strong_source") or "").strip(),
                        earliest_strong_timestamp=_parse_optional_dt(row.get("earliest_strong_timestamp") or ""),
                        okx_confirmed_at=_parse_optional_dt(row.get("okx_confirmed_at") or ""),
                        binance_confirmed_at=_parse_optional_dt(row.get("binance_confirmed_at") or ""),
                        signal_sequence=(row.get("signal_sequence") or "single_source").strip() or "single_source",
                        lead_time_minutes=_parse_optional_float(row.get("lead_time_minutes") or ""),
                    )
                )
        return BridgeLoadResult(self.name, records, {"status": "ok", "path": str(path), "count": len(records)})


class SquareFeedBridge:
    name = "square_feed"

    def __init__(self, max_age_minutes: int = 30, stale_hotness_multiplier: float = 0.25) -> None:
        self.max_age_minutes = max_age_minutes
        self.stale_hotness_multiplier = stale_hotness_multiplier

    def load(self, path: Path) -> BridgeLoadResult:
        if not path.exists():
            return BridgeLoadResult(self.name, [], {"status": "missing", "path": str(path)})
        raw_payload = path.read_text(encoding="utf-8").strip()
        if not raw_payload:
            return BridgeLoadResult(self.name, [], {"status": "empty", "path": str(path), "count": 0})
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            return BridgeLoadResult(self.name, [], {"status": "invalid_json", "path": str(path), "error": str(exc), "count": 0})
        summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
        captured_at = _parse_optional_dt(str(summary.get("captured_at") or payload.get("captured_at") or ""))
        now = datetime.now(timezone.utc)
        age_minutes = (now - captured_at).total_seconds() / 60.0 if captured_at is not None else None
        status = str(summary.get("status") or "").lower()
        is_stale = bool(status in {"stale", "error", "expired"} or (age_minutes is not None and age_minutes > self.max_age_minutes))
        freshness_multiplier = self.stale_hotness_multiplier if is_stale else 1.0
        symbol_mentions: dict[str, dict[str, float]] = {}
        for post in payload.get("posts", []):
            bot_filter = post.get("bot_filter", {})
            if bot_filter and not bool(bot_filter.get("keep", True)):
                continue
            raw_symbols = [str(symbol).upper() for symbol in post.get("trading_pairs", []) if str(symbol).strip()]
            regex_symbols = [str(symbol).upper() for symbol in post.get("bot_features", {}).get("regex_ticker_mentions", []) if str(symbol).strip()]
            symbols = list(dict.fromkeys([*raw_symbols, *regex_symbols]))
            if not symbols:
                continue
            engagement = float(post.get("bot_features", {}).get("engagement_total", 0.0))
            views = float(post.get("view_count", 0.0))
            view_ratio = float(post.get("bot_features", {}).get("engagement_view_ratio", 0.0))
            regex_boost = 0.35 * len(regex_symbols)
            base_weight = 1.0 + min(4.0, engagement / 25.0) + min(2.0, views / 20000.0) + min(3.0, view_ratio * 50.0) + regex_boost
            for symbol in symbols:
                slot = symbol_mentions.setdefault(symbol, {"mentions": 0.0, "weight": 0.0})
                mention_increment = 1.0 + (0.5 if symbol in regex_symbols else 0.0)
                slot["mentions"] += mention_increment
                slot["weight"] += base_weight + (0.35 if symbol in regex_symbols else 0.0)

        if not symbol_mentions:
            return BridgeLoadResult(self.name, [], {"status": "ok", "path": str(path), "count": 0})

        max_weight = max(item["weight"] for item in symbol_mentions.values())
        records = []
        for symbol, values in sorted(symbol_mentions.items()):
            hotness = round(((values["weight"] / max_weight) * 100.0) * freshness_multiplier, 2) if max_weight > 0 else 0.0
            records.append(
                SentimentSignal(
                    symbol=symbol,
                    hotness_score=hotness,
                    mention_count=max(1, int(round(values["mentions"]))),
                    source=self.name,
                    updated_at=now,
                    notes=(
                        "derived from square feed engagement + $ticker regex"
                        + (f"; stale_square_feed age_minutes={age_minutes:.1f}" if is_stale and age_minutes is not None else "")
                    ),
                )
            )
        return BridgeLoadResult(self.name, records, {"status": "stale" if is_stale else "ok", "path": str(path), "count": len(records), "age_minutes": age_minutes})


class SkillBridgeDirectory:
    name = "skill_bridge_directory"

    def load(self, path: Path) -> BridgeLoadResult:
        records: list[SentimentSignal] = []
        metadata = {"status": "missing", "path": str(path), "files": [], "skipped_stale": []}
        if not path.exists():
            return BridgeLoadResult(self.name, records, metadata)
        files = sorted(path.glob("*.json"))
        metadata = {"status": "ok", "path": str(path), "files": [str(item) for item in files], "skipped_stale": []}
        now = datetime.now(timezone.utc)
        max_age_hours = float(os.getenv("TS_SKILL_BRIDGE_MAX_AGE_HOURS", "48") or 48)
        for file_path in files:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            bridge_name = payload.get("bridge_name") or file_path.stem
            try:
                age_hours = max(0.0, (now - datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)).total_seconds() / 3600.0)
            except Exception:
                age_hours = 0.0
            if max_age_hours > 0 and age_hours > max_age_hours:
                metadata["skipped_stale"].append(
                    {
                        "bridge_name": bridge_name,
                        "path": str(file_path),
                        "age_hours": round(age_hours, 3),
                        "max_age_hours": max_age_hours,
                    }
                )
                continue
            for item in payload.get("signals", []):
                symbol = str(item.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                records.append(
                    SentimentSignal(
                        symbol=symbol,
                        hotness_score=float(item.get("hotness_score") or 0.0),
                        mention_count=int(float(item.get("mention_count") or 0)),
                        source=str(item.get("source") or bridge_name),
                        updated_at=now,
                        notes=str(item.get("notes") or ""),
                    )
                )
        metadata["count"] = len(records)
        return BridgeLoadResult(self.name, records, metadata)


class CoinGeckoTrendingBridge:
    name = "coingecko_trending"
    url = "https://api.coingecko.com/api/v3/search/trending"

    def load(self) -> BridgeLoadResult:
        try:
            response = requests.get(self.url, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return BridgeLoadResult(self.name, [], {"status": "error", "error": str(exc)})
        coins = payload.get("coins", [])
        now = datetime.now(timezone.utc)
        records: list[SentimentSignal] = []
        total = max(1, len(coins))
        for idx, item in enumerate(coins, start=1):
            coin = item.get("item") or {}
            symbol = str(coin.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            records.append(
                SentimentSignal(
                    symbol=f"{symbol}USDT",
                    hotness_score=round(max(0.0, 100.0 - ((idx - 1) * (80.0 / total))), 2),
                    mention_count=max(1, total - idx + 1),
                    source=self.name,
                    updated_at=now,
                    notes=f"coingecko_rank={idx}",
                )
            )
        return BridgeLoadResult(self.name, records, {"status": "ok", "count": len(records)})


class BinanceTopMoversBridge:
    name = "binance_top_movers"

    def __init__(self, base_url: str = "https://fapi.binance.com") -> None:
        self.base_url = base_url

    def load(self) -> BridgeLoadResult:
        try:
            response = requests.get(f"{self.base_url}/fapi/v1/ticker/24hr", timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return BridgeLoadResult(self.name, [], {"status": "error", "error": str(exc)})
        ranked = sorted(
            [item for item in payload if str(item.get("symbol") or "").endswith("USDT")],
            key=lambda item: float(item.get("priceChangePercent") or 0.0),
            reverse=True,
        )[:20]
        now = datetime.now(timezone.utc)
        records: list[SentimentSignal] = []
        total = max(1, len(ranked))
        for idx, item in enumerate(ranked, start=1):
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            move = float(item.get("priceChangePercent") or 0.0)
            volume = float(item.get("quoteVolume") or 0.0)
            hotness = min(100.0, max(0.0, move * 2.2) + min(25.0, volume / 20_000_000.0))
            records.append(
                SentimentSignal(
                    symbol=symbol,
                    hotness_score=round(hotness, 2),
                    mention_count=max(1, total - idx + 1),
                    source=self.name,
                    updated_at=now,
                    notes=f"24h_rank={idx};move={move:.2f};quote_volume={volume:.0f}",
                )
            )
        return BridgeLoadResult(self.name, records, {"status": "ok", "count": len(records)})


class OptionsDeliveryBridge:
    name = "options_delivery"
    binance_base_url = "https://eapi.binance.com"
    okx_base_url = "https://www.okx.com"

    def __init__(
        self,
        *,
        underlyings: tuple[str, ...] | list[str] | None = None,
        limit: int = 5,
        snapshot_path: Path | None = None,
        binance_base_url: str | None = None,
        okx_base_url: str | None = None,
    ) -> None:
        self.underlyings = tuple(dict.fromkeys(_normalize_option_underlying(item) for item in (underlyings or ("BTCUSDT", "ETHUSDT"))))
        self.underlyings = tuple(item for item in self.underlyings if item)
        self.limit = max(1, int(limit or 80))
        self.snapshot_path = snapshot_path
        if binance_base_url:
            self.binance_base_url = binance_base_url.rstrip("/")
        if okx_base_url:
            self.okx_base_url = okx_base_url.rstrip("/")

    def load(self) -> BridgeLoadResult:
        records: list[SentimentSignal] = []
        raw: dict[str, dict[str, object]] = {"binance": {}, "okx": {}}
        errors: list[dict[str, str]] = []
        for underlying in self.underlyings:
            try:
                binance_rows = self._fetch_binance(underlying)
                raw["binance"][underlying] = binance_rows
                signal = _build_binance_options_delivery_signal(underlying, binance_rows)
                if signal is not None:
                    records.append(signal)
            except Exception as exc:
                errors.append({"venue": "binance", "underlying": underlying, "error": f"{type(exc).__name__}: {exc}"})
            try:
                okx_rows = self._fetch_okx(underlying)
                raw["okx"][underlying] = okx_rows
                signal = _build_okx_options_delivery_signal(underlying, okx_rows)
                if signal is not None:
                    records.append(signal)
            except Exception as exc:
                errors.append({"venue": "okx", "underlying": underlying, "error": f"{type(exc).__name__}: {exc}"})

        status = "ok" if not errors else ("partial" if records else "error")
        metadata = {
            "status": status,
            "count": len(records),
            "underlyings": list(self.underlyings),
            "venue_count": 2,
            "errors": errors,
        }
        if self.snapshot_path is not None:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_path.write_text(
                json.dumps(
                    {
                        "bridge_name": self.name,
                        "summary": metadata,
                        "raw": raw,
                        "records": [
                            {
                                "symbol": record.symbol,
                                "hotness_score": record.hotness_score,
                                "mention_count": record.mention_count,
                                "source": record.source,
                                "updated_at": record.updated_at.isoformat(),
                                "notes": record.notes,
                            }
                            for record in records
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            metadata["snapshot_path"] = str(self.snapshot_path)
        return BridgeLoadResult(self.name, records, metadata)

    def _fetch_binance(self, underlying: str) -> list[dict]:
        response = requests.get(
            f"{self.binance_base_url}/eapi/v1/exerciseHistory",
            params={"underlying": underlying, "limit": self.limit},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload or []) if isinstance(payload, list) else []

    def _fetch_okx(self, underlying: str) -> list[dict]:
        response = requests.get(
            f"{self.okx_base_url}/api/v5/public/delivery-exercise-history",
            params={"instType": "OPTION", "uly": _okx_uly_from_underlying(underlying), "limit": self.limit},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return []
        if str(payload.get("code") or "0") != "0":
            raise RuntimeError(str(payload.get("msg") or payload.get("message") or payload))
        return list(payload.get("data") or [])


class OpenNewsBridge:
    name = "opennews"
    url = "https://ai.6551.io/open/news_search"

    def __init__(
        self,
        token: str | list[str] | tuple[str, ...] = "",
        lookback_hours: int = 24,
        page_limit: int = 20,
        max_pages: int = 2,
        snapshot_path: Path | None = None,
    ) -> None:
        self.tokens = _normalize_token_pool(token)
        self.token = self.tokens[0] if self.tokens else ""
        self.lookback_hours = lookback_hours
        self.page_limit = page_limit
        self.max_pages = max_pages
        self.snapshot_path = snapshot_path

    def load(self) -> BridgeLoadResult:
        payload = {"limit": self.page_limit, "page": 1, "hasCoin": True, "lookbackHours": self.lookback_hours}
        return self._fetch(payload, self.max_pages)

    def fetch_for_symbols(self, symbols: list[str]) -> BridgeLoadResult:
        bases = [_symbol_to_base(symbol) for symbol in symbols]
        bases = [item for item in dict.fromkeys(bases) if item]
        if not bases:
            return BridgeLoadResult(self.name, [], {"status": "skipped", "reason": "no_symbols"})
        payload = {"limit": self.page_limit, "page": 1, "coins": bases, "lookbackHours": self.lookback_hours}
        result = self._fetch(payload, self.max_pages)
        result.metadata["target_symbols"] = [f"{item}USDT" for item in bases]
        return result

    def _fetch(self, payload: dict, max_pages: int) -> BridgeLoadResult:
        if not self.tokens:
            return BridgeLoadResult(self.name, [], {"status": "missing_token", "token_pool_size": 0})
        articles: list[dict] = []
        token_meta: dict = {"token_pool_size": len(self.tokens)}
        for page in range(1, max(1, max_pages) + 1):
            body = dict(payload)
            body["page"] = page
            result, token_meta = _post_json_with_token_pool(self.url, self.tokens, body)
            if token_meta.get("status") != "ok":
                return BridgeLoadResult(self.name, [], {**token_meta, "page": page})
            batch = result.get("data") or []
            articles.extend(batch)
            if len(batch) < int(body.get("limit") or self.page_limit):
                break
        records = _build_opennews_records(articles)
        metadata = {"status": "ok", "count": len(records), "article_count": len(articles), **token_meta}
        if self.snapshot_path is not None:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_path.write_text(
                json.dumps(
                    {
                        "summary": metadata,
                        "articles": articles,
                        "records": [
                            {
                                "symbol": record.symbol,
                                "hotness_score": record.hotness_score,
                                "mention_count": record.mention_count,
                                "source": record.source,
                                "updated_at": record.updated_at.isoformat(),
                                "notes": record.notes,
                            }
                            for record in records
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            metadata["snapshot_path"] = str(self.snapshot_path)
        return BridgeLoadResult(self.name, records, metadata)


DEFAULT_OFFICIAL_TWITTER_QUERIES: dict[str, tuple[str, ...]] = {
    "JTO": ("from:jito_sol JTX",),
}


def _official_twitter_queries_for_base(base: str) -> list[str]:
    queries = list(DEFAULT_OFFICIAL_TWITTER_QUERIES.get(base.upper(), ()))
    raw = os.getenv("TS_OFFICIAL_TWITTER_QUERIES", "").strip()
    if not raw:
        return queries
    # Format: BASE=query one||query two;OTHER=from:official handle
    for chunk in raw.split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        if key.strip().upper() != base.upper():
            continue
        queries.extend(item.strip() for item in value.split("||") if item.strip())
    return list(dict.fromkeys(queries))


class OpenTwitterBridge:
    name = "opentwitter"
    url = "https://ai.6551.io/open/twitter_search"

    def __init__(self, token: str | list[str] | tuple[str, ...] = "", max_results: int = 8, snapshot_path: Path | None = None) -> None:
        self.tokens = _normalize_token_pool(token)
        self.token = self.tokens[0] if self.tokens else ""
        self.max_results = max_results
        self.snapshot_path = snapshot_path

    def fetch_for_symbols(self, symbols: list[str]) -> BridgeLoadResult:
        bases = [_symbol_to_base(symbol) for symbol in symbols]
        bases = [item for item in dict.fromkeys(bases) if item]
        if not bases:
            return BridgeLoadResult(self.name, [], {"status": "skipped", "reason": "no_symbols"})
        if not self.tokens:
            return BridgeLoadResult(self.name, [], {"status": "missing_token", "target_symbols": [f"{item}USDT" for item in bases], "token_pool_size": 0})

        tweets_by_symbol: dict[str, list[dict]] = {}
        token_meta: dict = {"token_pool_size": len(self.tokens)}
        failed_symbols: list[dict] = []
        failed_official_queries: list[dict] = []
        official_query_count = 0
        for base in bases:
            body = {
                "keywords": f"${base} OR {base}",
                "maxResults": self.max_results,
                "product": "Latest",
                "excludeReplies": True,
                "excludeRetweets": True,
            }
            result, token_meta = _post_json_with_token_pool(self.url, self.tokens, body)
            if token_meta.get("status") != "ok":
                failed_symbols.append({**token_meta, "symbol": f"{base}USDT"})
                continue
            tweets = list(result.get("data") or [])
            for query in _official_twitter_queries_for_base(base):
                official_body = {
                    "keywords": query,
                    "maxResults": min(8, self.max_results),
                    "product": "Latest",
                    "excludeReplies": False,
                    "excludeRetweets": False,
                }
                official_query_count += 1
                official_result, official_meta = _post_json_with_token_pool(self.url, self.tokens, official_body)
                if official_meta.get("status") != "ok":
                    failed_official_queries.append({**official_meta, "symbol": f"{base}USDT", "query": query})
                    continue
                for tweet in official_result.get("data") or []:
                    if not isinstance(tweet, dict):
                        continue
                    enriched = dict(tweet)
                    enriched["_somnus_source"] = "opentwitter_official"
                    enriched["_somnus_official_query"] = query
                    tweets.append(enriched)
            tweets_by_symbol[base] = tweets

        if not tweets_by_symbol:
            return BridgeLoadResult(
                self.name,
                [],
                {
                    "status": "error",
                    "target_symbols": [f"{item}USDT" for item in bases],
                    "failed_symbols": failed_symbols,
                    **token_meta,
                },
            )

        records = _build_opentwitter_records(tweets_by_symbol)
        metadata = {
            "status": "partial" if failed_symbols else "ok",
            "count": len(records),
            "target_symbols": [f"{item}USDT" for item in bases],
            "tweet_count": sum(len(items) for items in tweets_by_symbol.values()),
            "failed_symbols": failed_symbols,
            "official_query_count": official_query_count,
            "failed_official_queries": failed_official_queries,
            **token_meta,
        }
        if self.snapshot_path is not None:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_path.write_text(
                json.dumps(
                    {
                        "summary": metadata,
                        "tweets_by_symbol": tweets_by_symbol,
                        "records": [
                            {
                                "symbol": record.symbol,
                                "hotness_score": record.hotness_score,
                                "mention_count": record.mention_count,
                                "source": record.source,
                                "updated_at": record.updated_at.isoformat(),
                                "notes": record.notes,
                            }
                            for record in records
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            metadata["snapshot_path"] = str(self.snapshot_path)
        return BridgeLoadResult(self.name, records, metadata)


def _parse_dt(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.now(timezone.utc)


def _parse_optional_dt(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _parse_optional_float(value: str) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _symbol_to_base(symbol: str) -> str:
    token = str(symbol or "").strip().upper()
    if token.endswith("USDT"):
        token = token[:-4]
    if token in STABLECOIN_BASES:
        return ""
    return token if token.isalnum() else ""


def _normalize_option_underlying(value: str) -> str:
    token = str(value or "").strip().upper().replace("-", "")
    if token.endswith("USD") and not token.endswith("USDT"):
        token = f"{token[:-3]}USDT"
    if not token.endswith("USDT"):
        token = f"{token}USDT"
    base = token[:-4]
    if not base or base in STABLECOIN_BASES or not base.isalnum():
        return ""
    return token


def _okx_uly_from_underlying(underlying: str) -> str:
    base = _symbol_to_base(_normalize_option_underlying(underlying))
    return f"{base}-USD" if base else ""


def _option_expiry_dt(value) -> datetime:
    return _parse_any_dt(value)


def _option_delivery_score(total_rows: int, exercised_rows: int, timestamp: datetime) -> float:
    total_rows = max(0, int(total_rows or 0))
    exercised_rows = max(0, int(exercised_rows or 0))
    age_hours = abs((datetime.now(timezone.utc) - timestamp).total_seconds()) / 3600.0
    freshness_bonus = 10.0 if age_hours <= 48 else (5.0 if age_hours <= 168 else 0.0)
    score = 45.0 + min(18.0, total_rows / 4.0) + min(15.0, exercised_rows / 3.0) + freshness_bonus
    return round(min(72.0, max(35.0, score)), 2)


def _option_settlement_price(values: list[str]) -> str:
    for value in values:
        value = str(value or "").strip()
        if value:
            return value
    return "--"


def _build_binance_options_delivery_signal(underlying: str, rows: list[dict]) -> SentimentSignal | None:
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if not valid_rows:
        return None
    latest_ts = max(_option_expiry_dt(row.get("expiryDate")) for row in valid_rows)
    latest_rows = [row for row in valid_rows if _option_expiry_dt(row.get("expiryDate")) == latest_ts] or valid_rows
    exercised = sum(1 for row in latest_rows if str(row.get("strikeResult") or "").upper() == "REALISTIC_VALUE_STRICKEN")
    expired = max(0, len(latest_rows) - exercised)
    settlement = _option_settlement_price([str(row.get("realStrikePrice") or "") for row in latest_rows])
    symbols = [str(row.get("symbol") or "") for row in latest_rows[:4] if row.get("symbol")]
    notes = (
        f"Binance期权交割 {underlying} 到期={latest_ts.date().isoformat()} 交割价={settlement} "
        f"行权/实值={exercised} 失效/虚值={expired} 样本={len(latest_rows)}"
        + (f" 代表={','.join(symbols)}" if symbols else "")
    )
    return SentimentSignal(
        symbol=underlying,
        hotness_score=_option_delivery_score(len(latest_rows), exercised, latest_ts),
        mention_count=max(1, len(latest_rows)),
        source="options_delivery_binance",
        updated_at=latest_ts,
        notes=notes[:400],
    )


def _build_okx_options_delivery_signal(underlying: str, rows: list[dict]) -> SentimentSignal | None:
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if not valid_rows:
        return None
    latest_ts = max(_option_expiry_dt(row.get("ts")) for row in valid_rows)
    latest_rows = [row for row in valid_rows if _option_expiry_dt(row.get("ts")) == latest_ts] or valid_rows
    details: list[dict] = []
    for row in latest_rows:
        details.extend(item for item in (row.get("details") or []) if isinstance(item, dict))
    if not details:
        return None
    exercised = sum(1 for item in details if str(item.get("type") or "").lower() == "exercised")
    expired = sum(1 for item in details if str(item.get("type") or "").lower().startswith("expired"))
    settlement = _option_settlement_price([str(item.get("px") or "") for item in details])
    symbols = [str(item.get("insId") or "") for item in details[:4] if item.get("insId")]
    notes = (
        f"OKX期权交割 {underlying} 到期={latest_ts.date().isoformat()} 交割价={settlement} "
        f"行权={exercised} 失效={expired} 样本={len(details)}"
        + (f" 代表={','.join(symbols)}" if symbols else "")
    )
    return SentimentSignal(
        symbol=underlying,
        hotness_score=_option_delivery_score(len(details), exercised, latest_ts),
        mention_count=max(1, len(details)),
        source="options_delivery_okx",
        updated_at=latest_ts,
        notes=notes[:400],
    )


def _normalize_symbol(symbol: str) -> str:
    base = _symbol_to_base(symbol)
    return f"{base}USDT" if base else ""


def _parse_any_dt(value) -> datetime:
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    text = str(value or "").strip()
    if text.isdigit():
        ts = float(text)
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return _parse_dt(text)


def _freshness_multiplier(timestamp: datetime) -> float:
    age_hours = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600.0)
    if age_hours <= 3:
        return 1.0
    if age_hours <= 12:
        return 0.9
    if age_hours <= 24:
        return 0.8
    return 0.65


def _build_opennews_records(articles: list[dict]) -> list[SentimentSignal]:
    grouped: dict[str, dict] = {}
    for article in articles:
        timestamp = _parse_any_dt(article.get("ts"))
        rating = article.get("aiRating") or {}
        engine_type = str(article.get("engineType") or "news").strip().lower() or "news"
        source_name = f"opennews_{engine_type}"
        score = float(rating.get("score") or 35.0)
        weighted_score = score * OPENNEWS_ENGINE_WEIGHTS.get(engine_type, 0.75) * _freshness_multiplier(timestamp)
        note = str(rating.get("summary") or rating.get("enSummary") or article.get("text") or article.get("description") or "").strip()
        for coin in article.get("coins") or []:
            symbol = _normalize_symbol(str((coin or {}).get("symbol") or ""))
            if not symbol:
                continue
            bucket = grouped.setdefault(symbol, {"score": 0.0, "mentions": 0, "latest": timestamp, "sources": set(), "notes": []})
            bucket["score"] += weighted_score
            bucket["mentions"] += 1
            bucket["latest"] = max(bucket["latest"], timestamp)
            bucket["sources"].add(source_name)
            if note:
                bucket["notes"].append(note[:160])
    records: list[SentimentSignal] = []
    for symbol, bucket in sorted(grouped.items()):
        mentions = max(1, bucket["mentions"])
        hotness = min(100.0, round(bucket["score"] / mentions, 2))
        records.append(
            SentimentSignal(
                symbol=symbol,
                hotness_score=hotness,
                mention_count=mentions,
                source=",".join(sorted(bucket["sources"])),
                updated_at=bucket["latest"],
                notes="; ".join(dict.fromkeys(bucket["notes"]))[:400],
            )
        )
    return records


def _build_opentwitter_records(tweets_by_symbol: dict[str, list[dict]]) -> list[SentimentSignal]:
    records: list[SentimentSignal] = []
    for base, tweets in sorted(tweets_by_symbol.items()):
        seen_ids: set[str] = set()
        score_total = 0.0
        mention_count = 0
        latest = datetime.now(timezone.utc)
        sources: set[str] = set()
        notes: list[str] = []
        for tweet in tweets:
            tweet_id = str(tweet.get("id") or "")
            if tweet_id and tweet_id in seen_ids:
                continue
            seen_ids.add(tweet_id)
            text = str(tweet.get("text") or "")
            upper_text = text.upper()
            source_name = str(tweet.get("_somnus_source") or "opentwitter_search")
            is_official = source_name == "opentwitter_official"
            if not is_official and f"${base}" not in upper_text and base not in upper_text:
                continue
            favorite = float(tweet.get("favoriteCount") or 0.0)
            retweet = float(tweet.get("retweetCount") or 0.0)
            reply = float(tweet.get("replyCount") or 0.0)
            followers = float(tweet.get("userFollowers") or 0.0)
            created_at = _parse_dt(str(tweet.get("createdAt") or ""))
            latest = max(latest, created_at)
            base_score = 30.0 + min(35.0, favorite / 20.0) + min(20.0, retweet * 2.0) + min(10.0, reply * 2.0) + min(15.0, followers / 5000.0)
            if is_official:
                base_score += 15.0
            score_total += min(100.0, base_score)
            mention_count += 1
            sources.add(source_name)
            handle = str(tweet.get("userScreenName") or "")
            prefix = "official:" if is_official else ""
            notes.append(f"{prefix}@{handle}:{text[:120]}")
        if mention_count <= 0:
            continue
        records.append(
            SentimentSignal(
                symbol=f"{base}USDT",
                hotness_score=round(min(100.0, score_total / mention_count), 2),
                mention_count=mention_count,
                source=",".join(sorted(sources or {"opentwitter_search"})),
                updated_at=latest,
                notes="; ".join(dict.fromkeys(notes))[:400],
            )
        )
    return records
