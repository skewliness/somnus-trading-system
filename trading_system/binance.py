from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any

import requests

from trading_system.bb_structure import compute_bb_structure
from trading_system.binance_public_guard import (
    BinancePublicRateLimitError,
    assert_binance_public_not_banned,
    note_binance_public_response,
    throttle_binance_public_request,
)
from trading_system.config import StrategyConfig
from trading_system.h4_structure import compute_h4_structure
from trading_system.models import MarketSnapshot
from trading_system.sentiment import SentimentBook


LOGGER = logging.getLogger(__name__)
KLINE_ENDPOINT_MAX_LIMIT = 1500
KLINE_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}
MARKET_SNAPSHOT_FIELDS = {f.name for f in fields(MarketSnapshot)}


class BinanceAccessError(requests.RequestException):
    """Raised when Binance blocks the environment or the endpoint is unavailable."""


@dataclass(slots=True)
class BinanceFuturesClient:
    base_url: str
    timeout_seconds: int = 20
    session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "codex-trading-system/1.0"})

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        errors: list[str] = []
        for base_url in _public_base_urls(self.base_url):
            try:
                return self._get_json_from_base(base_url, path, params)
            except BinanceAccessError as exc:
                errors.append(f"{base_url}: {exc}")
                continue
        raise BinanceAccessError("Binance Futures request failed on all public base URLs: " + " | ".join(errors))

    def _get_json_from_base(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{base_url}{path}"
        if "fapi.binance.com" in base_url:
            try:
                assert_binance_public_not_banned(f"v5:{path}")
            except BinancePublicRateLimitError as exc:
                raise BinanceAccessError(str(exc)) from exc
        last_exc: Exception | None = None
        response: requests.Response | None = None
        for attempt in range(3):
            try:
                if "fapi.binance.com" in base_url:
                    throttle_binance_public_request()
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
                note_binance_public_response(response.status_code, response.text, f"v5:{path}")
                if response.status_code == 418:
                    break
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except requests.RequestException as exc:
                last_exc = exc
                response = None
            if attempt < 2:
                time.sleep(0.4 * (2 ** attempt))
        if response is None:
            raise BinanceAccessError(f"Binance Futures request failed after retries: {last_exc}")
        text = response.text
        if response.status_code == 451 or "restricted location" in text.lower():
            raise BinanceAccessError(
                "Binance Futures returned HTTP 451 restricted location. "
                "Run live mode from a region/network where Binance Futures market-data access is allowed."
            )
        response.raise_for_status()
        return response.json()


def _public_base_urls(primary: str) -> list[str]:
    urls = [primary.rstrip("/")]
    raw = os.getenv("BINANCE_FUTURES_PUBLIC_FALLBACK_BASE_URLS", "")
    for item in raw.split(","):
        url = item.strip().rstrip("/")
        if url and url not in urls:
            urls.append(url)
    return urls


class KlineCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, symbol: str, interval: str) -> list[Any]:
        path = self._path(symbol, interval)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(payload, dict):
            payload = payload.get("klines") or []
        if not isinstance(payload, list):
            return []
        return _merge_klines(payload)

    def save(self, symbol: str, interval: str, klines: list[Any]) -> None:
        path = self._path(symbol, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = _merge_klines(klines)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        tmp.write_text(json.dumps(merged, separators=(",", ":")), encoding="utf-8")
        try:
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def count(self, symbol: str, interval: str) -> int:
        return len(self.load(symbol, interval))

    def is_fresh(self, symbol: str, interval: str, max_age_seconds: int) -> bool:
        if max_age_seconds <= 0:
            return False
        path = self._path(symbol, interval)
        if not path.exists():
            return False
        return (time.time() - path.stat().st_mtime) <= max_age_seconds

    def _path(self, symbol: str, interval: str) -> Path:
        safe_symbol = "".join(ch for ch in symbol.upper() if ch.isalnum() or ch in {"_", "-"})
        return self.root / interval / f"{safe_symbol}.json"


def _kline_open_time(kline: Any) -> int:
    try:
        return int(kline[0])
    except (IndexError, TypeError, ValueError):
        return 0


def _merge_klines(klines: list[Any]) -> list[Any]:
    rows: dict[int, Any] = {}
    for kline in klines:
        open_time = _kline_open_time(kline)
        if open_time > 0:
            rows[open_time] = kline
    return [rows[key] for key in sorted(rows)]


def _select_momo_symbols(
    ranked: list[dict[str, Any]],
    *,
    sentiment: SentimentBook,
    mark_price_map: dict[str, float],
    funding_map: dict[str, float],
    max_symbols: int,
) -> set[str]:
    if max_symbols <= 0:
        return set()
    scored = []
    for rank_index, item in enumerate(ranked):
        symbol = str(item.get("symbol") or "")
        if not symbol:
            continue
        signal = sentiment.lookup(symbol)
        scored.append((
            _momo_prefilter_score(
                item,
                rank_index=rank_index,
                sentiment_score=signal.hotness_score if signal else 0.0,
                sentiment_mentions=signal.mention_count if signal else 0,
                funding_rate=funding_map.get(symbol, 0.0),
                mark_price=mark_price_map.get(symbol, float(item.get("lastPrice") or 0.0)),
            ),
            symbol,
        ))
    scored.sort(reverse=True)
    return {symbol for _, symbol in scored[:max_symbols]}


def _momo_prefilter_score(
    item: dict[str, Any],
    *,
    rank_index: int,
    sentiment_score: float,
    sentiment_mentions: int,
    funding_rate: float,
    mark_price: float,
) -> float:
    price_change = max(0.0, float(item.get("priceChangePercent") or 0.0))
    quote_volume = max(0.0, float(item.get("quoteVolume") or 0.0))
    volume_component = min(30.0, math.log10(max(quote_volume, 1.0)) * 3.0)
    rank_component = max(0.0, 25.0 - rank_index * 0.4)
    sentiment_component = min(35.0, sentiment_score * 0.25 + min(sentiment_mentions, 100) * 0.1)
    funding_penalty = 8.0 if funding_rate < 0 else 0.0
    price_penalty = 10.0 if mark_price <= 0 else 0.0
    return round(price_change * 0.6 + volume_component + rank_component + sentiment_component - funding_penalty - price_penalty, 6)


def _rank_long_scan_candidates(
    tickers: list[dict[str, Any]],
    symbol_meta: dict[str, Any],
    config: StrategyConfig,
) -> list[dict[str, Any]]:
    return sorted(
        (
            item for item in tickers
            if item.get("symbol") in symbol_meta
            and _quote_volume_usdt(item) >= config.min_quote_volume_usdt
            and _price_change_percent(item) >= config.min_price_change_pct
        ),
        key=lambda item: (_price_change_percent(item), _quote_volume_usdt(item)),
        reverse=True,
    )


def _rank_short_scan_candidates(
    tickers: list[dict[str, Any]],
    symbol_meta: dict[str, Any],
    funding_map: dict[str, float],
    config: StrategyConfig,
) -> list[dict[str, Any]]:
    breakdown_candidates: list[dict[str, Any]] = []
    squeeze_fade_candidates: list[dict[str, Any]] = []
    for item in tickers:
        symbol = str(item.get("symbol") or "")
        if symbol not in symbol_meta:
            continue
        if _quote_volume_usdt(item) < config.short_scan_min_quote_volume_usdt:
            continue
        price_change = _price_change_percent(item)
        funding_rate = funding_map.get(symbol, 0.0)
        if price_change <= -abs(config.short_scan_min_abs_price_change_pct):
            breakdown_candidates.append(item)
        elif (
            price_change >= config.short_scan_squeeze_min_price_change_pct
            and funding_rate <= config.anomaly_negative_funding_rate
        ):
            squeeze_fade_candidates.append(item)

    breakdown_ranked = sorted(
        breakdown_candidates,
        key=lambda item: (abs(_price_change_percent(item)), _quote_volume_usdt(item)),
        reverse=True,
    )
    squeeze_ranked = sorted(
        squeeze_fade_candidates,
        key=lambda item: (
            _price_change_percent(item),
            abs(funding_map.get(str(item.get("symbol") or ""), 0.0)),
            _quote_volume_usdt(item),
        ),
        reverse=True,
    )
    return _merge_scan_candidates(breakdown_ranked, squeeze_ranked)


def _merge_scan_candidates(*candidate_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in candidate_groups:
        for item in group:
            symbol = str(item.get("symbol") or "")
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            merged.append(item)
    return merged


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _select_deep_strategy_candidates(
    long_ranked: list[dict[str, Any]],
    short_ranked: list[dict[str, Any]],
    sentiment_ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    max_symbols = _env_int("TS_DEEP_STRATEGY_MAX_SYMBOLS", 15)
    if max_symbols <= 0:
        return _merge_scan_candidates(long_ranked, short_ranked, sentiment_ranked)

    long_limit = max(0, _env_int("TS_DEEP_STRATEGY_LONG_SYMBOLS", 8))
    short_limit = max(0, _env_int("TS_DEEP_STRATEGY_SHORT_SYMBOLS", 5))
    sentiment_limit = max(0, _env_int("TS_DEEP_STRATEGY_SENTIMENT_SYMBOLS", 5))
    deep_ranked = _merge_scan_candidates(
        long_ranked[:long_limit],
        short_ranked[:short_limit],
        sentiment_ranked[:sentiment_limit],
    )
    if len(deep_ranked) < max_symbols:
        deep_ranked = _merge_scan_candidates(deep_ranked, long_ranked, short_ranked, sentiment_ranked)
    return deep_ranked[:max_symbols]


def _env_symbol_set(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name, default)
    return {item.strip().upper() for item in raw.split(",") if item.strip()}


def _rank_sentiment_scan_candidates(
    tickers: list[dict[str, Any]],
    symbol_meta: dict[str, Any],
    sentiment: SentimentBook,
    max_symbols: int,
) -> list[dict[str, Any]]:
    if max_symbols <= 0:
        return []
    excluded = _env_symbol_set(
        "TS_SENTIMENT_UNIVERSE_EXCLUDE_SYMBOLS",
        "BTCUSDT,ETHUSDT,HYPEUSDT,BNBUSDT,SOLUSDT,ZECUSDT",
    )
    ticker_map = {str(item.get("symbol") or ""): item for item in tickers}
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for symbol in sentiment.symbols():
        if symbol in excluded or symbol not in symbol_meta or symbol not in ticker_map:
            continue
        signal = sentiment.lookup(symbol)
        if signal is None:
            continue
        item = ticker_map[symbol]
        liquidity = min(25.0, math.log10(max(_quote_volume_usdt(item), 1.0)) * 2.5)
        score = float(signal.hotness_score) + min(int(signal.mention_count), 500) * 8.0 + liquidity
        scored.append((score, symbol, item))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [item for _, _, item in scored[:max_symbols]]


def _quote_volume_usdt(item: dict[str, Any]) -> float:
    return float(item.get("quoteVolume") or 0.0)


def _price_change_percent(item: dict[str, Any]) -> float:
    return float(item.get("priceChangePercent") or 0.0)


class BinanceMarketDataProvider:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.client = BinanceFuturesClient(config.base_url)
        self.kline_cache = KlineCache(config.momo_structure_cache_dir)

    def collect(self, sentiment: SentimentBook) -> list[MarketSnapshot]:
        exchange_info = self.client.get_json("/fapi/v1/exchangeInfo")
        tickers = self.client.get_json("/fapi/v1/ticker/24hr")
        mark_prices = self.client.get_json("/fapi/v1/premiumIndex")

        symbol_meta = {
            item["symbol"]: item
            for item in exchange_info.get("symbols", [])
            if item.get("status") == "TRADING"
            and item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
        }
        mark_price_map = {item["symbol"]: float(item["markPrice"]) for item in mark_prices if "symbol" in item}
        funding_map = {item["symbol"]: float(item.get("lastFundingRate") or 0.0) for item in mark_prices if "symbol" in item}

        long_ranked = _rank_long_scan_candidates(tickers, symbol_meta, self.config)
        short_ranked = (
            _rank_short_scan_candidates(tickers, symbol_meta, funding_map, self.config)
            if self.config.short_scan_enabled
            else []
        )
        sentiment_ranked = _rank_sentiment_scan_candidates(
            tickers,
            symbol_meta,
            sentiment,
            _env_int("TS_SENTIMENT_UNIVERSE_MAX_SYMBOLS", 0),
        )
        ranked = _merge_scan_candidates(
            long_ranked[: self.config.max_universe_size],
            short_ranked[: self.config.short_scan_max_universe_size],
            sentiment_ranked,
        )
        deep_ranked = _select_deep_strategy_candidates(long_ranked, short_ranked, sentiment_ranked)
        LOGGER.info(
            "V5 scan layers: quick_candidates=%d deep_strategy_candidates=%d long=%d short=%d sentiment=%d",
            len(ranked),
            len(deep_ranked),
            len(long_ranked),
            len(short_ranked),
            len(sentiment_ranked),
        )

        momo_symbols = _select_momo_symbols(
            deep_ranked,
            sentiment=sentiment,
            mark_price_map=mark_price_map,
            funding_map=funding_map,
            max_symbols=self.config.momo_structure_top_n,
        )
        momo_backfills_used = 0
        snapshots: list[MarketSnapshot] = []
        for item in deep_ranked:
            symbol = item["symbol"]
            meta = symbol_meta[symbol]
            sentiment_signal = sentiment.lookup(symbol)
            mark_price = mark_price_map.get(symbol, float(item.get("lastPrice") or 0))
            price_change_pct_5m = self._get_price_change_pct_5m(symbol, mark_price)
            momo_target = (
                self.config.bb_structure_enabled
                and self.config.momo_structure_enabled
                and symbol in momo_symbols
            )
            use_momo_structure = False
            if momo_target:
                cache_ready = self._momo_cache_ready(symbol)
                if cache_ready or momo_backfills_used < max(0, self.config.momo_structure_backfill_symbols_per_run):
                    use_momo_structure = True
                    if not cache_ready:
                        momo_backfills_used += 1
            normal_h4_limit = 700 if self.config.bb_structure_enabled else 28
            h4_limit = self.config.momo_4h_kline_limit if use_momo_structure else normal_h4_limit
            try:
                h4_klines = (
                    self._get_momo_klines(symbol, "4h", h4_limit, self.config.momo_4h_refresh_seconds)
                    if use_momo_structure
                    else self._get_cached_or_live_klines(symbol, "4h", h4_limit, self.config.momo_4h_refresh_seconds)
                )
            except requests.RequestException as exc:
                if not use_momo_structure:
                    raise
                LOGGER.warning("Momo 4h structure fetch failed for %s; falling back to V5 limits: %s", symbol, exc)
                use_momo_structure = False
                h4_klines = self._get_cached_or_live_klines(symbol, "4h", normal_h4_limit, self.config.momo_4h_refresh_seconds)
            price_change_pct_4h = _price_change_pct_from_klines(h4_klines[-2:], current_price=mark_price)
            h4_structure = compute_h4_structure(symbol, h4_klines, mark_price)
            daily_limit = self.config.momo_daily_kline_limit if use_momo_structure else 20
            try:
                daily_klines = (
                    self._get_momo_klines(symbol, "1d", daily_limit, self.config.momo_daily_refresh_seconds)
                    if use_momo_structure
                    else self._get_cached_or_live_klines(symbol, "1d", daily_limit, self.config.momo_daily_refresh_seconds)
                )
            except requests.RequestException as exc:
                if not use_momo_structure:
                    raise
                LOGGER.warning("Momo daily structure fetch failed for %s; falling back to V5 limits: %s", symbol, exc)
                use_momo_structure = False
                daily_klines = self._get_cached_or_live_klines(symbol, "1d", 20, self.config.momo_daily_refresh_seconds)

            bb_signal = None
            if self.config.bb_structure_enabled:
                klines_15m_limit = self.config.momo_15m_kline_limit if use_momo_structure else 215
                try:
                    klines_15m = (
                        self._get_momo_klines(symbol, "15m", klines_15m_limit, self.config.momo_15m_refresh_seconds)
                        if use_momo_structure
                        else self._get_cached_or_live_klines(symbol, "15m", klines_15m_limit, self.config.momo_15m_refresh_seconds)
                    )
                    klines_1m = []
                    if use_momo_structure:
                        klines_1m = self._get_momo_klines(
                            symbol,
                            "1m",
                            self.config.momo_1m_kline_limit,
                            self.config.momo_1m_refresh_seconds,
                        )
                except requests.RequestException as exc:
                    if not use_momo_structure:
                        raise
                    LOGGER.warning("Momo intraday structure fetch failed for %s; falling back to V5 limits: %s", symbol, exc)
                    use_momo_structure = False
                    klines_1m = []
                    klines_15m = self._get_cached_or_live_klines(symbol, "15m", 215, self.config.momo_15m_refresh_seconds)
                bb_signal = compute_bb_structure(
                    klines_15m,
                    h4_klines,
                    mark_price,
                    self.config.bb_touch_threshold_pct,
                    klines_1m=klines_1m,
                    klines_1d=daily_klines if use_momo_structure else None,
                )

            structure = _market_structure_from_klines(
                daily_klines,
                mark_price=mark_price,
            )
            kline_stats = _kline_statistics(daily_klines, mark_price=mark_price)
            pattern_flags = _detect_kline_patterns(daily_klines)

            open_interest = self._get_optional_json("/fapi/v1/openInterest", {"symbol": symbol}, default={})
            oi_hist = self._get_optional_json(
                "/futures/data/openInterestHist",
                {"symbol": symbol, "period": "4h", "limit": 2},
                default=[],
            )
            taker = self._get_optional_json(
                "/futures/data/takerlongshortRatio",
                {"symbol": symbol, "period": "1h", "limit": 2},
                default=[],
            )

            open_interest_qty = float(open_interest.get("openInterest") or 0.0)
            open_interest_usdt = open_interest_qty * mark_price
            oi_change_pct = _calc_oi_change_pct(oi_hist)
            taker_ratio = float(taker[-1]["buySellRatio"]) if taker else 1.0

            onboard_date_ms = int(meta.get("onboardDate") or 0)
            listing_days = _listing_days(onboard_date_ms)
            setup_type, setup_quality = classify_setup(
                price_change_pct_24h=float(item.get("priceChangePercent") or 0.0),
                funding_rate=funding_map.get(symbol, 0.0),
                oi_change_pct_4h=oi_change_pct,
                taker_buy_sell_ratio=taker_ratio,
                sentiment_score=sentiment_signal.hotness_score if sentiment_signal else 0.0,
                listing_days=listing_days,
                config=self.config,
                price_change_pct_4h=price_change_pct_4h,
                price_position_7d=structure["price_position_7d"],
                price_position_14d=structure["price_position_14d"],
                price_position_20d=structure["price_position_20d"],
                breakout_distance_7d=structure["breakout_distance_7d"],
                compression_score=structure["compression_score"],
                price_change_pct_5m=price_change_pct_5m,
                quote_volume_usdt_24h=float(item.get("quoteVolume") or 0.0),
                sentiment_mentions=sentiment_signal.mention_count if sentiment_signal else 0,
                bb_price_position_96=getattr(bb_signal, "bb_price_position_96", 0.5) if bb_signal else 0.5,
                bb_price_position_105_4h=getattr(bb_signal, "bb_price_position_105_4h", 0.5) if bb_signal else 0.5,
                bb_price_position_672_4h=getattr(bb_signal, "bb_price_position_672_4h", 0.5) if bb_signal else 0.5,
                bb_price_position_1440_4h=getattr(bb_signal, "bb_price_position_1440_4h", 0.5) if bb_signal else 0.5,
                bb_price_position_105_1m=getattr(bb_signal, "bb_price_position_105_1m", 0.5) if bb_signal else 0.5,
                bb_price_position_300_1m=getattr(bb_signal, "bb_price_position_300_1m", 0.5) if bb_signal else 0.5,
            )
            manipulation_stage, manipulation_quality, manipulation_flags = _manipulation_context(sentiment_signal)
            oi_divergence_score = _oi_price_divergence_score(oi_change_pct, price_change_pct_4h)

            snapshots.append(
                MarketSnapshot(
                    symbol=symbol,
                    mark_price=mark_price,
                    price_change_pct_5m=price_change_pct_5m,
                    price_change_pct_4h=price_change_pct_4h,
                    price_change_pct_24h=float(item.get("priceChangePercent") or 0.0),
                    quote_volume_usdt_24h=float(item.get("quoteVolume") or 0.0),
                    funding_rate=funding_map.get(symbol, 0.0),
                    open_interest_usdt=open_interest_usdt,
                    oi_change_pct_4h=oi_change_pct,
                    taker_buy_sell_ratio=taker_ratio,
                    listing_days=listing_days,
                    sentiment_score=sentiment_signal.hotness_score if sentiment_signal else 0.0,
                    sentiment_mentions=sentiment_signal.mention_count if sentiment_signal else 0,
                    sentiment_source=sentiment_signal.source if sentiment_signal else "none",
                    price_position_7d=structure["price_position_7d"],
                    price_position_14d=structure["price_position_14d"],
                    price_position_20d=structure["price_position_20d"],
                    breakout_distance_7d=structure["breakout_distance_7d"],
                    compression_score=structure["compression_score"],
                    kline_lower_shadow_pct=kline_stats.get("kline_lower_shadow_pct", 0.0),
                    kline_upper_shadow_pct=kline_stats.get("kline_upper_shadow_pct", 0.0),
                    kline_body_pct=kline_stats.get("kline_body_pct", 0.0),
                    kline_direction=kline_stats.get("kline_direction", 0.0),
                    kline_volume_vs_ma5=kline_stats.get("kline_volume_vs_ma5", 0.0),
                    kline_range_vs_ma5_range=kline_stats.get("kline_range_vs_ma5_range", 0.0),
                    kline_close_position=kline_stats.get("kline_close_position", 0.0),
                    kline_consecutive_direction=kline_stats.get("kline_consecutive_direction", 0.0),
                    kline_atr_pct=kline_stats.get("kline_atr_pct", 0.0),
                    kline_ma20_distance_pct=kline_stats.get("kline_ma20_distance_pct", 0.0),
                    kline_volume_trend=kline_stats.get("kline_volume_trend", 0.0),
                    kline_max_drawdown_5d=kline_stats.get("kline_max_drawdown_5d", 0.0),
                    kline_pattern_bullish_count=pattern_flags.get("bullish_count", 0.0),
                    kline_pattern_bearish_count=pattern_flags.get("bearish_count", 0.0),
                    kline_pattern_flags=pattern_flags.get("flags", ""),
                    h4_close_above_ma20=h4_structure.h4_close_above_ma20,
                    h4_ma20_slope_pct=h4_structure.h4_ma20_slope_pct,
                    h4_price_vs_ma20_pct=h4_structure.h4_price_vs_ma20_pct,
                    h4_close_position_20bars=h4_structure.h4_close_position_20bars,
                    h4_low_break_recent_20bars=h4_structure.h4_low_break_recent_20bars,
                    h4_high_break_recent_20bars=h4_structure.h4_high_break_recent_20bars,
                    h4_atr_compression=h4_structure.h4_atr_compression,
                    h4_lower_highs_count=h4_structure.h4_lower_highs_count,
                    h4_higher_lows_count=h4_structure.h4_higher_lows_count,
                    h4_structure_state=h4_structure.h4_structure_state,
                    h4_price_location=h4_structure.h4_price_location,
                    h4_pattern_type=h4_structure.h4_pattern_type,
                    h4_range_low=h4_structure.h4_range_low,
                    h4_recent_swing_low=h4_structure.h4_recent_swing_low,
                    setup_type=setup_type,
                    setup_quality=setup_quality,
                    manipulation_stage=manipulation_stage,
                    manipulation_quality=manipulation_quality,
                    manipulation_flags=manipulation_flags,
                    oi_price_divergence_score=oi_divergence_score,
                    **_bb_signal_kwargs(bb_signal),
                )
            )
        return snapshots

    def _get_klines(self, symbol: str, interval: str, limit: int, **params: Any) -> list[Any]:
        request_params = {"symbol": symbol, "interval": interval, "limit": _binance_kline_limit(limit), **params}
        return self.client.get_json("/fapi/v1/klines", request_params)

    def _get_optional_json(self, path: str, params: dict[str, Any], *, default: Any) -> Any:
        try:
            return self.client.get_json(path, params)
        except requests.RequestException as exc:
            LOGGER.warning("Optional Binance request failed path=%s params=%s: %s", path, params, exc)
            return default

    def _get_price_change_pct_5m(self, symbol: str, mark_price: float) -> float:
        try:
            klines = self._get_cached_or_live_klines(symbol, "5m", 2, self.config.momo_1m_refresh_seconds)
        except requests.RequestException as exc:
            LOGGER.warning("Optional Binance 5m kline request failed symbol=%s: %s", symbol, exc)
            return 0.0
        return _price_change_pct_from_klines(klines[-2:], current_price=mark_price)

    def _get_cached_or_live_klines(self, symbol: str, interval: str, desired_limit: int, refresh_seconds: int) -> list[Any]:
        if not self.config.momo_structure_cache_enabled:
            return self._get_klines(symbol, interval, desired_limit)
        return self._get_momo_klines(symbol, interval, desired_limit, refresh_seconds)

    def _get_momo_klines(self, symbol: str, interval: str, desired_limit: int, refresh_seconds: int) -> list[Any]:
        desired_limit = _desired_kline_count(desired_limit)
        if not self.config.momo_structure_cache_enabled:
            return self._get_klines(symbol, interval, desired_limit)

        cached = self.kline_cache.load(symbol, interval)
        if len(cached) >= desired_limit and self.kline_cache.is_fresh(symbol, interval, refresh_seconds):
            return cached[-desired_limit:]

        try:
            cached = self._refresh_cached_klines(symbol, interval, cached, desired_limit, refresh_seconds)
        except requests.RequestException:
            if cached:
                LOGGER.warning(
                    "Using stale cached %s %s klines after Binance request failure: cached=%s desired=%s",
                    symbol,
                    interval,
                    len(cached),
                    desired_limit,
                )
                return cached[-desired_limit:]
            raise
        return cached[-desired_limit:]

    def _refresh_cached_klines(
        self,
        symbol: str,
        interval: str,
        cached: list[Any],
        desired_limit: int,
        refresh_seconds: int,
    ) -> list[Any]:
        interval_ms = KLINE_INTERVAL_MS.get(interval)
        if interval_ms is None:
            return self._get_klines(symbol, interval, desired_limit)

        if cached and not self.kline_cache.is_fresh(symbol, interval, refresh_seconds):
            start_time = _kline_open_time(cached[-1]) + interval_ms
            if start_time > 0:
                elapsed_bars = max(2, int(((time.time() * 1000) - start_time) / interval_ms) + 2)
                latest = self._get_klines(
                    symbol,
                    interval,
                    min(KLINE_ENDPOINT_MAX_LIMIT, elapsed_bars),
                    startTime=start_time,
                )
                cached = _merge_klines([*cached, *latest])

        while len(cached) < desired_limit:
            missing = desired_limit - len(cached)
            params: dict[str, Any] = {}
            if cached:
                first_open_time = _kline_open_time(cached[0])
                if first_open_time <= 0:
                    break
                params["endTime"] = first_open_time - 1
            page = self._get_klines(symbol, interval, min(KLINE_ENDPOINT_MAX_LIMIT, missing), **params)
            if not page:
                break
            cached = _merge_klines([*page, *cached])
            if len(page) < min(KLINE_ENDPOINT_MAX_LIMIT, missing):
                break

        if cached:
            keep_limit = max(desired_limit, KLINE_ENDPOINT_MAX_LIMIT, len(cached))
            self.kline_cache.save(symbol, interval, cached[-keep_limit:])
        return cached

    def _momo_cache_ready(self, symbol: str) -> bool:
        if not self.config.momo_structure_cache_enabled:
            return False
        required = {
            "15m": min(_desired_kline_count(self.config.momo_15m_kline_limit), 1440),
            "4h": min(_desired_kline_count(self.config.momo_4h_kline_limit), 672),
            "1m": min(_desired_kline_count(self.config.momo_1m_kline_limit), 300),
            "1d": min(_desired_kline_count(self.config.momo_daily_kline_limit), 20),
        }
        return all(self.kline_cache.count(symbol, interval) >= count for interval, count in required.items())


def _kline_limit(value: int) -> int:
    return max(1, min(KLINE_ENDPOINT_MAX_LIMIT, int(value)))


def _desired_kline_count(value: int) -> int:
    return max(1, int(value))


def _binance_kline_limit(value: int) -> int:
    return max(1, min(KLINE_ENDPOINT_MAX_LIMIT, int(value)))


def _bb_signal_kwargs(bb_signal: Any | None) -> dict[str, Any]:
    if not bb_signal:
        return {}
    return {key: value for key, value in asdict(bb_signal).items() if key in MARKET_SNAPSHOT_FIELDS}


class ReplayMarketDataProvider:
    def __init__(self, replay_path: Path, config: StrategyConfig | None = None) -> None:
        self.replay_path = replay_path
        self.config = config or StrategyConfig()

    def collect(self, sentiment: SentimentBook) -> list[MarketSnapshot]:
        payload = json.loads(self.replay_path.read_text(encoding="utf-8"))
        snapshots: list[MarketSnapshot] = []
        for item in payload:
            sentiment_signal = sentiment.lookup(item["symbol"])
            listing_days = int(item["listing_days"]) if item.get("listing_days") is not None else None
            structure = _market_structure_from_payload(item, mark_price=float(item["mark_price"]))
            setup_type, setup_quality = classify_setup(
                price_change_pct_24h=float(item["price_change_pct_24h"]),
                price_change_pct_4h=_coerce_optional_float(
                    item.get("price_change_pct_4h"),
                    _price_change_pct_from_closes(item.get("closes_4h") or [], current_price=float(item["mark_price"])),
                )
                or 0.0,
                funding_rate=float(item["funding_rate"]),
                oi_change_pct_4h=float(item["oi_change_pct_4h"]),
                taker_buy_sell_ratio=float(item["taker_buy_sell_ratio"]),
                sentiment_score=sentiment_signal.hotness_score if sentiment_signal else float(item.get("sentiment_score", 0.0)),
                listing_days=listing_days,
                config=self.config,
                price_position_7d=structure["price_position_7d"],
                price_position_14d=structure["price_position_14d"],
                price_position_20d=structure["price_position_20d"],
                breakout_distance_7d=structure["breakout_distance_7d"],
                compression_score=structure["compression_score"],
                price_change_pct_5m=_coerce_optional_float(
                    item.get("price_change_pct_5m"),
                    _price_change_pct_from_closes(item.get("closes_5m") or [], current_price=float(item["mark_price"])),
                )
                or 0.0,
                quote_volume_usdt_24h=float(item["quote_volume_usdt_24h"]),
                sentiment_mentions=sentiment_signal.mention_count if sentiment_signal else int(item.get("sentiment_mentions", 0)),
                bb_price_position_96=_coerce_optional_float(item.get("bb_price_position_96"), 0.5),
                bb_price_position_105_4h=_coerce_optional_float(item.get("bb_price_position_105_4h"), 0.5),
                bb_price_position_672_4h=_coerce_optional_float(item.get("bb_price_position_672_4h"), 0.5),
                bb_price_position_1440_4h=_coerce_optional_float(item.get("bb_price_position_1440_4h"), 0.5),
                bb_price_position_105_1m=_coerce_optional_float(item.get("bb_price_position_105_1m"), 0.5),
                bb_price_position_300_1m=_coerce_optional_float(item.get("bb_price_position_300_1m"), 0.5),
            )
            sentiment_stage, sentiment_quality, sentiment_flags = _manipulation_context(sentiment_signal)
            oi_divergence_score = _oi_price_divergence_score(
                float(item["oi_change_pct_4h"]),
                _coerce_optional_float(
                    item.get("price_change_pct_4h"),
                    _price_change_pct_from_closes(item.get("closes_4h") or [], current_price=float(item["mark_price"])),
                )
                or 0.0,
            )
            snapshots.append(
                MarketSnapshot(
                    symbol=item["symbol"],
                    mark_price=float(item["mark_price"]),
                    price_change_pct_5m=_coerce_optional_float(
                        item.get("price_change_pct_5m"),
                        _price_change_pct_from_closes(item.get("closes_5m") or [], current_price=float(item["mark_price"])),
                    )
                    or 0.0,
                    price_change_pct_4h=_coerce_optional_float(
                        item.get("price_change_pct_4h"),
                        _price_change_pct_from_closes(item.get("closes_4h") or [], current_price=float(item["mark_price"])),
                    )
                    or 0.0,
                    price_change_pct_24h=float(item["price_change_pct_24h"]),
                    quote_volume_usdt_24h=float(item["quote_volume_usdt_24h"]),
                    funding_rate=float(item["funding_rate"]),
                    open_interest_usdt=float(item["open_interest_usdt"]),
                    oi_change_pct_4h=float(item["oi_change_pct_4h"]),
                    taker_buy_sell_ratio=float(item["taker_buy_sell_ratio"]),
                    listing_days=listing_days,
                    sentiment_score=sentiment_signal.hotness_score if sentiment_signal else float(item.get("sentiment_score", 0.0)),
                    sentiment_mentions=sentiment_signal.mention_count if sentiment_signal else int(item.get("sentiment_mentions", 0)),
                    sentiment_source=sentiment_signal.source if sentiment_signal else str(item.get("sentiment_source", "replay")),
                    price_position_7d=_coerce_optional_float(item.get("price_position_7d"), structure["price_position_7d"]),
                    price_position_14d=_coerce_optional_float(item.get("price_position_14d"), structure["price_position_14d"]),
                    price_position_20d=_coerce_optional_float(item.get("price_position_20d"), structure["price_position_20d"]),
                    breakout_distance_7d=_coerce_optional_float(item.get("breakout_distance_7d"), structure["breakout_distance_7d"]),
                    compression_score=_coerce_optional_float(item.get("compression_score"), structure["compression_score"]),
                    setup_type=str(item.get("setup_type") or setup_type),
                    setup_quality=float(item.get("setup_quality") or setup_quality),
                    manipulation_stage=str(item.get("manipulation_stage") or sentiment_stage),
                    manipulation_quality=float(item.get("manipulation_quality") or sentiment_quality),
                    manipulation_flags=list(item.get("manipulation_flags") or sentiment_flags),
                    oi_price_divergence_score=float(item.get("oi_price_divergence_score") or oi_divergence_score),
                )
            )
        return snapshots


def _manipulation_context(sentiment_signal: Any) -> tuple[str, float, list[str]]:
    if not sentiment_signal:
        return "trend", 0.0, []
    text = f"{getattr(sentiment_signal, 'source', '')} {getattr(sentiment_signal, 'notes', '')}".lower()
    flags: list[str] = []
    keyword_map = {
        "controlled_supply": ["控盘", "高度控盘", "筹码集中", "筹码分布", "locked", "concentrated supply"],
        "multisig_cluster": ["多签钱包", "multisig", "6 个多签", "6个多签"],
        "single_wallet_dominance": ["独占77%", "独占 77%", "77%", "dominant wallet", "单地址独占"],
        "retail_absent": ["散户持仓", "不到0.1%", "不到 0.1%", "retail"],
        "venue_concentration": ["bitget", "88%", "单平台", "现货主战场", "venue concentration"],
        "distribution": ["庄家控盘术", "distribution", "派发", "拉高出货", "庄盘"],
    }
    for flag, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            flags.append(flag)
    if flags:
        quality = min(95.0, 55.0 + (len(flags) * 8.0))
        return "distribution", quality, flags
    return "trend", 0.0, []


def _listing_days(onboard_date_ms: int) -> int | None:
    if onboard_date_ms <= 0:
        return None
    onboard = datetime.fromtimestamp(onboard_date_ms / 1000, tz=timezone.utc)
    delta = datetime.now(timezone.utc) - onboard
    return max(0, delta.days)


def _calc_oi_change_pct(history: list[dict[str, Any]]) -> float:
    if len(history) < 2:
        return 0.0
    previous = float(history[0].get("sumOpenInterestValue") or 0.0)
    latest = float(history[-1].get("sumOpenInterestValue") or 0.0)
    if previous <= 0:
        return 0.0
    return round(((latest - previous) / previous) * 100, 3)


def _oi_price_divergence_score(oi_change_pct_4h: float, price_change_pct_4h: float) -> float:
    damping = min(0.95, abs(price_change_pct_4h) / 10.0)
    return round(max(0.0, oi_change_pct_4h * (1.0 - damping) * 10.0), 2)


def _bb_position(value: float | None, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _short_bb_structure_context(
    *,
    bb_price_position_96: float | None,
    bb_price_position_105_4h: float | None,
    bb_price_position_672_4h: float | None,
    bb_price_position_1440_4h: float | None,
    bb_price_position_105_1m: float | None,
    bb_price_position_300_1m: float | None,
) -> dict[str, float | bool]:
    h4_positions = [
        _bb_position(bb_price_position_105_4h),
        _bb_position(bb_price_position_672_4h),
        _bb_position(bb_price_position_1440_4h),
    ]
    m1_positions = [
        _bb_position(bb_price_position_105_1m),
        _bb_position(bb_price_position_300_1m),
    ]
    bb96_position = _bb_position(bb_price_position_96)
    h4_mid_retest = any(0.42 <= value <= 0.58 for value in h4_positions)
    h4_below_or_at_mid = any(value <= 0.58 for value in h4_positions)
    m1_upper_pressure = max(m1_positions)
    m1_upper_ready = m1_upper_pressure >= 0.82
    m15_bb96_ready = bb96_position >= 0.50
    bonus = 0.0
    if h4_mid_retest:
        bonus += 8.0
    elif h4_below_or_at_mid:
        bonus += 3.0
    if m1_upper_ready:
        bonus += min(8.0, max(0.0, m1_upper_pressure - 0.82) * 45.0 + 3.0)
    if m15_bb96_ready:
        bonus += min(5.0, max(0.0, bb96_position - 0.50) * 20.0 + 1.0)
    return {
        "h4_mid_retest": h4_mid_retest,
        "h4_below_or_at_mid": h4_below_or_at_mid,
        "m1_upper_ready": m1_upper_ready,
        "m1_upper_pressure": m1_upper_pressure,
        "m15_bb96_ready": m15_bb96_ready,
        "bb96_position": bb96_position,
        "active": bool((h4_mid_retest or h4_below_or_at_mid) and m1_upper_ready and m15_bb96_ready),
        "bonus": bonus,
    }


def classify_setup(
    *,
    price_change_pct_24h: float,
    price_change_pct_4h: float = 0.0,
    funding_rate: float,
    oi_change_pct_4h: float,
    taker_buy_sell_ratio: float,
    sentiment_score: float,
    listing_days: int | None,
    config: StrategyConfig,
    price_position_7d: float | None = None,
    price_position_14d: float | None = None,
    price_position_20d: float | None = None,
    breakout_distance_7d: float | None = None,
    compression_score: float | None = None,
    price_change_pct_5m: float = 0.0,
    quote_volume_usdt_24h: float = 0.0,
    sentiment_mentions: int = 0,
    bb_price_position_96: float | None = None,
    bb_price_position_105_4h: float | None = None,
    bb_price_position_672_4h: float | None = None,
    bb_price_position_1440_4h: float | None = None,
    bb_price_position_105_1m: float | None = None,
    bb_price_position_300_1m: float | None = None,
) -> tuple[str, float]:
    short_bb = _short_bb_structure_context(
        bb_price_position_96=bb_price_position_96,
        bb_price_position_105_4h=bb_price_position_105_4h,
        bb_price_position_672_4h=bb_price_position_672_4h,
        bb_price_position_1440_4h=bb_price_position_1440_4h,
        bb_price_position_105_1m=bb_price_position_105_1m,
        bb_price_position_300_1m=bb_price_position_300_1m,
    )
    if (
        price_change_pct_24h >= config.anomaly_price_change_pct
        and funding_rate <= config.anomaly_negative_funding_rate
        and oi_change_pct_4h >= config.min_oi_change_pct
    ):
        quality = min(100.0, 55.0 + min(25.0, price_change_pct_24h / 6.0) + min(20.0, abs(funding_rate) * 5000.0))
        return "squeeze_short_candidate", round(quality, 2)

    if (
        price_change_pct_24h <= -config.base_breakout_min_price_change_pct
        and funding_rate <= 0.0
        and oi_change_pct_4h >= config.min_oi_change_pct
        and taker_buy_sell_ratio <= 1.0
        and sentiment_score >= config.min_sentiment_score
        and price_position_7d is not None
        and price_position_7d <= (1.0 - config.base_breakout_min_price_position_7d + 0.05)
        and price_position_20d is not None
        and price_position_20d >= 0.28
        and breakout_distance_7d is not None
        and breakout_distance_7d <= config.base_breakout_max_breakout_distance_7d
        and compression_score is not None
        and compression_score >= config.base_breakout_min_compression_score
    ):
        quality = min(
            100.0,
            36.0
            + abs(price_change_pct_24h)
            + oi_change_pct_4h
            + ((1.0 - taker_buy_sell_ratio) * 80.0)
            + (compression_score * 12.0)
            + min(12.0, float(short_bb["bonus"])),
        )
        return "base_breakdown_confirmed", round(quality, 2)

    if (
        price_change_pct_24h <= -abs(config.short_scan_min_abs_price_change_pct)
        and quote_volume_usdt_24h >= config.short_scan_min_quote_volume_usdt
        and taker_buy_sell_ratio <= 1.0
        and price_change_pct_4h <= 2.0
        and (
            oi_change_pct_4h >= 0.0
            or funding_rate <= 0.0
            or taker_buy_sell_ratio <= 0.92
        )
        and (
            price_position_7d is None
            or price_position_7d <= 0.45
            or (price_position_20d is not None and price_position_20d <= 0.7)
        )
    ):
        quality = min(
            79.0,
            30.0
            + min(22.0, abs(price_change_pct_24h) * 0.8)
            + min(10.0, max(0.0, -price_change_pct_4h) * 2.5)
            + min(12.0, max(0.0, 1.0 - taker_buy_sell_ratio) * 60.0)
            + min(10.0, max(0.0, oi_change_pct_4h) * 1.5)
            + min(8.0, quote_volume_usdt_24h / 30_000_000.0),
        )
        return "base_breakdown_watch", round(quality, 2)

    if (
        bool(short_bb["active"])
        and quote_volume_usdt_24h >= config.short_scan_min_quote_volume_usdt
        and taker_buy_sell_ratio <= 1.05
        and price_change_pct_4h <= 2.0
    ):
        quality = min(
            79.0,
            34.0
            + min(18.0, float(short_bb["bonus"]))
            + min(8.0, max(0.0, 1.05 - taker_buy_sell_ratio) * 50.0)
            + min(8.0, max(0.0, -price_change_pct_4h) * 2.0)
            + min(6.0, quote_volume_usdt_24h / 50_000_000.0),
        )
        return "base_breakdown_watch", round(quality, 2)

    if (
        oi_change_pct_4h >= config.oi_divergence_min_oi_change_pct_4h
        and abs(price_change_pct_4h) <= config.oi_divergence_max_abs_price_change_pct_4h
        and sentiment_score >= config.oi_divergence_min_sentiment_score
        and sentiment_mentions >= config.oi_divergence_min_sentiment_mentions
        and quote_volume_usdt_24h >= config.oi_divergence_min_quote_volume_usdt
        and taker_buy_sell_ratio >= config.oi_divergence_min_taker_ratio
    ):
        divergence = _oi_price_divergence_score(oi_change_pct_4h, price_change_pct_4h)
        if divergence < config.oi_divergence_min_divergence_score:
            return "trend", 0.0
        quality = min(
            100.0,
            58.0
            + min(22.0, divergence * 0.18)
            + min(10.0, sentiment_mentions * 0.8)
            + min(10.0, max(0.0, sentiment_score - config.oi_divergence_min_sentiment_score) * 0.35),
        )
        return "oi_divergence_long_candidate", round(quality, 2)

    if (
        config.mania_followthrough_min_price_change_pct_24h <= price_change_pct_24h <= config.mania_followthrough_max_price_change_pct_24h
        and price_change_pct_5m >= config.mania_followthrough_min_price_change_pct_5m
        and sentiment_score >= config.mania_followthrough_min_sentiment_score
        and sentiment_mentions >= config.mania_followthrough_min_sentiment_mentions
        and quote_volume_usdt_24h >= config.mania_followthrough_min_quote_volume_usdt
        and oi_change_pct_4h >= config.mania_followthrough_min_oi_change_pct_4h
        and taker_buy_sell_ratio >= config.mania_followthrough_min_taker_ratio
        and (price_position_7d is None or price_position_7d >= 0.96)
        and (compression_score is None or compression_score <= 0.35)
    ):
        continuation_bonus = min(18.0, price_change_pct_5m * 2.5)
        sentiment_bonus = min(16.0, (sentiment_score - config.mania_followthrough_min_sentiment_score) * 0.6)
        taker_bonus = max(0.0, (taker_buy_sell_ratio - config.mania_followthrough_min_taker_ratio) * 60.0)
        quality = min(100.0, 56.0 + continuation_bonus + sentiment_bonus + taker_bonus + min(10.0, sentiment_mentions * 1.5))
        return "mania_followthrough", round(quality, 2)

    if (
        config.trend_followthrough_min_score <= sentiment_score
        and config.trend_followthrough_min_sentiment_mentions <= sentiment_mentions
        and price_change_pct_24h >= config.trend_followthrough_min_price_change_pct_24h
        and price_change_pct_5m >= 0.0
        and taker_buy_sell_ratio >= config.mania_followthrough_min_taker_ratio
        and quote_volume_usdt_24h >= config.mania_followthrough_min_quote_volume_usdt
        and (price_position_7d is None or price_position_7d >= 0.90)
        and (compression_score is None or compression_score <= 0.45)
    ):
        hotness_bonus = max(0.0, min(12.0, (price_change_pct_24h - config.trend_followthrough_min_price_change_pct_24h) * 0.5))
        quality = min(100.0, 52.0 + hotness_bonus + min(12.0, sentiment_mentions * 1.0) + min(12.0, (sentiment_score - config.trend_followthrough_min_score) * 0.4))
        return "trend_followthrough", round(quality, 2)

    if (
        config.base_breakout_min_price_change_pct <= price_change_pct_24h <= config.base_breakout_max_price_change_pct
        and oi_change_pct_4h >= config.min_oi_change_pct
        and taker_buy_sell_ratio >= config.min_taker_ratio
        and sentiment_score >= config.min_sentiment_score
        and price_position_7d is not None
        and price_position_7d >= config.base_breakout_min_price_position_7d
        and price_position_20d is not None
        and price_position_20d <= config.base_breakout_max_price_position_20d
        and breakout_distance_7d is not None
        and breakout_distance_7d <= config.base_breakout_max_breakout_distance_7d
        and compression_score is not None
        and compression_score >= config.base_breakout_min_compression_score
    ):
        freshness_bonus = 8.0 if listing_days is not None and listing_days <= 60 else 0.0
        mid_base_bonus = 0.0 if price_position_14d is None else max(0.0, (0.85 - price_position_14d) * 10.0)
        structure_bonus = (
            ((price_position_7d - 0.85) * 35.0)
            + ((1.0 - price_position_20d) * 18.0)
            + (compression_score * 12.0)
            + ((max(0.0, 0.035 - breakout_distance_7d) / 0.035) * 10.0)
            + mid_base_bonus
        )
        quality = min(
            100.0,
            34.0 + price_change_pct_24h + oi_change_pct_4h + ((taker_buy_sell_ratio - 1.0) * 40.0) + freshness_bonus + structure_bonus,
        )
        setup_name = "base_breakout_confirmed" if quality >= 80.0 else "base_breakout_weak"
        return setup_name, round(quality, 2)

    return "trend", 0.0


def _market_structure_from_payload(item: dict[str, Any], mark_price: float) -> dict[str, float | None]:
    explicit = {
        "price_position_7d": _coerce_optional_float(item.get("price_position_7d")),
        "price_position_14d": _coerce_optional_float(item.get("price_position_14d")),
        "price_position_20d": _coerce_optional_float(item.get("price_position_20d")),
        "breakout_distance_7d": _coerce_optional_float(item.get("breakout_distance_7d")),
        "compression_score": _coerce_optional_float(item.get("compression_score")),
    }
    if all(value is not None for value in explicit.values()):
        return explicit
    closes = item.get("daily_closes") or []
    if closes:
        return _market_structure_from_closes([float(price) for price in closes], mark_price=mark_price)
    return explicit


KLINE_STAT_KEYS = (
    "kline_lower_shadow_pct", "kline_upper_shadow_pct", "kline_body_pct",
    "kline_direction", "kline_volume_vs_ma5", "kline_range_vs_ma5_range",
    "kline_close_position", "kline_consecutive_direction", "kline_atr_pct",
    "kline_ma20_distance_pct", "kline_volume_trend", "kline_max_drawdown_5d",
)


def _kline_statistics(klines: list[Any], *, mark_price: float) -> dict[str, float]:
    if not klines or len(klines) < 5:
        return {k: 0.0 for k in KLINE_STAT_KEYS}
    try:
        latest = klines[-1]
        o, h, l, c, v = float(latest[1]), float(latest[2]), float(latest[3]), float(latest[4]), float(latest[5])
    except (IndexError, TypeError, ValueError):
        return {k: 0.0 for k in KLINE_STAT_KEYS}
    range_ = max(h - l, 1e-9)
    lower_shadow_pct = (min(o, c) - l) / range_
    upper_shadow_pct = (h - max(o, c)) / range_
    body_pct = abs(c - o) / range_
    direction = 1.0 if c >= o else -1.0
    close_position = (c - l) / range_

    volumes = []
    for k in klines[-5:]:
        try:
            volumes.append(float(k[5]))
        except (IndexError, TypeError, ValueError):
            pass
    ma5_vol = sum(volumes) / len(volumes) if volumes else 1.0
    volume_vs_ma5 = v / max(ma5_vol, 1e-9)

    ranges = []
    for k in klines[-5:]:
        try:
            ranges.append(float(k[2]) - float(k[3]))
        except (IndexError, TypeError, ValueError):
            pass
    ma5_range = sum(ranges) / len(ranges) if ranges else 1.0
    range_vs_ma5 = range_ / max(ma5_range, 1e-9)

    consecutive = 0
    for k in reversed(klines[:-1]):
        try:
            kc, ko = float(k[4]), float(k[1])
        except (IndexError, TypeError, ValueError):
            break
        if (kc >= ko) == (direction > 0):
            consecutive += 1
        else:
            break

    atr_values = []
    for k in klines[-20:]:
        try:
            atr_values.append(float(k[2]) - float(k[3]))
        except (IndexError, TypeError, ValueError):
            pass
    atr = sum(atr_values) / len(atr_values) if atr_values else 0.0
    atr_pct = atr / max(mark_price, 1e-9)

    closes_20 = []
    for k in klines[-20:]:
        try:
            closes_20.append(float(k[4]))
        except (IndexError, TypeError, ValueError):
            pass
    ma20 = sum(closes_20) / len(closes_20) if closes_20 else mark_price
    ma20_distance_pct = (mark_price - ma20) / max(abs(ma20), 1e-9)

    volumes_20 = []
    for k in klines[-20:]:
        try:
            volumes_20.append(float(k[5]))
        except (IndexError, TypeError, ValueError):
            pass
    ma20_vol = sum(volumes_20) / len(volumes_20) if volumes_20 else 1.0
    volume_trend = ma5_vol / max(ma20_vol, 1e-9)

    max_drawdown_5d = 0.0
    for k in klines[-5:]:
        try:
            ko, kc = float(k[1]), float(k[4])
            if ko > 0:
                dd = (kc - ko) / ko
                if dd < max_drawdown_5d:
                    max_drawdown_5d = dd
        except (IndexError, TypeError, ValueError):
            pass

    return {
        "kline_lower_shadow_pct": round(lower_shadow_pct, 4),
        "kline_upper_shadow_pct": round(upper_shadow_pct, 4),
        "kline_body_pct": round(body_pct, 4),
        "kline_direction": direction,
        "kline_volume_vs_ma5": round(volume_vs_ma5, 4),
        "kline_range_vs_ma5_range": round(range_vs_ma5, 4),
        "kline_close_position": round(close_position, 4),
        "kline_consecutive_direction": float(consecutive) * direction,
        "kline_atr_pct": round(atr_pct, 6),
        "kline_ma20_distance_pct": round(ma20_distance_pct, 4),
        "kline_volume_trend": round(volume_trend, 4),
        "kline_max_drawdown_5d": round(max_drawdown_5d, 4),
    }


def _detect_kline_patterns(klines: list[Any]) -> dict[str, Any]:
    """Detect K-line patterns from daily klines (observation-only side-channel)."""
    try:
        from trading_system.kline_pattern import detect_patterns
        patterns = detect_patterns(klines)
        bullish = sum(1 for p in patterns if p.direction == "bullish")
        bearish = sum(1 for p in patterns if p.direction == "bearish")
        flags = ",".join(p.name for p in patterns) if patterns else ""
        return {"bullish_count": float(bullish), "bearish_count": float(bearish), "flags": flags}
    except Exception:
        return {"bullish_count": 0.0, "bearish_count": 0.0, "flags": ""}


def _market_structure_from_klines(klines: list[Any], *, mark_price: float) -> dict[str, float | None]:
    closes = [float(entry[4]) for entry in klines if len(entry) >= 5]
    return _market_structure_from_closes(closes, mark_price=mark_price)


def _market_structure_from_closes(closes: list[float], *, mark_price: float) -> dict[str, float | None]:
    if not closes or mark_price <= 0:
        return {
            "price_position_7d": None,
            "price_position_14d": None,
            "price_position_20d": None,
            "breakout_distance_7d": None,
            "compression_score": None,
        }
    prices = closes[-20:]
    return {
        "price_position_7d": _position_in_window(prices[-7:], mark_price),
        "price_position_14d": _position_in_window(prices[-14:], mark_price),
        "price_position_20d": _position_in_window(prices, mark_price),
        "breakout_distance_7d": _breakout_distance(prices[-7:], mark_price),
        "compression_score": _compression_score(prices),
    }


def _position_in_window(prices: list[float], mark_price: float) -> float | None:
    if len(prices) < 2:
        return None
    low = min(prices)
    high = max(prices)
    if high <= low:
        return 1.0
    return round(max(0.0, min(1.0, (mark_price - low) / (high - low))), 4)


def _breakout_distance(prices: list[float], mark_price: float) -> float | None:
    if len(prices) < 2:
        return None
    prior_high = max(prices[:-1])
    if prior_high <= 0:
        return None
    return round(max(0.0, (mark_price - prior_high) / prior_high), 4)


def _compression_score(prices: list[float]) -> float | None:
    if len(prices) < 10:
        return None
    short_range = max(prices[-7:]) - min(prices[-7:])
    long_range = max(prices) - min(prices)
    if long_range <= 0:
        return 1.0
    ratio = short_range / long_range
    return round(max(0.0, min(1.0, 1.0 - ratio)), 4)


def _price_change_pct_from_klines(klines: list[Any], *, current_price: float) -> float:
    closes = [float(entry[4]) for entry in klines if len(entry) >= 5]
    return _price_change_pct_from_closes(closes, current_price=current_price) or 0.0


def _price_change_pct_from_closes(closes: list[Any], *, current_price: float) -> float | None:
    if not closes or current_price <= 0:
        return None
    try:
        start_price = float(closes[0])
    except (TypeError, ValueError):
        return None
    if start_price <= 0:
        return None
    return round(((current_price - start_price) / start_price) * 100.0, 4)


def _coerce_optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
