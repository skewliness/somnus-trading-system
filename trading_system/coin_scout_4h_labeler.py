"""
妖币模型 - 4H Klines 精确标签生成器 (4H Klines Precise Labeler)
====================================================================

用 4H klines 数据为训练样本生成精确标签。

核心策略：
  - 主要数据源：decision_samples（13,351 条，含完整特征）
  - 标签方式：用 captured_at 时间 + symbol，获取后续 4H klines
  - 批量优化：按 symbol 分组，每个 symbol 只获取一次 klines

标签定义（4H 框架，side-aware）：
  - 方向有效 (1): 后续 4H/24H 内顺方向走势有效且逆向回撤可接受
  - 方向无效 (0): 其他

Author: OWL for snow
Date: 2026-05-08
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import parse

import requests

from trading_system.binance_public_guard import (
    BinancePublicRateLimitError,
    assert_binance_public_not_banned,
    note_binance_public_response,
    throttle_binance_public_request,
)
from trading_system.coin_scout_model import (
    CoinFeatures,
    CoinScoutModel,
)

logger = logging.getLogger(__name__)


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _local_http_proxy_available(proxy_url: str, *, timeout: float = 0.25) -> bool:
    try:
        parsed = parse.urlparse(proxy_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host or port <= 0:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_binance_proxy_url() -> str:
    configured = _first_env("TS_BINANCE_PROXY_URL", "TS_V6_BINANCE_PROXY_URL", "BINANCE_PROXY_URL")
    if configured.strip().lower() in {"0", "false", "off", "none", "direct"}:
        return ""
    if configured.strip():
        return configured.strip()
    if not _env_bool("TS_BINANCE_AUTO_LOCAL_PROXY", True):
        return ""
    local_proxy = _first_env("TS_LOCAL_PROXY_URL", "LOCAL_PROXY_URL", default="http://127.0.0.1:7897").strip()
    if local_proxy and _local_http_proxy_available(local_proxy):
        return local_proxy
    return ""

# Binance Futures API
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_API_TIMEOUT = 15
API_RATE_LIMIT_DELAY = 0.25  # 250ms between calls → ~4 req/s
INTERVAL_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}
BINANCE_FAPI_SYMBOL_ALIASES = {
    "PEPEUSDT": ["1000PEPEUSDT"],
    "SHIBUSDT": ["1000SHIBUSDT"],
    "BONKUSDT": ["1000BONKUSDT"],
    "FLOKIUSDT": ["1000FLOKIUSDT"],
    "LUNCUSDT": ["1000LUNCUSDT"],
    "XECUSDT": ["1000XECUSDT"],
    "SATSUSDT": ["1000SATSUSDT"],
    "RATSUSDT": ["1000RATSUSDT"],
}


def binance_fapi_symbol_candidates(symbol: str) -> list[str]:
    """Return Binance Futures symbols to try for a venue-normalized symbol."""
    normalized = str(symbol or "").upper().replace("_", "").strip()
    if not normalized:
        return []
    candidates = [normalized, *BINANCE_FAPI_SYMBOL_ALIASES.get(normalized, [])]
    if normalized.endswith("USDT") and not normalized.startswith("1000"):
        candidates.append(f"1000{normalized}")
    deduped: list[str] = []
    for item in candidates:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


class CoinScout4HLabeler:
    """
    4H klines 精确标签生成器

    标签逻辑（用户的交易哲学）：
    - "4H 拿" → 用 4H 框架评估
    - "回踩加" → drawdown 要小，才能加仓
    - "破位走" → drawdown 太大就是破位
    """

    # 标签阈值（基于用户交易哲学：早发现，小仓进，4H 拿，回踩加，破位走）
    # drawdown 从入场价算起（不是峰值），因为"破位走"是相对入场价
    #
    DIRECTION_GAIN = 0.025
    DIRECTION_CLOSE_RETURN = 0.015
    DIRECTION_DD = 0.055

    # 评估窗口：入场后 6 根 4H klines ≈ 24H
    EVAL_BARS = 6

    def __init__(self, cache_dir: str = "data/trading_system/model_training/klines_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, list[dict]] = {}
        self.session = requests.Session()
        self.session.trust_env = True
        proxy_url = resolve_binance_proxy_url()
        if proxy_url:
            self.session.proxies.update({"http": proxy_url, "https": proxy_url})

    def _get(self, url: str, *, params: dict[str, Any], timeout: float) -> requests.Response:
        retries = max(1, int(os.getenv("TS_BINANCE_REQUEST_RETRIES", "3") or "3"))
        last_exc: requests.RequestException | None = None
        for attempt in range(retries):
            try:
                return self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt + 1 < retries:
                    time.sleep(0.6 * (2 ** attempt))
        raise last_exc if last_exc is not None else RuntimeError("Binance kline request failed before response")

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        *,
        force_refresh: bool = False,
    ) -> list[dict]:
        """获取指定周期 klines。

        默认保留训练/回放路径的内存+磁盘缓存；命令面板、report/fc 等临盘路径
        传入 force_refresh=True 时必须绕过缓存，直接从 Binance 补齐窗口K线后再写回缓存。
        """
        interval = str(interval or "4h").lower()
        if interval not in INTERVAL_MS:
            raise ValueError(f"unsupported kline interval: {interval}")
        cache_key = f"{symbol}_{interval}_{start_dt.strftime('%Y%m%d%H%M')}_{end_dt.strftime('%Y%m%d%H%M')}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not force_refresh and cache_key in self._mem_cache:
            return self._mem_cache[cache_key]

        if not force_refresh and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                self._mem_cache[cache_key] = data
                return data
            except Exception:
                pass

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        all_klines: list[dict] = []
        errors: list[str] = []

        for api_symbol in binance_fapi_symbol_candidates(symbol):
            fetched, error = self._fetch_klines_for_api_symbol(api_symbol, interval, start_ms, end_ms)
            if fetched:
                all_klines = fetched
                if api_symbol != symbol:
                    logger.info("Resolved Binance Futures symbol alias: %s -> %s", symbol, api_symbol)
                break
            if error:
                errors.append(f"{api_symbol}: {error}")

        if not all_klines and errors:
            logger.warning("Binance API error for %s %s: %s", symbol, interval, " | ".join(errors[:3]))

        if all_klines:
            try:
                cache_file.write_text(json.dumps(all_klines), encoding="utf-8")
            except Exception:
                pass
            self._mem_cache[cache_key] = all_klines

        return all_klines

    def fetch_4h_klines(self, symbol: str, start_dt: datetime, end_dt: datetime) -> list[dict]:
        """获取 4H klines，带内存+磁盘缓存。"""
        return self.fetch_klines(symbol, "4h", start_dt, end_dt)

    def _fetch_klines_for_api_symbol(
        self,
        api_symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> tuple[list[dict], str]:
        all_klines: list[dict] = []
        current_start = start_ms
        step_ms = INTERVAL_MS[interval]

        while current_start < end_ms:
            try:
                assert_binance_public_not_banned(f"labeler:/fapi/v1/klines:{api_symbol}:{interval}")
                throttle_binance_public_request()
                resp = self._get(
                    BINANCE_FAPI,
                    params={
                        "symbol": api_symbol,
                        "interval": interval,
                        "startTime": current_start,
                        "endTime": end_ms,
                        "limit": 500,
                    },
                    timeout=BINANCE_API_TIMEOUT,
                )
                note_binance_public_response(
                    resp.status_code,
                    resp.text,
                    f"labeler:/fapi/v1/klines:{api_symbol}:{interval}",
                )
                if resp.status_code == 418:
                    return all_klines, resp.text
                if resp.status_code == 429:
                    logger.warning("Rate limited, waiting 10s...")
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                klines = resp.json()
                if not klines:
                    break

                for k in klines:
                    all_klines.append({
                        "open_time": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "binance_symbol": api_symbol,
                        "interval": interval,
                    })

                current_start = klines[-1][0] + step_ms
                time.sleep(API_RATE_LIMIT_DELAY)

            except BinancePublicRateLimitError as exc:
                return all_klines, str(exc)
            except Exception as exc:
                time.sleep(2)
                return all_klines, str(exc)

        return all_klines, ""

    def compute_4h_backtest_metrics(
        self, entry_price: float, klines_after: list[dict], side: str = "long"
    ) -> dict[str, float | int | bool]:
        """
        从入场后的 4H klines 计算 side-aware 回测指标。

        Args:
            entry_price: 入场价
            klines_after: 入场后的 4H klines
            side: "long" or "short"

        Returns: label/max_gain/max_drawdown/final_unrealized_pnl metrics.
        """
        if not klines_after or entry_price <= 0:
            return {
                "label": 0,
                "max_gain": 0.0,
                "max_drawdown": 0.0,
                "final_unrealized_pnl": 0.0,
                "direction_valid": False,
                "eval_bars": 0,
            }

        max_gain = 0.0
        max_dd_from_entry = 0.0
        close_return = 0.0

        usable = klines_after[:self.EVAL_BARS]
        for k in usable:
            high = k["high"]
            low = k["low"]
            close = k["close"]
            if side == "short":
                gain = (entry_price - low) / entry_price
                dd = (high - entry_price) / entry_price
                close_return = (entry_price - close) / entry_price
            else:
                gain = (high - entry_price) / entry_price
                dd = (entry_price - low) / entry_price
                close_return = (close - entry_price) / entry_price
            if gain > max_gain:
                max_gain = gain
            if dd > max_dd_from_entry:
                max_dd_from_entry = dd

        max_dd = max_dd_from_entry

        if (
            (max_gain >= self.DIRECTION_GAIN and max_dd <= self.DIRECTION_DD)
            or (close_return >= self.DIRECTION_CLOSE_RETURN and max_gain >= 0.018 and max_dd <= self.DIRECTION_DD)
        ):
            label = 1
        else:
            label = 0
        return {
            "label": label,
            "max_gain": max_gain,
            "max_drawdown": max_dd,
            "final_unrealized_pnl": close_return,
            "direction_valid": bool(label),
            "eval_bars": len(usable),
        }

    def compute_4h_label(
        self, entry_price: float, klines_after: list[dict], side: str = "long"
    ) -> tuple[int, float, float]:
        """
        从入场后的 4H klines 计算标签（side-aware）。

        Returns: (label, max_gain, max_drawdown)
        """
        metrics = self.compute_4h_backtest_metrics(entry_price, klines_after, side=side)
        return int(metrics["label"]), float(metrics["max_gain"]), float(metrics["max_drawdown"])

    def build_dataset_from_decision_samples(
        self,
        decision_samples_path: str | Path,
        max_symbols: int | None = None,
        min_score_for_positive: float = 50.0,
        max_neg_per_pos: int = 3,
    ) -> dict:
        """
        从 decision_samples 构建 4H 标签训练集。

        核心策略：
        1. 按 symbol 分组，每个 symbol 只获取一次 klines
        2. 对每个 sample，用 captured_at 之后的 klines 计算 4H 标签
        3. 保留所有特征完整的样本
        4. 控制正负比例

        Args:
            decision_samples_path: decision_samples.jsonl 路径
            max_symbols: 最多处理多少个 symbol（None=全部）
            min_score_for_positive: 只对这个分数以上的样本做 4H 验证
            max_neg_per_pos: 正负样本最大比例
        """
        decision_samples_path = Path(decision_samples_path)
        if not decision_samples_path.exists():
            return {"status": "error", "message": f"File not found: {decision_samples_path}"}

        # 加载所有 samples
        all_samples = self._load_jsonl(decision_samples_path)
        logger.info("Loaded %d decision samples", len(all_samples))

        # 按 symbol 分组
        by_symbol = defaultdict(list)
        for s in all_samples:
            sym = s.get("symbol", "")
            if sym:
                by_symbol[sym].append(s)

        logger.info("Found %d unique symbols", len(by_symbol))

        # 选择要处理的 symbol（优先处理样本多的）
        sorted_symbols = sorted(by_symbol.keys(), key=lambda s: len(by_symbol[s]), reverse=True)
        if max_symbols:
            sorted_symbols = sorted_symbols[:max_symbols]

        all_features = []
        all_labels = []
        api_call_count = 0
        labeled_count = 0
        skipped_count = 0

        for sym_idx, symbol in enumerate(sorted_symbols):
            samples = by_symbol[symbol]

            # 确定这个 symbol 的时间范围
            timestamps = []
            for s in samples:
                cap = s.get("captured_at", "")
                if cap:
                    try:
                        dt = datetime.fromisoformat(cap.replace("Z", "+00:00"))
                        timestamps.append(dt)
                    except Exception:
                        pass

            if not timestamps:
                continue

            min_ts = min(timestamps)
            max_ts = max(timestamps)

            # 获取 klines：从最早 sample 前 1 天到最晚 sample 后 2 天
            klines_start = min_ts - timedelta(days=1)
            klines_end = max_ts + timedelta(days=2)

            klines = self.fetch_4h_klines(symbol, klines_start, klines_end)
            api_call_count += 1

            if not klines:
                logger.debug("No klines for %s, skipping %d samples", symbol, len(samples))
                skipped_count += len(samples)
                continue

            # 对这个 symbol 的每条 sample 计算 4H 标签
            sym_labeled = 0
            for s in samples:
                cap_at = s.get("captured_at", "")
                if not cap_at:
                    continue

                try:
                    cap_dt = datetime.fromisoformat(cap_at.replace("Z", "+00:00"))
                except Exception:
                    continue

                # 找 captured_at 之后的 klines
                cap_ms = int(cap_dt.timestamp() * 1000)
                klines_after = [k for k in klines if k["open_time"] >= cap_ms]

                if len(klines_after) < 3:
                    continue

                # 入场参考价：用第一根 kline 的 open
                entry_price = klines_after[0]["open"]
                if entry_price <= 0:
                    continue

                label, max_gain, max_dd = self.compute_4h_label(entry_price, klines_after)

                # 构建特征
                features = self._features_from_decision(s)
                all_features.append(features)
                all_labels.append(label)
                labeled_count += 1
                sym_labeled += 1

            if (sym_idx + 1) % 10 == 0:
                dist = {0: 0, 1: 0, 2: 0}
                for l in all_labels:
                    dist[l] = dist.get(l, 0) + 1
                logger.info(
                    "Progress: %d/%d symbols, %d labeled, dist=%s",
                    sym_idx + 1, len(sorted_symbols), labeled_count, dist,
                )

        # 统计
        dist = {0: 0, 1: 0, 2: 0}
        for l in all_labels:
            dist[l] = dist.get(l, 0) + 1
        total = len(all_features)
        pos = dist[1] + dist[2]
        neg = dist[0]

        logger.info(
            "Dataset built: %d samples (%d pos, %d neg), API calls: %d, skipped: %d",
            total, pos, neg, api_call_count, skipped_count,
        )
        logger.info("Label distribution: %s", dist)

        return {
            "status": "ok",
            "total_samples": total,
            "label_distribution": {
                "方向无效(0)": dist[0],
                "方向有效(1)": dist[1],
            },
            "pos_neg_ratio": f"{pos}:{neg}" if neg > 0 else "all_pos",
            "api_calls": api_call_count,
            "skipped": skipped_count,
            "features_list": all_features,
            "labels": all_labels,
        }

    def _features_from_decision(self, ds: dict) -> CoinFeatures:
        """从 decision_sample 构建 CoinFeatures"""
        f = CoinFeatures(symbol=ds.get("symbol", "UNKNOWN"))

        # OI
        f.oi_change_pct_4h = float(ds.get("oi_change_pct_4h") or 0)
        f.funding_rate = float(ds.get("funding_rate") or 0)
        f.oi_price_divergence_score = float(ds.get("oi_price_divergence_score") or 0)

        # 情绪
        f.sentiment_score = float(ds.get("sentiment_score") or 0)
        f.sentiment_mentions = float(ds.get("sentiment_mentions") or 0)

        # 市场微结构
        f.price_change_pct_5m = float(ds.get("price_change_pct_5m") or 0)
        f.price_change_pct_24h = float(ds.get("price_change_pct_24h") or 0)
        f.taker_buy_sell_ratio = float(ds.get("taker_buy_sell_ratio") or 0)
        f.price_position_7d = float(ds.get("price_position_7d") or 0)
        f.price_position_14d = float(ds.get("price_position_14d") or 0)
        f.price_position_20d = float(ds.get("price_position_20d") or 0)
        f.breakout_distance_7d = float(ds.get("breakout_distance_7d") or 0)
        f.compression_score = float(ds.get("compression_score") or 0)
        f.quote_volume_usdt_24h = float(ds.get("quote_volume_usdt_24h") or 0)

        def _sf(key: str, default: float = 0.0) -> float:
            try:
                return float(ds.get(key, default) or default)
            except Exception:
                return default

        def _pos(key: str, default: float = 0.5) -> float:
            value = _sf(key, default)
            return max(0.0, min(1.0, value))

        price = _sf("mark_price", _sf("entry_price", _sf("current_price", 0.0)))

        def _distance_pct(reference_key: str) -> float:
            return 0.0

        f.bb_price_position_96 = _pos("bb_price_position_96", 0.5)
        f.bb_price_position_105_4h = _pos("bb_price_position_105_4h", 0.5)
        f.bb_price_position_672_4h = _pos("bb_price_position_672_4h", 0.5)
        f.bb_price_position_1440_4h = _pos("bb_price_position_1440_4h", 0.5)
        f.bb_price_position_105_1m = _pos("bb_price_position_105_1m", 0.5)
        f.bb_price_position_300_1m = _pos("bb_price_position_300_1m", 0.5)
        f.bb105_mid_4h_distance_pct = _distance_pct("bb105_mid_4h")
        f.bb672_mid_4h_distance_pct = _distance_pct("bb672_mid_4h")
        f.bb1440_mid_4h_distance_pct = _distance_pct("bb1440_mid_4h")
        f.bb105_upper_1m_distance_pct = _distance_pct("bb105_upper_1m")
        f.bb300_upper_1m_distance_pct = _distance_pct("bb300_upper_1m")
        f.bb96_upper_15m_distance_pct = _distance_pct("bb96_upper_15m")
        f.short_4h_mid_retest_distance_min_pct = min(
            [
                value
                for value in (
                    -f.bb105_mid_4h_distance_pct,
                    -f.bb672_mid_4h_distance_pct,
                    -f.bb1440_mid_4h_distance_pct,
                )
                if value > 0
            ],
            default=0.0,
        )
        f.short_1m_upper_pressure_max = max(f.bb_price_position_105_1m, f.bb_price_position_300_1m)
        f.short_15m_bb96_position = f.bb_price_position_96

        # 鲸鱼/聪明钱（decision_samples 中可能有）
        f.whale_long_count = float(ds.get("whale_long_count") or 0)
        f.whale_short_count = float(ds.get("whale_short_count") or 0)
        f.whale_long_notional_usdt = float(ds.get("whale_long_notional_usdt") or 0)
        f.whale_short_notional_usdt = float(ds.get("whale_short_notional_usdt") or 0)
        f.whale_pressure_score = float(ds.get("whale_pressure_score") or 0)
        f.smart_money_long_count = float(ds.get("smart_money_long_count") or 0)
        f.smart_money_short_count = float(ds.get("smart_money_short_count") or 0)
        f.smart_money_long_profit_ratio = float(ds.get("smart_money_long_profit_ratio") or 0)
        f.smart_money_short_profit_ratio = float(ds.get("smart_money_short_profit_ratio") or 0)
        f.smart_money_single_side_pct = float(ds.get("smart_money_single_side_pct") or 0)

        # 交互特征 — aligned with FeatureEngine._calc_momentum_score
        f.sentiment_oi_interaction = f.sentiment_score * f.oi_change_pct_4h / 100.0
        sentiment_momentum = f.sentiment_score_delta_4h
        price_momentum = f.price_change_pct_4h if f.price_change_pct_4h else f.price_change_pct_24h
        oi_momentum = f.oi_change_pct_4h
        if price_momentum > 0 and oi_momentum > 0 and sentiment_momentum > 0:
            f.momentum_score = (price_momentum + oi_momentum + sentiment_momentum) * 1.5
        elif price_momentum > 0 and oi_momentum > 0:
            f.momentum_score = (price_momentum + oi_momentum) * 1.2
        else:
            f.momentum_score = price_momentum + oi_momentum * 0.5 + sentiment_momentum * 0.3

        if f.compression_score > 0.5:
            f.structure_quality += f.compression_score * 30
        if f.breakout_distance_7d < 0.03:
            f.structure_quality += (0.03 - f.breakout_distance_7d) * 500
        f.structure_quality = min(100.0, f.structure_quality)

        # Use sample's captured_at for time features, not datetime.now()
        # to avoid distribution shift between training and inference.
        cap_at = ds.get("captured_at", "")
        try:
            sample_dt = datetime.fromisoformat(cap_at.replace("Z", "+00:00")) if cap_at else datetime.now(timezone.utc)
        except Exception:
            sample_dt = datetime.now(timezone.utc)
        f.hour_of_day = float(sample_dt.hour)
        f.day_of_week = float(sample_dt.weekday())
        f.is_asian_session = 1.0 if 0 <= sample_dt.hour < 8 else 0.0
        f.is_us_session = 1.0 if 13 <= sample_dt.hour < 21 else 0.0

        return f

    def _load_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows


def build_and_train_4h(
    decision_samples_path: str | Path = "data/trading_system/model_training/paper_replay_autolearn/paper_decision_samples.jsonl",
    model_dir: str = "data/trading_system/model_training/coin_scout",
    max_symbols: int | None = None,
) -> dict:
    """
    便捷函数：4H 标签 + 训练一条龙

    Args:
        decision_samples_path: decision_samples.jsonl 路径
        model_dir: 模型输出目录
        max_symbols: 最多处理 symbol 数（None=全部，用于测试可设小值）
    """
    labeler = CoinScout4HLabeler()
    dataset = labeler.build_dataset_from_decision_samples(
        decision_samples_path=decision_samples_path,
        max_symbols=max_symbols,
    )

    if dataset.get("status") != "ok":
        return dataset

    total = dataset["total_samples"]
    if total < 30:
        return {
            "status": "insufficient",
            "total_samples": total,
            "label_distribution": dataset["label_distribution"],
        }

    model = CoinScoutModel(model_dir=model_dir)
    train_result = model.train(dataset["features_list"], dataset["labels"])

    summary = {
        "status": train_result.get("status", "unknown"),
        "total_samples": total,
        "label_distribution": dataset["label_distribution"],
        "train_result": train_result,
        "source": "4h_klines_labeled",
        "api_calls": dataset.get("api_calls", 0),
        "feature_count": len(CoinFeatures.feature_names()),
    }
    summary_path = Path(model_dir) / "training_summary_4h_v2.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    max_sym = int(sys.argv[1]) if len(sys.argv) > 1 else None
    result = build_and_train_4h(max_symbols=max_sym)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
