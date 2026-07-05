#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse, request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.bb_structure import SHARED_BB_PERIODS


BINANCE_FAPI_BASE = "https://fapi.binance.com"
INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}
HORIZONS = {
    "15m": [16, 32, 96],  # 4h, 8h, 24h
    "4h": [6, 12, 30],  # 24h, 48h, 5d
    "1d": [7, 14, 30],
}
PRIMARY_HORIZON = {
    "15m": 32,
    "4h": 12,
    "1d": 14,
}
COOLDOWN_BARS = {
    "15m": 4,
    "4h": 2,
    "1d": 1,
}
BANDS = ("lower", "mid", "upper")


@dataclass(slots=True)
class KlineSet:
    open_time: np.ndarray
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


def utc_iso(ms: int | None = None) -> str:
    if ms is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc).isoformat()


def parse_utc_date_to_ms(value: str, *, end_of_day: bool = False) -> int:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty date")
    if "T" not in text and len(text) == 10:
        text = f"{text}T23:59:59+00:00" if end_of_day else f"{text}T00:00:00+00:00"
    text = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def pct_diff(price: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return (price - reference) / reference * 100.0


def fetch_json(url: str, retries: int = 10) -> list:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if "429" in str(exc):
                time.sleep(min(30.0, 6.0 * (attempt + 1)))
            else:
                time.sleep(min(3.0, 0.5 * (attempt + 1)))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    interval_ms = INTERVAL_MS[interval]
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        }
        url = f"{BINANCE_FAPI_BASE}/fapi/v1/klines?{parse.urlencode(params)}"
        batch = fetch_json(url)
        if not batch:
            break
        for row in batch:
            if not rows or int(row[0]) > int(rows[-1][0]):
                rows.append(row)
        last_open = int(batch[-1][0])
        next_cursor = last_open + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.12)
    return rows


def normalize_klines(rows: list[list]) -> list[list]:
    by_open: dict[int, list] = {}
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        try:
            open_time = int(row[0])
        except (TypeError, ValueError):
            continue
        by_open[open_time] = row
    return [by_open[key] for key in sorted(by_open)]


def cache_path(cache_dir: Path, symbol: str, interval: str) -> Path:
    return cache_dir / interval / f"{symbol.upper()}.json"


def load_cached_klines(cache_dir: Path, symbol: str, interval: str) -> list[list]:
    path = cache_path(cache_dir, symbol, interval)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return normalize_klines(payload)
    if isinstance(payload, dict):
        return normalize_klines(payload.get("klines") or [])
    return []


def save_cached_klines(cache_dir: Path, symbol: str, interval: str, rows: list[list], *, earliest_exhausted: bool) -> None:
    rows = normalize_klines(rows)
    path = cache_path(cache_dir, symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "binance_futures_mainnet",
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": len(rows),
        "earliest_exhausted": bool(earliest_exhausted),
        "first_open_time": int(rows[0][0]) if rows else None,
        "last_open_time": int(rows[-1][0]) if rows else None,
        "updated_at": utc_iso(),
        "klines": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def fetch_and_cache_klines(symbol: str, interval: str, start_ms: int, end_ms: int, cache_dir: Path) -> list[list]:
    cached = load_cached_klines(cache_dir, symbol, interval)
    if cached:
        cached = [row for row in cached if start_ms <= int(row[0]) <= end_ms]
    if cached and int(cached[0][0]) <= start_ms + INTERVAL_MS[interval] and int(cached[-1][0]) >= end_ms - INTERVAL_MS[interval]:
        return cached
    fetched = fetch_klines(symbol, interval, start_ms, end_ms)
    merged = normalize_klines(cached + fetched)
    save_cached_klines(cache_dir, symbol, interval, merged, earliest_exhausted=False)
    return [row for row in merged if start_ms <= int(row[0]) <= end_ms]


def to_kline_set(rows: list[list]) -> KlineSet:
    return KlineSet(
        open_time=np.array([int(row[0]) for row in rows], dtype=np.int64),
        open_=np.array([float(row[1]) for row in rows], dtype=np.float64),
        high=np.array([float(row[2]) for row in rows], dtype=np.float64),
        low=np.array([float(row[3]) for row in rows], dtype=np.float64),
        close=np.array([float(row[4]) for row in rows], dtype=np.float64),
        volume=np.array([float(row[5]) if len(row) > 5 else 0.0 for row in rows], dtype=np.float64),
    )


def rolling_band(closes: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = closes.size
    mid = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return upper, mid, lower
    csum = np.concatenate(([0.0], np.cumsum(closes)))
    csum2 = np.concatenate(([0.0], np.cumsum(closes * closes)))
    sums = csum[period:] - csum[:-period]
    sums2 = csum2[period:] - csum2[:-period]
    means = sums / period
    variance = np.maximum(sums2 / period - means * means, 0.0)
    std = np.sqrt(variance)
    mid[period - 1 :] = means
    upper[period - 1 :] = means + 2.0 * std
    lower[period - 1 :] = means - 2.0 * std
    return upper, mid, lower


def signal_side(band: str, distance_pct: float) -> tuple[str, str]:
    if band == "lower":
        if distance_pct >= 0:
            return "long", "lower_support_rebound"
        return "short", "lower_breakdown_retest"
    if band == "mid":
        if distance_pct >= 0:
            return "long", "mid_support_retest"
        return "short", "mid_resistance_retest"
    if distance_pct >= 0:
        return "long", "upper_breakout"
    return "short", "upper_rejection"


def update_bucket(
    bucket: dict[str, list[float] | int],
    *,
    side: str,
    entry: float,
    future_close: float,
    future_high: float,
    future_low: float,
) -> None:
    if side == "long":
        ret = pct_diff(future_close, entry)
        mfe = pct_diff(future_high, entry)
        mae = pct_diff(future_low, entry)
    else:
        ret = pct_diff(entry, future_close)
        mfe = pct_diff(entry, future_low)
        mae = pct_diff(entry, future_high)
    bucket["count"] = int(bucket.get("count", 0)) + 1
    bucket.setdefault("returns", []).append(ret)  # type: ignore[union-attr]
    bucket.setdefault("mfe", []).append(mfe)  # type: ignore[union-attr]
    bucket.setdefault("mae", []).append(mae)  # type: ignore[union-attr]


def summarize_bucket(key: tuple, bucket: dict[str, list[float] | int]) -> dict:
    returns = np.array(bucket.get("returns", []), dtype=np.float64)
    mfe = np.array(bucket.get("mfe", []), dtype=np.float64)
    mae = np.array(bucket.get("mae", []), dtype=np.float64)
    count = int(bucket.get("count", 0))
    if count <= 0:
        raise ValueError("empty bucket")
    win_rate = float((returns > 0).mean() * 100.0)
    avg_ret = float(returns.mean())
    avg_mfe = float(mfe.mean())
    avg_mae = float(mae.mean())
    edge_score = avg_ret * math.sqrt(count) * max(0.5, win_rate / 50.0)
    sensitivity_score = avg_mfe * math.sqrt(count)
    return {
        "symbol": key[0],
        "timeframe": key[1],
        "period": key[2],
        "band": key[3],
        "side": key[4],
        "pattern": key[5],
        "horizon_bars": key[6],
        "events": count,
        "win_rate_pct": round(win_rate, 2),
        "avg_return_pct": round(avg_ret, 4),
        "median_return_pct": round(float(np.median(returns)), 4),
        "avg_mfe_pct": round(avg_mfe, 4),
        "avg_mae_pct": round(avg_mae, 4),
        "worst_mae_pct": round(float(mae.min()), 4),
        "best_mfe_pct": round(float(mfe.max()), 4),
        "edge_score": round(edge_score, 4),
        "sensitivity_score": round(sensitivity_score, 4),
    }


def backtest_one(
    *,
    symbol: str,
    timeframe: str,
    rows: list[list],
    backtest_start_ms: int,
    near_pct: float,
) -> tuple[list[dict], dict]:
    data = to_kline_set(rows)
    stats: dict[tuple, dict[str, list[float] | int]] = defaultdict(dict)
    coverage: dict[str, dict] = {}
    cooldown = COOLDOWN_BARS[timeframe]
    last_seen: dict[tuple[int, str, str, str, int], int] = {}
    n = data.close.size
    for period in SHARED_BB_PERIODS:
        upper, mid, lower = rolling_band(data.close, period)
        ready_count = int(np.isfinite(mid).sum())
        coverage[str(period)] = {
            "ready": bool(ready_count > 0),
            "ready_bars": ready_count,
            "total_bars": int(n),
        }
        bands = {"upper": upper, "mid": mid, "lower": lower}
        for band_name, values in bands.items():
            for idx in range(n):
                if data.open_time[idx] < backtest_start_ms:
                    continue
                level = float(values[idx])
                if not math.isfinite(level) or level <= 0:
                    continue
                entry = float(data.close[idx])
                distance = pct_diff(entry, level)
                if abs(distance) > near_pct:
                    continue
                side, pattern = signal_side(band_name, distance)
                for horizon in HORIZONS[timeframe]:
                    if idx + horizon >= n:
                        continue
                    cooldown_key = (period, band_name, side, pattern, horizon)
                    last_idx = last_seen.get(cooldown_key)
                    if last_idx is not None and idx - last_idx < cooldown:
                        continue
                    last_seen[cooldown_key] = idx
                    future_close = float(data.close[idx + horizon])
                    future_high = float(data.high[idx + 1 : idx + horizon + 1].max())
                    future_low = float(data.low[idx + 1 : idx + horizon + 1].min())
                    key = (symbol, timeframe, period, band_name, side, pattern, horizon)
                    update_bucket(
                        stats[key],
                        side=side,
                        entry=entry,
                        future_close=future_close,
                        future_high=future_high,
                        future_low=future_low,
                    )
    summaries = [summarize_bucket(key, bucket) for key, bucket in stats.items() if int(bucket.get("count", 0)) > 0]
    return summaries, coverage


def min_events_for(timeframe: str) -> int:
    return {"15m": 25, "4h": 10, "1d": 5}[timeframe]


def top_rows(rows: list[dict], *, symbol: str, timeframe: str, horizon: int, metric: str, limit: int = 10) -> list[dict]:
    subset = [
        row for row in rows
        if row["symbol"] == symbol
        and row["timeframe"] == timeframe
        and row["horizon_bars"] == horizon
        and row["events"] >= min_events_for(timeframe)
    ]
    subset.sort(key=lambda row: (float(row[metric]), float(row["avg_return_pct"]), float(row["win_rate_pct"])), reverse=True)
    return subset[:limit]


def row_label(row: dict) -> str:
    return f"{row['timeframe']} BB{row['period']} {row['band']} {row['side']}"


def write_markdown(payload: dict, path: Path) -> None:
    rows = payload["results"]
    lines: list[str] = []
    lines.append("# V6 BB Sensitivity Backtest")
    lines.append("")
    lines.append(f"- generated_at: {payload['generated_at']}")
    lines.append(f"- window: {payload['backtest_start']} -> {payload['backtest_end']}")
    lines.append(f"- source: Binance USDT futures klines")
    lines.append(f"- touch_near_pct: {payload['near_pct']}%")
    lines.append("- signal semantics: same as V6 structure_entry_scan near BB levels")
    lines.append("")
    lines.append("## Best Edge By Symbol")
    lines.append("")
    for symbol in payload["symbols"]:
        lines.append(f"### {symbol}")
        for timeframe in payload["timeframes"]:
            horizon = PRIMARY_HORIZON[timeframe]
            best = top_rows(rows, symbol=symbol, timeframe=timeframe, horizon=horizon, metric="edge_score", limit=5)
            label = f"{timeframe} horizon={horizon} bars"
            lines.append("")
            lines.append(f"**{label}**")
            if not best:
                lines.append("- no enough events")
                continue
            for row in best:
                lines.append(
                    "- "
                    f"{row_label(row)} pattern={row['pattern']} "
                    f"events={row['events']} win={row['win_rate_pct']}% "
                    f"avg={row['avg_return_pct']}% med={row['median_return_pct']}% "
                    f"MFE={row['avg_mfe_pct']}% MAE={row['avg_mae_pct']}% "
                    f"score={row['edge_score']}"
                )
    lines.append("")
    lines.append("## Most Sensitive By MFE")
    lines.append("")
    for symbol in payload["symbols"]:
        lines.append(f"### {symbol}")
        for timeframe in payload["timeframes"]:
            horizon = PRIMARY_HORIZON[timeframe]
            best = top_rows(rows, symbol=symbol, timeframe=timeframe, horizon=horizon, metric="sensitivity_score", limit=3)
            if not best:
                continue
            lines.append(f"- {timeframe}: " + "; ".join(
                f"{row_label(row)} events={row['events']} avgMFE={row['avg_mfe_pct']}% avg={row['avg_return_pct']}%"
                for row in best
            ))
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    for symbol, by_tf in payload["coverage"].items():
        lines.append(f"### {symbol}")
        for timeframe, cov in by_tf.items():
            missing = [period for period, item in cov.items() if not item["ready"]]
            total = next(iter(cov.values()))["total_bars"] if cov else 0
            lines.append(f"- {timeframe}: bars={total}; missing_periods={','.join(missing) if missing else 'none'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT")
    parser.add_argument("--timeframes", default="15m,4h,1d")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--start-date", default="", help="UTC start date, for example 2020-06-01")
    parser.add_argument("--end-date", default="", help="UTC end date. Defaults to current time.")
    parser.add_argument("--near-pct", type=float, default=0.3)
    parser.add_argument("--output-dir", default="data/trading_system/v6_backtests")
    parser.add_argument("--cache-dir", default="data/trading_system/cache/backtest_klines")
    args = parser.parse_args()

    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    timeframes = [item.strip().lower() for item in args.timeframes.split(",") if item.strip()]
    for timeframe in timeframes:
        if timeframe not in INTERVAL_MS:
            raise ValueError(f"unsupported timeframe: {timeframe}")
    now_ms = parse_utc_date_to_ms(args.end_date, end_of_day=True) if args.end_date else int(time.time() * 1000)
    backtest_start_ms = (
        parse_utc_date_to_ms(args.start_date)
        if args.start_date
        else now_ms - args.days * 24 * 60 * 60 * 1000
    )
    if backtest_start_ms >= now_ms:
        raise ValueError("--start-date must be before --end-date/current time")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    all_results: list[dict] = []
    coverage: dict[str, dict] = {}
    for symbol in symbols:
        coverage[symbol] = {}
        for timeframe in timeframes:
            lookback_ms = max(SHARED_BB_PERIODS) * INTERVAL_MS[timeframe]
            fetch_start_ms = max(0, backtest_start_ms - lookback_ms)
            print(f"[fetch] {symbol} {timeframe} {utc_iso(fetch_start_ms)} -> {utc_iso(now_ms)}", flush=True)
            rows = fetch_and_cache_klines(symbol, timeframe, fetch_start_ms, now_ms, cache_dir)
            print(f"[backtest] {symbol} {timeframe} rows={len(rows)}", flush=True)
            summaries, cov = backtest_one(
                symbol=symbol,
                timeframe=timeframe,
                rows=rows,
                backtest_start_ms=backtest_start_ms,
                near_pct=args.near_pct,
            )
            coverage[symbol][timeframe] = cov
            all_results.extend(summaries)

    payload = {
        "generated_at": utc_iso(),
        "symbols": symbols,
        "timeframes": timeframes,
        "periods": list(SHARED_BB_PERIODS),
        "bands": list(BANDS),
        "near_pct": args.near_pct,
        "days": round((now_ms - backtest_start_ms) / 1000 / 86400, 4),
        "backtest_start": utc_iso(backtest_start_ms),
        "backtest_end": utc_iso(now_ms),
        "primary_horizons": PRIMARY_HORIZON,
        "horizons": HORIZONS,
        "coverage": coverage,
        "results": sorted(
            all_results,
            key=lambda row: (row["symbol"], row["timeframe"], row["period"], row["band"], row["side"], row["horizon_bars"]),
        ),
    }
    if args.start_date:
        start_slug = datetime.fromtimestamp(backtest_start_ms / 1000, timezone.utc).strftime("%Y%m%d")
        end_slug = datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime("%Y%m%d")
        stem = f"bb_sensitivity_{'_'.join(symbol.lower().replace('usdt', '') for symbol in symbols)}_{start_slug}_{end_slug}"
    else:
        stem = f"bb_sensitivity_{'_'.join(symbol.lower().replace('usdt', '') for symbol in symbols)}_{args.days}d"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "result_count": len(all_results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
