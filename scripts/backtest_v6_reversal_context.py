#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_v6_bb_sensitivity import fetch_and_cache_klines, parse_utc_date_to_ms, pct_diff, rolling_band, utc_iso  # noqa: E402
from trading_system.bb_structure import SHARED_BB_PERIODS  # noqa: E402


INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}
HORIZONS_15M = [32, 96, 192]
HORIZONS_4H = [12, 30]
COOLDOWN = {"15m": 16, "4h": 3}
GOOD_4H_LOWER = {
    "BTCUSDT": [300],
    "ETHUSDT": [2880],
    "HYPEUSDT": [120, 210, 105, 96, 672, 300],
    "BNBUSDT": [1440, 2880, 1920, 672, 300],
    "ZECUSDT": [1920, 1440, 672, 210],
    "SOLUSDT": [4707, 300, 210],
}


def to_arrays(rows: list[list[Any]]) -> dict[str, np.ndarray]:
    rows = sorted(rows, key=lambda row: int(row[0]))
    return {
        "open_time": np.array([int(row[0]) for row in rows], dtype=np.int64),
        "high": np.array([float(row[2]) for row in rows], dtype=np.float64),
        "low": np.array([float(row[3]) for row in rows], dtype=np.float64),
        "close": np.array([float(row[4]) for row in rows], dtype=np.float64),
    }


def sma(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(values.size, np.nan, dtype=np.float64)
    if values.size < period:
        return out
    csum = np.concatenate(([0.0], np.cumsum(values)))
    out[period - 1 :] = (csum[period:] - csum[:-period]) / period
    return out


def future_result(data: dict[str, np.ndarray], idx: int, horizon: int, side: str) -> dict[str, float] | None:
    if idx + horizon >= data["close"].size:
        return None
    entry = float(data["close"][idx])
    future_close = float(data["close"][idx + horizon])
    future_high = float(data["high"][idx + 1 : idx + horizon + 1].max())
    future_low = float(data["low"][idx + 1 : idx + horizon + 1].min())
    if side == "long":
        ret = pct_diff(future_close, entry)
        mfe = pct_diff(future_high, entry)
        mae = pct_diff(future_low, entry)
    else:
        ret = pct_diff(entry, future_close)
        mfe = pct_diff(entry, future_low)
        mae = pct_diff(entry, future_high)
    return {"ret": ret, "mfe": mfe, "mae": mae}


def bucket_add(bucket: dict[str, list[float]], result: dict[str, float]) -> None:
    bucket["returns"].append(result["ret"])
    bucket["mfe"].append(result["mfe"])
    bucket["mae"].append(result["mae"])
    bucket["giveback"].append(result["mfe"] - result["ret"])


def summarize(meta: dict[str, Any], bucket: dict[str, list[float]]) -> dict[str, Any]:
    returns = bucket.get("returns") or []
    if not returns:
        return {**meta, "events": 0}
    mfe = bucket.get("mfe") or []
    mae = bucket.get("mae") or []
    giveback = bucket.get("giveback") or []
    count = len(returns)
    win = sum(1 for value in returns if value > 0) / count * 100.0
    return {
        **meta,
        "events": count,
        "win_rate_pct": round(win, 2),
        "avg_return_pct": round(sum(returns) / count, 4),
        "median_return_pct": round(statistics.median(returns), 4),
        "avg_mfe_pct": round(sum(mfe) / count, 4),
        "avg_mae_pct": round(sum(mae) / count, 4),
        "avg_giveback_pct": round(sum(giveback) / count, 4),
        "avg_mfe_8x_pct": round(sum(mfe) / count * 8.0, 2),
        "avg_giveback_8x_pct": round(sum(giveback) / count * 8.0, 2),
    }


def last_closed_index(open_time: np.ndarray, interval_ms: int, event_ms: int) -> int:
    idx = int(np.searchsorted(open_time, event_ms, side="right") - 1)
    if idx >= 0 and event_ms < int(open_time[idx]) + interval_ms:
        idx -= 1
    return idx


def h4_state_at(h4: dict[str, np.ndarray], ma20: np.ndarray, event_ms: int) -> tuple[str, int]:
    idx = last_closed_index(h4["open_time"], INTERVAL_MS["4h"], event_ms)
    if idx < 23 or idx >= h4["close"].size or not math.isfinite(float(ma20[idx])):
        return "h4_unknown", idx
    close = float(h4["close"][idx])
    current = float(ma20[idx])
    previous = float(ma20[idx - 3])
    if close > current and current > previous:
        return "h4_bullish", idx
    if close < current and current < previous:
        return "h4_bearish", idx
    if close > current:
        return "h4_above_ma20_mixed", idx
    if close < current:
        return "h4_below_ma20_mixed", idx
    return "h4_mixed", idx


def future_h4_bearish_flip(h4: dict[str, np.ndarray], ma20: np.ndarray, event_ms: int, horizon_ms: int) -> bool:
    start = last_closed_index(h4["open_time"], INTERVAL_MS["4h"], event_ms)
    if start < 23:
        return False
    end_ms = event_ms + horizon_ms
    end = int(np.searchsorted(h4["open_time"], end_ms, side="right") - 1)
    end = min(end, h4["close"].size - 1)
    for idx in range(start + 1, end + 1):
        if idx < 3 or not math.isfinite(float(ma20[idx])):
            continue
        if float(h4["close"][idx]) < float(ma20[idx]) and float(ma20[idx]) < float(ma20[idx - 3]):
            return True
    return False


def future_h4_bullish_flip(h4: dict[str, np.ndarray], ma20: np.ndarray, event_ms: int, horizon_ms: int) -> bool:
    start = last_closed_index(h4["open_time"], INTERVAL_MS["4h"], event_ms)
    if start < 23:
        return False
    end_ms = event_ms + horizon_ms
    end = int(np.searchsorted(h4["open_time"], end_ms, side="right") - 1)
    end = min(end, h4["close"].size - 1)
    for idx in range(start + 1, end + 1):
        if idx < 3 or not math.isfinite(float(ma20[idx])):
            continue
        if float(h4["close"][idx]) > float(ma20[idx]) and float(ma20[idx]) > float(ma20[idx - 3]):
            return True
    return False


def daily_state_at(daily: dict[str, np.ndarray], ma7: np.ndarray, ma20: np.ndarray, event_ms: int) -> tuple[str, int]:
    idx = last_closed_index(daily["open_time"], INTERVAL_MS["1d"], event_ms)
    if idx < 21 or idx >= daily["close"].size or not math.isfinite(float(ma7[idx])) or not math.isfinite(float(ma20[idx])):
        return "1d_unknown", idx
    close = float(daily["close"][idx])
    m7 = float(ma7[idx])
    m20 = float(ma20[idx])
    if close > m7 and m7 >= m20:
        return "1d_bullish_ma7_above_ma20", idx
    if close > m7:
        return "1d_above_ma7_bear_stack", idx
    if close < m7 and m7 <= m20:
        return "1d_bearish_ma7_below_ma20", idx
    if close < m7:
        return "1d_below_ma7_bull_stack", idx
    return "1d_mixed", idx


def future_daily_reclaim(daily: dict[str, np.ndarray], ma7: np.ndarray, ma20: np.ndarray, event_ms: int, horizon_ms: int) -> tuple[bool, bool]:
    start = last_closed_index(daily["open_time"], INTERVAL_MS["1d"], event_ms)
    if start < 21:
        return False, False
    end_ms = event_ms + horizon_ms
    end = int(np.searchsorted(daily["open_time"], end_ms, side="right") - 1)
    end = min(end, daily["close"].size - 1)
    ma7_hit = False
    ma20_hit = False
    for idx in range(start + 1, end + 1):
        close = float(daily["close"][idx])
        if math.isfinite(float(ma7[idx])) and close > float(ma7[idx]):
            ma7_hit = True
        if math.isfinite(float(ma20[idx])) and close > float(ma20[idx]):
            ma20_hit = True
    return ma7_hit, ma20_hit


def generate_15m_events(symbol: str, data: dict[str, np.ndarray], start_ms: int) -> list[dict[str, Any]]:
    upper96, _, lower96 = rolling_band(data["close"], 96)
    _, _, lower210 = rolling_band(data["close"], 210)
    events: list[dict[str, Any]] = []
    last_seen: dict[str, int] = {}
    recent_lookback = 6
    prior_lookback = 96
    for idx in range(210, data["close"].size):
        if int(data["open_time"][idx]) < start_ms:
            continue
        low = float(data["low"][idx])
        close = float(data["close"][idx])
        l96 = float(lower96[idx])
        if math.isfinite(l96) and l96 > 0 and low <= l96:
            rule = "15m_bb96_lower_once_touch_long"
            if idx - last_seen.get(rule, -10_000) >= COOLDOWN["15m"]:
                last_seen[rule] = idx
                events.append({"symbol": symbol, "rule": rule, "idx": idx, "time": int(data["open_time"][idx])})

        l210 = float(lower210[idx])
        if not (math.isfinite(l96) and math.isfinite(l210) and l96 > 0 and l210 > 0):
            continue
        zone_low = min(l96, l210)
        zone_high = max(l96, l210)
        zone_mid = (zone_low + zone_high) / 2.0
        width_pct = pct_diff(zone_high, zone_low)
        if width_pct > 0.8 or idx < recent_lookback + 12:
            continue
        recent_slice = slice(idx - recent_lookback + 1, idx + 1)
        prior_start = max(0, idx - recent_lookback - prior_lookback + 1)
        prior_end = idx - recent_lookback + 1
        prior_slice = slice(prior_start, prior_end)
        recent_low = float(data["low"][recent_slice].min())
        prior_low = float(data["low"][prior_slice].min()) if prior_end > prior_start else 0.0
        confirmed = (
            recent_low <= zone_high * 1.008
            and close >= zone_low * 0.992
            and prior_low > 0
            and abs(pct_diff(recent_low, prior_low)) <= 0.8
            and abs(pct_diff(close, zone_mid)) <= 0.8
        )
        if confirmed:
            rule = "15m_bb96_210_lower_second_test_long"
            if idx - last_seen.get(rule, -10_000) >= COOLDOWN["15m"]:
                last_seen[rule] = idx
                events.append({"symbol": symbol, "rule": rule, "idx": idx, "time": int(data["open_time"][idx])})
    return events


def generate_4h_events(symbol: str, data: dict[str, np.ndarray], start_ms: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    periods = GOOD_4H_LOWER.get(symbol, [])
    for period in periods:
        _, _, lower = rolling_band(data["close"], int(period))
        last_idx = -10_000
        for idx in range(int(period), data["close"].size):
            if int(data["open_time"][idx]) < start_ms:
                continue
            level = float(lower[idx])
            if not math.isfinite(level) or level <= 0 or float(data["low"][idx]) > level:
                continue
            if idx - last_idx < COOLDOWN["4h"]:
                continue
            last_idx = idx
            events.append({
                "symbol": symbol,
                "rule": "4h_bb_lower_touch_long",
                "period": int(period),
                "idx": idx,
                "time": int(data["open_time"][idx]),
            })
    return events


def run(symbols: list[str], starts: dict[str, int], end_ms: int, cache_dir: Path) -> dict[str, Any]:
    results: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: {"returns": [], "mfe": [], "mae": [], "giveback": []})
    coverage: dict[str, Any] = {}
    for symbol in symbols:
        rows_15m = fetch_and_cache_klines(symbol, "15m", starts["15m"] - 210 * INTERVAL_MS["15m"], end_ms, cache_dir)
        rows_4h = fetch_and_cache_klines(symbol, "4h", starts["4h"] - max(SHARED_BB_PERIODS) * INTERVAL_MS["4h"], end_ms, cache_dir)
        rows_1d = fetch_and_cache_klines(symbol, "1d", starts["1d"] - 40 * INTERVAL_MS["1d"], end_ms, cache_dir)
        d15 = to_arrays(rows_15m)
        d4h = to_arrays(rows_4h)
        d1d = to_arrays(rows_1d)
        h4_ma20 = sma(d4h["close"], 20)
        d1_ma7 = sma(d1d["close"], 7)
        d1_ma20 = sma(d1d["close"], 20)
        coverage[symbol] = {
            "15m": {"rows": len(rows_15m), "first": utc_iso(int(rows_15m[0][0])) if rows_15m else "", "last": utc_iso(int(rows_15m[-1][0])) if rows_15m else ""},
            "4h": {"rows": len(rows_4h), "first": utc_iso(int(rows_4h[0][0])) if rows_4h else "", "last": utc_iso(int(rows_4h[-1][0])) if rows_4h else ""},
            "1d": {"rows": len(rows_1d), "first": utc_iso(int(rows_1d[0][0])) if rows_1d else "", "last": utc_iso(int(rows_1d[-1][0])) if rows_1d else ""},
        }

        for event in generate_15m_events(symbol, d15, starts["15m"]):
            h4_state, _ = h4_state_at(d4h, h4_ma20, int(event["time"]))
            for horizon in HORIZONS_15M:
                result = future_result(d15, int(event["idx"]), horizon, "long")
                if not result:
                    continue
                h4_flip = future_h4_bearish_flip(d4h, h4_ma20, int(event["time"]), horizon * INTERVAL_MS["15m"])
                keys = [
                    ("15m_by_h4_entry_state", event["rule"], symbol, horizon, h4_state),
                    ("15m_by_future_h4_bear_flip", event["rule"], symbol, horizon, "future_h4_bear_flip_yes" if h4_flip else "future_h4_bear_flip_no"),
                    ("15m_all", event["rule"], symbol, horizon, "all"),
                ]
                for key in keys:
                    bucket_add(results[key], result)

        for event in generate_4h_events(symbol, d4h, starts["4h"]):
            d_state, _ = daily_state_at(d1d, d1_ma7, d1_ma20, int(event["time"]))
            for horizon in HORIZONS_4H:
                result = future_result(d4h, int(event["idx"]), horizon, "long")
                if not result:
                    continue
                horizon_ms = horizon * INTERVAL_MS["4h"]
                h4_bull = future_h4_bullish_flip(d4h, h4_ma20, int(event["time"]), horizon_ms)
                d_ma7, d_ma20 = future_daily_reclaim(d1d, d1_ma7, d1_ma20, int(event["time"]), horizon_ms)
                keys = [
                    ("4h_by_future_h4_bull_flip", event["rule"], symbol, int(event["period"]), horizon, "future_h4_bull_flip_yes" if h4_bull else "future_h4_bull_flip_no"),
                    ("4h_by_entry_1d_state", event["rule"], symbol, int(event["period"]), horizon, d_state),
                    ("4h_by_future_1d_ma7", event["rule"], symbol, int(event["period"]), horizon, "future_1d_ma7_reclaim_yes" if d_ma7 else "future_1d_ma7_reclaim_no"),
                    ("4h_by_future_1d_ma20", event["rule"], symbol, int(event["period"]), horizon, "future_1d_ma20_reclaim_yes" if d_ma20 else "future_1d_ma20_reclaim_no"),
                    ("4h_all", event["rule"], symbol, int(event["period"]), horizon, "all"),
                ]
                for key in keys:
                    bucket_add(results[key], result)

    summaries: list[dict[str, Any]] = []
    for key, bucket in results.items():
        if key[0].startswith("15m"):
            group, rule, symbol, horizon, label = key
            meta = {"group": group, "rule": rule, "symbol": symbol, "timeframe": "15m", "horizon_bars": horizon, "label": label}
        else:
            group, rule, symbol, period, horizon, label = key
            meta = {"group": group, "rule": rule, "symbol": symbol, "timeframe": "4h", "period": period, "horizon_bars": horizon, "label": label}
        summaries.append(summarize(meta, bucket))
    summaries.sort(key=lambda row: (row["group"], row["symbol"], row["rule"], int(row.get("period") or 0), int(row["horizon_bars"]), row["label"]))
    return {"coverage": coverage, "results": summaries}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,HYPEUSDT,BNBUSDT,ZECUSDT,SOLUSDT")
    parser.add_argument("--start-15m", default="2025-06-14")
    parser.add_argument("--start-4h", default="2021-06-14")
    parser.add_argument("--start-1d", default="2021-06-14")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--cache-dir", default="data/trading_system/cache/backtest_klines")
    parser.add_argument("--output-dir", default="data/trading_system/v6_backtests")
    args = parser.parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    end_ms = parse_utc_date_to_ms(args.end_date, end_of_day=True) if args.end_date else int(datetime.now(timezone.utc).timestamp() * 1000)
    starts = {
        "15m": parse_utc_date_to_ms(args.start_15m),
        "4h": parse_utc_date_to_ms(args.start_4h),
        "1d": parse_utc_date_to_ms(args.start_1d),
    }
    payload = {
        "generated_at": utc_iso(),
        "symbols": symbols,
        "windows": {key: {"start": utc_iso(value), "end": utc_iso(end_ms)} for key, value in starts.items()},
        **run(symbols, starts, end_ms, Path(args.cache_dir)),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"v6_reversal_context_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"json": str(path), "result_count": len(payload["results"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
