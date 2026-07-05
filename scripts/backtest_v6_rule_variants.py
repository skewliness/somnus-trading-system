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

from scripts.backtest_v6_bb_sensitivity import (  # noqa: E402
    fetch_and_cache_klines,
    parse_utc_date_to_ms,
    pct_diff,
    rolling_band,
    utc_iso,
)
from trading_system.bb_structure import SHARED_BB_PERIODS  # noqa: E402


INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

HORIZONS = {
    "15m": [32, 96, 192],  # 8h, 24h, 48h
    "4h": [6, 12, 30],  # 24h, 48h, 5d
    "1d": [7, 14, 30],
}

COOLDOWN = {
    "15m": 16,
    "4h": 3,
    "1d": 3,
}


def to_arrays(rows: list[list[Any]]) -> dict[str, np.ndarray]:
    rows = sorted(rows, key=lambda row: int(row[0]))
    return {
        "open_time": np.array([int(row[0]) for row in rows], dtype=np.int64),
        "open": np.array([float(row[1]) for row in rows], dtype=np.float64),
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


def add_event(bucket: dict[str, list[float]], result: dict[str, float]) -> None:
    bucket["returns"].append(float(result["ret"]))
    bucket["mfe"].append(float(result["mfe"]))
    bucket["mae"].append(float(result["mae"]))


def summarize(name: str, meta: dict[str, Any], bucket: dict[str, list[float]]) -> dict[str, Any]:
    returns = bucket.get("returns") or []
    mfe = bucket.get("mfe") or []
    mae = bucket.get("mae") or []
    count = len(returns)
    if count <= 0:
        return {**meta, "rule": name, "events": 0}
    win_rate = sum(1 for value in returns if value > 0) / count * 100.0
    avg_return = sum(returns) / count
    return {
        **meta,
        "rule": name,
        "events": count,
        "win_rate_pct": round(win_rate, 2),
        "avg_return_pct": round(avg_return, 4),
        "median_return_pct": round(statistics.median(returns), 4),
        "avg_mfe_pct": round(sum(mfe) / count, 4),
        "avg_mae_pct": round(sum(mae) / count, 4),
        "best_mfe_pct": round(max(mfe), 4),
        "worst_mae_pct": round(min(mae), 4),
        "edge_score": round(avg_return * math.sqrt(count) * max(0.5, win_rate / 50.0), 4),
    }


def run_15m_bb96_tests(symbol: str, rows: list[list[Any]], start_ms: int) -> list[dict[str, Any]]:
    data = to_arrays(rows)
    n = data["close"].size
    upper96, _, lower96 = rolling_band(data["close"], 96)
    _, _, lower210 = rolling_band(data["close"], 210)
    stats: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: {"returns": [], "mfe": [], "mae": []})
    last_seen: dict[str, int] = {}
    recent_lookback = 6
    prior_lookback = 96
    overlap_max_pct = 0.8
    touch_near_pct = 0.8
    previous_near_pct = 0.8
    current_near_pct = 0.8
    for idx in range(210, n):
        if int(data["open_time"][idx]) < start_ms:
            continue
        low = float(data["low"][idx])
        close = float(data["close"][idx])
        l96 = float(lower96[idx])
        if math.isfinite(l96) and l96 > 0 and low <= l96:
            key = "15m_bb96_lower_once_touch_long"
            if idx - last_seen.get(key, -10_000) >= COOLDOWN["15m"]:
                last_seen[key] = idx
                for horizon in HORIZONS["15m"]:
                    result = future_result(data, idx, horizon, "long")
                    if result:
                        add_event(stats[(key, horizon)], result)

        l210 = float(lower210[idx])
        if not (math.isfinite(l96) and math.isfinite(l210) and l96 > 0 and l210 > 0):
            continue
        zone_low = min(l96, l210)
        zone_high = max(l96, l210)
        zone_mid = (zone_low + zone_high) / 2.0
        width_pct = pct_diff(zone_high, zone_low)
        if width_pct > overlap_max_pct:
            continue
        if idx < recent_lookback + 12:
            continue
        recent_slice = slice(idx - recent_lookback + 1, idx + 1)
        prior_start = max(0, idx - recent_lookback - prior_lookback + 1)
        prior_end = idx - recent_lookback + 1
        prior_slice = slice(prior_start, prior_end)
        recent_low = float(data["low"][recent_slice].min())
        prior_low = float(data["low"][prior_slice].min()) if prior_end > prior_start else 0.0
        recent_touched = recent_low <= zone_high * (1.0 + touch_near_pct / 100.0)
        close_rebounded = close >= zone_low * (1.0 - touch_near_pct / 100.0)
        near_previous = prior_low > 0 and abs(pct_diff(recent_low, prior_low)) <= previous_near_pct
        current_near_zone = abs(pct_diff(close, zone_mid)) <= current_near_pct
        confirmed = recent_touched and close_rebounded and near_previous and current_near_zone
        if confirmed:
            key = "15m_bb96_210_lower_second_test_long"
            if idx - last_seen.get(key, -10_000) >= COOLDOWN["15m"]:
                last_seen[key] = idx
                for horizon in HORIZONS["15m"]:
                    result = future_result(data, idx, horizon, "long")
                    if result:
                        add_event(stats[(key, horizon)], result)

    rows_out: list[dict[str, Any]] = []
    for (rule, horizon), bucket in stats.items():
        rows_out.append(
            summarize(
                rule,
                {"symbol": symbol, "timeframe": "15m", "side": "long", "horizon_bars": horizon},
                bucket,
            )
        )
    return rows_out


def run_4h_lower_tests(symbol: str, rows: list[list[Any]], start_ms: int) -> list[dict[str, Any]]:
    data = to_arrays(rows)
    stats: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: {"returns": [], "mfe": [], "mae": []})
    last_seen: dict[int, int] = {}
    for period in SHARED_BB_PERIODS:
        _, _, lower = rolling_band(data["close"], int(period))
        for idx in range(int(period), data["close"].size):
            if int(data["open_time"][idx]) < start_ms:
                continue
            level = float(lower[idx])
            if not math.isfinite(level) or level <= 0:
                continue
            if float(data["low"][idx]) > level:
                continue
            if idx - last_seen.get(int(period), -10_000) < COOLDOWN["4h"]:
                continue
            last_seen[int(period)] = idx
            for horizon in HORIZONS["4h"]:
                result = future_result(data, idx, horizon, "long")
                if result:
                    add_event(stats[(int(period), horizon)], result)
    return [
        summarize(
            "4h_bb_lower_touch_long",
            {"symbol": symbol, "timeframe": "4h", "side": "long", "period": period, "horizon_bars": horizon},
            bucket,
        )
        for (period, horizon), bucket in stats.items()
    ]


def run_daily_ma_tests(symbol: str, rows: list[list[Any]], start_ms: int) -> list[dict[str, Any]]:
    data = to_arrays(rows)
    ma7 = sma(data["close"], 7)
    ma20 = sma(data["close"], 20)
    stats: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: {"returns": [], "mfe": [], "mae": []})
    last_seen: dict[str, int] = {}
    pending_reclaim_idx = -1
    pending_break_idx = -1
    for idx in range(21, data["close"].size):
        if int(data["open_time"][idx]) < start_ms:
            continue
        close = float(data["close"][idx])
        prev_close = float(data["close"][idx - 1])
        m7 = float(ma7[idx])
        pm7 = float(ma7[idx - 1])
        m20 = float(ma20[idx])
        pm20 = float(ma20[idx - 1])
        if not all(math.isfinite(value) and value > 0 for value in (m7, pm7, m20, pm20)):
            continue

        if prev_close <= pm7 and close > m7:
            pending_reclaim_idx = idx
            key = "1d_close_reclaim_ma7_long"
            if idx - last_seen.get(key, -10_000) >= COOLDOWN["1d"]:
                last_seen[key] = idx
                for horizon in HORIZONS["1d"]:
                    result = future_result(data, idx, horizon, "long")
                    if result:
                        add_event(stats[(key, horizon)], result)

        if pending_reclaim_idx > 0 and 0 < idx - pending_reclaim_idx <= 30:
            if prev_close <= pm20 and close > m20:
                key = "1d_ma7_reclaim_then_ma20_second_reclaim_long"
                if idx - last_seen.get(key, -10_000) >= COOLDOWN["1d"]:
                    last_seen[key] = idx
                    for horizon in HORIZONS["1d"]:
                        result = future_result(data, idx, horizon, "long")
                        if result:
                            add_event(stats[(key, horizon)], result)
                pending_reclaim_idx = -1
        elif pending_reclaim_idx > 0 and idx - pending_reclaim_idx > 30:
            pending_reclaim_idx = -1

        if prev_close >= pm7 and close < m7:
            pending_break_idx = idx
            key = "1d_close_break_ma7_short"
            if idx - last_seen.get(key, -10_000) >= COOLDOWN["1d"]:
                last_seen[key] = idx
                for horizon in HORIZONS["1d"]:
                    result = future_result(data, idx, horizon, "short")
                    if result:
                        add_event(stats[(key, horizon)], result)

        if pending_break_idx > 0 and 0 < idx - pending_break_idx <= 30:
            if prev_close >= pm20 and close < m20:
                key = "1d_ma7_break_then_ma20_second_break_short"
                if idx - last_seen.get(key, -10_000) >= COOLDOWN["1d"]:
                    last_seen[key] = idx
                    for horizon in HORIZONS["1d"]:
                        result = future_result(data, idx, horizon, "short")
                        if result:
                            add_event(stats[(key, horizon)], result)
                pending_break_idx = -1
        elif pending_break_idx > 0 and idx - pending_break_idx > 30:
            pending_break_idx = -1

    rows_out: list[dict[str, Any]] = []
    for (rule, horizon), bucket in stats.items():
        side = "short" if "short" in rule else "long"
        rows_out.append(
            summarize(
                rule,
                {"symbol": symbol, "timeframe": "1d", "side": side, "horizon_bars": horizon},
                bucket,
            )
        )
    return rows_out


def top(rows: list[dict[str, Any]], *, rule: str | None = None, timeframe: str | None = None, horizon: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
    subset = rows
    if rule:
        subset = [row for row in subset if row.get("rule") == rule]
    if timeframe:
        subset = [row for row in subset if row.get("timeframe") == timeframe]
    if horizon:
        subset = [row for row in subset if row.get("horizon_bars") == horizon]
    subset = [row for row in subset if int(row.get("events") or 0) > 0]
    subset.sort(key=lambda row: (float(row.get("edge_score") or -999), float(row.get("avg_return_pct") or -999)), reverse=True)
    return subset[:limit]


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = payload["results"]
    lines = [
        "# V6 Rule Variant Backtest",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- symbols: {', '.join(payload['symbols'])}",
        f"- 15m_window: {payload['windows']['15m']['start']} -> {payload['windows']['15m']['end']}",
        f"- 4h_window: {payload['windows']['4h']['start']} -> {payload['windows']['4h']['end']}",
        f"- 1d_window: {payload['windows']['1d']['start']} -> {payload['windows']['1d']['end']}",
        "- returns are close-to-close after horizon; MFE/MAE use high/low inside horizon.",
        "",
        "## 15M BB96 once vs BB96/210 second test",
    ]
    for rule in ("15m_bb96_lower_once_touch_long", "15m_bb96_210_lower_second_test_long"):
        lines.append(f"### {rule}")
        for horizon in HORIZONS["15m"]:
            best = top(rows, rule=rule, timeframe="15m", horizon=horizon, limit=20)
            lines.append(f"**horizon={horizon} bars**")
            if not best:
                lines.append("- no events")
            for row in best:
                lines.append(
                    f"- {row['symbol']}: n={row['events']} win={row['win_rate_pct']}% "
                    f"avg={row['avg_return_pct']}% med={row['median_return_pct']}% "
                    f"MFE={row['avg_mfe_pct']}% MAE={row['avg_mae_pct']}%"
                )
    lines.append("")
    lines.append("## 4H lower-band long/add tests")
    for horizon in HORIZONS["4h"]:
        lines.append(f"### horizon={horizon} bars")
        best = top(rows, rule="4h_bb_lower_touch_long", timeframe="4h", horizon=horizon, limit=30)
        for row in best:
            lines.append(
                f"- {row['symbol']} BB{row.get('period')}: n={row['events']} win={row['win_rate_pct']}% "
                f"avg={row['avg_return_pct']}% med={row['median_return_pct']}% "
                f"MFE={row['avg_mfe_pct']}% MAE={row['avg_mae_pct']}%"
            )
    lines.append("")
    lines.append("## Daily MA7/MA20 tests")
    for rule in (
        "1d_close_reclaim_ma7_long",
        "1d_ma7_reclaim_then_ma20_second_reclaim_long",
        "1d_close_break_ma7_short",
        "1d_ma7_break_then_ma20_second_break_short",
    ):
        lines.append(f"### {rule}")
        for horizon in HORIZONS["1d"]:
            best = top(rows, rule=rule, timeframe="1d", horizon=horizon, limit=20)
            lines.append(f"**horizon={horizon} days**")
            if not best:
                lines.append("- no events")
            for row in best:
                lines.append(
                    f"- {row['symbol']}: n={row['events']} win={row['win_rate_pct']}% "
                    f"avg={row['avg_return_pct']}% med={row['median_return_pct']}% "
                    f"MFE={row['avg_mfe_pct']}% MAE={row['avg_mae_pct']}%"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    cache_dir = Path(args.cache_dir)
    results: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for symbol in symbols:
        coverage[symbol] = {}
        for timeframe in ("15m", "4h", "1d"):
            lookback = max(SHARED_BB_PERIODS) * INTERVAL_MS[timeframe] if timeframe != "1d" else 30 * INTERVAL_MS[timeframe]
            fetch_start = max(0, starts[timeframe] - lookback)
            print(f"[fetch] {symbol} {timeframe} {utc_iso(fetch_start)} -> {utc_iso(end_ms)}", flush=True)
            rows = fetch_and_cache_klines(symbol, timeframe, fetch_start, end_ms, cache_dir)
            rows = [row for row in rows if int(row[0]) <= end_ms]
            coverage[symbol][timeframe] = {
                "rows": len(rows),
                "first": utc_iso(int(rows[0][0])) if rows else "",
                "last": utc_iso(int(rows[-1][0])) if rows else "",
            }
            if timeframe == "15m":
                results.extend(run_15m_bb96_tests(symbol, rows, starts[timeframe]))
            elif timeframe == "4h":
                results.extend(run_4h_lower_tests(symbol, rows, starts[timeframe]))
            else:
                results.extend(run_daily_ma_tests(symbol, rows, starts[timeframe]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "generated_at": utc_iso(),
        "symbols": symbols,
        "windows": {
            key: {"start": utc_iso(value), "end": utc_iso(end_ms)}
            for key, value in starts.items()
        },
        "coverage": coverage,
        "results": sorted(results, key=lambda row: (row.get("timeframe", ""), row.get("rule", ""), row.get("symbol", ""), int(row.get("period") or 0), int(row.get("horizon_bars") or 0))),
    }
    json_path = output_dir / f"v6_rule_variants_{stamp}.json"
    md_path = output_dir / f"v6_rule_variants_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "result_count": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
