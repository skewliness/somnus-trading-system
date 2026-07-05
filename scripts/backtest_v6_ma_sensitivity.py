#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
import sys
import time
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_v6_bb_sensitivity import INTERVAL_MS, fetch_and_cache_klines, utc_iso  # noqa: E402

CACHE = ROOT / "data/trading_system/cache/backtest_klines"
OUT = ROOT / "data/trading_system/v6_backtests"
DEFAULT_SYMBOLS = ["SUIUSDT", "NEARUSDT", "SOLUSDT", "XRPUSDT"]
DEFAULT_TIMEFRAMES = ["15m", "4h"]
MA_PERIODS = [7, 20, 60, 96, 105, 120]
HORIZONS = {"15m": [16, 32, 96], "4h": [6, 12, 30]}
PRIMARY = {"15m": 32, "4h": 12}
MIN_EVENTS = {"15m": 25, "4h": 10}
NEAR_PCT = 0.3
COOLDOWN = {"15m": 4, "4h": 2}
DEFAULT_DAYS = 365


def load_rows(symbol: str, timeframe: str) -> list[list]:
    path = CACHE / timeframe / f"{symbol}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("klines", payload if isinstance(payload, list) else [])
    return sorted(rows, key=lambda row: int(row[0]))


def pct(price: float, reference: float) -> float:
    return (price - reference) / reference * 100.0 if reference else 0.0


def update_bucket(bucket: dict[str, list[float]], side: str, entry: float, future_close: float, future_high: float, future_low: float) -> None:
    if side == "long":
        ret = pct(future_close, entry)
        mfe = pct(future_high, entry)
        mae = pct(future_low, entry)
    else:
        ret = pct(entry, future_close)
        mfe = pct(entry, future_low)
        mae = pct(entry, future_high)
    bucket["returns"].append(ret)
    bucket["mfe"].append(mfe)
    bucket["mae"].append(mae)


def summarize(key: tuple, bucket: dict[str, list[float]]) -> dict:
    returns = bucket["returns"]
    mfe = bucket["mfe"]
    mae = bucket["mae"]
    count = len(returns)
    win_rate = sum(1 for item in returns if item > 0) / count * 100.0
    avg_return = sum(returns) / count
    avg_mfe = sum(mfe) / count
    avg_mae = sum(mae) / count
    edge_score = avg_return * math.sqrt(count) * max(0.5, win_rate / 50.0)
    sensitivity_score = avg_mfe * math.sqrt(count)
    return {
        "symbol": key[0],
        "timeframe": key[1],
        "indicator": key[2],
        "period": key[3],
        "side": key[4],
        "pattern": key[5],
        "horizon_bars": key[6],
        "events": count,
        "win_rate_pct": round(win_rate, 2),
        "avg_return_pct": round(avg_return, 4),
        "median_return_pct": round(statistics.median(returns), 4),
        "avg_mfe_pct": round(avg_mfe, 4),
        "avg_mae_pct": round(avg_mae, 4),
        "worst_mae_pct": round(min(mae), 4),
        "best_mfe_pct": round(max(mfe), 4),
        "edge_score": round(edge_score, 4),
        "sensitivity_score": round(sensitivity_score, 4),
    }


def rolling_ma(close: list[float], period: int) -> list[float | None]:
    values: list[float | None] = [None] * len(close)
    rolling_sum = 0.0
    for idx, value in enumerate(close):
        rolling_sum += value
        if idx >= period:
            rolling_sum -= close[idx - period]
        if idx >= period - 1:
            values[idx] = rolling_sum / period
    return values


def backtest_symbol_timeframe(symbol: str, timeframe: str, rows: list[list], backtest_start_ms: int) -> tuple[list[dict], dict]:
    open_time = [int(row[0]) for row in rows]
    high = [float(row[2]) for row in rows]
    low = [float(row[3]) for row in rows]
    close = [float(row[4]) for row in rows]
    n = len(rows)
    stats: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: {"returns": [], "mfe": [], "mae": []})
    coverage: dict[str, dict] = {}
    last_seen: dict[tuple, int] = {}

    for period in MA_PERIODS:
        ma = rolling_ma(close, period)
        ready = sum(1 for item in ma if item)
        coverage[str(period)] = {"ready": ready > 0, "ready_bars": ready, "total_bars": n}
        for idx in range(period, n):
            if open_time[idx] < backtest_start_ms:
                continue
            level = ma[idx]
            if not level or level <= 0:
                continue
            entry = close[idx]
            distance = pct(entry, level)
            if abs(distance) <= NEAR_PCT:
                side = "long" if distance >= 0 else "short"
                pattern = "ma_support_retest" if side == "long" else "ma_resistance_retest"
                for horizon in HORIZONS[timeframe]:
                    if idx + horizon >= n:
                        continue
                    cooldown_key = (period, side, pattern, horizon)
                    if cooldown_key in last_seen and idx - last_seen[cooldown_key] < COOLDOWN[timeframe]:
                        continue
                    last_seen[cooldown_key] = idx
                    key = (symbol, timeframe, "MA", period, side, pattern, horizon)
                    update_bucket(stats[key], side, entry, close[idx + horizon], max(high[idx + 1 : idx + horizon + 1]), min(low[idx + 1 : idx + horizon + 1]))

            previous_level = ma[idx - 1]
            if not previous_level:
                continue
            if close[idx - 1] <= previous_level and entry > level:
                side = "long"
                pattern = "ma_reclaim_cross"
            elif close[idx - 1] >= previous_level and entry < level:
                side = "short"
                pattern = "ma_break_cross"
            else:
                continue
            for horizon in HORIZONS[timeframe]:
                if idx + horizon >= n:
                    continue
                cooldown_key = (period, side, pattern, horizon)
                if cooldown_key in last_seen and idx - last_seen[cooldown_key] < COOLDOWN[timeframe]:
                    continue
                last_seen[cooldown_key] = idx
                key = (symbol, timeframe, "MA", period, side, pattern, horizon)
                update_bucket(stats[key], side, entry, close[idx + horizon], max(high[idx + 1 : idx + horizon + 1]), min(low[idx + 1 : idx + horizon + 1]))

    return [summarize(key, bucket) for key, bucket in stats.items() if bucket["returns"]], coverage


def top_rows(rows: list[dict], symbol: str, timeframe: str, side: str, limit: int = 5) -> list[dict]:
    horizon = PRIMARY[timeframe]
    subset = [
        row for row in rows
        if row["symbol"] == symbol
        and row["timeframe"] == timeframe
        and row["side"] == side
        and row["horizon_bars"] == horizon
        and row["events"] >= MIN_EVENTS[timeframe]
    ]
    subset.sort(key=lambda row: (row["edge_score"], row["avg_return_pct"], row["win_rate_pct"]), reverse=True)
    return subset[:limit]


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# V6 MA Sensitivity Backtest",
        "",
        f"- generated_at: {payload['generated_at']}",
        "- source: existing Binance futures kline cache",
        f"- touch_near_pct: {payload['near_pct']}%",
        "- semantics: near MA within touch_near_pct, plus MA reclaim/break crosses",
        "",
    ]
    rows = payload["results"]
    for symbol in payload["symbols"]:
        lines.append(f"## {symbol}")
        for timeframe in payload["timeframes"]:
            lines.append(f"### {timeframe} horizon={PRIMARY[timeframe]} bars")
            for side in ("long", "short"):
                lines.append(f"**{side}**")
                best = top_rows(rows, symbol, timeframe, side)
                if not best:
                    lines.append("- no enough events")
                for row in best:
                    lines.append(
                        f"- MA{row['period']} {row['pattern']} "
                        f"events={row['events']} win={row['win_rate_pct']}% "
                        f"avg={row['avg_return_pct']}% med={row['median_return_pct']}% "
                        f"MFE={row['avg_mfe_pct']}% MAE={row['avg_mae_pct']}% "
                        f"score={row['edge_score']}"
                    )
            lines.append("")
    lines.append("## Coverage")
    for symbol, by_timeframe in payload["coverage"].items():
        lines.append(f"### {symbol}")
        for timeframe, coverage in by_timeframe.items():
            total = next(iter(coverage.values()))["total_bars"] if coverage else 0
            missing = [period for period, item in coverage.items() if not item["ready"]]
            lines.append(f"- {timeframe}: bars={total}; missing_periods={','.join(missing) if missing else 'none'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    args = parser.parse_args()

    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    timeframes = [item.strip().lower() for item in args.timeframes.split(",") if item.strip()]
    for timeframe in timeframes:
        if timeframe not in HORIZONS:
            raise ValueError(f"unsupported timeframe: {timeframe}")

    OUT.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    backtest_start_ms = now_ms - args.days * 24 * 60 * 60 * 1000
    all_results: list[dict] = []
    coverage: dict[str, dict] = {}
    for symbol in symbols:
        coverage[symbol] = {}
        for timeframe in timeframes:
            fetch_start_ms = max(0, backtest_start_ms - max(MA_PERIODS) * INTERVAL_MS[timeframe])
            rows = fetch_and_cache_klines(symbol, timeframe, fetch_start_ms, now_ms, CACHE)
            results, timeframe_coverage = backtest_symbol_timeframe(symbol, timeframe, rows, backtest_start_ms)
            all_results.extend(results)
            coverage[symbol][timeframe] = timeframe_coverage

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backtest_start": utc_iso(backtest_start_ms),
        "backtest_end": utc_iso(now_ms),
        "days": args.days,
        "symbols": symbols,
        "timeframes": timeframes,
        "indicator_family": "MA",
        "ma_periods": MA_PERIODS,
        "near_pct": NEAR_PCT,
        "primary_horizon_bars": PRIMARY,
        "coverage": coverage,
        "result_count": len(all_results),
        "results": all_results,
    }
    stem_symbols = "_".join(symbol.replace("USDT", "").lower() for symbol in symbols)
    json_path = OUT / f"ma_sensitivity_{stem_symbols}_{args.days}d.json"
    md_path = OUT / f"ma_sensitivity_{stem_symbols}_{args.days}d.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, md_path)
    print(json_path)
    print(md_path)
    print(f"result_count={len(all_results)}")
    for symbol in symbols:
        for timeframe in timeframes:
            total = next(iter(coverage[symbol][timeframe].values()))["total_bars"]
            missing = [period for period, item in coverage[symbol][timeframe].items() if not item["ready"]]
            print(f"{symbol} {timeframe} bars={total} missing={','.join(missing) if missing else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
