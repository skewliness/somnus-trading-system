#!/usr/bin/env python3
"""Cross-symbol Gann zone outcome stratification / 跨币种 Gann 区域收益分层评估.

Walk-forward evaluation of trading_system.gann_feature across symbols:
for each day (after warmup) compute the Gann feature as-of that day (no
future data enters anchor selection), then measure forward returns and
MFE/MAE stratified by zone (above_1x1 / 1x2_to_1x1 / below_1x2).

对每个交易日逐日前推计算 Gann 结构特征（锚点只用历史数据），再统计各
zone 的未来 1/3/7/14/30 日收益、MFE/MAE，检验 zone 排序是否成立。

Usage:
    python3 scripts/evaluate_gann_structural_features.py \
        [--symbols BTCUSDT ETHUSDT ...] [--start-date 2021-01-01] \
        [--lookback-days 365] [--offline] [--cache-ttl-hours 24]

Data: Binance USDS futures public daily klines (no API key). Raw klines
cached under data/gann_eval_cache/{symbol}_1d.json. Outputs:
    reports/gann_eval/gann_zone_stratification_rows.csv
    reports/gann_eval/gann_zone_stratification_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading_system.gann_feature import compute_gann_feature  # noqa: E402

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "HYPEUSDT", "ZECUSDT"]
BINANCE_FAPI_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
CACHE_DIR = REPO_ROOT / "data" / "gann_eval_cache"
REPORTS_DIR = REPO_ROOT / "reports" / "gann_eval"
ROWS_CSV = REPORTS_DIR / "gann_zone_stratification_rows.csv"
SUMMARY_CSV = REPORTS_DIR / "gann_zone_stratification_summary.csv"

HORIZONS = (1, 3, 7, 14, 30)
MFE_MAE_WINDOW_DAYS = 30
VERDICT_HORIZONS = (7, 14, 30)
ZONE_RANKING = ("above_1x1", "1x2_to_1x1", "below_1x2")
MAX_KLINES_PER_REQUEST = 1500

ROW_COLUMNS = [
    "symbol", "date", "close", "zone", "bias", "confidence",
    "anchor_high_time", "anchor_low_time", "cycle_days",
    "line_1x2", "line_1x1", "line_2x1", "anchor_stability",
    "future_return_1d", "future_return_3d", "future_return_7d",
    "future_return_14d", "future_return_30d", "mfe_30d", "mae_30d",
]
SUMMARY_COLUMNS = [
    "symbol", "zone", "horizon_days", "n", "avg_pct", "median_pct",
    "win_rate_pct", "profit_factor", "avg_mfe_pct", "avg_mae_pct",
]


# ---------------------------------------------------------------------------
# Pure computation helpers (unit-testable, no network / no I/O)
# 纯计算函数（可单测，无网络、无 I/O）
# ---------------------------------------------------------------------------

def compute_future_return_pct(closes: list[float], index: int, horizon_days: int) -> float | None:
    """% return from closes[index] to closes[index + horizon_days]; None if unavailable."""
    target = index + horizon_days
    if index < 0 or target >= len(closes):
        return None
    entry = closes[index]
    if entry <= 0:
        return None
    return (closes[target] - entry) / entry * 100.0


def compute_mfe_mae_pct(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    index: int,
    window_days: int = MFE_MAE_WINDOW_DAYS,
) -> tuple[float | None, float | None]:
    """Max favorable / adverse excursion % over the NEXT window_days bars.

    MFE = (max high - entry) / entry * 100 ; MAE = (min low - entry) / entry * 100.
    Requires the full window; otherwise (None, None).
    """
    end = index + window_days
    if index < 0 or end >= len(closes):
        return None, None
    entry = closes[index]
    if entry <= 0:
        return None, None
    window_highs = highs[index + 1:end + 1]
    window_lows = lows[index + 1:end + 1]
    if not window_highs or not window_lows:
        return None, None
    mfe = (max(window_highs) - entry) / entry * 100.0
    mae = (min(window_lows) - entry) / entry * 100.0
    return mfe, mae


def profit_factor(returns: list[float]) -> float | None:
    """Sum of wins / abs sum of losses. inf if wins and no losses; None if no trades either way."""
    wins = sum(r for r in returns if r > 0)
    losses = sum(-r for r in returns if r < 0)
    if losses == 0:
        return math.inf if wins > 0 else None
    return wins / losses


def summarize_rows(
    rows: list[dict[str, Any]],
    horizons: tuple[int, ...] = HORIZONS,
) -> list[dict[str, Any]]:
    """Group row-level records by symbol x zone x horizon into summary stats.

    Each input row needs: symbol, zone, future_return_{h}d for h in horizons,
    mfe_30d, mae_30d (numeric or None). Returns sorted summary dicts.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["symbol"]), str(row["zone"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (symbol, zone), group in sorted(groups.items()):
        mfe_values = [r["mfe_30d"] for r in group if r.get("mfe_30d") is not None]
        mae_values = [r["mae_30d"] for r in group if r.get("mae_30d") is not None]
        avg_mfe = statistics.fmean(mfe_values) if mfe_values else None
        avg_mae = statistics.fmean(mae_values) if mae_values else None
        for horizon in horizons:
            key = f"future_return_{horizon}d"
            values = [r[key] for r in group if r.get(key) is not None]
            if not values:
                continue
            summary.append({
                "symbol": symbol,
                "zone": zone,
                "horizon_days": horizon,
                "n": len(values),
                "avg_pct": statistics.fmean(values),
                "median_pct": statistics.median(values),
                "win_rate_pct": sum(1 for v in values if v > 0) / len(values) * 100.0,
                "profit_factor": profit_factor(values),
                "avg_mfe_pct": avg_mfe,
                "avg_mae_pct": avg_mae,
            })
    return summary


def zone_ranking_verdict(
    summary: list[dict[str, Any]],
    symbol: str,
    horizon_days: int,
    ranking: tuple[str, ...] = ZONE_RANKING,
) -> bool | None:
    """True if avg returns rank strictly ranking[0] > ranking[1] > ranking[2].

    None when any zone has no data for this symbol/horizon (undecidable).
    """
    averages: dict[str, float] = {}
    for entry in summary:
        if entry["symbol"] == symbol and entry["horizon_days"] == horizon_days:
            averages[str(entry["zone"])] = float(entry["avg_pct"])
    if any(zone not in averages for zone in ranking):
        return None
    return all(averages[ranking[i]] > averages[ranking[i + 1]] for i in range(len(ranking) - 1))


def build_outcome_row(
    feature_dict: dict[str, Any],
    date_str: str,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    index: int,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, Any]:
    """Assemble one row-level record from a computed feature + forward outcomes."""
    mfe, mae = compute_mfe_mae_pct(highs, lows, closes, index)
    row: dict[str, Any] = {
        "symbol": feature_dict["symbol"],
        "date": date_str,
        "close": closes[index],
        "zone": feature_dict["zone"],
        "bias": feature_dict["bias"],
        "confidence": feature_dict["confidence"],
        "anchor_high_time": feature_dict["anchor_high_time"],
        "anchor_low_time": feature_dict["anchor_low_time"],
        "cycle_days": feature_dict["cycle_days"],
        "line_1x2": feature_dict["line_values"].get("1x2"),
        "line_1x1": feature_dict["line_values"].get("1x1"),
        "line_2x1": feature_dict["line_values"].get("2x1"),
        "anchor_stability": feature_dict["anchor_stability"],
        "mfe_30d": mfe,
        "mae_30d": mae,
    }
    for horizon in horizons:
        row[f"future_return_{horizon}d"] = compute_future_return_pct(closes, index, horizon)
    return row


# ---------------------------------------------------------------------------
# Data plane: Binance USDS futures daily klines + local JSON cache
# ---------------------------------------------------------------------------

def fetch_binance_daily_klines(symbol: str, start_ms: int) -> list[list[Any]]:
    import requests

    klines: list[list[Any]] = []
    cursor = start_ms
    while True:
        response = requests.get(
            BINANCE_FAPI_KLINES_URL,
            params={"symbol": symbol, "interval": "1d", "startTime": cursor, "limit": MAX_KLINES_PER_REQUEST},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break
        klines.extend(batch)
        if len(batch) < MAX_KLINES_PER_REQUEST:
            break
        cursor = int(batch[-1][0]) + 86_400_000
        time.sleep(0.3)
    return klines


def load_symbol_klines(
    symbol: str,
    start_ms: int,
    *,
    offline: bool,
    cache_ttl_hours: float,
) -> list[list[Any]] | None:
    """Return raw klines from a fresh cache, else fetch and cache. None on failure."""
    cache_path = CACHE_DIR / f"{symbol}_1d.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            fetched_at = float(payload.get("fetched_at") or 0.0)
            cached = payload.get("klines") or []
            fresh = (time.time() - fetched_at) <= cache_ttl_hours * 3600
            if cached and (offline or fresh):
                print(f"[{symbol}] cache hit: {len(cached)} klines ({cache_path})")
                return cached
        except (ValueError, OSError) as exc:
            print(f"[{symbol}] cache unreadable ({exc}); refetching", file=sys.stderr)
    if offline:
        print(f"[{symbol}] --offline and no usable cache; skipping", file=sys.stderr)
        return None
    try:
        klines = fetch_binance_daily_klines(symbol, start_ms)
    except Exception as exc:  # network / HTTP / JSON errors: skip symbol, keep going
        print(f"[{symbol}] fetch failed ({exc}); skipping", file=sys.stderr)
        return None
    if not klines:
        print(f"[{symbol}] no klines returned; skipping", file=sys.stderr)
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"fetched_at": time.time(), "symbol": symbol, "klines": klines}))
    print(f"[{symbol}] fetched {len(klines)} klines -> {cache_path}")
    return klines


def klines_to_daily_rows(klines: list[list[Any]]) -> list[dict[str, Any]]:
    """Binance kline arrays -> normalized daily rows with ISO dates."""
    rows: list[dict[str, Any]] = []
    for entry in klines:
        try:
            rows.append({
                "time": datetime.fromtimestamp(int(entry[0]) / 1000, tz=timezone.utc).date().isoformat(),
                "open": float(entry[1]),
                "high": float(entry[2]),
                "low": float(entry[3]),
                "close": float(entry[4]),
            })
        except (TypeError, ValueError, IndexError):
            continue
    return rows


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward_symbol(
    symbol: str,
    daily_rows: list[dict[str, Any]],
    lookback_days: int,
) -> list[dict[str, Any]]:
    closes = [float(r["close"]) for r in daily_rows]
    highs = [float(r["high"]) for r in daily_rows]
    lows = [float(r["low"]) for r in daily_rows]
    records: list[dict[str, Any]] = []
    for index in range(lookback_days, len(daily_rows)):
        feature = compute_gann_feature(symbol, daily_rows, asof_index=index, lookback_days=lookback_days)
        if feature is None:
            continue
        records.append(build_outcome_row(feature.to_dict(), str(daily_rows[index]["time"]), closes, highs, lows, index))
    return records


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return round(value, 6)
    return value


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _fmt(row.get(col)) for col in columns})


def print_summary_table(summary: list[dict[str, Any]]) -> None:
    header = f"{'symbol':<10} {'zone':<12} {'hz':>4} {'n':>6} {'avg%':>8} {'med%':>8} {'win%':>7} {'pf':>7} {'mfe%':>8} {'mae%':>8}"
    print("\n=== Gann zone stratification summary / Gann 区域分层汇总 ===")
    print(header)
    print("-" * len(header))
    for entry in summary:
        pf = entry["profit_factor"]
        pf_text = "inf" if (pf is not None and math.isinf(pf)) else ("" if pf is None else f"{pf:.2f}")
        mfe = entry["avg_mfe_pct"]
        mae = entry["avg_mae_pct"]
        print(
            f"{entry['symbol']:<10} {entry['zone']:<12} {entry['horizon_days']:>3}d {entry['n']:>6} "
            f"{entry['avg_pct']:>8.3f} {entry['median_pct']:>8.3f} {entry['win_rate_pct']:>7.2f} {pf_text:>7} "
            f"{('' if mfe is None else f'{mfe:.2f}'):>8} {('' if mae is None else f'{mae:.2f}'):>8}"
        )


def print_verdicts(summary: list[dict[str, Any]], symbols: list[str]) -> None:
    print("\n=== Zone ranking verdicts (above_1x1 > 1x2_to_1x1 > below_1x2 on avg return) ===")
    for symbol in symbols:
        parts = []
        for horizon in VERDICT_HORIZONS:
            verdict = zone_ranking_verdict(summary, symbol, horizon)
            parts.append(f"{horizon}d={'HOLDS' if verdict else 'n/a' if verdict is None else 'FAILS'}")
        decided = [zone_ranking_verdict(summary, symbol, h) for h in VERDICT_HORIZONS]
        if all(v is None for v in decided):
            overall = "NO DATA"
        elif all(v for v in decided if v is not None) and any(v is not None for v in decided):
            overall = "HOLDS"
        else:
            overall = "FAILS"
        print(f"VERDICT {symbol}: {' '.join(parts)} -> {overall}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gann zone outcome stratification / Gann 区域收益分层评估")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS), help="Binance USDS futures symbols")
    parser.add_argument("--start-date", default="2021-01-01", help="Kline history start date (UTC, YYYY-MM-DD)")
    parser.add_argument("--lookback-days", type=int, default=365, help="Gann anchor lookback / warmup days")
    parser.add_argument("--offline", action="store_true", help="Use cached klines only; never hit the network")
    parser.add_argument("--cache-ttl-hours", type=float, default=24.0, help="Cache freshness window in hours")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start_ms = int(datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_rows: list[dict[str, Any]] = []
    evaluated_symbols: list[str] = []
    for symbol in args.symbols:
        klines = load_symbol_klines(symbol, start_ms, offline=args.offline, cache_ttl_hours=args.cache_ttl_hours)
        if not klines:
            continue
        daily_rows = klines_to_daily_rows(klines)
        if len(daily_rows) <= args.lookback_days:
            print(f"[{symbol}] only {len(daily_rows)} days (< warmup {args.lookback_days}); skipping", file=sys.stderr)
            continue
        started = time.time()
        records = walk_forward_symbol(symbol, daily_rows, args.lookback_days)
        print(f"[{symbol}] walk-forward: {len(records)} feature days from {len(daily_rows)} bars ({time.time() - started:.1f}s)")
        all_rows.extend(records)
        evaluated_symbols.append(symbol)

    if not all_rows:
        print("No rows produced (all symbols skipped?)", file=sys.stderr)
        return 1

    summary = summarize_rows(all_rows)
    write_csv(ROWS_CSV, ROW_COLUMNS, all_rows)
    write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, summary)
    print(f"\nWrote {len(all_rows)} rows -> {ROWS_CSV}")
    print(f"Wrote {len(summary)} summary rows -> {SUMMARY_CSV}")
    print_summary_table(summary)
    print_verdicts(summary, evaluated_symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
