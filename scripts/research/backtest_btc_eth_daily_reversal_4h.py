#!/usr/bin/env python3
"""Backtest BTC/ETH daily MA7/MA20 reversal signals on subsequent 4H paths.

The script intentionally keeps the research cache separate from execution and
chart caches. Signal confirmation uses a closed 1D candle and execution starts
at the next 4H open, so no information from the confirming close is traded
early.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


BASE_URL = "https://fapi.binance.com"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
INTERVAL_MS = {"4h": 4 * 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}
HORIZONS = (1, 2, 3, 6, 12, 18, 24, 30, 42)
THRESHOLDS = (0.5, 1.0, 2.0, 3.0, 5.0)
MAX_PATH_BARS = max(HORIZONS)


def utc_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def iso_to_ms(value: str) -> int:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    return statistics.fmean(numbers) if numbers else None


def median(values: Iterable[float]) -> float | None:
    numbers = list(values)
    return statistics.median(numbers) if numbers else None


def percentile(values: Iterable[float], q: float) -> float | None:
    numbers = sorted(values)
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    position = (len(numbers) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numbers[lower]
    weight = position - lower
    return numbers[lower] * (1.0 - weight) + numbers[upper] * weight


def request_json(session: requests.Session, path: str, params: dict[str, Any] | None = None) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            response = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network and exchange errors are retried together
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
    raise RuntimeError(f"Binance request failed after retries: {path} {params}: {last_error}")


def fetch_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    cursor = start_ms
    rows_by_open_time: dict[int, dict[str, Any]] = {}
    step_ms = INTERVAL_MS[interval]
    while cursor <= end_ms:
        payload = request_json(
            session,
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        if not payload:
            break
        for raw in payload:
            row = {
                "open_time": int(raw[0]),
                "open": float(raw[1]),
                "high": float(raw[2]),
                "low": float(raw[3]),
                "close": float(raw[4]),
                "volume": float(raw[5]),
                "close_time": int(raw[6]),
                "quote_volume": float(raw[7]),
                "trade_count": int(raw[8]),
                "taker_buy_base_volume": float(raw[9]),
                "taker_buy_quote_volume": float(raw[10]),
            }
            if row["close_time"] < end_ms:
                rows_by_open_time[row["open_time"]] = row
        last_open = int(payload[-1][0])
        next_cursor = last_open + step_ms
        if next_cursor <= cursor:
            raise RuntimeError(f"Pagination stalled for {symbol} {interval} at {cursor}")
        cursor = next_cursor
        if len(payload) < 1500:
            break
        time.sleep(0.05)
    return [rows_by_open_time[key] for key in sorted(rows_by_open_time)]


def data_quality(symbol: str, interval: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    step_ms = INTERVAL_MS[interval]
    gaps: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        delta = current["open_time"] - previous["open_time"]
        if delta != step_ms:
            gaps.append(
                {
                    "after": utc_iso(previous["open_time"]),
                    "before": utc_iso(current["open_time"]),
                    "delta_ms": delta,
                    "estimated_missing_bars": max(0, round(delta / step_ms) - 1),
                }
            )
    return {
        "symbol": symbol,
        "interval": interval,
        "rows": len(rows),
        "first_open_time": utc_iso(rows[0]["open_time"]) if rows else None,
        "last_open_time": utc_iso(rows[-1]["open_time"]) if rows else None,
        "last_close_time": utc_iso(rows[-1]["close_time"]) if rows else None,
        "duplicate_open_times": len(rows) - len({row["open_time"] for row in rows}),
        "gap_count": len(gaps),
        "estimated_missing_bars": sum(item["estimated_missing_bars"] for item in gaps),
        "gaps": gaps[:50],
    }


def add_daily_state(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closes = [float(row["close"]) for row in rows]
    prefix = [0.0]
    for close in closes:
        prefix.append(prefix[-1] + close)

    def sma(index: int, period: int) -> float | None:
        if index + 1 < period:
            return None
        return (prefix[index + 1] - prefix[index + 1 - period]) / period

    states: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        ma7 = sma(index, 7)
        ma20 = sma(index, 20)
        if ma7 is None or ma20 is None:
            continue
        close = float(row["close"])
        states.append(
            {
                **row,
                "ma7": ma7,
                "ma20": ma20,
                "bearish_stack": ma20 > ma7,
                "bullish_stack": ma7 >= ma20,
                "above_ma7": close >= ma7,
                "above_ma20": close >= ma20,
                "below_ma7": close <= ma7,
                "below_ma20": close <= ma20,
                "above_both": close >= ma7 and close >= ma20,
                "below_both": close <= ma7 and close <= ma20,
            }
        )
    return states


def transition_events(states: list[dict[str, Any]], predicate: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous = False
    for state in states:
        current = bool(predicate(state))
        if current and not previous:
            events.append(state)
        previous = current
    return events


def event_context(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "daily_open_time": utc_iso(int(state["open_time"])),
        "daily_close_time": utc_iso(int(state["close_time"])),
        "daily_close": round(float(state["close"]), 8),
        "ma7": round(float(state["ma7"]), 8),
        "ma20": round(float(state["ma20"]), 8),
        "bearish_stack": bool(state["bearish_stack"]),
        "bullish_stack": bool(state["bullish_stack"]),
        "above_both": bool(state["above_both"]),
        "below_both": bool(state["below_both"]),
    }


def build_signal_events(daily_states: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    # Exact current production market rule for invalidating shorts: both BTC and
    # ETH have a closed daily candle above MA7 and MA20. The current code does
    # not require MA20 > MA7 in this market-level guard.
    state_maps = {
        symbol: {state["open_time"]: state for state in states}
        for symbol, states in daily_states.items()
    }
    common_times = sorted(set(state_maps["BTCUSDT"]) & set(state_maps["ETHUSDT"]))
    joint_rows: list[dict[str, Any]] = []
    for open_time in common_times:
        btc = state_maps["BTCUSDT"][open_time]
        eth = state_maps["ETHUSDT"][open_time]
        joint_rows.append(
            {
                "open_time": open_time,
                "close_time": max(int(btc["close_time"]), int(eth["close_time"])),
                "condition": bool(btc["above_both"] and eth["above_both"]),
                "drivers": {"BTCUSDT": event_context(btc), "ETHUSDT": event_context(eth)},
            }
        )
    previous = False
    for row in joint_rows:
        current = bool(row["condition"])
        if current and not previous:
            for symbol in SYMBOLS:
                events.append(
                    {
                        "signal_family": "production_joint_bullish",
                        "signal_description": "BTC+ETH closed above daily MA7 and MA20",
                        "symbol": symbol,
                        "direction": "long",
                        "signal_time": int(row["close_time"]),
                        "signal_daily_open_time": int(row["open_time"]),
                        "drivers": row["drivers"],
                    }
                )
        previous = current

    # Exact current production per-symbol long invalidation rule.
    for symbol in SYMBOLS:
        bearish_events = transition_events(
            daily_states[symbol],
            lambda state: state["bullish_stack"] and state["below_both"],
        )
        for state in bearish_events:
            events.append(
                {
                    "signal_family": "production_asset_bearish",
                    "signal_description": "daily MA7>=MA20 and close lost both MA7 and MA20",
                    "symbol": symbol,
                    "direction": "short",
                    "signal_time": int(state["close_time"]),
                    "signal_daily_open_time": int(state["open_time"]),
                    "drivers": {symbol: event_context(state)},
                }
            )

    # Symmetric per-asset reference. This is not an extra production rule; it
    # measures whether the intended MA7/MA20 reversal concept behaves similarly
    # on each asset in both directions.
    for symbol in SYMBOLS:
        bullish_events = transition_events(
            daily_states[symbol],
            lambda state: state["bearish_stack"] and state["above_both"],
        )
        for state in bullish_events:
            events.append(
                {
                    "signal_family": "reference_asset_bullish_strict",
                    "signal_description": "daily MA20>MA7 and close reclaimed both MA7 and MA20",
                    "symbol": symbol,
                    "direction": "long",
                    "signal_time": int(state["close_time"]),
                    "signal_daily_open_time": int(state["open_time"]),
                    "drivers": {symbol: event_context(state)},
                }
            )

    return sorted(events, key=lambda item: (item["signal_time"], item["signal_family"], item["symbol"]))


def directional_return(entry: float, value: float, direction: str) -> float:
    raw = (value / entry - 1.0) * 100.0
    return raw if direction == "long" else -raw


def evaluate_event(event: dict[str, Any], rows_4h: list[dict[str, Any]]) -> dict[str, Any] | None:
    open_times = [int(row["open_time"]) for row in rows_4h]
    entry_index = bisect.bisect_right(open_times, int(event["signal_time"]))
    if entry_index >= len(rows_4h):
        return None
    entry_row = rows_4h[entry_index]
    entry_price = float(entry_row["open"])
    direction = str(event["direction"])
    available = len(rows_4h) - entry_index
    evaluated: dict[str, Any] = {
        **event,
        "signal_time_iso": utc_iso(int(event["signal_time"])),
        "signal_daily_open_time_iso": utc_iso(int(event["signal_daily_open_time"])),
        "entry_index": entry_index,
        "entry_time": int(entry_row["open_time"]),
        "entry_time_iso": utc_iso(int(entry_row["open_time"])),
        "entry_price": entry_price,
        "available_4h_bars": available,
    }

    for horizon in HORIZONS:
        if available < horizon:
            continue
        path = rows_4h[entry_index : entry_index + horizon]
        exit_close = float(path[-1]["close"])
        raw_return = (exit_close / entry_price - 1.0) * 100.0
        if direction == "long":
            favorable_moves = [(float(row["high"]) / entry_price - 1.0) * 100.0 for row in path]
            adverse_moves = [(float(row["low"]) / entry_price - 1.0) * 100.0 for row in path]
        else:
            favorable_moves = [(1.0 - float(row["low"]) / entry_price) * 100.0 for row in path]
            adverse_moves = [(1.0 - float(row["high"]) / entry_price) * 100.0 for row in path]
        mfe = max(favorable_moves)
        mae = min(adverse_moves)
        evaluated[f"close_raw_return_{horizon}"] = raw_return
        evaluated[f"close_directional_return_{horizon}"] = raw_return if direction == "long" else -raw_return
        evaluated[f"mfe_{horizon}"] = mfe
        evaluated[f"mae_{horizon}"] = mae

    path = rows_4h[entry_index : entry_index + min(MAX_PATH_BARS, available)]
    if path:
        if direction == "long":
            favorable_moves = [(float(row["high"]) / entry_price - 1.0) * 100.0 for row in path]
            adverse_moves = [(float(row["low"]) / entry_price - 1.0) * 100.0 for row in path]
        else:
            favorable_moves = [(1.0 - float(row["low"]) / entry_price) * 100.0 for row in path]
            adverse_moves = [(1.0 - float(row["high"]) / entry_price) * 100.0 for row in path]
        evaluated["peak_favorable_bar"] = favorable_moves.index(max(favorable_moves)) + 1
        evaluated["peak_adverse_bar"] = adverse_moves.index(min(adverse_moves)) + 1
        evaluated["peak_favorable_pct"] = max(favorable_moves)
        evaluated["peak_adverse_pct"] = min(adverse_moves)

    for threshold in THRESHOLDS:
        key = str(threshold).replace(".", "p")
        favorable_bar: int | None = None
        adverse_bar: int | None = None
        threshold_fraction = threshold / 100.0
        for offset, row in enumerate(path, start=1):
            if direction == "long":
                favorable_hit = float(row["high"]) >= entry_price * (1.0 + threshold_fraction)
                adverse_hit = float(row["low"]) <= entry_price * (1.0 - threshold_fraction)
            else:
                favorable_hit = float(row["low"]) <= entry_price * (1.0 - threshold_fraction)
                adverse_hit = float(row["high"]) >= entry_price * (1.0 + threshold_fraction)
            if favorable_hit and favorable_bar is None:
                favorable_bar = offset
            if adverse_hit and adverse_bar is None:
                adverse_bar = offset
            if favorable_bar is not None and adverse_bar is not None:
                break
        if favorable_bar is None and adverse_bar is None:
            first_outcome = "none"
        elif favorable_bar is None:
            first_outcome = "adverse"
        elif adverse_bar is None:
            first_outcome = "favorable"
        elif favorable_bar < adverse_bar:
            first_outcome = "favorable"
        elif adverse_bar < favorable_bar:
            first_outcome = "adverse"
        else:
            first_outcome = "same_bar_ambiguous"
        evaluated[f"favorable_{key}_bar"] = favorable_bar
        evaluated[f"adverse_{key}_bar"] = adverse_bar
        evaluated[f"first_{key}_outcome"] = first_outcome

    return evaluated


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["signal_family"]), str(row["symbol"]), str(row["direction"])


def mark_nonoverlap_42bar(evaluated: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        row["complete_42bar"] = int(row.get("available_4h_bars") or 0) >= MAX_PATH_BARS
        row["nonoverlap_42bar"] = False
        groups[group_key(row)].append(row)
    minimum_gap_ms = MAX_PATH_BARS * INTERVAL_MS["4h"]
    for rows in groups.values():
        last_kept_entry: int | None = None
        for row in sorted(rows, key=lambda item: int(item["entry_time"])):
            entry_time = int(row["entry_time"])
            if last_kept_entry is None or entry_time - last_kept_entry >= minimum_gap_ms:
                row["nonoverlap_42bar"] = True
                last_kept_entry = entry_time


def summarize_horizons(evaluated: list[dict[str, Any]], sample_set: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        groups[group_key(row)].append(row)
    output: list[dict[str, Any]] = []
    for (family, symbol, direction), rows in sorted(groups.items()):
        for horizon in HORIZONS:
            eligible = [row for row in rows if f"close_directional_return_{horizon}" in row]
            directional = [float(row[f"close_directional_return_{horizon}"]) for row in eligible]
            raw = [float(row[f"close_raw_return_{horizon}"]) for row in eligible]
            mfes = [float(row[f"mfe_{horizon}"]) for row in eligible]
            maes = [float(row[f"mae_{horizon}"]) for row in eligible]
            output.append(
                {
                    "sample_set": sample_set,
                    "signal_family": family,
                    "symbol": symbol,
                    "direction": direction,
                    "horizon_4h_bars": horizon,
                    "elapsed_hours": horizon * 4,
                    "samples": len(eligible),
                    "directional_win_rate_pct": round_or_none(100.0 * sum(value > 0 for value in directional) / len(directional)) if directional else None,
                    "mean_raw_return_pct": round_or_none(mean(raw)),
                    "median_raw_return_pct": round_or_none(median(raw)),
                    "mean_directional_return_pct": round_or_none(mean(directional)),
                    "median_directional_return_pct": round_or_none(median(directional)),
                    "p25_directional_return_pct": round_or_none(percentile(directional, 0.25)),
                    "p75_directional_return_pct": round_or_none(percentile(directional, 0.75)),
                    "mean_mfe_pct": round_or_none(mean(mfes)),
                    "median_mfe_pct": round_or_none(median(mfes)),
                    "mean_mae_pct": round_or_none(mean(maes)),
                    "median_mae_pct": round_or_none(median(maes)),
                }
            )
    return output


def summarize_thresholds(evaluated: list[dict[str, Any]], sample_set: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        groups[group_key(row)].append(row)
    output: list[dict[str, Any]] = []
    for (family, symbol, direction), rows in sorted(groups.items()):
        for threshold in THRESHOLDS:
            key = str(threshold).replace(".", "p")
            favorable_bars = [int(row[f"favorable_{key}_bar"]) for row in rows if row.get(f"favorable_{key}_bar") is not None]
            adverse_bars = [int(row[f"adverse_{key}_bar"]) for row in rows if row.get(f"adverse_{key}_bar") is not None]
            outcomes = [str(row.get(f"first_{key}_outcome") or "none") for row in rows]
            output.append(
                {
                    "sample_set": sample_set,
                    "signal_family": family,
                    "symbol": symbol,
                    "direction": direction,
                    "threshold_pct": threshold,
                    "max_4h_bars": MAX_PATH_BARS,
                    "samples": len(rows),
                    "favorable_hit_count": len(favorable_bars),
                    "favorable_hit_rate_pct": round_or_none(100.0 * len(favorable_bars) / len(rows)) if rows else None,
                    "favorable_median_4h_bars": round_or_none(median(favorable_bars), 3),
                    "favorable_mean_4h_bars": round_or_none(mean(favorable_bars), 3),
                    "adverse_hit_count": len(adverse_bars),
                    "adverse_hit_rate_pct": round_or_none(100.0 * len(adverse_bars) / len(rows)) if rows else None,
                    "adverse_median_4h_bars": round_or_none(median(adverse_bars), 3),
                    "adverse_mean_4h_bars": round_or_none(mean(adverse_bars), 3),
                    "favorable_first_count": outcomes.count("favorable"),
                    "adverse_first_count": outcomes.count("adverse"),
                    "same_bar_ambiguous_count": outcomes.count("same_bar_ambiguous"),
                    "neither_hit_count": outcomes.count("none"),
                    "favorable_first_rate_pct": round_or_none(100.0 * outcomes.count("favorable") / len(rows)) if rows else None,
                    "adverse_first_rate_pct": round_or_none(100.0 * outcomes.count("adverse") / len(rows)) if rows else None,
                }
            )
    return output


def summarize_peaks(evaluated: list[dict[str, Any]], sample_set: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        groups[group_key(row)].append(row)
    output: list[dict[str, Any]] = []
    for (family, symbol, direction), rows in sorted(groups.items()):
        favorable_bars = [float(row["peak_favorable_bar"]) for row in rows if row.get("peak_favorable_bar") is not None]
        adverse_bars = [float(row["peak_adverse_bar"]) for row in rows if row.get("peak_adverse_bar") is not None]
        favorable = [float(row["peak_favorable_pct"]) for row in rows if row.get("peak_favorable_pct") is not None]
        adverse = [float(row["peak_adverse_pct"]) for row in rows if row.get("peak_adverse_pct") is not None]
        output.append(
            {
                "sample_set": sample_set,
                "signal_family": family,
                "symbol": symbol,
                "direction": direction,
                "samples": len(rows),
                "window_4h_bars": MAX_PATH_BARS,
                "median_peak_favorable_bar": round_or_none(median(favorable_bars), 3),
                "mean_peak_favorable_bar": round_or_none(mean(favorable_bars), 3),
                "median_peak_favorable_pct": round_or_none(median(favorable)),
                "mean_peak_favorable_pct": round_or_none(mean(favorable)),
                "median_peak_adverse_bar": round_or_none(median(adverse_bars), 3),
                "mean_peak_adverse_bar": round_or_none(mean(adverse_bars), 3),
                "median_peak_adverse_pct": round_or_none(median(adverse)),
                "mean_peak_adverse_pct": round_or_none(mean(adverse)),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/trading_system/backtests/btc_eth_daily_reversal_4h_20260719",
    )
    parser.add_argument("--start", default="2019-09-01T00:00:00Z")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    kline_dir = output_dir / "klines"
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    server_payload = request_json(session, "/fapi/v1/time")
    server_time_ms = int(server_payload["serverTime"])
    start_ms = iso_to_ms(args.start)

    datasets: dict[str, dict[str, list[dict[str, Any]]]] = {symbol: {} for symbol in SYMBOLS}
    qualities: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for interval in ("1d", "4h"):
            rows = fetch_klines(session, symbol, interval, start_ms, server_time_ms)
            if not rows:
                raise RuntimeError(f"No closed klines returned for {symbol} {interval}")
            datasets[symbol][interval] = rows
            quality = data_quality(symbol, interval, rows)
            qualities.append(quality)
            json_dump(
                kline_dir / f"{symbol}_{interval}_binance_usdm_futures_utc.json",
                {
                    "metadata": {
                        "source": "Binance USDM Futures REST /fapi/v1/klines",
                        "market": "USDM perpetual futures",
                        "timezone": "UTC",
                        "symbol": symbol,
                        "interval": interval,
                        "fetched_at": utc_iso(server_time_ms),
                        "closed_candles_only": True,
                        "quality": quality,
                    },
                    "rows": rows,
                },
            )

    daily_states = {symbol: add_daily_state(datasets[symbol]["1d"]) for symbol in SYMBOLS}
    signal_events = build_signal_events(daily_states)
    evaluated: list[dict[str, Any]] = []
    for event in signal_events:
        result = evaluate_event(event, datasets[event["symbol"]]["4h"])
        if result is not None:
            evaluated.append(result)

    mark_nonoverlap_42bar(evaluated)
    nonoverlap = [row for row in evaluated if row["nonoverlap_42bar"]]
    complete_42bar = [row for row in evaluated if row["complete_42bar"]]
    nonoverlap_complete_42bar = [row for row in nonoverlap if row["complete_42bar"]]
    horizon_stats = (
        summarize_horizons(evaluated, "all_transitions")
        + summarize_horizons(nonoverlap, "nonoverlap_42bar")
    )
    threshold_stats = (
        summarize_thresholds(complete_42bar, "all_complete_42bar")
        + summarize_thresholds(nonoverlap_complete_42bar, "nonoverlap_complete_42bar")
    )
    peak_stats = (
        summarize_peaks(complete_42bar, "all_complete_42bar")
        + summarize_peaks(nonoverlap_complete_42bar, "nonoverlap_complete_42bar")
    )
    event_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"all": 0, "complete_42bar": 0, "nonoverlap_42bar": 0, "nonoverlap_complete_42bar": 0})
    for row in evaluated:
        counts = event_counts["|".join(group_key(row))]
        counts["all"] += 1
        counts["complete_42bar"] += int(bool(row["complete_42bar"]))
        counts["nonoverlap_42bar"] += int(bool(row["nonoverlap_42bar"]))
        counts["nonoverlap_complete_42bar"] += int(bool(row["nonoverlap_42bar"] and row["complete_42bar"]))

    summary = {
        "generated_at": utc_iso(server_time_ms),
        "source": "Binance USDM Futures REST /fapi/v1/klines",
        "timezone": "UTC",
        "execution_assumption": "signal confirmed on closed 1D candle; entry at next 4H open",
        "production_config_verified": {
            "daily_ma7_short_reversal_required_1d_candles": 1,
            "daily_ma7_short_reversal_tolerance_pct": 0.0,
            "daily_ma7_short_reversal_market_symbols": list(SYMBOLS),
            "daily_ma7_short_reversal_require_all_market_symbols": True,
            "environment_override_found": False,
        },
        "signal_definitions": {
            "production_joint_bullish": "BTC and ETH each close above daily MA7 and MA20; first transition only",
            "production_asset_bearish": "per asset: MA7>=MA20 and close below both daily MA7 and MA20; first transition only",
            "reference_asset_bullish_strict": "per asset symmetric reference: MA20>MA7 and close above both daily MA7 and MA20; first transition only",
        },
        "horizons_4h_bars": list(HORIZONS),
        "thresholds_pct": list(THRESHOLDS),
        "event_counts": dict(sorted(event_counts.items())),
        "data_quality": qualities,
        "latest_daily_states": {
            symbol: event_context(daily_states[symbol][-1]) for symbol in SYMBOLS
        },
        "horizon_stats": horizon_stats,
        "threshold_stats": threshold_stats,
        "peak_stats": peak_stats,
    }
    json_dump(output_dir / "summary.json", summary)
    json_dump(output_dir / "data_quality.json", qualities)
    write_csv(output_dir / "events.csv", evaluated)
    write_csv(output_dir / "horizon_stats.csv", horizon_stats)
    write_csv(output_dir / "threshold_stats.csv", threshold_stats)
    write_csv(output_dir / "peak_stats.csv", peak_stats)
    print(json.dumps({
        "output_dir": str(output_dir),
        "generated_at": summary["generated_at"],
        "event_counts": summary["event_counts"],
        "data_quality": qualities,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
