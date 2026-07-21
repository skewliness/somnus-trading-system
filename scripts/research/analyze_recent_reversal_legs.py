#!/usr/bin/env python3
"""Analyze recent production daily-reversal legs until the next signal."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRODUCTION_FAMILIES = {"production_joint_bullish", "production_asset_bearish"}


def utc_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_klines(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["rows"])


def max_streak(directions: list[str], wanted: str) -> int:
    best = 0
    current = 0
    for direction in directions:
        if direction == wanted:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def round_value(value: float) -> float:
    return round(value, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="data/trading_system/backtests/btc_eth_daily_reversal_4h_20260719",
    )
    parser.add_argument("--cutoff", default="2026-07-01T00:00:00Z")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    cutoff_ms = int(datetime.fromisoformat(args.cutoff.replace("Z", "+00:00")).timestamp() * 1000)
    all_events = [
        row
        for row in load_csv(base_dir / "events.csv")
        if row["signal_family"] in PRODUCTION_FAMILIES
    ]
    all_events.sort(key=lambda row: (int(row["entry_time"]), row["symbol"]))
    recent_events = [row for row in all_events if int(row["signal_time"]) >= cutoff_ms]
    klines = {
        symbol: load_klines(base_dir / "klines" / f"{symbol}_4h_binance_usdm_futures_utc.json")
        for symbol in ("BTCUSDT", "ETHUSDT")
    }

    legs: list[dict[str, Any]] = []
    for event in recent_events:
        symbol = event["symbol"]
        entry_time = int(event["entry_time"])
        next_events = [
            candidate
            for candidate in all_events
            if candidate["symbol"] == symbol and int(candidate["entry_time"]) > entry_time
        ]
        next_event = min(next_events, key=lambda row: int(row["entry_time"])) if next_events else None
        end_time = int(next_event["entry_time"]) if next_event else None
        path = [
            row
            for row in klines[symbol]
            if int(row["open_time"]) >= entry_time and (end_time is None or int(row["open_time"]) < end_time)
        ]
        if not path:
            continue
        entry_price = float(event["entry_price"])
        end_close = float(path[-1]["close"])
        max_high_index, max_high_row = max(enumerate(path), key=lambda item: float(item[1]["high"]))
        min_low_index, min_low_row = min(enumerate(path), key=lambda item: float(item[1]["low"]))
        max_high = float(max_high_row["high"])
        min_low = float(min_low_row["low"])
        raw_return = (end_close / entry_price - 1.0) * 100.0
        long_mfe = (max_high / entry_price - 1.0) * 100.0
        long_mae = (min_low / entry_price - 1.0) * 100.0
        short_mfe = (1.0 - min_low / entry_price) * 100.0
        short_mae = (1.0 - max_high / entry_price) * 100.0
        directions = [
            "up" if float(row["close"]) > float(row["open"]) else "down" if float(row["close"]) < float(row["open"]) else "flat"
            for row in path
        ]
        drivers = json.loads(event["drivers"])
        driver = drivers[symbol]
        signal_direction = event["direction"]
        signal_return = raw_return if signal_direction == "long" else -raw_return
        genuine_joint_bullish_flip = bool(
            event["signal_family"] == "production_joint_bullish"
            and drivers
            and all(bool(item.get("bearish_stack")) and bool(item.get("above_both")) for item in drivers.values())
        )
        if genuine_joint_bullish_flip:
            structural_class = "genuine_bearish_to_bullish_flip"
        elif event["signal_family"] == "production_joint_bullish":
            structural_class = "bullish_reclaim_reconfirmation"
        else:
            structural_class = "genuine_bullish_to_bearish_flip"
        daily_close = float(driver["daily_close"])
        daily_ma7 = float(driver["ma7"])
        daily_ma20 = float(driver["ma20"])
        legs.append(
            {
                "signal_time_utc": event["signal_time_iso"],
                "entry_time_utc": event["entry_time_iso"],
                "symbol": symbol,
                "signal_family": event["signal_family"],
                "signal_direction": signal_direction,
                "structural_class": structural_class,
                "daily_close": daily_close,
                "daily_ma7": daily_ma7,
                "daily_ma20": daily_ma20,
                "daily_close_vs_ma7_pct": round_value((daily_close / daily_ma7 - 1.0) * 100.0),
                "daily_close_vs_ma20_pct": round_value((daily_close / daily_ma20 - 1.0) * 100.0),
                "entry_price": round_value(entry_price),
                "segment_end_reason": "next_daily_signal" if next_event else "latest_closed_4h",
                "segment_end_time_utc": utc_iso(int(path[-1]["close_time"])),
                "segment_end_close": round_value(end_close),
                "four_h_bars": len(path),
                "up_4h_bars": directions.count("up"),
                "down_4h_bars": directions.count("down"),
                "flat_4h_bars": directions.count("flat"),
                "max_consecutive_up_bars": max_streak(directions, "up"),
                "max_consecutive_down_bars": max_streak(directions, "down"),
                "price_net_return_pct": round_value(raw_return),
                "signal_side_return_pct": round_value(signal_return),
                "hypothetical_long_return_pct": round_value(raw_return),
                "hypothetical_short_return_pct": round_value(-raw_return),
                "long_peak_price": round_value(max_high),
                "long_peak_bar": max_high_index + 1,
                "long_peak_time_utc": utc_iso(int(max_high_row["open_time"])),
                "long_mfe_pct": round_value(long_mfe),
                "long_mae_pct": round_value(long_mae),
                "short_trough_price": round_value(min_low),
                "short_trough_bar": min_low_index + 1,
                "short_trough_time_utc": utc_iso(int(min_low_row["open_time"])),
                "short_mfe_pct": round_value(short_mfe),
                "short_mae_pct": round_value(short_mae),
                "next_signal_time_utc": next_event["signal_time_iso"] if next_event else "",
                "next_signal_direction": next_event["direction"] if next_event else "",
            }
        )

    unique_market_signals: dict[tuple[str, int], dict[str, Any]] = {}
    unique_asset_signals: dict[tuple[str, str, int], dict[str, Any]] = {}
    for event in recent_events:
        signal_time = int(event["signal_time"])
        if event["signal_family"] == "production_joint_bullish":
            unique_market_signals[(event["direction"], signal_time)] = event
        else:
            unique_asset_signals[(event["direction"], event["symbol"], signal_time)] = event

    genuine_joint_long_times: set[int] = set()
    reconfirmation_joint_long_times: set[int] = set()
    for event in unique_market_signals.values():
        drivers = json.loads(event["drivers"])
        genuine = bool(
            event["direction"] == "long"
            and drivers
            and all(bool(item.get("bearish_stack")) and bool(item.get("above_both")) for item in drivers.values())
        )
        target = genuine_joint_long_times if genuine else reconfirmation_joint_long_times
        target.add(int(event["signal_time"]))

    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for leg in legs:
        by_symbol[leg["symbol"]].append(leg)
    aggregates: dict[str, Any] = {}
    for symbol, rows in sorted(by_symbol.items()):
        long_returns = [float(row["hypothetical_long_return_pct"]) for row in rows]
        short_returns = [float(row["hypothetical_short_return_pct"]) for row in rows]
        completed = [row for row in rows if row["segment_end_reason"] == "next_daily_signal"]
        aggregates[symbol] = {
            "segments": len(rows),
            "completed_segments": len(completed),
            "partial_segments": len(rows) - len(completed),
            "total_4h_bars": sum(int(row["four_h_bars"]) for row in rows),
            "up_4h_bars": sum(int(row["up_4h_bars"]) for row in rows),
            "down_4h_bars": sum(int(row["down_4h_bars"]) for row in rows),
            "long_profitable_segments": sum(value > 0 for value in long_returns),
            "short_profitable_segments": sum(value > 0 for value in short_returns),
            "additive_long_return_pct": round_value(sum(long_returns)),
            "additive_short_return_pct": round_value(sum(short_returns)),
            "compounded_long_return_pct": round_value((__import__("math").prod(1.0 + value / 100.0 for value in long_returns) - 1.0) * 100.0),
            "compounded_short_return_pct": round_value((__import__("math").prod(1.0 + value / 100.0 for value in short_returns) - 1.0) * 100.0),
        }

    result = {
        "cutoff_utc": args.cutoff,
        "latest_closed_4h": {
            symbol: utc_iso(int(rows[-1]["close_time"])) for symbol, rows in klines.items()
        },
        "signal_counts": {
            "market_level_long_signals": sum(key[0] == "long" for key in unique_market_signals),
            "market_level_short_signals": sum(key[0] == "short" for key in unique_market_signals),
            "genuine_structural_long_flips": len(genuine_joint_long_times),
            "bullish_reclaim_reconfirmations": len(reconfirmation_joint_long_times),
            "asset_level_long_signals": sum(key[0] == "long" for key in unique_asset_signals),
            "asset_level_short_signals": sum(key[0] == "short" for key in unique_asset_signals),
            "coin_level_long_applications": sum(event["direction"] == "long" for event in recent_events),
            "coin_level_short_applications": sum(event["direction"] == "short" for event in recent_events),
        },
        "market_signal_times": sorted({event["signal_time_iso"] for event in unique_market_signals.values()}),
        "aggregates": aggregates,
        "legs": legs,
    }
    output_json = base_dir / "recent_reversal_legs_20260701.json"
    output_csv = base_dir / "recent_reversal_legs_20260701.csv"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_csv, legs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
