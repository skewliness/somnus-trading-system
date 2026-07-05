#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HORIZONS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
MEANINGFUL_THRESHOLDS = (0.5, 1.0, 2.0)
MA_PERIODS = (7, 20, 60, 96, 120, 210, 300)


def finite_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_dt_ms(value: Any) -> int | None:
    parsed = parse_dt(value)
    return int(parsed.timestamp() * 1000) if parsed else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def parse_kline_row(row: Any) -> dict[str, float] | None:
    if isinstance(row, Mapping):
        open_time = row.get("open_time") or row.get("openTime") or row.get("t") or row.get("openTimeMs")
        open_price = row.get("open") or row.get("o")
        high = row.get("high") or row.get("h")
        low = row.get("low") or row.get("l")
        close = row.get("close") or row.get("c")
        volume = row.get("volume") or row.get("v")
    elif isinstance(row, list) and len(row) >= 6:
        open_time, open_price, high, low, close, volume = row[:6]
    else:
        return None
    values = {
        "open_time": finite_float(open_time, None),
        "open": finite_float(open_price, None),
        "high": finite_float(high, None),
        "low": finite_float(low, None),
        "close": finite_float(close, None),
        "volume": finite_float(volume, None),
    }
    if any(value is None for value in values.values()):
        return None
    return values  # type: ignore[return-value]


def load_klines(kline_dir: Path, symbol: str) -> list[dict[str, float]]:
    path = kline_dir / f"{symbol}.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, Mapping):
        raw_rows = payload.get("data") or payload.get("klines") or payload.get("rows") or []
    else:
        raw_rows = payload
    if not isinstance(raw_rows, list):
        return []
    rows = [parsed for raw in raw_rows if (parsed := parse_kline_row(raw))]
    return sorted(rows, key=lambda item: item["open_time"])


def find_entry_bar(rows: list[dict[str, float]], opened_ms: int) -> int | None:
    if not rows:
        return None
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid]["open_time"] <= opened_ms:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    return idx if idx >= 0 else None


def pct_diff(value: float, base: float) -> float:
    return (value - base) / base * 100.0 if base > 0 else 0.0


def sma(values: list[float], end_idx: int, period: int) -> float | None:
    start = end_idx - period + 1
    if start < 0:
        return None
    window = values[start : end_idx + 1]
    if len(window) != period:
        return None
    return sum(window) / period


def stddev(values: list[float], end_idx: int, period: int) -> float | None:
    avg = sma(values, end_idx, period)
    if avg is None:
        return None
    start = end_idx - period + 1
    window = values[start : end_idx + 1]
    return math.sqrt(sum((value - avg) ** 2 for value in window) / period)


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def profit_factor(rows: list[Mapping[str, Any]]) -> float | None:
    profit = sum(max(0.0, finite_float(row.get("pnl_usdt"), 0.0) or 0.0) for row in rows)
    loss = abs(sum(min(0.0, finite_float(row.get("pnl_usdt"), 0.0) or 0.0) for row in rows))
    return profit / loss if loss > 0 else None


def indicator_snapshot(rows: list[dict[str, float]], entry_idx: int, side: str) -> dict[str, Any]:
    indicator_idx = max(0, entry_idx - 1)
    closes = [row["close"] for row in rows]
    volumes = [row["volume"] for row in rows]
    row = rows[indicator_idx]
    close = row["close"]
    open_price = row["open"]
    high = row["high"]
    low = row["low"]
    out: dict[str, Any] = {
        "indicator_idx": indicator_idx,
        "indicator_open_time": int(row["open_time"]),
        "indicator_close": close,
        "indicator_open": open_price,
        "indicator_high": high,
        "indicator_low": low,
    }
    ma: dict[int, float] = {}
    for period in MA_PERIODS:
        value = sma(closes, indicator_idx, period)
        if value is not None:
            ma[period] = value
            out[f"ma{period}"] = value
            out[f"close_vs_ma{period}_pct"] = pct_diff(close, value)
    ma20_now = ma.get(20)
    ma20_prev4 = sma(closes, indicator_idx - 4, 20) if indicator_idx >= 4 else None
    if ma20_now is not None and ma20_prev4:
        out["ma20_slope_4_pct"] = pct_diff(ma20_now, ma20_prev4)
    ma7 = ma.get(7)
    ma20 = ma.get(20)
    if ma7 is not None and ma20 is not None:
        out["ma7_vs_ma20_pct"] = pct_diff(ma7, ma20)
    vol5 = sma(volumes, indicator_idx, 5)
    vol10 = sma(volumes, indicator_idx, 10)
    if vol5:
        out["volume_vs_ma5"] = row["volume"] / vol5
    if vol10:
        out["volume_vs_ma10"] = row["volume"] / vol10
    bb_mid = sma(closes, indicator_idx, 20)
    bb_sd = stddev(closes, indicator_idx, 20)
    if bb_mid is not None and bb_sd is not None and bb_sd > 0:
        upper = bb_mid + 2.0 * bb_sd
        lower = bb_mid - 2.0 * bb_sd
        out["bb20_mid"] = bb_mid
        out["bb20_upper"] = upper
        out["bb20_lower"] = lower
        out["bb20_position"] = (close - lower) / (upper - lower) if upper > lower else 0.5
        out["bb20_width_pct"] = (upper - lower) / bb_mid * 100.0 if bb_mid > 0 else 0.0
    body = abs(close - open_price)
    out["candle_body_pct"] = body / open_price * 100.0 if open_price > 0 else 0.0
    out["upper_wick_pct"] = (high - max(open_price, close)) / open_price * 100.0 if open_price > 0 else 0.0
    out["lower_wick_pct"] = (min(open_price, close) - low) / open_price * 100.0 if open_price > 0 else 0.0
    out["indicator_tags"] = sorted(indicator_tags(out, side))
    return out


def indicator_tags(snapshot: Mapping[str, Any], side: str) -> set[str]:
    tags: set[str] = set()
    close = finite_float(snapshot.get("indicator_close"), None)
    open_price = finite_float(snapshot.get("indicator_open"), None)
    ma7 = finite_float(snapshot.get("ma7"), None)
    ma20 = finite_float(snapshot.get("ma20"), None)
    ma60 = finite_float(snapshot.get("ma60"), None)
    ma20_slope = finite_float(snapshot.get("ma20_slope_4_pct"), None)
    volume_vs_ma5 = finite_float(snapshot.get("volume_vs_ma5"), None)
    volume_vs_ma10 = finite_float(snapshot.get("volume_vs_ma10"), None)
    bb_pos = finite_float(snapshot.get("bb20_position"), None)
    upper_wick = finite_float(snapshot.get("upper_wick_pct"), 0.0) or 0.0
    lower_wick = finite_float(snapshot.get("lower_wick_pct"), 0.0) or 0.0
    body = finite_float(snapshot.get("candle_body_pct"), 0.0) or 0.0
    if close is None:
        return tags
    if ma20:
        dist20 = abs(pct_diff(close, ma20))
        if dist20 <= 2.0:
            tags.add("entry_close_near_ma20_le_2pct")
        if dist20 <= 5.0:
            tags.add("entry_close_near_ma20_le_5pct")
    if side == "long":
        if ma20 and close > ma20:
            tags.add("side_aligned_close_above_ma20")
        if ma7 and ma20 and ma7 > ma20:
            tags.add("side_aligned_ma7_vs_ma20")
        if ma20_slope is not None and ma20_slope > 0:
            tags.add("side_aligned_ma20_slope")
        if ma60 and close > ma60:
            tags.add("side_aligned_close_vs_ma60")
        if bb_pos is not None and bb_pos < 0.8:
            tags.add("bb20_not_near_upper_for_long")
        if bb_pos is not None and bb_pos >= 0.5:
            tags.add("bb20_upper_half_for_long")
        if open_price is not None and close > open_price:
            tags.add("entry_candle_green_for_long")
        if upper_wick <= max(body, 0.01):
            tags.add("no_large_upper_wick_for_long")
    elif side == "short":
        if ma20 and close < ma20:
            tags.add("side_aligned_close_below_ma20")
        if ma7 and ma20 and ma7 < ma20:
            tags.add("side_aligned_ma7_vs_ma20")
        if ma20_slope is not None and ma20_slope < 0:
            tags.add("side_aligned_ma20_slope")
        if ma60 and close < ma60:
            tags.add("side_aligned_close_vs_ma60")
        if bb_pos is not None and bb_pos > 0.2:
            tags.add("bb20_not_near_lower_for_short")
        if bb_pos is not None and bb_pos <= 0.5:
            tags.add("bb20_lower_half_for_short")
        if open_price is not None and close < open_price:
            tags.add("entry_candle_red_for_short")
        if lower_wick <= max(body, 0.01):
            tags.add("no_large_lower_wick_for_short")
    if volume_vs_ma5 is not None and volume_vs_ma5 >= 1.2:
        tags.add("volume_vs_ma5_ge_1_2")
    if volume_vs_ma10 is not None and volume_vs_ma10 >= 1.2:
        tags.add("volume_vs_ma10_ge_1_2")
    return tags


def build_path_rows(trades: list[dict[str, Any]], kline_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kline_cache: dict[str, list[dict[str, float]]] = {}
    path_rows: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for trade in trades:
        symbol = str(trade.get("symbol") or "")
        side = str(trade.get("side") or "")
        opened_ms = parse_dt_ms(trade.get("opened_at"))
        entry = finite_float(trade.get("entry_price"), None)
        if not symbol or opened_ms is None or entry is None or entry <= 0:
            missing.append({"record_id": trade.get("record_id"), "symbol": symbol, "reason": "missing_entry"})
            continue
        if symbol not in kline_cache:
            kline_cache[symbol] = load_klines(kline_dir, symbol)
        klines = kline_cache[symbol]
        entry_idx = find_entry_bar(klines, opened_ms)
        if entry_idx is None:
            missing.append({"record_id": trade.get("record_id"), "symbol": symbol, "reason": "missing_15m_klines"})
            continue
        snapshot = indicator_snapshot(klines, entry_idx, side)
        base = {key: trade.get(key) for key in [
            "record_id",
            "source",
            "symbol",
            "side",
            "opened_at",
            "closed_at",
            "entry_price",
            "close_price",
            "pnl_pct",
            "pnl_usdt",
            "horizon",
            "entry_timeframe",
            "selected_level",
            "selected_signal",
            "risk_labels",
        ]}
        base.update(
            {
                "entry_bar_idx": entry_idx,
                "entry_bar_open_time": int(klines[entry_idx]["open_time"]),
                **snapshot,
            }
        )
        enriched.append(base)
        for n in HORIZONS:
            future_idx = entry_idx + n
            if future_idx >= len(klines):
                continue
            future = klines[future_idx]
            raw_return = pct_diff(future["close"], entry)
            favorable_return = raw_return if side == "long" else -raw_return
            path_rows.append(
                {
                    **{key: base.get(key) for key in [
                        "record_id",
                        "source",
                        "symbol",
                        "side",
                        "opened_at",
                        "closed_at",
                        "entry_price",
                        "pnl_pct",
                        "pnl_usdt",
                        "horizon",
                        "entry_timeframe",
                        "selected_level",
                        "selected_signal",
                    ]},
                    "n_15m": n,
                    "future_open_time": int(future["open_time"]),
                    "future_close": future["close"],
                    "raw_return_pct": raw_return,
                    "favorable_return_pct": favorable_return,
                    "raw_result": "up" if raw_return > 0 else "down" if raw_return < 0 else "flat",
                    "side_result": "favorable" if favorable_return > 0 else "adverse" if favorable_return < 0 else "flat",
                    "favorable_ge_0_5pct": favorable_return >= 0.5,
                    "adverse_le_minus_0_5pct": favorable_return <= -0.5,
                    "indicator_tags": base.get("indicator_tags") or [],
                }
            )
    return path_rows, enriched + missing


def aggregate_path(rows: list[Mapping[str, Any]], group_keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(group_key) for group_key in group_keys) + (row.get("n_15m"),)
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    def _sort_key(item: tuple[tuple[Any, ...], list[Mapping[str, Any]]]) -> tuple[Any, ...]:
        key_values = item[0]
        prefix = tuple(str(value) for value in key_values[:-1])
        return prefix + (int(key_values[-1] or 0),)

    for key, items in sorted(grouped.items(), key=_sort_key):
        n = len(items)
        raw_up = sum(1 for item in items if item.get("raw_result") == "up")
        raw_down = sum(1 for item in items if item.get("raw_result") == "down")
        favorable = sum(1 for item in items if item.get("side_result") == "favorable")
        adverse = sum(1 for item in items if item.get("side_result") == "adverse")
        fav_returns = [finite_float(item.get("favorable_return_pct"), 0.0) or 0.0 for item in items]
        raw_returns = [finite_float(item.get("raw_return_pct"), 0.0) or 0.0 for item in items]
        row = {group_keys[i]: key[i] for i in range(len(group_keys))}
        row.update(
            {
                "n_15m": key[-1],
                "records_count": n,
                "raw_up_count": raw_up,
                "raw_down_count": raw_down,
                "raw_up_prob": raw_up / n if n else 0.0,
                "raw_down_prob": raw_down / n if n else 0.0,
                "side_favorable_count": favorable,
                "side_adverse_count": adverse,
                "side_favorable_prob": favorable / n if n else 0.0,
                "side_adverse_prob": adverse / n if n else 0.0,
                "favorable_ge_0_5pct_prob": sum(1 for value in fav_returns if value >= 0.5) / n if n else 0.0,
                "adverse_le_minus_0_5pct_prob": sum(1 for value in fav_returns if value <= -0.5) / n if n else 0.0,
                "avg_raw_return_pct": sum(raw_returns) / n if n else 0.0,
                "median_raw_return_pct": median(raw_returns),
                "avg_favorable_return_pct": sum(fav_returns) / n if n else 0.0,
                "median_favorable_return_pct": median(fav_returns),
            }
        )
        out.append(row)
    return out


def first_meaningful_rows(path_rows: list[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    by_record: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in path_rows:
        by_record[str(row.get("record_id"))].append(row)
    out: list[dict[str, Any]] = []
    for record_id, rows in by_record.items():
        ordered = sorted(rows, key=lambda row: int(row.get("n_15m") or 0))
        first = None
        for row in ordered:
            favorable = finite_float(row.get("favorable_return_pct"), 0.0) or 0.0
            if abs(favorable) >= threshold:
                first = row
                break
        if first:
            out.append(
                {
                    "record_id": record_id,
                    "symbol": first.get("symbol"),
                    "side": first.get("side"),
                    "threshold_pct": threshold,
                    "first_n_15m": first.get("n_15m"),
                    "first_result": "favorable" if (finite_float(first.get("favorable_return_pct"), 0.0) or 0.0) > 0 else "adverse",
                    "favorable_return_pct": first.get("favorable_return_pct"),
                    "selected_level": first.get("selected_level"),
                    "selected_signal": first.get("selected_signal"),
                    "horizon": first.get("horizon"),
                }
            )
    return out


def aggregate_first(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("threshold_pct"), row.get("first_n_15m"))].append(row)
    out: list[dict[str, Any]] = []
    for (threshold, first_n), items in sorted(grouped.items()):
        n = len(items)
        fav = sum(1 for item in items if item.get("first_result") == "favorable")
        adv = sum(1 for item in items if item.get("first_result") == "adverse")
        out.append(
            {
                "threshold_pct": threshold,
                "first_n_15m": first_n,
                "records_count": n,
                "first_favorable_count": fav,
                "first_adverse_count": adv,
                "first_favorable_prob": fav / n if n else 0.0,
                "first_adverse_prob": adv / n if n else 0.0,
            }
        )
    return out


def aggregate_indicator_scan(enriched: list[Mapping[str, Any]], path_rows: list[Mapping[str, Any]], n_for_path: int = 4) -> list[dict[str, Any]]:
    by_record_path = {str(row.get("record_id")): row for row in path_rows if int(row.get("n_15m") or 0) == n_for_path}
    tags: set[str] = set()
    for row in enriched:
        for tag in row.get("indicator_tags") or []:
            tags.add(str(tag))
    out: list[dict[str, Any]] = []
    base_rows = [row for row in enriched if "record_id" in row and finite_float(row.get("pnl_usdt"), None) is not None]
    for tag in sorted(tags):
        matched = [row for row in base_rows if tag in (row.get("indicator_tags") or [])]
        if not matched:
            continue
        winners = [row for row in matched if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) > 0]
        losers = [row for row in matched if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) < 0]
        path = [by_record_path[str(row.get("record_id"))] for row in matched if str(row.get("record_id")) in by_record_path]
        side_fav = sum(1 for row in path if row.get("side_result") == "favorable")
        side_adv = sum(1 for row in path if row.get("side_result") == "adverse")
        out.append(
            {
                "indicator_tag": tag,
                "records_count": len(matched),
                "winners_count": len(winners),
                "losers_count": len(losers),
                "win_rate": len(winners) / len(matched) if matched else 0.0,
                "gross_profit_usdt": sum(max(0.0, finite_float(row.get("pnl_usdt"), 0.0) or 0.0) for row in matched),
                "gross_loss_usdt": abs(sum(min(0.0, finite_float(row.get("pnl_usdt"), 0.0) or 0.0) for row in matched)),
                "net_pnl_usdt": sum(finite_float(row.get("pnl_usdt"), 0.0) or 0.0 for row in matched),
                "profit_factor": profit_factor(matched),
                f"n{n_for_path}_side_favorable_prob": side_fav / len(path) if path else "",
                f"n{n_for_path}_side_adverse_prob": side_adv / len(path) if path else "",
                f"n{n_for_path}_avg_favorable_return_pct": (
                    sum(finite_float(row.get("favorable_return_pct"), 0.0) or 0.0 for row in path) / len(path) if path else ""
                ),
            }
        )
    return sorted(out, key=lambda row: (finite_float(row.get("profit_factor"), 0.0) or 0.0, row.get("win_rate") or 0.0), reverse=True)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float):
        return round(value, 8)
    return value


def top_rows(rows: list[Mapping[str, Any]], sort_key: str, limit: int = 8) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: finite_float(row.get(sort_key), 0.0) or 0.0, reverse=True)[:limit]


def write_report(
    path: Path,
    loss_rows: list[Mapping[str, Any]],
    all_rows: list[Mapping[str, Any]],
    loss_path_summary: list[Mapping[str, Any]],
    first_summary: list[Mapping[str, Any]],
    indicator_summary: list[Mapping[str, Any]],
    grouped_summaries: dict[str, list[Mapping[str, Any]]],
    missing_rows: list[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    lines = [
        "# V5 Losing Order 15m Path Backtest",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "- Live execution changed: **No**.",
        "- This is a report-only path backtest using historical closed outcome records and cached 15m klines.",
        "- Main sample: losing closed outcome records.",
        "- Control sample: all closed outcome records, used only to judge whether an indicator is actually better or just overfit to losses.",
        "- Unit: closed outcome record. Partial closes and closed-queue rows are not merged into position-level trades.",
        "",
        "## Sample",
        "",
        f"- Losing records with 15m path data: {len(loss_rows)}",
        f"- All closed records with 15m path data: {len(all_rows)}",
        f"- Missing/insufficient records: {len(missing_rows)}",
        f"- Output dir: `{output_dir}`",
        "",
        "## Loss-Only 15m Path",
        "",
        "| N 15m candles after entry | Records | Raw up | Raw down | Side favorable | Side adverse | Avg side return | Median side return | P(adverse <= -0.5%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in loss_path_summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["n_15m"]),
                    str(row["records_count"]),
                    f"{row['raw_up_prob']:.2%}",
                    f"{row['raw_down_prob']:.2%}",
                    f"{row['side_favorable_prob']:.2%}",
                    f"{row['side_adverse_prob']:.2%}",
                    f"{row['avg_favorable_return_pct']:.3f}%",
                    f"{row['median_favorable_return_pct']:.3f}%",
                    f"{row['adverse_le_minus_0_5pct_prob']:.2%}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## First Meaningful Move On Losing Records",
            "",
            "A meaningful move is the first observed side-normalized move whose absolute size reaches the threshold.",
            "",
            "| Threshold | First N 15m | Records | First favorable | First adverse |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in first_summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['threshold_pct']}%",
                    str(row["first_n_15m"]),
                    str(row["records_count"]),
                    f"{row['first_favorable_prob']:.2%}",
                    f"{row['first_adverse_prob']:.2%}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Weakest Original Entry Groups On Losing Records", ""])
    for group_key, title in [
        ("selected_signal", "By Selected Signal"),
        ("selected_level", "By Selected Level"),
        ("horizon", "By Horizon"),
        ("entry_timeframe", "By Entry Timeframe"),
    ]:
        lines.extend(
            [
                f"### {title}",
                "",
                "| Group | N 15m | Records | Side favorable | Side adverse | Avg side return |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        group_rows = [row for row in grouped_summaries[group_key] if int(row.get("records_count") or 0) >= 3 and int(row.get("n_15m") or 0) in {2, 4, 8}]
        group_rows = sorted(group_rows, key=lambda row: (finite_float(row.get("side_adverse_prob"), 0.0) or 0.0, -(finite_float(row.get("avg_favorable_return_pct"), 0.0) or 0.0)), reverse=True)[:10]
        for row in group_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row.get(group_key)}`",
                        str(row["n_15m"]),
                        str(row["records_count"]),
                        f"{row['side_favorable_prob']:.2%}",
                        f"{row['side_adverse_prob']:.2%}",
                        f"{row['avg_favorable_return_pct']:.3f}%",
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Entry Indicator Control Scan",
            "",
            "These rows use all closed outcomes. They are better candidates only if sample size is acceptable, win rate/profit factor are better, and they do not simply explain losses after the fact.",
            "",
            "| Indicator tag | Records | Win rate | Profit factor | Net PnL | N4 side favorable | N4 avg side return |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    candidates = [row for row in indicator_summary if int(row.get("records_count") or 0) >= 20]
    for row in top_rows(candidates, "profit_factor", 12):
        pf = row.get("profit_factor")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['indicator_tag']}`",
                    str(row["records_count"]),
                    f"{row['win_rate']:.2%}",
                    "" if pf in ("", None) else f"{float(pf):.3f}",
                    f"{row['net_pnl_usdt']:.2f}",
                    "" if row.get("n4_side_favorable_prob") == "" else f"{row['n4_side_favorable_prob']:.2%}",
                    "" if row.get("n4_avg_favorable_return_pct") == "" else f"{row['n4_avg_favorable_return_pct']:.3f}%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The loss-only path table answers how quickly bad entries become adverse after entry.",
            "- If adverse probability is already high by N=2 or N=4, the original entry indicator needs either stronger confirmation or earlier rejection.",
            "- Replacement indicators must be judged on the all-outcome control scan, not only on losing orders.",
            "- Any candidate with small sample size should remain a research tag, not an execution rule.",
            "",
            "## Output Files",
            "",
            "- `loss_order_15m_paths.jsonl`",
            "- `loss_order_path_summary.csv`",
            "- `loss_order_first_meaningful_move.csv`",
            "- `loss_order_by_selected_signal.csv`",
            "- `loss_order_by_selected_level.csv`",
            "- `loss_order_by_horizon.csv`",
            "- `loss_order_by_entry_timeframe.csv`",
            "- `all_outcome_indicator_scan.csv`",
            "- `enriched_closed_outcomes.jsonl`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    trades = load_jsonl(Path(args.trades))
    kline_dir = Path(args.kline_dir)
    output_dir = Path(args.out)
    docs_out = Path(args.docs_out)
    all_path_rows, all_enriched_or_missing = build_path_rows(trades, kline_dir)
    all_enriched = [row for row in all_enriched_or_missing if "record_id" in row and "indicator_tags" in row]
    missing_rows = [row for row in all_enriched_or_missing if "reason" in row and "record_id" in row]
    loss_records = {str(row.get("record_id")) for row in trades if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) < 0}
    loss_path_rows = [row for row in all_path_rows if str(row.get("record_id")) in loss_records]
    loss_enriched = [row for row in all_enriched if str(row.get("record_id")) in loss_records]

    loss_summary = aggregate_path(loss_path_rows)
    first_rows = []
    for threshold in MEANINGFUL_THRESHOLDS:
        first_rows.extend(first_meaningful_rows(loss_path_rows, threshold))
    first_summary = aggregate_first(first_rows)
    grouped_summaries = {
        "selected_signal": aggregate_path(loss_path_rows, ("selected_signal",)),
        "selected_level": aggregate_path(loss_path_rows, ("selected_level",)),
        "horizon": aggregate_path(loss_path_rows, ("horizon",)),
        "entry_timeframe": aggregate_path(loss_path_rows, ("entry_timeframe",)),
    }
    indicator_summary = aggregate_indicator_scan(all_enriched, all_path_rows, n_for_path=4)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "loss_order_15m_paths.jsonl", loss_path_rows)
    write_jsonl(output_dir / "enriched_closed_outcomes.jsonl", all_enriched)
    write_csv(output_dir / "loss_order_path_summary.csv", loss_summary)
    write_csv(output_dir / "loss_order_first_meaningful_move.csv", first_summary)
    for group_key, rows in grouped_summaries.items():
        write_csv(output_dir / f"loss_order_by_{group_key}.csv", rows)
    write_csv(output_dir / "all_outcome_indicator_scan.csv", indicator_summary)
    write_csv(output_dir / "missing_path_data.csv", missing_rows)
    write_report(docs_out, loss_enriched, all_enriched, loss_summary, first_summary, indicator_summary, grouped_summaries, missing_rows, output_dir)

    return {
        "closed_records": len(trades),
        "closed_records_with_15m_path": len(all_enriched),
        "loss_records": len(loss_records),
        "loss_records_with_15m_path": len(loss_enriched),
        "path_rows": len(all_path_rows),
        "missing_records": len(missing_rows),
        "output_dir": str(output_dir),
        "report": str(docs_out),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest losing V5 orders by 15m post-entry path.")
    parser.add_argument("--trades", default="data/trading_system/v5_loss_prevention/risk_labeled_closed_trades.jsonl")
    parser.add_argument("--kline-dir", default="data/trading_system/cache/backtest_klines/15m")
    parser.add_argument("--out", default="data/trading_system/v5_loss_prevention_15m_paths")
    parser.add_argument("--docs-out", default="docs/V5_LOSS_ORDER_15M_PATH_BACKTEST_REPORT.md")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
