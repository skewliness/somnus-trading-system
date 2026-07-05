#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_v6_bb_sensitivity import fetch_klines, pct_diff, rolling_band, to_kline_set  # noqa: E402
from scripts.run_v6_execution import load_local_env  # noqa: E402
from trading_system.bb_structure import SHARED_BB_PERIODS  # noqa: E402
from trading_system.persistence import append_text_line, atomic_write_json, load_json  # noqa: E402
from trading_system.v6_execution import V6ExecutionConfig  # noqa: E402


INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}
REVIEW_TIMEFRAMES = ("15m", "4h")
REVIEW_BANDS = ("mid", "lower", "upper")
SYMBOL_CODE = {"BTCUSDT": 1.0, "ETHUSDT": 2.0, "BNBUSDT": 3.0, "HYPEUSDT": 4.0, "ZECUSDT": 5.0}
TIMEFRAME_CODE = {"15m": 0.15, "4h": 0.40}
BAND_CODE = {"lower": -1.0, "mid": 0.0, "upper": 1.0}
SIDE_CODE = {"short": -1.0, "long": 1.0}
VISIBLE_MA_PERIODS = (7, 20, 60, 96, 105, 120)
VISIBLE_VOLUME_MA_PERIODS = (5, 10)


def parse_iso_ms(value: str | None) -> int:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return 0
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def utc_iso(ms: int | None = None) -> str:
    if ms is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc).isoformat()


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def review_key(item: dict[str, Any]) -> str:
    return "|".join([
        str(item.get("symbol") or ""),
        str(item.get("side") or ""),
        str(item.get("opened_at") or ""),
        str(item.get("closed_at") or ""),
        str(item.get("target_name") or ""),
    ])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def trade_profit_pct(side: str, entry: float, exit_price: float) -> float:
    if entry <= 0 or exit_price <= 0:
        return 0.0
    if side == "long":
        return pct_diff(exit_price, entry)
    return pct_diff(entry, exit_price)


def adverse_before_touch(side: str, data: Any, entry_idx: int, touch_idx: int, entry: float) -> float:
    if touch_idx <= entry_idx or entry <= 0:
        return 0.0
    if side == "long":
        low = float(data.low[entry_idx + 1 : touch_idx + 1].min())
        return pct_diff(low, entry)
    high = float(data.high[entry_idx + 1 : touch_idx + 1].max())
    return pct_diff(entry, high)


def realized_weighted_profit(review: dict[str, Any]) -> dict[str, Any]:
    side = str(review.get("side") or "").lower()
    entry = finite_float(review.get("entry_price"))
    stages = [stage for stage in review.get("take_profit_stages") or [] if isinstance(stage, dict)]
    target_names = {name for name in str(review.get("target_name") or "").split(",") if name}
    total_fraction = 0.0
    weighted = 0.0
    legs: list[dict[str, Any]] = []
    for stage in stages:
        status = str(stage.get("status") or "")
        name = str(stage.get("name") or "")
        if status != "executed" and name not in target_names:
            continue
        fraction = finite_float(stage.get("fraction_of_initial"))
        if fraction <= 0:
            continue
        price = finite_float(stage.get("avg_price")) or finite_float(stage.get("trigger_price")) or finite_float(stage.get("price"))
        profit = trade_profit_pct(side, entry, price)
        total_fraction += fraction
        weighted += profit * fraction
        legs.append({"name": name, "fraction": round(fraction, 6), "price": price, "profit_pct": round(profit, 6)})
    if total_fraction < 0.999:
        close_price = closing_price(review)
        fraction = max(0.0, 1.0 - total_fraction)
        profit = trade_profit_pct(side, entry, close_price)
        weighted += profit * fraction
        total_fraction += fraction
        legs.append({"name": "final_close", "fraction": round(fraction, 6), "price": close_price, "profit_pct": round(profit, 6)})
    return {
        "weighted_profit_pct": round(weighted, 6),
        "total_fraction": round(total_fraction, 6),
        "legs": legs,
    }


def closing_price(review: dict[str, Any]) -> float:
    order = review.get("closing_order") if isinstance(review.get("closing_order"), dict) else {}
    response = order.get("response") if isinstance(order.get("response"), dict) else {}
    avg = finite_float(response.get("avgPrice"))
    if avg > 0:
        return avg
    return finite_float(review.get("close_price"))


def candidate_is_favorable(side: str, band: str, level: float, entry: float) -> bool:
    if level <= 0 or entry <= 0:
        return False
    if side == "short":
        return band in {"mid", "lower"} and level < entry
    return band in {"mid", "upper"} and level > entry


def rolling_sma(values: np.ndarray, period: int) -> np.ndarray:
    n = values.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def entry_indicator_features(data: Any, entry_idx: int, entry: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for period in VISIBLE_MA_PERIODS:
        ma = rolling_sma(data.close, period)
        value = finite_float(ma[entry_idx])
        out[f"ma{period}"] = {
            "value": round(value, 8),
            "distance_pct": round(pct_diff(entry, value), 6) if value > 0 else 0.0,
            "ready": value > 0,
        }
    for period in VISIBLE_VOLUME_MA_PERIODS:
        vol_ma = rolling_sma(data.volume, period)
        value = finite_float(vol_ma[entry_idx])
        current = float(data.volume[entry_idx]) if getattr(data, "volume", np.array([])).size > entry_idx else 0.0
        out[f"volume_vs_ma{period}"] = round(current / value - 1.0, 6) if current > 0 and value > 0 else 0.0
    return out


def find_touch_index(side: str, band: str, data: Any, start_idx: int, end_idx: int, level: float) -> int | None:
    if side == "short":
        values = data.low[start_idx + 1 : end_idx + 1]
        hits = np.where(values <= level)[0]
    else:
        values = data.high[start_idx + 1 : end_idx + 1]
        hits = np.where(values >= level)[0]
    if hits.size == 0:
        return None
    return int(start_idx + 1 + hits[0])


def evaluate_timeframe(review: dict[str, Any], timeframe: str, rows: list[list], review_ms: int) -> dict[str, Any]:
    side = str(review.get("side") or "").lower()
    entry = finite_float(review.get("entry_price"))
    entry_ms = parse_iso_ms(str(review.get("opened_at") or ""))
    close_ms = parse_iso_ms(str(review.get("closed_at") or ""))
    data = to_kline_set(rows)
    if data.close.size == 0:
        return {"timeframe": timeframe, "status": "no_data", "candidates": []}
    entry_idx = int(np.searchsorted(data.open_time, entry_ms, side="right") - 1)
    end_idx = int(np.searchsorted(data.open_time, review_ms, side="right") - 1)
    if entry_idx < 0 or end_idx <= entry_idx:
        return {"timeframe": timeframe, "status": "insufficient_window", "candidates": []}
    entry_indicators = entry_indicator_features(data, entry_idx, entry)
    candidates: list[dict[str, Any]] = []
    for period in SHARED_BB_PERIODS:
        upper, mid, lower = rolling_band(data.close, period)
        for band, values in {"upper": upper, "mid": mid, "lower": lower}.items():
            level = finite_float(values[entry_idx])
            if not candidate_is_favorable(side, band, level, entry):
                continue
            touch_idx = find_touch_index(side, band, data, entry_idx, end_idx, level)
            touched = touch_idx is not None
            profit = trade_profit_pct(side, entry, level)
            row = {
                "name": f"bb{period}_{band}_{timeframe}",
                "timeframe": timeframe,
                "period": period,
                "band": band,
                "level_price": round(level, 8),
                "entry_distance_pct": round(abs(pct_diff(entry, level)), 6),
                "profit_pct_if_touched": round(profit, 6),
                "touched": touched,
            }
            if touched and touch_idx is not None:
                touch_ms = int(data.open_time[touch_idx])
                row.update({
                    "touched_at": utc_iso(touch_ms),
                    "touched_after_final_close": touch_ms > close_ms,
                    "hold_minutes": round((touch_ms - entry_ms) / 60000.0, 2),
                    "adverse_before_touch_pct": round(adverse_before_touch(side, data, entry_idx, touch_idx, entry), 6),
                })
            candidates.append(row)
    touched_rows = [row for row in candidates if row.get("touched")]
    touched_rows.sort(key=lambda row: (finite_float(row.get("profit_pct_if_touched")), -finite_float(row.get("hold_minutes"))), reverse=True)
    return {
        "timeframe": timeframe,
        "status": "ok",
        "candidate_count": len(candidates),
        "touched_count": len(touched_rows),
        "entry_indicators": entry_indicators,
        "best_touched": touched_rows[0] if touched_rows else None,
        "top_touched": touched_rows[:8],
    }


def build_review(review: dict[str, Any]) -> dict[str, Any]:
    symbol = str(review.get("symbol") or "").upper()
    opened_ms = parse_iso_ms(str(review.get("opened_at") or ""))
    review_ms = parse_iso_ms(str(review.get("review_due_at") or "")) or utc_now_ms()
    realized = realized_weighted_profit(review)
    timeframe_results: list[dict[str, Any]] = []
    for timeframe in REVIEW_TIMEFRAMES:
        lookback_ms = max(SHARED_BB_PERIODS) * INTERVAL_MS[timeframe]
        start_ms = max(0, opened_ms - lookback_ms)
        rows = fetch_klines(symbol, timeframe, start_ms, review_ms)
        timeframe_results.append(evaluate_timeframe(review, timeframe, rows, review_ms))
    best_rows = [item.get("best_touched") for item in timeframe_results if isinstance(item.get("best_touched"), dict)]
    best_rows.sort(key=lambda row: finite_float(row.get("profit_pct_if_touched")), reverse=True)
    best = best_rows[0] if best_rows else None
    actual_profit = finite_float(realized.get("weighted_profit_pct"))
    best_profit = finite_float(best.get("profit_pct_if_touched")) if isinstance(best, dict) else 0.0
    improvement = best_profit - actual_profit
    return {
        "event": "v6_post_trade_review",
        "reviewed_at": utc_iso(),
        "review_key": review_key(review),
        "symbol": symbol,
        "side": review.get("side"),
        "opened_at": review.get("opened_at"),
        "closed_at": review.get("closed_at"),
        "review_due_at": review.get("review_due_at"),
        "entry_price": review.get("entry_price"),
        "close_price": closing_price(review),
        "actual": realized,
        "best_candidate": best,
        "improvement_pct": round(improvement, 6),
        "better_tp_available": bool(best and improvement > 0.5),
        "hold_after_final_tp_better": bool(best and best.get("touched_after_final_close") and improvement > 0.5),
        "timeframes": timeframe_results,
    }


def build_training_sample(review: dict[str, Any]) -> dict[str, Any]:
    best = review.get("best_candidate") if isinstance(review.get("best_candidate"), dict) else {}
    symbol = str(review.get("symbol") or "").upper()
    side = str(review.get("side") or "").lower()
    best_timeframe = str(best.get("timeframe") or "")
    entry_indicators: dict[str, Any] = {}
    for item in review.get("timeframes") or []:
        if isinstance(item, dict) and str(item.get("timeframe") or "") == best_timeframe:
            entry_indicators = item.get("entry_indicators") if isinstance(item.get("entry_indicators"), dict) else {}
            break
    indicator_features: dict[str, float] = {}
    for period in VISIBLE_MA_PERIODS:
        item = entry_indicators.get(f"ma{period}") if isinstance(entry_indicators.get(f"ma{period}"), dict) else {}
        indicator_features[f"entry_ma{period}_distance_pct"] = finite_float(item.get("distance_pct"))
    for period in VISIBLE_VOLUME_MA_PERIODS:
        indicator_features[f"entry_volume_vs_ma{period}"] = finite_float(entry_indicators.get(f"volume_vs_ma{period}"))
    return {
        "sample_type": "v6_post_trade_tp_review",
        "review_key": review.get("review_key"),
        "symbol": symbol,
        "side": side,
        "opened_at": review.get("opened_at"),
        "closed_at": review.get("closed_at"),
        "label_better_tp_available": 1 if review.get("better_tp_available") else 0,
        "label_hold_after_final_tp_better": 1 if review.get("hold_after_final_tp_better") else 0,
        "target_best_tp_name": best.get("name", ""),
        "target_best_tp_profit_pct": finite_float(best.get("profit_pct_if_touched")),
        "actual_weighted_profit_pct": finite_float((review.get("actual") or {}).get("weighted_profit_pct")),
        "improvement_pct": finite_float(review.get("improvement_pct")),
        "features": {
            "symbol_code": SYMBOL_CODE.get(symbol, 0.0),
            "side_code": SIDE_CODE.get(side, 0.0),
            "entry_price": finite_float(review.get("entry_price")),
            "close_price": finite_float(review.get("close_price")),
            "actual_weighted_profit_pct": finite_float((review.get("actual") or {}).get("weighted_profit_pct")),
            "best_timeframe_code": TIMEFRAME_CODE.get(str(best.get("timeframe") or ""), 0.0),
            "best_period": finite_float(best.get("period")),
            "best_band_code": BAND_CODE.get(str(best.get("band") or ""), 0.0),
            "best_entry_distance_pct": finite_float(best.get("entry_distance_pct")),
            "best_hold_minutes": finite_float(best.get("hold_minutes")),
            "best_adverse_before_touch_pct": finite_float(best.get("adverse_before_touch_pct")),
            **indicator_features,
        },
    }


def process_due_reviews(config: V6ExecutionConfig, *, force: bool, limit: int) -> dict[str, Any]:
    queue_file = config.closed_trade_review_queue_file
    status_file = config.state_dir / "post_trade_review_status.json"
    reviews_file = config.state_dir / "post_trade_reviews.jsonl"
    status = load_json(status_file, default={})
    processed = set(status.get("processed_keys") or [])
    now_ms = utc_now_ms()
    queued = load_jsonl(queue_file)
    processed_count = 0
    skipped_not_due = 0
    errors: list[dict[str, Any]] = []
    for item in queued:
        key = review_key(item)
        if not key or key in processed:
            continue
        due_ms = parse_iso_ms(str(item.get("review_due_at") or ""))
        if not force and due_ms > now_ms:
            skipped_not_due += 1
            continue
        try:
            review = build_review(item)
            sample = build_training_sample(review)
            append_text_line(reviews_file, json.dumps(review, ensure_ascii=False))
            append_text_line(config.post_trade_training_samples_file, json.dumps(sample, ensure_ascii=False))
            processed.add(key)
            processed_count += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"review_key": key, "error": f"{type(exc).__name__}: {exc}"})
        if limit > 0 and processed_count >= limit:
            break
    status_payload = {
        "updated_at": utc_iso(),
        "processed_keys": sorted(processed)[-5000:],
        "last_result": {
            "queued": len(queued),
            "processed": processed_count,
            "skipped_not_due": skipped_not_due,
            "errors": errors[-20:],
        },
    }
    atomic_write_json(status_file, status_payload)
    return status_payload["last_result"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run due V6 post-trade take-profit reviews.")
    parser.add_argument("--force", action="store_true", help="Process pending queue items even if 12h delay has not elapsed.")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    load_local_env(ROOT / ".env.local")
    config = V6ExecutionConfig.from_env()
    config.ensure_dirs()
    result = process_due_reviews(config, force=args.force, limit=args.limit)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"queued={result['queued']} processed={result['processed']} skipped_not_due={result['skipped_not_due']} errors={len(result['errors'])}")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
