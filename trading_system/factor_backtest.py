from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from trading_system.binance import BinanceFuturesClient
from trading_system.config import StrategyConfig
from trading_system.self_train_predictor import VALID_USDT_SYMBOL_RE


RESERVED_NON_FACTOR_KEYS = {
    "symbol",
    "timestamp",
    "entry_ts",
    "exit_ts",
    "side",
    "is_win",
    "pnl_pct",
    "pnl_usdt",
    "entry_price",
    "exit_price",
    "hf_forward_return_pct",
    "reason",
    "reason_family",
    "source",
    "status",
}

NOISE_REASON_FAMILIES = {"stop_loss_hit", "score_faded", "oi_divergence_signal_fade", "market_snapshot_missing_timeout"}

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _load_json_or_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("data") or payload.get("items")
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]
            return [payload]
    return _load_jsonl(path)


def _f(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_win(row: dict) -> bool:
    return bool(row.get("is_win")) or _f(row.get("pnl_pct")) > 0


def _bucket(row: dict, factor: str) -> str:
    v = _f(row.get(factor))
    if factor in {"price_change_pct_5m", "price_change_pct_24h", "oi_change_pct_4h", "hf_momentum_pct", "hf_forward_return_pct"}:
        if v < 0:
            return "negative"
        if v < 3:
            return "flat"
        if v < 12:
            return "warm"
        return "hot"
    if factor in {"taker_buy_sell_ratio", "hf_volume_z"}:
        if v < 0.98:
            return "sell_dominant"
        if v < 1.03:
            return "neutral"
        if v < 1.12:
            return "buying"
        return "aggressive_buying"
    if factor == "funding_rate":
        if v < -0.001:
            return "negative_extreme"
        if v < 0:
            return "negative"
        if v < 0.001:
            return "normal_positive"
        return "positive_extreme"
    if factor == "whale_long_short_ratio":
        if v <= 0:
            return "missing"
        if v < 0.45:
            return "short_heavy"
        if v < 0.60:
            return "balanced"
        if v < 0.75:
            return "long_heavy"
        return "crowded_long"
    if factor in {"sentiment_mentions", "whale_total_count"}:
        if v < 5:
            return "thin"
        if v < 20:
            return "moderate"
        if v < 80:
            return "active"
        return "crowded"
    return "unknown"


def _summarize(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"count": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0, "loss_rate": 0.0, "stop_loss_rate": 0.0, "fixed_horizon_loss_rate": 0.0}
    pnls = [_f(row.get("pnl_pct")) for row in rows]
    losses = [1 for row in rows if not _is_win(row)]
    stop_losses = [1 for row in rows if str(row.get("reason_family") or "").lower() == "stop_loss_hit"]
    return {
        "count": len(rows),
        "win_rate": round(sum(1 for row in rows if _is_win(row)) / len(rows), 4),
        "avg_pnl_pct": round(mean(pnls), 6),
        "loss_rate": round(len(losses) / len(rows), 4),
        "stop_loss_rate": round(len(stop_losses) / len(rows), 4),
        "fixed_horizon_loss_rate": round(sum(1 for row in rows if str(row.get("reason_family") or "").lower() == "fixed_horizon_loss") / len(rows), 4),
    }


def _write_payload(output_path: Path | None, payload: dict) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def _write_jsonl(output_path: Path | None, rows: list[dict]) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _recommendation_lookup(factor_report: dict) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    boost: set[tuple[str, str]] = set()
    penalize: set[tuple[str, str]] = set()
    for item in factor_report.get("recommendations") or []:
        factor = str(item.get("factor") or "")
        bucket = str(item.get("bucket") or "")
        if not factor or not bucket:
            continue
        if item.get("action") == "boost":
            boost.add((factor, bucket))
        elif item.get("action") == "penalize":
            penalize.add((factor, bucket))
    return boost, penalize


def distill_factor_training_samples(
    closed_trade_dataset_path: Path,
    factor_report: dict | None = None,
    output_path: Path | None = None,
    *,
    min_rows_for_model: int = 8,
) -> dict:
    """Summarize factor output, remove obvious noise, and emit clean samples for main training."""
    rows = [row for row in _load_jsonl(closed_trade_dataset_path) if VALID_USDT_SYMBOL_RE.fullmatch(str(row.get("symbol") or ""))]
    if factor_report is None:
        factor_report = run_factor_backtest(closed_trade_dataset_path, min_samples=3)
    boost_lookup, penalize_lookup = _recommendation_lookup(factor_report or {})
    distilled: list[dict[str, Any]] = []
    reject_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        symbol = str(row.get("symbol") or "")
        reason_family = str(row.get("reason_family") or "").lower()
        pnl = _f(row.get("pnl_pct"))
        matched_boost = []
        matched_penalize = []
        for factor in _numeric_factor_names([row], min_samples=1):
            bucket = _bucket(row, factor)
            if (factor, bucket) in boost_lookup:
                matched_boost.append({"factor": factor, "bucket": bucket})
            if (factor, bucket) in penalize_lookup:
                matched_penalize.append({"factor": factor, "bucket": bucket})
        is_noise = False
        if matched_penalize and pnl <= 0:
            is_noise = True
            reject_reasons["penalized_loser_bucket"] += 1
        elif reason_family in NOISE_REASON_FAMILIES and not matched_boost:
            is_noise = True
            reject_reasons[f"noisy_reason:{reason_family}"] += 1
        if is_noise:
            continue
        sample = dict(row)
        sample["sample_key"] = str(row.get("sample_key") or f"factor_distilled|{row.get('timestamp') or row.get('entry_ts') or ''}|{symbol}|{row.get('reason') or reason_family}|{row.get('entry_price') or ''}|{row.get('exit_price') or ''}")
        sample["sample_type"] = "factor_distilled_closed_trade"
        sample["factor_distilled"] = True
        sample["factor_matched_boosts"] = matched_boost
        sample["factor_matched_penalties"] = matched_penalize
        sample["noise_filter_version"] = "factor_distill_v1"
        distilled.append(sample)
    payload = {
        "status": "ok" if distilled else "missing",
        "source_rows": len(rows),
        "distilled_rows": len(distilled),
        "rejected_rows": len(rows) - len(distilled),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "model_training_ready": len(distilled) >= min_rows_for_model,
        "output_file": str(output_path) if output_path else "",
    }
    _write_jsonl(output_path, distilled)
    return payload


def _numeric_factor_names(rows: list[dict], *, min_samples: int) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if key in RESERVED_NON_FACTOR_KEYS:
                continue
            try:
                float(value)
            except Exception:
                continue
            counts[key] += 1
    return sorted(key for key, count in counts.items() if count >= min_samples)


def _factor_edge(rows: list[dict], factor: str) -> dict[str, Any] | None:
    values = [(row, _f(row.get(factor))) for row in rows if row.get(factor) is not None]
    if len(values) < 2:
        return None
    ordered = sorted(values, key=lambda item: item[1])
    split = max(1, len(ordered) // 3)
    low_rows = [row for row, _ in ordered[:split]]
    high_rows = [row for row, _ in ordered[-split:]]
    low = _summarize(low_rows)
    high = _summarize(high_rows)
    high_edge = (high["win_rate"] - low["win_rate"]) + (high["avg_pnl_pct"] - low["avg_pnl_pct"]) * 10
    low_edge = -high_edge
    direction = "higher_is_better" if high_edge >= 0 else "lower_is_better"
    edge_score = abs(high_edge)
    return {
        "factor": factor,
        "direction": direction,
        "edge_score": round(edge_score, 6),
        "low_bucket": low,
        "high_bucket": high,
        "coverage": round(len(values) / len(rows), 4),
    }


def discover_quant_factors(
    dataset_path: Path,
    output_path: Path | None = None,
    *,
    min_samples: int = 10,
    top_n: int = 20,
) -> dict:
    rows = [row for row in _load_jsonl(dataset_path) if VALID_USDT_SYMBOL_RE.fullmatch(str(row.get("symbol") or ""))]
    if not rows:
        payload = {"status": "missing", "rows": 0, "top_factors": []}
        _write_payload(output_path, payload)
        return payload
    factors = []
    for factor in _numeric_factor_names(rows, min_samples=min_samples):
        edge = _factor_edge(rows, factor)
        if edge is not None:
            factors.append(edge)
    factors.sort(key=lambda item: (float(item["edge_score"]), float(item["coverage"])), reverse=True)
    payload = {"status": "ok", "rows": len(rows), "top_factors": factors[:top_n]}
    _write_payload(output_path, payload)
    return payload


def run_factor_backtest(closed_trade_dataset_path: Path, output_path: Path | None = None, min_samples: int = 3) -> dict:
    """Backtest high-frequency/event factors from closed trade samples.

    This is intentionally model-free: bucket each pre-trade/event feature and compare
    realized win rate / avg PnL against the global baseline, then emit scoring knobs.
    """
    rows = [row for row in _load_jsonl(closed_trade_dataset_path) if VALID_USDT_SYMBOL_RE.fullmatch(str(row.get("symbol") or ""))]
    if not rows:
        payload = {"status": "missing", "closed_rows": 0, "baseline": {}, "factors": {}, "recommendations": []}
        _write_payload(output_path, payload)
        return payload

    baseline = _summarize(rows)
    factor_names = tuple(
        dict.fromkeys(
            [
                "price_change_pct_5m",
                "price_change_pct_24h",
                "oi_change_pct_4h",
                "taker_buy_sell_ratio",
                "funding_rate",
                "sentiment_mentions",
                "whale_long_short_ratio",
                "whale_total_count",
                *_numeric_factor_names(rows, min_samples=min_samples),
            ]
        )
    )
    factors: dict[str, list[dict[str, Any]]] = {}
    recommendations: list[dict[str, Any]] = []
    for factor in factor_names:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[_bucket(row, factor)].append(row)
        factor_rows: list[dict[str, Any]] = []
        for bucket_name, bucket_rows in sorted(buckets.items()):
            stats = _summarize(bucket_rows)
            lift = round(stats["win_rate"] - baseline["win_rate"], 4)
            avg_pnl_lift = round(stats["avg_pnl_pct"] - baseline["avg_pnl_pct"], 6)
            item = {"bucket": bucket_name, **stats, "win_rate_lift": lift, "avg_pnl_lift": avg_pnl_lift}
            factor_rows.append(item)
            if stats["count"] >= min_samples:
                if lift >= 0.12 and avg_pnl_lift >= 0:
                    recommendations.append({"factor": factor, "bucket": bucket_name, "action": "boost", "score_adjustment": 4.0, **stats})
                elif lift <= -0.12 or stats["stop_loss_rate"] >= max(0.55, baseline["stop_loss_rate"] + 0.18):
                    recommendations.append({"factor": factor, "bucket": bucket_name, "action": "penalize", "score_adjustment": -5.0, **stats})
        factors[factor] = factor_rows

    quant_discovery = discover_quant_factors(closed_trade_dataset_path, min_samples=min_samples, top_n=20)
    recommendations.sort(key=lambda item: (abs(float(item["score_adjustment"])), int(item["count"])), reverse=True)
    payload = {
        "status": "ok",
        "closed_rows": len(rows),
        "baseline": baseline,
        "factors": factors,
        "quant_factor_discovery": quant_discovery,
        "recommendations": recommendations[:24],
    }
    _write_payload(output_path, payload)
    return payload


def _market_rows_by_symbol(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _load_json_or_jsonl(path):
        symbol = str(row.get("symbol") or "")
        if VALID_USDT_SYMBOL_RE.fullmatch(symbol):
            grouped[symbol].append(row)
    for symbol in list(grouped):
        grouped[symbol].sort(key=lambda item: item.get("timestamp") or item.get("open_time") or item.get("ts") or 0)
    return grouped


def _symbols_from_replay_snapshot(path: Path, *, limit: int) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for row in _load_json_or_jsonl(path):
        symbol = str(row.get("symbol") or "")
        if VALID_USDT_SYMBOL_RE.fullmatch(symbol) and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
        if len(symbols) >= limit:
            break
    return symbols


def build_binance_hf_market_jsonl(
    symbols: list[str],
    output_path: Path,
    *,
    interval: str = "5m",
    limit: int = 120,
    config: StrategyConfig | None = None,
    client: BinanceFuturesClient | None = None,
) -> dict:
    """Fetch Binance Futures klines and write hf-backtest compatible JSONL rows."""
    cfg = config or StrategyConfig()
    binance = client or BinanceFuturesClient(cfg.base_url)
    rows: list[dict[str, Any]] = []
    per_symbol: dict[str, int] = {}
    errors: dict[str, str] = {}
    for symbol in symbols:
        if not VALID_USDT_SYMBOL_RE.fullmatch(symbol):
            continue
        try:
            klines = binance.get_json("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
            oi_hist = binance.get_json("/futures/data/openInterestHist", {"symbol": symbol, "period": interval, "limit": limit})
        except Exception as exc:
            per_symbol[symbol] = 0
            errors[symbol] = str(exc)[:240]
            continue
        oi_by_ts: dict[int, float] = {}
        prev_oi = 0.0
        for oi_item in oi_hist or []:
            try:
                oi_by_ts[int(oi_item.get("timestamp") or 0)] = _f(oi_item.get("sumOpenInterestValue") or oi_item.get("sumOpenInterest"))
            except Exception:
                continue
        count = 0
        for item in klines or []:
            try:
                ts = int(item[0])
                oi_value = oi_by_ts.get(ts, 0.0)
                oi_change_pct = ((oi_value / prev_oi - 1) * 100) if oi_value > 0 and prev_oi > 0 else 0.0
                row = {
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "quote_volume": float(item[7]) if len(item) > 7 else 0.0,
                    "trade_count": int(item[8]) if len(item) > 8 else 0,
                    "open_interest_usdt": round(oi_value, 6),
                    "oi_change_pct": round(oi_change_pct, 6),
                }
            except Exception:
                continue
            rows.append(row)
            if oi_value > 0:
                prev_oi = oi_value
            count += 1
        per_symbol[symbol] = count
    rows.sort(key=lambda item: (str(item["symbol"]), int(item["timestamp"])))
    _write_jsonl(output_path, rows)
    status = "ok" if rows else "missing"
    payload = {
        "status": status if not errors else f"{status}_partial",
        "symbols": len(per_symbol),
        "rows": len(rows),
        "per_symbol": per_symbol,
        "errors": errors,
        "output_file": str(output_path),
        "interval": interval,
        "limit": limit,
    }
    return payload


def build_binance_hf_market_jsonl_v2(
    symbols: list[str],
    output_path: Path,
    *,
    interval: str = "5m",
    limit: int = 120,
    config: StrategyConfig | None = None,
    client: BinanceFuturesClient | None = None,
) -> dict:
    """V2 market data builder: klines + OI + taker ratio + funding rate."""
    cfg = config or StrategyConfig()
    binance = client or BinanceFuturesClient(cfg.base_url)
    rows: list[dict[str, Any]] = []
    per_symbol: dict[str, int] = {}
    errors: dict[str, str] = {}

    funding_map: dict[str, float] = {}
    try:
        premium_data = binance.get_json("/fapi/v1/premiumIndex", {})
        for item in premium_data or []:
            sym = str(item.get("symbol") or "")
            fr = float(item.get("lastFundingRate") or 0)
            if sym:
                funding_map[sym] = fr
    except Exception:
        pass

    for symbol in symbols:
        if not VALID_USDT_SYMBOL_RE.fullmatch(symbol):
            continue
        try:
            klines = binance.get_json("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
            oi_hist = binance.get_json("/futures/data/openInterestHist", {"symbol": symbol, "period": interval, "limit": limit})
        except Exception as exc:
            per_symbol[symbol] = 0
            errors[symbol] = str(exc)[:240]
            continue

        taker_by_ts: dict[int, float] = {}
        try:
            taker_hist = binance.get_json("/futures/data/takerlongshortRatio", {"symbol": symbol, "period": interval, "limit": limit})
            for t_item in taker_hist or []:
                ts = int(t_item.get("timestamp") or 0)
                buy_vol = float(t_item.get("buyVol") or 0)
                sell_vol = float(t_item.get("sellVol") or 0)
                total = buy_vol + sell_vol
                if ts and total > 0:
                    taker_by_ts[ts] = round(buy_vol / total, 6)
        except Exception:
            pass

        oi_by_ts: dict[int, float] = {}
        for oi_item in oi_hist or []:
            try:
                oi_by_ts[int(oi_item.get("timestamp") or 0)] = _f(oi_item.get("sumOpenInterestValue") or oi_item.get("sumOpenInterest"))
            except Exception:
                continue

        prev_oi = 0.0
        funding = funding_map.get(symbol, 0.0)
        count = 0
        for item in klines or []:
            try:
                ts = int(item[0])
                oi_value = oi_by_ts.get(ts, 0.0)
                oi_change_pct = ((oi_value / prev_oi - 1) * 100) if oi_value > 0 and prev_oi > 0 else 0.0
                row = {
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "quote_volume": float(item[7]) if len(item) > 7 else 0.0,
                    "trade_count": int(item[8]) if len(item) > 8 else 0,
                    "open_interest_usdt": round(oi_value, 6),
                    "oi_change_pct": round(oi_change_pct, 6),
                    "taker_buy_ratio": taker_by_ts.get(ts, 0.0),
                    "funding_rate": funding,
                }
            except Exception:
                continue
            rows.append(row)
            if oi_value > 0:
                prev_oi = oi_value
            count += 1
        per_symbol[symbol] = count

    rows.sort(key=lambda item: (str(item["symbol"]), int(item["timestamp"])))
    _write_jsonl(output_path, rows)
    status = "ok" if rows else "missing"
    return {
        "status": status if not errors else f"{status}_partial",
        "symbols": len(per_symbol),
        "rows": len(rows),
        "per_symbol": per_symbol,
        "errors": errors,
        "output_file": str(output_path),
        "interval": interval,
        "limit": limit,
    }


def _zscore(value: float, sample: list[float]) -> float:
    if len(sample) < 2:
        return 0.0
    sd = pstdev(sample)
    if sd == 0:
        return 0.0
    return (value - mean(sample)) / sd


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return round(max_dd, 6)


def run_high_frequency_backtest(
    market_jsonl_path: Path,
    output_path: Path | None = None,
    *,
    lookback: int = 12,
    hold_bars: int = 3,
    min_momentum_pct: float = 1.2,
    volume_z_min: float = 1.0,
    min_oi_change_pct: float = 0.5,
    max_momentum_pct: float | None = None,
    volume_z_max: float | None = None,
    max_oi_change_pct: float | None = None,
) -> dict:
    """Build event-driven high-frequency replay trades from kline/tick JSONL rows.

    Input rows need symbol, timestamp/open_time, close, volume and OI change. The strategy
    enters long only when momentum, volume expansion and OI expansion align, then marks
    fixed-horizon PnL after hold_bars. Optional max thresholds model the observed edge:
    chase warm starts, but skip overheated momentum/volume/OI spikes.
    """
    events: list[dict[str, Any]] = []
    skipped_overheated: dict[str, int] = {"momentum": 0, "volume": 0, "oi": 0, "combo": 0}
    for symbol, rows in _market_rows_by_symbol(market_jsonl_path).items():
        closes = [_f(row.get("close") or row.get("mark_price")) for row in rows]
        volumes = [_f(row.get("volume") or row.get("quote_volume")) for row in rows]
        for idx in range(lookback, len(rows) - hold_bars):
            base = closes[idx - lookback]
            if base <= 0:
                continue
            momentum = (closes[idx] / base - 1) * 100
            vol_z = _zscore(volumes[idx], volumes[idx - lookback : idx])
            oi_change = _f(rows[idx].get("oi_change_pct") or rows[idx].get("oi_change_pct_5m") or rows[idx].get("oi_change_pct_4h"))
            if momentum < min_momentum_pct or vol_z < volume_z_min or oi_change < min_oi_change_pct:
                continue
            overheated_momentum = max_momentum_pct is not None and momentum > max_momentum_pct
            overheated_volume = volume_z_max is not None and vol_z > volume_z_max
            overheated_oi = max_oi_change_pct is not None and oi_change > max_oi_change_pct
            if overheated_momentum or overheated_volume or overheated_oi:
                skipped_overheated["combo"] += 1
                if overheated_momentum:
                    skipped_overheated["momentum"] += 1
                if overheated_volume:
                    skipped_overheated["volume"] += 1
                if overheated_oi:
                    skipped_overheated["oi"] += 1
                continue
            forward_return = (closes[idx + hold_bars] / closes[idx] - 1) * 100 if closes[idx] else 0.0
            smart = rows[idx].get("smart_money_context") if isinstance(rows[idx].get("smart_money_context"), dict) else {}
            events.append(
                {
                    "symbol": symbol,
                    "timestamp": rows[idx].get("timestamp") or rows[idx].get("open_time") or rows[idx].get("ts"),
                    "side": "long",
                    "entry_price": closes[idx],
                    "exit_price": closes[idx + hold_bars],
                    "pnl_pct": round(forward_return / 100, 8),
                    "is_win": forward_return > 0,
                    "reason_family": "hf_oi_momentum_followthrough" if forward_return > 0 else "fixed_horizon_loss",
                    "hf_momentum_pct": round(momentum, 6),
                    "hf_volume_z": round(vol_z, 6),
                    "hf_oi_change_pct": round(oi_change, 6),
                    "oi_signal": "oi_expansion",
                    "hf_forward_return_pct": round(forward_return, 6),
                    "funding_rate": _f(rows[idx].get("funding_rate")),
                    "smart_money_single_side_pct": _f(smart.get("single_side_pct")),
                    "smart_money_oi_composition": str(smart.get("oi_composition") or ""),
                    "smart_money_symbol_class": str(smart.get("symbol_class") or ""),
                    "smart_money_monitor_action": str(smart.get("monitor_action") or ""),
                    "whale_long_position_usdt": _f(smart.get("whale_long_position_usdt")),
                    "whale_short_position_usdt": _f(smart.get("whale_short_position_usdt")),
                    "smart_money_long_count": int(_f(smart.get("smart_money_long_count"))),
                    "smart_money_short_count": int(_f(smart.get("smart_money_short_count"))),
                }
            )
    if not events:
        payload = {
            "status": "missing",
            "events": 0,
            "baseline": {},
            "factor_rankings": [],
            "recommendations": [],
            "risk": {},
            "strategy": {
                "name": "hf_oi_momentum_warm_start_overheat_filter_v3",
                "entry": "long only when momentum, volume expansion, and OI expansion pass minimum thresholds without overheating",
                "exit": f"fixed_horizon_{hold_bars}_bars",
                "lookback": lookback,
                "hold_bars": hold_bars,
                "min_momentum_pct": min_momentum_pct,
                "volume_z_min": volume_z_min,
                "min_oi_change_pct": min_oi_change_pct,
                "max_momentum_pct": max_momentum_pct,
                "volume_z_max": volume_z_max,
                "max_oi_change_pct": max_oi_change_pct,
            },
            "filters": {"skipped_overheated": skipped_overheated},
        }
        _write_payload(output_path, payload)
        return payload
    baseline = _summarize(events)
    hft_allowed_factors = {"hf_momentum_pct", "hf_volume_z", "hf_oi_change_pct", "funding_rate"}
    factor_rankings = [
        item
        for item in (
            _factor_edge(events, factor)
            for factor in _numeric_factor_names(events, min_samples=max(1, min(2, len(events))))
            if factor in hft_allowed_factors
        )
        if item
    ]
    factor_rankings.sort(key=lambda item: float(item["edge_score"]), reverse=True)
    recommendations = [
        {
            "factor": item["factor"],
            "direction": item["direction"],
            "action": "boost" if item["direction"] == "higher_is_better" else "penalize",
            "score_adjustment": 3.0 if item["direction"] == "higher_is_better" else -3.0,
            "edge_score": item["edge_score"],
        }
        for item in factor_rankings[:12]
        if float(item["edge_score"]) > 0
    ]
    pnls = [_f(row.get("pnl_pct")) for row in events]
    payload = {
        "status": "ok",
        "events": len(events),
        "baseline": baseline,
        "strategy": {
            "name": "hf_oi_momentum_warm_start_overheat_filter_v3",
            "entry": "long only when momentum, volume expansion, and OI expansion pass minimum thresholds without overheating",
            "exit": f"fixed_horizon_{hold_bars}_bars",
            "lookback": lookback,
            "hold_bars": hold_bars,
            "min_momentum_pct": min_momentum_pct,
            "volume_z_min": volume_z_min,
            "min_oi_change_pct": min_oi_change_pct,
            "max_momentum_pct": max_momentum_pct,
            "volume_z_max": volume_z_max,
            "max_oi_change_pct": max_oi_change_pct,
        },
        "filters": {"skipped_overheated": skipped_overheated},
        "risk": {
            "max_drawdown_pct": _max_drawdown(pnls),
            "avg_trade_pnl_pct": baseline["avg_pnl_pct"],
            "sharpe_like": round((mean(pnls) / pstdev(pnls)) * sqrt(len(pnls)), 6) if len(pnls) > 1 and pstdev(pnls) else 0.0,
        },
        "factor_rankings": factor_rankings[:20],
        "recommendations": recommendations,
        "sample_events": events[:20],
    }
    _write_payload(output_path, payload)
    return payload


def run_hf_backtest_v2(
    market_jsonl_path: Path,
    output_path: Path | None = None,
    report_path: Path | None = None,
    *,
    lookback: int = 12,
    momentum_threshold_pct: float = 0.3,
    volume_z_threshold: float = 1.0,
    oi_change_threshold_pct: float = 0.8,
    funding_extreme_threshold: float = 0.0005,
    taker_imbalance_threshold: float = 0.55,
    label_horizons: tuple[int, ...] = (3, 6, 12),
    mfe_mae_window: int = 12,
) -> dict:
    """Wide-entry, multi-horizon labeled HF backtest for model training."""
    grouped = _market_rows_by_symbol(market_jsonl_path)
    if not grouped:
        payload = {"status": "missing", "version": "hf_backtest_v2", "total_entries": 0}
        _write_payload(report_path, payload)
        return payload

    max_forward = max(label_horizons) if label_horizons else mfe_mae_window
    max_forward = max(max_forward, mfe_mae_window)
    entries: list[dict[str, Any]] = []
    conflict_bar_count = 0

    for symbol, rows in grouped.items():
        if len(rows) < lookback + max_forward + 1:
            continue
        closes = [_f(r.get("close") or r.get("mark_price")) for r in rows]
        highs = [_f(r.get("high") or r.get("close")) for r in rows]
        lows = [_f(r.get("low") or r.get("close")) for r in rows]
        volumes = [_f(r.get("volume") or r.get("quote_volume")) for r in rows]
        trade_counts = [_f(r.get("trade_count")) for r in rows]

        for idx in range(lookback, len(rows) - max_forward):
            if closes[idx] <= 0:
                continue
            momentum = (closes[idx] / closes[idx - lookback] - 1) * 100 if closes[idx - lookback] > 0 else 0.0
            vol_window = volumes[idx - lookback:idx]
            vol_z = _zscore(volumes[idx], vol_window)
            tc_window = trade_counts[idx - lookback:idx]
            tc_z = _zscore(trade_counts[idx], tc_window)
            oi_change = _f(rows[idx].get("oi_change_pct") or rows[idx].get("oi_change_pct_5m") or rows[idx].get("oi_change_pct_4h"))
            funding = _f(rows[idx].get("funding_rate"))
            taker = _f(rows[idx].get("taker_buy_ratio"))

            lookback_highs = highs[idx - lookback:idx + 1]
            lookback_lows = lows[idx - lookback:idx + 1]
            hl_range = max(lookback_highs) - min(lookback_lows) if lookback_highs else 0
            price_pos = (closes[idx] - min(lookback_lows)) / hl_range if hl_range > 0 else 0.5

            signals_long: list[str] = []
            signals_short: list[str] = []

            if momentum >= momentum_threshold_pct:
                signals_long.append("momentum")
            if momentum <= -momentum_threshold_pct:
                signals_short.append("momentum")

            if vol_z >= volume_z_threshold:
                if momentum > 0:
                    signals_long.append("volume_spike")
                elif momentum < 0:
                    signals_short.append("volume_spike")

            if oi_change >= oi_change_threshold_pct:
                if momentum > 0:
                    signals_long.append("oi_surge")
                elif momentum < 0:
                    signals_short.append("oi_surge")

            if funding_extreme_threshold > 0 and funding != 0:
                if funding <= -funding_extreme_threshold:
                    signals_long.append("funding_extreme")
                elif funding >= funding_extreme_threshold:
                    signals_short.append("funding_extreme")

            if taker_imbalance_threshold > 0 and taker > 0:
                if taker >= taker_imbalance_threshold:
                    signals_long.append("taker_imbalance")
                elif taker <= (1 - taker_imbalance_threshold):
                    signals_short.append("taker_imbalance")

            candidates: list[tuple[str, list[str]]] = []
            if signals_long and signals_short:
                conflict_bar_count += 1
                if len(signals_long) > len(signals_short):
                    candidates.append(("long", signals_long))
                elif len(signals_short) > len(signals_long):
                    candidates.append(("short", signals_short))
            else:
                if signals_long:
                    candidates.append(("long", signals_long))
                if signals_short:
                    candidates.append(("short", signals_short))

            for side, signals in candidates:
                direction = 1.0 if side == "long" else -1.0
                entry_price = closes[idx]
                ts = rows[idx].get("timestamp") or rows[idx].get("open_time") or rows[idx].get("ts")

                labels: dict[str, Any] = {}
                for h in label_horizons:
                    if idx + h < len(closes) and closes[idx + h] > 0:
                        ret = (closes[idx + h] / entry_price - 1) * 100 * direction
                        labels[f"return_{h}bar"] = round(ret, 6)
                        labels[f"is_profitable_{h}bar"] = ret > 0
                    else:
                        labels[f"return_{h}bar"] = None
                        labels[f"is_profitable_{h}bar"] = None

                fwd_highs = highs[idx + 1:idx + 1 + mfe_mae_window]
                fwd_lows = lows[idx + 1:idx + 1 + mfe_mae_window]
                if fwd_highs and fwd_lows and entry_price > 0:
                    if side == "long":
                        mfe = (max(fwd_highs) / entry_price - 1) * 100
                        mae = (min(fwd_lows) / entry_price - 1) * 100
                    else:
                        mfe = (1 - min(fwd_lows) / entry_price) * 100
                        mae = (1 - max(fwd_highs) / entry_price) * 100
                else:
                    mfe = 0.0
                    mae = 0.0

                rr: float | None = None
                if abs(mae) >= 0.02:
                    rr = min(round(mfe / abs(mae), 4), 20.0)

                entry = {
                    "sample_key": f"hf_v2|{symbol}|{ts}|{side}",
                    "sample_type": "hf_entry_candidate",
                    "symbol": symbol,
                    "timestamp": ts,
                    "side": side,
                    "triggered_signals": signals,
                    "signal_count": len(signals),
                    "entry_price": round(entry_price, 8),
                    "hf_momentum_pct": round(momentum, 6),
                    "hf_volume_z": round(vol_z, 6),
                    "hf_oi_change_pct": round(oi_change, 6),
                    "hf_funding_snapshot": round(funding, 8),
                    "hf_funding_source": "current_snapshot",
                    "hf_taker_buy_ratio": round(taker, 6),
                    "hf_price_position": round(price_pos, 6),
                    "hf_trade_intensity": round(tc_z, 6),
                    "mfe_12bar": round(mfe, 6),
                    "mae_12bar": round(mae, 6),
                    "risk_reward_12bar": rr,
                    **labels,
                }
                entries.append(entry)

    if output_path:
        existing_keys: set[str] = set()
        existing_rows: list[dict[str, Any]] = []
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    key = row.get("sample_key", "")
                    if key and key not in existing_keys:
                        existing_keys.add(key)
                        existing_rows.append(row)
                except Exception:
                    pass
        new_count = 0
        for e in entries:
            key = e.get("sample_key", "")
            if key not in existing_keys:
                existing_keys.add(key)
                existing_rows.append(e)
                new_count += 1
        _write_jsonl(output_path, existing_rows)
        entries = existing_rows
    else:
        new_count = len(entries)

    report = _build_v2_report(entries, output_path, conflict_bar_count, new_count)
    _write_payload(report_path, report)
    return report


def _build_v2_report(entries: list[dict[str, Any]], output_path: Path | None, conflict_bar_count: int = 0, new_count: int = 0) -> dict[str, Any]:
    if not entries:
        return {"status": "missing", "version": "hf_backtest_v2", "total_entries": 0}

    long_entries = [e for e in entries if e["side"] == "long"]
    short_entries = [e for e in entries if e["side"] == "short"]
    symbols_with = len({e["symbol"] for e in entries})
    avg_sig = mean([e["signal_count"] for e in entries]) if entries else 0

    horizon_stats: dict[str, dict[str, Any]] = {}
    for h in (3, 6, 12):
        key = f"return_{h}bar"
        vals = [e[key] for e in entries if e.get(key) is not None]
        if vals:
            horizon_stats[f"{h}bar"] = {
                "win_rate": round(sum(1 for v in vals if v > 0) / len(vals), 4),
                "avg_return": round(mean(vals), 6),
                "count": len(vals),
            }

    mfes = [e["mfe_12bar"] for e in entries if e.get("mfe_12bar")]
    maes = [e["mae_12bar"] for e in entries if e.get("mae_12bar")]
    rrs = [e["risk_reward_12bar"] for e in entries if e.get("risk_reward_12bar") and e["risk_reward_12bar"] != 0]

    signal_breakdown: dict[str, dict[str, Any]] = {}
    for sig_name in ("momentum", "volume_spike", "oi_surge", "funding_extreme", "taker_imbalance"):
        matched = [e for e in entries if sig_name in e.get("triggered_signals", [])]
        if matched:
            r3 = [e["return_3bar"] for e in matched if e.get("return_3bar") is not None]
            signal_breakdown[sig_name] = {
                "count": len(matched),
                "avg_return_3bar": round(mean(r3), 6) if r3 else 0,
            }

    per_symbol: dict[str, int] = {}
    for e in entries:
        per_symbol[e["symbol"]] = per_symbol.get(e["symbol"], 0) + 1

    return {
        "status": "ok",
        "version": "hf_backtest_v2",
        "total_entries": len(entries),
        "long_entries": len(long_entries),
        "short_entries": len(short_entries),
        "symbols_with_entries": symbols_with,
        "avg_signal_count": round(avg_sig, 2),
        "horizon_stats": horizon_stats,
        "mfe_mae_stats": {
            "avg_mfe": round(mean(mfes), 6) if mfes else 0,
            "avg_mae": round(mean(maes), 6) if maes else 0,
            "avg_risk_reward": round(mean(rrs), 4) if rrs else 0,
        },
        "signal_breakdown": signal_breakdown,
        "conflict_bar_count": conflict_bar_count,
        "new_entries_this_run": new_count,
        "entries_per_symbol": per_symbol,
        "output_file": str(output_path) if output_path else None,
    }