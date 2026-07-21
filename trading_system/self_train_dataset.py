from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.pdf_strategy_labeler import label_trade as _pdf_label_trade
from trading_system.persistence import file_lock


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


TRUSTED_DISTILL_SOURCE_HINTS = (
    "distill",
    "gmgn",
    "wallet",
    "okx",
    "opentwitter",
    "opennews",
    "smart_money",
    "kol",
)


VALID_USDT_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,30}USDT$")
INVALID_SYMBOL_ALIASES = {
    "BUSDT",
    "QUSDT",
}


def is_valid_usdt_symbol(symbol: Any) -> bool:
    value = str(symbol or "").upper()
    if value in INVALID_SYMBOL_ALIASES:
        return False
    return bool(VALID_USDT_SYMBOL_RE.fullmatch(value))


def parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _direction_from_event(event: dict) -> str:
    raw = str(
        event.get("direction")
        or event.get("signal_direction")
        or event.get("stance")
        or event.get("side")
        or event.get("bias")
        or ""
    ).lower()
    if raw in {"long", "bull", "bullish", "buy", "same", "up", "positive"}:
        return "long"
    if raw in {"short", "bear", "bearish", "sell", "reverse", "down", "negative"}:
        return "short"
    return ""


def _is_trusted_distill_source(source: str) -> bool:
    lowered = source.lower()
    return any(hint in lowered for hint in TRUSTED_DISTILL_SOURCE_HINTS)


def _is_strong_distill_event(event: dict) -> bool:
    if str(event.get("strength") or "").lower() in {"strong", "high", "very_high"}:
        return True
    confidence = max(
        to_float(event.get("confidence"), -1.0),
        to_float(event.get("confidence_score"), -1.0),
        to_float(event.get("hotness_score"), -1.0),
        to_float(event.get("score"), -1.0),
    )
    return confidence >= 70.0


def _extract_first_float(text: str, key: str) -> float:
    match = re.search(rf"(?:^|[|,;\s]){re.escape(key)}=([-+]?\d+(?:\.\d+)?)", text)
    return to_float(match.group(1), 0.0) if match else 0.0


def _extract_whale_metrics(source: dict | None) -> dict:
    """Normalize Binance/OKX whale-overview style fields into model features.

    Supports flat fields, compact bridge notes, and the nested payload emitted by
    trading_system.whale_overview.build_whale_overview(). Missing data is neutral.
    """
    source = source or {}
    summary = source.get("summary") if isinstance(source.get("summary"), dict) else {}
    long_group = source.get("long_group") if isinstance(source.get("long_group"), dict) else {}
    short_group = source.get("short_group") if isinstance(source.get("short_group"), dict) else {}
    market = source.get("market") if isinstance(source.get("market"), dict) else {}
    okx = source.get("okx_signal_context") if isinstance(source.get("okx_signal_context"), dict) else {}
    smart = source.get("smart_money_context") if isinstance(source.get("smart_money_context"), dict) else {}
    notes = str(source.get("notes") or source.get("whale_notes") or "")

    long_count = to_float(source.get("whale_long_count"), to_float(smart.get("whale_long_count"), to_float(long_group.get("whale_count"), _extract_first_float(notes, "whale_long_count"))))
    short_count = to_float(source.get("whale_short_count"), to_float(smart.get("whale_short_count"), to_float(short_group.get("whale_count"), _extract_first_float(notes, "whale_short_count"))))
    total_count = to_float(source.get("whale_total_count"), to_float(summary.get("whale_count"), _extract_first_float(notes, "whale_total_count"))) or long_count + short_count
    long_notional = to_float(source.get("whale_long_notional_usdt"), to_float(smart.get("whale_long_position_usdt"), to_float(long_group.get("current_position_usdt"), _extract_first_float(notes, "whale_long_notional_usdt"))))
    short_notional = to_float(source.get("whale_short_notional_usdt"), to_float(smart.get("whale_short_position_usdt"), to_float(short_group.get("current_position_usdt"), _extract_first_float(notes, "whale_short_notional_usdt"))))
    total_notional = to_float(source.get("whale_total_notional_usdt"), to_float(summary.get("total_position_usdt"), _extract_first_float(notes, "whale_total_notional_usdt"))) or long_notional + short_notional
    long_short_ratio = to_float(source.get("whale_long_short_ratio"), to_float(summary.get("long_ratio_pct"), _extract_first_float(notes, "whale_long_short_ratio")))
    if not long_short_ratio:
        denominator = total_count or total_notional
        numerator = long_count if total_count else long_notional
        long_short_ratio = (numerator / denominator * 100.0) if denominator else 0.0
    whale_pressure_score = max(-100.0, min(100.0, (long_short_ratio - 50.0) * 2.0)) if long_short_ratio else 0.0
    sold_ratio = to_float(source.get("whale_okx_sold_ratio"), to_float(okx.get("avg_sold_ratio"), _extract_first_float(notes, "sold_ratio")))
    if sold_ratio:
        whale_pressure_score -= max(-30.0, min(30.0, (sold_ratio - 50.0) * 0.6))
    return {
        "whale_total_notional_usdt": round(total_notional, 4),
        "whale_long_notional_usdt": round(long_notional, 4),
        "whale_short_notional_usdt": round(short_notional, 4),
        "whale_total_count": int(total_count),
        "whale_long_count": int(long_count),
        "whale_short_count": int(short_count),
        "whale_long_short_ratio": round(long_short_ratio, 4),
        "whale_long_avg_entry_price": to_float(source.get("whale_long_avg_entry_price"), to_float(long_group.get("average_entry_price"), _extract_first_float(notes, "whale_long_avg_entry_price"))),
        "whale_short_avg_entry_price": to_float(source.get("whale_short_avg_entry_price"), to_float(short_group.get("average_entry_price"), _extract_first_float(notes, "whale_short_avg_entry_price"))),
        "whale_long_pnl_usdt": to_float(source.get("whale_long_pnl_usdt"), to_float(long_group.get("unrealized_pnl_usdt", long_group.get("estimated_pnl_usdt")), _extract_first_float(notes, "whale_long_pnl_usdt"))),
        "whale_short_pnl_usdt": to_float(source.get("whale_short_pnl_usdt"), to_float(short_group.get("unrealized_pnl_usdt", short_group.get("estimated_pnl_usdt")), _extract_first_float(notes, "whale_short_pnl_usdt"))),
        "whale_long_profit_ratio": to_float(source.get("whale_long_profit_ratio"), to_float(long_group.get("profit_ratio_pct"), _extract_first_float(notes, "whale_long_profit_ratio"))),
        "whale_short_profit_ratio": to_float(source.get("whale_short_profit_ratio"), to_float(short_group.get("profit_ratio_pct"), _extract_first_float(notes, "whale_short_profit_ratio"))),
        "whale_pressure_score": round(max(-100.0, min(100.0, whale_pressure_score)), 4),
        "whale_okx_sold_ratio": round(sold_ratio, 4),
        "whale_okx_hold_ratio": to_float(source.get("whale_okx_hold_ratio"), to_float(okx.get("avg_hold_ratio"))),
        "whale_okx_hotness_score": to_float(source.get("whale_okx_hotness_score"), to_float(okx.get("hotness_score"))),
        "whale_okx_mention_count": int(to_float(source.get("whale_okx_mention_count"), to_float(okx.get("mention_count")))),
        "whale_market_oi_change_pct_4h": to_float(source.get("whale_market_oi_change_pct_4h"), to_float(market.get("oi_change_pct_4h"))),
        "whale_market_taker_buy_sell_ratio": to_float(source.get("whale_market_taker_buy_sell_ratio"), to_float(market.get("taker_buy_sell_ratio"))),
        "smart_money_single_side_pct": to_float(smart.get("single_side_pct")),
        "smart_money_oi_composition": str(smart.get("oi_composition") or ""),
        "smart_money_symbol_class": str(smart.get("symbol_class") or ""),
        "smart_money_monitor_action": str(smart.get("monitor_action") or ""),
        "smart_money_long_count": int(to_float(smart.get("smart_money_long_count"))),
        "smart_money_short_count": int(to_float(smart.get("smart_money_short_count"))),
        "smart_money_long_profit_ratio": to_float(smart.get("smart_money_long_profit_ratio")),
        "smart_money_short_profit_ratio": to_float(smart.get("smart_money_short_profit_ratio")),
    }


def _metric_source_for_symbol(report: dict, symbol: str) -> dict:
    symbol = str(symbol or "").upper()
    for container_name in ("paper_rebalance", "whale_overview", "whale_overviews", "top_ranked_symbols", "candidate_basket", "hotness_priority", "oi_divergence_candidates"):
        container = report.get(container_name) or []
        if isinstance(container, dict) and container_name == "paper_rebalance":
            container = container.get("entries") or []
        if isinstance(container, dict):
            candidate = container.get(symbol) or container.get(symbol.replace("USDT", "")) or {}
            if isinstance(candidate, dict):
                return candidate
        elif isinstance(container, list):
            for item in container:
                if isinstance(item, dict) and str(item.get("symbol") or "").upper() in {symbol, symbol.replace("USDT", "")}:
                    return item
    return {}


def _macro_trend_side_channel(report: dict) -> dict[str, Any]:
    macro = report.get("macro_trend") if isinstance(report.get("macro_trend"), dict) else {}
    trend_value = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}.get
    return {
        "macro_btc_ma7_trend": trend_value(str(macro.get("btc_trend") or ""), 0.0),
        "macro_eth_ma7_trend": trend_value(str(macro.get("eth_trend") or ""), 0.0),
        "macro_btc_ma7_distance_pct": to_float(macro.get("btc_distance_pct")),
        "macro_eth_ma7_distance_pct": to_float(macro.get("eth_distance_pct")),
        "macro_btc_ma7_slope_pct_3d": to_float(macro.get("btc_ma7_slope_pct_3d")),
        "macro_eth_ma7_slope_pct_3d": to_float(macro.get("eth_ma7_slope_pct_3d")),
        "macro_direction": str(macro.get("macro_direction") or "mixed"),
        "macro_confidence": str(macro.get("macro_confidence") or "neutral"),
        "macro_source": str(macro.get("source") or ""),
    }


def _distill_events_from_report(report: dict, symbol: str) -> list[dict]:
    containers = [
        report.get("distill_sequence"),
        report.get("distill_signal_sequence"),
        report.get("distill_signals"),
        report.get("sentiment_signal_sequence"),
    ]
    events: list[dict] = []
    for container in containers:
        if isinstance(container, dict):
            raw_events = container.get(symbol) or container.get(symbol.upper()) or container.get("signals") or []
        else:
            raw_events = container or []
        if not isinstance(raw_events, list):
            continue
        for event in raw_events:
            if not isinstance(event, dict):
                continue
            if str(event.get("symbol") or symbol).upper() != symbol.upper():
                continue
            source = str(event.get("source") or event.get("sentiment_source") or "")
            if not _is_trusted_distill_source(source):
                continue
            ts = parse_datetime(event.get("timestamp") or event.get("updated_at") or event.get("captured_at"))
            direction = _direction_from_event(event)
            if not ts or not direction:
                continue
            normalized = dict(event)
            normalized["_ts"] = ts
            normalized["_source"] = source
            normalized["_direction"] = direction
            normalized["_confidence"] = max(
                to_float(event.get("confidence"), -1.0),
                to_float(event.get("confidence_score"), -1.0),
                to_float(event.get("hotness_score"), -1.0),
                to_float(event.get("score"), -1.0),
            )
            events.append(normalized)
    return sorted(events, key=lambda item: item["_ts"])


def infer_distill_label(item: dict, report: dict) -> tuple[int, dict]:
    symbol = str(item.get("symbol") or "")
    entry_at = parse_datetime(item.get("entry_timestamp") or item.get("opened_at") or item.get("entry_at"))
    exit_at = parse_datetime(item.get("timestamp") or item.get("exit_timestamp") or item.get("closed_at"))
    if not symbol or not entry_at:
        return -1, {"reason": "missing_entry_time", "trusted_sources": []}
    side = _direction_from_event({"side": item.get("side") or item.get("position_side") or "long"}) or "long"
    opposite = "short" if side == "long" else "long"
    events = [event for event in _distill_events_from_report(report, symbol) if event["_ts"] >= entry_at and (not exit_at or event["_ts"] <= exit_at)]
    early_reverse = [
        event
        for event in events
        if (event["_ts"] - entry_at).total_seconds() <= 30 * 60
        and event["_direction"] == opposite
        and _is_strong_distill_event(event)
    ]
    early_reverse_sources = {event["_source"] for event in early_reverse}
    if len(early_reverse_sources) >= 2:
        return 0, {"reason": "early_reverse_strong_distill", "trusted_sources": sorted(early_reverse_sources)}

    same_direction = [event for event in events if event["_direction"] == side and _is_strong_distill_event(event)]
    if same_direction:
        first = same_direction[0]
        last = same_direction[-1]
        same_sources = {event["_source"] for event in same_direction}
        confidence_floor = first["_confidence"] - 10.0 if first["_confidence"] >= 0 else 0.0
        if (last["_ts"] - entry_at).total_seconds() >= 120 * 60 and len(same_sources) >= 2 and last["_confidence"] >= confidence_floor:
            return 1, {"reason": "stable_same_direction_distill", "trusted_sources": sorted(same_sources)}

    return -1, {"reason": "insufficient_distill_sequence", "trusted_sources": sorted({event["_source"] for event in events})}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with file_lock(path):
        return _load_existing_keys_unlocked(path)


def _load_existing_keys_unlocked(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        key = str(payload.get("sample_key") or "")
        if key:
            keys.add(key)
    return keys


def append_new_samples(path: Path, samples: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with file_lock(path):
        existing = _load_existing_keys_unlocked(path) if path.exists() else set()
        with path.open("a", encoding="utf-8") as handle:
            for sample in samples:
                key = str(sample.get("sample_key") or "")
                if not key or key in existing:
                    continue
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                existing.add(key)
                added += 1
            handle.flush()
            os.fsync(handle.fileno())
    return added


def build_trade_sample_key(item: dict) -> str:
    return "|".join(
        [
            str(item.get("timestamp") or ""),
            str(item.get("symbol") or ""),
            str(item.get("action") or ""),
            str(item.get("reason") or ""),
            str(item.get("entry_price") or ""),
            str(item.get("exit_price") or ""),
        ]
    )


LOW_QUALITY_LOSS_REASONS = {
    "stop_loss_hit",
    "score_faded",
    "market_snapshot_missing_timeout",
}


def _minutes_between(start: Any, end: Any) -> float:
    start_dt = parse_datetime(start)
    end_dt = parse_datetime(end)
    if not start_dt or not end_dt:
        return 0.0
    return max(0.0, (end_dt - start_dt).total_seconds() / 60.0)


def _sample_quality_flags(item: dict, basket_item: dict, hot: dict, oi_item: dict) -> list[str]:
    """Flag closed trades that should not dominate model training.

    This is deliberately signal-quality based, not symbol/name based. Raw journal
    rows remain untouched; only collected training samples receive quality tags.
    """
    pnl_pct = to_float(item.get("pnl_pct"))
    reason = str(item.get("reason") or "unknown").split(",", 1)[0].split(":", 1)[0]
    if pnl_pct >= 0:
        return []

    flags: list[str] = []
    score = to_float(item.get("score"), to_float(basket_item.get("score")))
    candidate_score = to_float(basket_item.get("score"))
    setup_type = str(basket_item.get("setup_type") or item.get("setup_type") or "").lower()
    hotness = to_float(hot.get("hotness_score"), to_float(item.get("hotness_score")))
    mentions = int(to_float(item.get("sentiment_mentions"), to_float(hot.get("mentions"))))
    oi_change = to_float(item.get("oi_change_pct_4h"), to_float(oi_item.get("oi_change_pct_4h")))
    taker = to_float(item.get("taker_buy_sell_ratio"), to_float(oi_item.get("taker_buy_sell_ratio")))
    quote_volume = to_float(item.get("quote_volume_usdt_24h"), to_float(oi_item.get("quote_volume_usdt_24h")))
    hold_minutes = _minutes_between(
        item.get("entry_timestamp") or item.get("opened_at") or item.get("entry_at"),
        item.get("timestamp") or item.get("exit_timestamp") or item.get("closed_at"),
    )

    if reason in LOW_QUALITY_LOSS_REASONS:
        flags.append(f"weak_exit_reason:{reason}")
    if score and score < 68.0 and candidate_score < 68.0:
        flags.append("low_entry_score")
    if not setup_type or setup_type in {"unknown", "none", "null"}:
        flags.append("missing_setup_type")
    if taker and taker < 1.0:
        flags.append("taker_weak")
    if oi_change <= 0.0:
        flags.append("oi_not_expanding")
    if hotness and hotness < 55.0:
        flags.append("low_hotness")
    if mentions and mentions < 3:
        flags.append("low_mentions")
    if quote_volume and quote_volume < 20_000_000:
        flags.append("low_liquidity")
    if hold_minutes and hold_minutes <= 45.0 and reason == "stop_loss_hit":
        flags.append("fast_stop_loss")
    if pnl_pct <= -0.08 and reason in LOW_QUALITY_LOSS_REASONS:
        flags.append("extreme_weak_loss")
    return flags


def _is_low_quality_training_loss(flags: list[str]) -> bool:
    if not flags:
        return False
    if any(flag.startswith("weak_exit_reason:market_snapshot_missing_timeout") for flag in flags):
        return True
    if "fast_stop_loss" in flags and len(flags) >= 2:
        return True
    signal_flags = {
        "low_entry_score",
        "missing_setup_type",
        "taker_weak",
        "oi_not_expanding",
        "low_hotness",
        "low_mentions",
        "low_liquidity",
        "extreme_weak_loss",
    }
    return bool(any(flag.startswith("weak_exit_reason:") for flag in flags) and len(signal_flags.intersection(flags)) >= 2)


_SHORT_SETUP_TYPES = {"squeeze_short_candidate", "base_breakdown_confirmed"}


def _infer_side_from_setup(setup_type: str) -> str:
    return "short" if setup_type in _SHORT_SETUP_TYPES else "long"


_HF_QUALITY_KEYS = (
    "hf_win_rate_3bar",
    "hf_avg_mfe_12bar",
    "hf_avg_risk_reward",
    "hf_best_signal_win_rate",
    "hf_momentum_edge",
    "hf_oi_surge_edge",
    "hf_signal_count_total",
)


def _extract_hf_quality(source: dict, side: str = "long") -> dict:
    by_side = source.get("by_side")
    if by_side is not None:
        directional = by_side.get(side)
        if directional is None:
            return {k: 0.0 for k in _HF_QUALITY_KEYS}
        return {k: to_float(directional.get(k)) for k in _HF_QUALITY_KEYS}
    # No by_side → old data or already-directional journal snapshot, use as-is
    return {k: to_float(source.get(k)) for k in _HF_QUALITY_KEYS}


def _extract_h4_structure(h4: dict | None) -> dict:
    if not h4:
        return {
            "h4_close_above_ma20": True,
            "h4_ma20_slope_pct": 0.0,
            "h4_price_vs_ma20_pct": 0.0,
            "h4_close_position_20bars": 0.5,
            "h4_low_break_recent_20bars": False,
            "h4_high_break_recent_20bars": False,
            "h4_atr_compression": 1.0,
            "h4_lower_highs_count": 0,
            "h4_higher_lows_count": 0,
            "h4_structure_state": "unknown",
            "h4_price_location": "middle",
            "h4_pattern_type": "unknown",
        }
    return {
        "h4_close_above_ma20": bool(h4.get("h4_close_above_ma20", True)),
        "h4_ma20_slope_pct": to_float(h4.get("h4_ma20_slope_pct")),
        "h4_price_vs_ma20_pct": to_float(h4.get("h4_price_vs_ma20_pct")),
        "h4_close_position_20bars": to_float(h4.get("h4_close_position_20bars"), 0.5),
        "h4_low_break_recent_20bars": bool(h4.get("h4_low_break_recent_20bars", False)),
        "h4_high_break_recent_20bars": bool(h4.get("h4_high_break_recent_20bars", False)),
        "h4_atr_compression": to_float(h4.get("h4_atr_compression"), 1.0),
        "h4_lower_highs_count": int(to_float(h4.get("h4_lower_highs_count"))),
        "h4_higher_lows_count": int(to_float(h4.get("h4_higher_lows_count"))),
        "h4_structure_state": str(h4.get("h4_structure_state") or "unknown"),
        "h4_price_location": str(h4.get("h4_price_location") or "middle"),
        "h4_pattern_type": str(h4.get("h4_pattern_type") or "unknown"),
    }


_BB_ENTRY_FEATURE_KEYS = (
    "daily_price_vs_ma7_pct",
    "daily_price_vs_ma20_pct",
    "daily_price_position_365",
    "bb_price_position_96",
    "bb_price_position_210",
    "bb_price_position_672_15m",
    "bb_price_position_105_4h",
    "bb_price_position_672_4h",
    "bb_price_position_1440_4h",
    "bb_price_position_105_1m",
    "bb_price_position_300_1m",
    "bb105_mid_4h_distance_pct",
    "bb672_mid_4h_distance_pct",
    "bb1440_mid_4h_distance_pct",
    "bb105_upper_1m_distance_pct",
    "bb300_upper_1m_distance_pct",
    "bb96_upper_15m_distance_pct",
    "short_4h_mid_retest_distance_min_pct",
    "short_1m_upper_pressure_max",
    "short_15m_bb96_position",
    "williams_r_15m",
    "williams_r_15m_120",
    "williams_r_1m",
)

_BB_ENTRY_HEAT_FIELDS = (
    "bb_price_position_96",
    "bb_price_position_210",
    "bb_price_position_672_15m",
    "bb_price_position_105_4h",
)

_BB_ENTRY_EXTENSION_FIELDS = (
    "bb_price_position_96",
    "bb_price_position_210",
    "bb_price_position_672_15m",
    "bb_price_position_105_4h",
    "bb_price_position_672_4h",
)


def _source_get(source: dict | None, key: str, default: object = None) -> object:
    if not isinstance(source, dict):
        return default
    return source.get(key, default)


def _bb_touch_upper_value(zone: object) -> float:
    return 1.0 if "upper" in str(zone or "").lower() else 0.0


def _extract_bb_entry_features(cf: dict, item: dict, kline_source: dict, h4_features: dict) -> dict:
    """Copy live-computable BB context from the entry snapshot into closed-trade samples."""
    features = {
        key: to_float(_source_get(cf, key, _source_get(item, key)))
        for key in _BB_ENTRY_FEATURE_KEYS
    }
    bb_touch_zone = str(_source_get(cf, "bb_touch_zone", _source_get(item, "bb_touch_zone", "")) or "")
    features["bb_touch_zone"] = bb_touch_zone
    features["bb_upper_heat_score"] = float(
        sum(1 for key in _BB_ENTRY_HEAT_FIELDS if to_float(features.get(key), 0.5) >= 0.85)
    )
    features["bb_extension_max"] = max(
        to_float(features.get(key), 0.5) for key in _BB_ENTRY_EXTENSION_FIELDS
    )
    features["atr_regime_high"] = 1.0 if to_float(_source_get(kline_source, "kline_atr_pct")) >= 0.10 else 0.0
    features["false_breakout_4h"] = (
        1.0
        if to_float(features.get("bb_price_position_105_4h"), 0.5) >= 0.70
        and to_float(h4_features.get("h4_ma20_slope_pct")) < 0.0
        else 0.0
    )
    features["bb_touch_upper"] = _bb_touch_upper_value(bb_touch_zone)
    return features


def _classify_loss(pnl_pct: float, hold_minutes: float | None, reason_family: str) -> tuple[str, float]:
    """Classify a loss trade and assign sample weight for training."""
    if pnl_pct >= 0:
        return "winner", 1.0
    if hold_minutes and hold_minutes <= 45.0 and reason_family == "stop_loss_hit":
        return "execution_noise", 0.3
    if pnl_pct <= -0.08:
        return "extreme_loss", 0.5
    if hold_minutes and hold_minutes >= 360.0 and reason_family == "stop_loss_hit":
        return "signal_failure", 1.5
    if reason_family == "score_faded":
        return "signal_fade", 0.8
    return "normal_loss", 1.0


def _compute_pdf_label(item: dict, prev_exit: dict | None = None, daily_loss_usdt: float = 0.0):
    """Build context from a journal exit item and run PDF strategy labeler."""
    from trading_system.pdf_strategy_labeler import PdfStrategyLabel

    cf = item.get("coin_features") or {}
    hold_minutes = _minutes_between(item.get("entry_timestamp"), item.get("timestamp"))

    minutes_since_last_sl = 9999.0
    last_exit_reason = ""
    if prev_exit:
        last_exit_reason = str(prev_exit.get("reason") or "")
        if last_exit_reason.startswith("stop_loss"):
            elapsed = _minutes_between(prev_exit.get("timestamp"), item.get("entry_timestamp") or item.get("timestamp"))
            if elapsed > 0:
                minutes_since_last_sl = elapsed

    ctx = {
        "kline_direction": to_float(cf.get("kline_direction") or item.get("kline_direction")),
        "kline_ma20_distance_pct": to_float(cf.get("kline_ma20_distance_pct") or item.get("kline_ma20_distance_pct")),
        "kline_consecutive_direction": to_float(cf.get("kline_consecutive_direction") or item.get("kline_consecutive_direction")),
        "kline_volume_vs_ma5": to_float(cf.get("kline_volume_vs_ma5") or item.get("kline_volume_vs_ma5")),
        "kline_upper_shadow_pct": to_float(cf.get("kline_upper_shadow_pct") or item.get("kline_upper_shadow_pct")),
        "kline_pattern_bullish_count": to_float(
            item.get("kline_pattern_bullish_count") if "kline_pattern_bullish_count" in item
            else cf.get("kline_pattern_bullish_count")
        ),
        "price_position_7d": to_float(cf.get("price_position_7d") or item.get("price_position_7d")),
        "score": to_float(item.get("entry_score") or item.get("score")),
        "entry_price": to_float(item.get("entry_price")),
        "stop_loss_price": to_float(item.get("stop_loss_price")),
        "take_profit_price": to_float(item.get("take_profit_price")),
        "initial_risk_pct": to_float(item.get("initial_risk_pct")),
        "notional_usdt": to_float(item.get("notional_usdt")),
        "total_capital_usdt": 1000.0,
        "pnl_pct": to_float(item.get("pnl_pct")),
        "exit_reason": str(item.get("reason") or ""),
        "hold_minutes": hold_minutes or 0.0,
        "size_fraction": to_float(item.get("size_fraction"), 1.0),
        "minutes_since_last_sl": minutes_since_last_sl,
        "last_exit_reason": last_exit_reason,
        "daily_loss_usdt": daily_loss_usdt,
    }
    try:
        return _pdf_label_trade(ctx)
    except Exception:
        return PdfStrategyLabel()


def collect_closed_trade_samples(state: dict, report: dict, captured_at: str | None = None, hf_quality_lookup: dict | None = None) -> tuple[list[dict], dict]:
    captured_at = captured_at or utc_now_iso()
    hf_quality_lookup = hf_quality_lookup or {}
    journal = list(state.get("journal") or [])
    journal.extend(item for item in (state.get("shadow_journal") or []) if isinstance(item, dict))
    journal.sort(key=lambda x: parse_datetime(x.get("timestamp")) or datetime.max.replace(tzinfo=timezone.utc))
    top_hot = report.get("hotness_priority") or []
    top_oi = report.get("oi_divergence_candidates") or []
    basket = report.get("candidate_basket") or []
    top_hot_by_symbol = {item.get("symbol"): item for item in top_hot if item.get("symbol")}
    basket_by_symbol = {item.get("symbol"): item for item in basket if item.get("symbol")}
    oi_by_symbol = {item.get("symbol"): item for item in top_oi if item.get("symbol")}
    samples: list[dict] = []
    _exit_history: dict[str, dict] = {}
    _daily_pnl: float = 0.0
    _last_date: str = ""
    for item in journal:
        action = str(item.get("action") or "")
        if action not in {"exit", "take_profit_partial"}:
            continue
        symbol = str(item.get("symbol") or "")
        if str(symbol or "").upper() in INVALID_SYMBOL_ALIASES:
            continue
        item_date = str(item.get("timestamp") or "")[:10]
        if item_date != _last_date:
            _daily_pnl = 0.0
            _last_date = item_date
        reason = str(item.get("reason") or "unknown")
        pnl_pct = to_float(item.get("pnl_pct"))
        notional = to_float(item.get("notional_usdt"))
        real_pnl_label = 1 if pnl_pct > 0 else 0
        distill_label, distill_meta = infer_distill_label(item, report)
        pdf_result = _compute_pdf_label(item, prev_exit=_exit_history.get(symbol.upper()), daily_loss_usdt=_daily_pnl)
        _daily_pnl += pnl_pct * notional
        _exit_history[symbol.upper()] = {"timestamp": item.get("timestamp"), "reason": reason, "pnl_pct": pnl_pct}
        pdf_label = pdf_result.pdf_label
        if pdf_label in {0, 1} and distill_label in {0, 1}:
            final_label = round(0.5 * real_pnl_label + 0.25 * distill_label + 0.25 * pdf_label, 4)
        elif pdf_label in {0, 1}:
            final_label = round(0.6 * real_pnl_label + 0.4 * pdf_label, 4)
        elif distill_label in {0, 1}:
            final_label = round(0.7 * real_pnl_label + 0.3 * distill_label, 4)
        else:
            final_label = -1
        hot = top_hot_by_symbol.get(symbol, {})
        basket_item = basket_by_symbol.get(symbol, {})
        oi_item = oi_by_symbol.get(symbol, {})
        whale = _extract_whale_metrics(_metric_source_for_symbol(report, symbol))
        quality_flags = _sample_quality_flags(item, basket_item, hot, oi_item)
        exclude_from_training = _is_low_quality_training_loss(quality_flags)
        reason_family = reason.split(",", 1)[0].split(":", 1)[0]
        hold_minutes = _minutes_between(item.get("entry_timestamp"), item.get("timestamp"))
        loss_category, sample_weight = _classify_loss(pnl_pct, hold_minutes, reason_family)
        initial_risk_pct = to_float(item.get("initial_risk_pct"))
        r_multiple = round(pnl_pct / initial_risk_pct, 4) if initial_risk_pct > 0 else 0.0
        r_multiple_clipped = max(-2.0, min(5.0, r_multiple))
        # Extract kline features from entry coin_features snapshot (primary) or basket (fallback)
        cf = item.get("coin_features") or {}
        _KLINE_KEYS = ("kline_lower_shadow_pct", "kline_body_pct", "kline_direction", "kline_atr_pct")
        kline_source = cf if any(k in cf for k in _KLINE_KEYS) else basket_item
        h4_features = _extract_h4_structure(item.get("entry_h4_structure"))
        bb_entry_features = _extract_bb_entry_features(cf, item, kline_source, h4_features)
        samples.append(
            {
                "sample_key": build_trade_sample_key(item),
                "captured_at": captured_at,
                "sample_type": "shadow_closed_trade" if item.get("source") == "shadow_paper" else "closed_trade",
                "source": str(item.get("source") or "paper"),
                "symbol": symbol,
                "timestamp": item.get("timestamp"),
                "action": action,
                "reason": reason,
                "reason_family": reason_family,
                "entry_price": to_float(item.get("entry_price")),
                "exit_price": to_float(item.get("exit_price")),
                "size_fraction": to_float(item.get("size_fraction"), 1.0),
                "notional_usdt": to_float(item.get("notional_usdt")),
                "pnl_pct": pnl_pct,
                "is_win": pnl_pct > 0,
                "initial_risk_pct": initial_risk_pct,
                "r_multiple": r_multiple_clipped,
                "loss_category": loss_category,
                "sample_weight": sample_weight,
                "exclude_from_training": exclude_from_training,
                "training_quality": "low_quality_loss" if exclude_from_training else "usable",
                "quality_flags": quality_flags,
                "real_pnl_label": real_pnl_label,
                "distill_label": distill_label,
                "pdf_label": pdf_label,
                "pdf_entry_score": pdf_result.pdf_entry_score,
                "pdf_risk_score": pdf_result.pdf_risk_score,
                "pdf_hold_score": pdf_result.pdf_hold_score,
                "pdf_overall_score": pdf_result.pdf_overall_score,
                "pdf_violation_count": len(pdf_result.violations),
                "pdf_revenge_entry_flag": 1.0 if "no_revenge_entry" in pdf_result.violations else 0.0,
                "pdf_violations": pdf_result.violations,
                "label": final_label,
                "distill_training_enabled": distill_label in {0, 1},
                "distill_label_reason": distill_meta.get("reason") or "",
                "distill_trusted_sources": distill_meta.get("trusted_sources") or [],
                "priority_score": to_float(hot.get("priority_score")),
                "hotness_score": to_float(hot.get("hotness_score")),
                "mentions": int(to_float(hot.get("mentions"))),
                "hot_source": hot.get("source") or "",
                "candidate_score": to_float(basket_item.get("score")),
                "risk_band": basket_item.get("risk_band") or "",
                "execution_status": basket_item.get("execution_status") or "",
                "setup_type": basket_item.get("setup_type") or item.get("setup_type") or "",
                "oi_divergence_score": to_float(oi_item.get("oi_price_divergence_score")),
                **whale,
                # --- 入场特征：优先用 exit journal 自带的（更准确），report 数据作 fallback ---
                "sentiment_score": to_float(
                    item.get("sentiment_score"),  # 来自 position.last_sentiment_score（入场时）
                    to_float(basket_item.get("sentiment_score"), to_float(hot.get("hotness_score"))),
                ),
                "sentiment_mentions": int(to_float(
                    item.get("sentiment_mentions"),  # 来自 position.last_sentiment_mentions（入场时）
                    to_float(hot.get("mentions")),
                )),
                "sentiment_source": str(item.get("sentiment_source") or hot.get("source") or ""),
                "oi_change_pct_4h": to_float(
                    item.get("oi_change_pct_4h"),  # 来自 position.last_oi_change_pct_4h（入场时）
                    to_float(oi_item.get("oi_change_pct_4h")),
                ),
                "taker_buy_sell_ratio": to_float(
                    item.get("taker_buy_sell_ratio"),  # 来自 position.last_taker_buy_sell_ratio（入场时）
                    to_float(oi_item.get("taker_buy_sell_ratio")),
                ),
                "quote_volume_usdt_24h": to_float(item.get("quote_volume_usdt_24h"), to_float(oi_item.get("quote_volume_usdt_24h"))),
                "price_change_pct_5m": to_float(item.get("price_change_pct_5m"), to_float(oi_item.get("price_change_pct_5m"))),
                "price_change_pct_24h": to_float(item.get("price_change_pct_24h"), to_float(oi_item.get("price_change_pct_24h"))),
                "timing_label": oi_item.get("timing_label") or "",
                # --- 结构 + K 线特征（从入场 coin_features 或 basket 提取） ---
                "price_position_7d": to_float(cf.get("price_position_7d"), to_float(basket_item.get("price_position_7d"))),
                "price_position_14d": to_float(cf.get("price_position_14d"), to_float(basket_item.get("price_position_14d"))),
                "price_position_20d": to_float(cf.get("price_position_20d"), to_float(basket_item.get("price_position_20d"))),
                "breakout_distance_7d": to_float(cf.get("breakout_distance_7d"), to_float(basket_item.get("breakout_distance_7d"))),
                "compression_score": to_float(cf.get("compression_score"), to_float(basket_item.get("compression_score"))),
                "kline_lower_shadow_pct": to_float(kline_source.get("kline_lower_shadow_pct")),
                "kline_upper_shadow_pct": to_float(kline_source.get("kline_upper_shadow_pct")),
                "kline_body_pct": to_float(kline_source.get("kline_body_pct")),
                "kline_direction": to_float(kline_source.get("kline_direction")),
                "kline_volume_vs_ma5": to_float(kline_source.get("kline_volume_vs_ma5")),
                "kline_range_vs_ma5_range": to_float(kline_source.get("kline_range_vs_ma5_range")),
                "kline_close_position": to_float(kline_source.get("kline_close_position")),
                "kline_consecutive_direction": to_float(kline_source.get("kline_consecutive_direction")),
                "kline_atr_pct": to_float(kline_source.get("kline_atr_pct")),
                "kline_ma20_distance_pct": to_float(kline_source.get("kline_ma20_distance_pct")),
                "kline_volume_trend": to_float(kline_source.get("kline_volume_trend")),
                "kline_max_drawdown_5d": to_float(kline_source.get("kline_max_drawdown_5d")),
                "kline_pattern_bullish_count": to_float(item.get("kline_pattern_bullish_count") if "kline_pattern_bullish_count" in item else kline_source.get("kline_pattern_bullish_count")),
                "kline_pattern_bearish_count": to_float(item.get("kline_pattern_bearish_count") if "kline_pattern_bearish_count" in item else kline_source.get("kline_pattern_bearish_count")),
                "kline_pattern_flags": str(item.get("kline_pattern_flags") if "kline_pattern_flags" in item else (kline_source.get("kline_pattern_flags") or "")),
                # --- 4H structure (入场时快照，避免后验污染) ---
                # --- BB multi-timeframe entry context (live-computable, not post-hoc labels) ---
                **bb_entry_features,
                **h4_features,
                # --- HF backtest 信号质量（仅用入场时 journal 保存的快照，避免后验回填） ---
                **_extract_hf_quality(item.get("hf_quality") or {}),
                # --- Coin Scout 模型需要的额外字段 ---
                "entry_timestamp": item.get("entry_timestamp"),
                "entry_score": to_float(item.get("entry_score")),
                "entry_mark_price": to_float(item.get("entry_mark_price")),
                # 入场时完整 CoinFeatures（65 维，来自 execution.py journal）
                "coin_features": item.get("coin_features"),
                # Shadow 模型打分（含 scores/errors/model_meta）
                "shadow_scores": item.get("shadow_scores"),
            }
        )
    context = {
        "top_hot_symbols": [item.get("symbol") for item in top_hot[:5]],
        "top_oi_symbols": [item.get("symbol") for item in top_oi[:5]],
        "candidate_symbols": [item.get("symbol") for item in basket[:5]],
    }
    return samples, context


def collect_decision_samples(report: dict, captured_at: str | None = None, hf_quality_lookup: dict | None = None) -> list[dict]:
    captured_at = captured_at or utc_now_iso()
    hf_quality_lookup = hf_quality_lookup or {}
    basket = report.get("candidate_basket") or []
    ranked = report.get("top_ranked_symbols") or []
    ranked_by_symbol = {str(item.get("symbol") or ""): item for item in ranked if item.get("symbol")}
    paper_rebalance = report.get("paper_rebalance") or {}
    entries = list(paper_rebalance.get("entries") or [])
    shadow_paper = paper_rebalance.get("shadow_paper") if isinstance(paper_rebalance, dict) else {}
    if isinstance(shadow_paper, dict):
        entries.extend(item for item in (shadow_paper.get("entries") or []) if isinstance(item, dict))
    entry_symbols = {str(item.get("symbol") or "") for item in entries if item.get("symbol")}
    macro_side_channel = _macro_trend_side_channel(report)
    samples: list[dict] = []
    for index, row in enumerate(basket):
        symbol = str(row.get("symbol") or "")
        if not is_valid_usdt_symbol(symbol):
            continue
        ranked_row = ranked_by_symbol.get(symbol, {})
        whale = _extract_whale_metrics(_metric_source_for_symbol(report, symbol))
        execution_status = str(row.get("execution_status") or "watch_only")
        blocked_reasons = list(row.get("blocked_reasons") or [])
        selected_for_entry = symbol in entry_symbols
        if selected_for_entry:
            decision_label = "entered"
        elif execution_status == "watch_only":
            decision_label = "blocked"
        elif execution_status == "confirm_needed":
            decision_label = "confirm_needed"
        else:
            decision_label = "watch"
        samples.append(
            {
                "sample_key": f"{captured_at}|{index}|{symbol}|{execution_status}|{row.get('setup_type') or ''}",
                "captured_at": captured_at,
                "sample_type": "decision",
                "symbol": symbol,
                "rank": index + 1,
                "score": to_float(row.get("score")),
                "setup_type": row.get("setup_type") or ranked_row.get("setup_type") or "",
                "execution_status": execution_status,
                "decision_label": decision_label,
                "selected_for_entry": selected_for_entry,
                "blocked_reasons": blocked_reasons,
                "risk_band": row.get("risk_band") or "",
                "risk_grade": row.get("risk_grade") or "",
                "setup_quality": to_float(row.get("setup_quality"), -1.0),
                "manipulation_stage": row.get("manipulation_stage"),
                "manipulation_quality": to_float(row.get("manipulation_quality"), -1.0),
                "price_position_7d": to_float(row.get("price_position_7d"), -1.0),
                "price_position_14d": to_float(row.get("price_position_14d"), -1.0),
                "price_position_20d": to_float(row.get("price_position_20d"), -1.0),
                "breakout_distance_7d": to_float(row.get("breakout_distance_7d"), -1.0),
                "compression_score": to_float(row.get("compression_score"), -1.0),
                "kline_lower_shadow_pct": to_float(row.get("kline_lower_shadow_pct")),
                "kline_upper_shadow_pct": to_float(row.get("kline_upper_shadow_pct")),
                "kline_body_pct": to_float(row.get("kline_body_pct")),
                "kline_direction": to_float(row.get("kline_direction")),
                "kline_volume_vs_ma5": to_float(row.get("kline_volume_vs_ma5")),
                "kline_range_vs_ma5_range": to_float(row.get("kline_range_vs_ma5_range")),
                "kline_close_position": to_float(row.get("kline_close_position")),
                "kline_consecutive_direction": to_float(row.get("kline_consecutive_direction")),
                "kline_atr_pct": to_float(row.get("kline_atr_pct")),
                "kline_ma20_distance_pct": to_float(row.get("kline_ma20_distance_pct")),
                "kline_volume_trend": to_float(row.get("kline_volume_trend")),
                "kline_max_drawdown_5d": to_float(row.get("kline_max_drawdown_5d")),
                "kline_pattern_bullish_count": to_float(row.get("kline_pattern_bullish_count")),
                "kline_pattern_bearish_count": to_float(row.get("kline_pattern_bearish_count")),
                "kline_pattern_flags": str(row.get("kline_pattern_flags") or ""),
                "h4_close_above_ma20": bool(row.get("h4_close_above_ma20", True)),
                "h4_ma20_slope_pct": to_float(row.get("h4_ma20_slope_pct")),
                "h4_price_vs_ma20_pct": to_float(row.get("h4_price_vs_ma20_pct")),
                "h4_close_position_20bars": to_float(row.get("h4_close_position_20bars"), 0.5),
                "h4_low_break_recent_20bars": bool(row.get("h4_low_break_recent_20bars", False)),
                "h4_high_break_recent_20bars": bool(row.get("h4_high_break_recent_20bars", False)),
                "h4_atr_compression": to_float(row.get("h4_atr_compression"), 1.0),
                "h4_lower_highs_count": int(to_float(row.get("h4_lower_highs_count"))),
                "h4_higher_lows_count": int(to_float(row.get("h4_higher_lows_count"))),
                "h4_structure_state": str(row.get("h4_structure_state") or "unknown"),
                "h4_price_location": str(row.get("h4_price_location") or "middle"),
                "h4_pattern_type": str(row.get("h4_pattern_type") or "unknown"),
                "funding_rate": to_float(row.get("funding_rate")),
                "oi_change_pct_4h": to_float(row.get("oi_change_pct_4h")),
                "price_change_pct_5m": to_float(row.get("price_change_pct_5m")),
                "price_change_pct_24h": to_float(row.get("price_change_pct_24h")),
                "sentiment_score": to_float(row.get("sentiment_score"), to_float(ranked_row.get("sentiment_score"))),
                "sentiment_mentions": int(to_float(ranked_row.get("sentiment_mentions"), 0.0)),
                "sentiment_source": str(ranked_row.get("sentiment_source") or ""),
                "taker_buy_sell_ratio": to_float(ranked_row.get("taker_buy_sell_ratio")),
                "quote_volume_usdt_24h": to_float(ranked_row.get("quote_volume_usdt_24h")),
                "oi_price_divergence_score": to_float(ranked_row.get("oi_price_divergence_score")),
                **macro_side_channel,
                **whale,
                **_extract_hf_quality(hf_quality_lookup.get(symbol, {}), side=_infer_side_from_setup(row.get("setup_type") or ranked_row.get("setup_type") or "")),
            }
        )
    return samples


def collect_distill_process_samples(state: dict, report: dict, captured_at: str | None = None) -> list[dict]:
    captured_at = captured_at or utc_now_iso()
    ranked = report.get("top_ranked_symbols") or []
    ranked_by_symbol = {str(item.get("symbol") or ""): item for item in ranked if item.get("symbol")}
    samples: list[dict] = []
    for pos in state.get("positions") or []:
        symbol = str(pos.get("symbol") or "")
        if not symbol:
            continue
        row = ranked_by_symbol.get(symbol, {})
        entry_price = to_float(pos.get("entry_price"), to_float(pos.get("current_price")))
        mark_price = to_float(row.get("mark_price"), to_float(pos.get("current_price"), entry_price))
        price_return = ((mark_price - entry_price) / entry_price) if entry_price else 0.0
        current_score = to_float(row.get("score"), to_float(pos.get("score")))
        last_score = to_float(pos.get("score"), current_score)
        sentiment_score = to_float(row.get("sentiment_score"), to_float(pos.get("last_sentiment_score")))
        last_sentiment_score = to_float(pos.get("last_sentiment_score"), sentiment_score)
        mentions = int(to_float(row.get("sentiment_mentions"), to_float(pos.get("last_sentiment_mentions"))))
        last_mentions = int(to_float(pos.get("last_sentiment_mentions"), mentions))
        oi = to_float(row.get("oi_change_pct_4h"), to_float(pos.get("last_oi_change_pct_4h")))
        taker = to_float(row.get("taker_buy_sell_ratio"), to_float(pos.get("last_taker_buy_sell_ratio")))
        health_score = 0
        warning_reasons: list[str] = []
        if price_return >= 0:
            health_score += 1
        else:
            warning_reasons.append("price_against_position")
        if sentiment_score >= last_sentiment_score - 10 and mentions >= max(1, int(last_mentions * 0.6)):
            health_score += 1
        else:
            warning_reasons.append("sentiment_faded")
        if oi >= 0:
            health_score += 1
        else:
            warning_reasons.append("oi_not_supporting")
        if taker >= 1.0:
            health_score += 1
        else:
            warning_reasons.append("taker_weak")
        if current_score < last_score - 15:
            warning_reasons.append("score_faded")
        health_score = min(3, health_score)
        process_label = "positive" if health_score >= 3 else "negative"
        samples.append(
            {
                "sample_key": f"{captured_at}|{symbol}|distill_process|{pos.get('opened_at') or ''}",
                "captured_at": captured_at,
                "sample_type": "distill_process",
                "symbol": symbol,
                "opened_at": pos.get("opened_at"),
                "entry_price": entry_price,
                "mark_price": mark_price,
                "price_return_pct": round(price_return * 100.0, 4),
                "score_at_entry": last_score,
                "current_score": current_score,
                "health_score": health_score,
                "process_label": process_label,
                "warning_tag": "" if process_label == "positive" else "distill_health_deteriorated",
                "warning_reasons": warning_reasons,
                "sentiment_score": sentiment_score,
                "sentiment_mentions": mentions,
                "sentiment_source": str(row.get("sentiment_source") or ""),
                "oi_change_pct_4h": oi,
                "taker_buy_sell_ratio": taker,
            }
        )
    return samples


def build_closed_trade_summary(dataset_path: Path) -> dict:
    rows: list[dict] = []
    if dataset_path.exists():
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    total = len(rows)
    wins = sum(1 for row in rows if row.get("is_win"))
    pnl_total = sum(to_float(row.get("pnl_pct")) for row in rows)
    reason_buckets: dict[str, dict[str, float]] = {}
    symbol_buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        reason = str(row.get("reason_family") or "unknown")
        rb = reason_buckets.setdefault(reason, {"count": 0, "wins": 0, "pnl_total": 0.0})
        rb["count"] += 1
        rb["wins"] += 1 if row.get("is_win") else 0
        rb["pnl_total"] += to_float(row.get("pnl_pct"))
        symbol = str(row.get("symbol") or "UNKNOWN")
        sb = symbol_buckets.setdefault(symbol, {"count": 0, "wins": 0, "pnl_total": 0.0})
        sb["count"] += 1
        sb["wins"] += 1 if row.get("is_win") else 0
        sb["pnl_total"] += to_float(row.get("pnl_pct"))
    return {
        "updated_at": utc_now_iso(),
        "dataset_file": str(dataset_path),
        "sample_count": total,
        "win_rate_pct": round((wins / total) * 100.0, 2) if total else 0.0,
        "avg_pnl_pct": round((pnl_total / total) * 100.0, 3) if total else 0.0,
        "total_pnl_pct": round(pnl_total * 100.0, 3),
        "by_reason": [
            {
                "reason": reason,
                "count": slot["count"],
                "win_rate_pct": round((slot["wins"] / slot["count"]) * 100.0, 2) if slot["count"] else 0.0,
                "avg_pnl_pct": round((slot["pnl_total"] / slot["count"]) * 100.0, 3) if slot["count"] else 0.0,
                "total_pnl_pct": round(slot["pnl_total"] * 100.0, 3),
            }
            for reason, slot in sorted(reason_buckets.items(), key=lambda item: (item[1]["pnl_total"], item[1]["count"]), reverse=True)
        ][:10],
        "top_symbols": [
            {
                "symbol": symbol,
                "count": slot["count"],
                "win_rate_pct": round((slot["wins"] / slot["count"]) * 100.0, 2) if slot["count"] else 0.0,
                "total_pnl_pct": round(slot["pnl_total"] * 100.0, 3),
            }
            for symbol, slot in sorted(symbol_buckets.items(), key=lambda item: (item[1]["pnl_total"], item[1]["count"]), reverse=True)
        ][:10],
    }


def build_decision_summary(dataset_path: Path) -> dict:
    rows: list[dict] = []
    if dataset_path.exists():
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    execution_status_counts = Counter(str(row.get("execution_status") or "unknown") for row in rows)
    decision_label_counts = Counter(str(row.get("decision_label") or "unknown") for row in rows)
    blocked_reason_counts = Counter()
    setup_type_counts = Counter(str(row.get("setup_type") or "unknown") for row in rows)
    for row in rows:
        blocked_reason_counts.update([str(item) for item in row.get("blocked_reasons") or [] if item])
    return {
        "updated_at": utc_now_iso(),
        "dataset_file": str(dataset_path),
        "sample_count": len(rows),
        "execution_status_counts": dict(execution_status_counts),
        "decision_label_counts": dict(decision_label_counts),
        "top_blocked_reasons": [
            {"reason": reason, "count": count}
            for reason, count in blocked_reason_counts.most_common(10)
        ],
        "top_setup_types": [
            {"setup_type": setup_type, "count": count}
            for setup_type, count in setup_type_counts.most_common(10)
        ],
    }


def build_analysis_maturity_summary(
    decision_dataset_path: Path,
    closed_trade_dataset_path: Path,
    state_path: Path | None = None,
    lookback_days: int = 30,
    min_win_rate_pct: float = 55.0,
    max_safe_drawdown_pct: float = 10.0,
    min_review_coverage_pct: float = 80.0,
    min_reviewed_samples: int = 30,
) -> dict:
    """Build a Nicjiang-style analysis tracking/review/maturity dashboard.

    Definitions:
    - analysis_count: decision samples generated in the rolling window.
    - reviewed_count: closed trade samples with a resolved win/loss outcome.
    - in_progress_count: open paper positions from state, plus currently entered decisions
      not yet represented by closed-trade samples.
    - pending_review_count: entries that are no longer in open positions but do not have
      a matching closed-trade review yet (best-effort; falls back to zero for old data).
    """
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - lookback_days * 86400

    def _read_jsonl(path: Path) -> list[dict]:
        rows: list[dict] = []
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

    def _in_window(row: dict, *fields: str) -> bool:
        for field in fields:
            ts = parse_datetime(row.get(field))
            if ts is not None:
                return ts.timestamp() >= cutoff
        return True

    decision_rows = [row for row in _read_jsonl(decision_dataset_path) if _in_window(row, "captured_at", "timestamp")]
    closed_rows = [row for row in _read_jsonl(closed_trade_dataset_path) if _in_window(row, "timestamp", "captured_at")]
    reviewed_rows = [row for row in closed_rows if "is_win" in row or "pnl_pct" in row]
    wins = sum(1 for row in reviewed_rows if bool(row.get("is_win")) or to_float(row.get("pnl_pct")) > 0)
    losses = max(0, len(reviewed_rows) - wins)

    open_positions: list[dict] = []
    if state_path and state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            open_positions = list(state.get("positions") or [])
        except Exception:
            open_positions = []

    entered_decisions = [row for row in decision_rows if row.get("selected_for_entry") or row.get("decision_label") == "entered"]
    closed_symbols = Counter(str(row.get("symbol") or "") for row in reviewed_rows)
    open_symbols = Counter(str(row.get("symbol") or "") for row in open_positions)
    pending_review_count = 0
    for row in entered_decisions:
        symbol = str(row.get("symbol") or "")
        if closed_symbols[symbol] > 0:
            closed_symbols[symbol] -= 1
        elif open_symbols[symbol] > 0:
            open_symbols[symbol] -= 1
        else:
            pending_review_count += 1

    in_progress_count = len(open_positions)
    analysis_count = len(decision_rows)
    reviewed_count = len(reviewed_rows)
    effective_total = reviewed_count + pending_review_count + in_progress_count
    review_coverage_pct = round((reviewed_count / effective_total) * 100.0, 2) if effective_total else 0.0
    real_win_rate_pct = round((wins / reviewed_count) * 100.0, 2) if reviewed_count else 0.0

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in sorted(reviewed_rows, key=lambda item: str(item.get("timestamp") or item.get("captured_at") or "")):
        equity += to_float(row.get("pnl_pct")) * to_float(row.get("size_fraction"), 1.0)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    max_drawdown_pct = round(max_drawdown * 100.0, 2)

    win_rate_ok = real_win_rate_pct >= min_win_rate_pct and reviewed_count >= min_reviewed_samples
    drawdown_ok = max_drawdown_pct <= max_safe_drawdown_pct
    coverage_ok = review_coverage_pct >= min_review_coverage_pct or reviewed_count >= max(min_reviewed_samples, analysis_count * 0.5)
    backlog_ok = pending_review_count <= max(5, int(max(1, analysis_count) * 0.05))
    if win_rate_ok and drawdown_ok and coverage_ok and backlog_ok:
        status = "mature"
    elif reviewed_count < min_reviewed_samples or not coverage_ok:
        status = "warming_up"
    elif not drawdown_ok:
        status = "risk_warning"
    elif not win_rate_ok:
        status = "needs_optimization"
    else:
        status = "watch"

    return {
        "updated_at": utc_now_iso(),
        "lookback_days": lookback_days,
        "analysis_count": analysis_count,
        "reviewed_count": reviewed_count,
        "real_win_rate_pct": real_win_rate_pct,
        "correct_count": wins,
        "wrong_count": losses,
        "pending_review_count": pending_review_count,
        "in_progress_count": in_progress_count,
        "review_coverage_pct": review_coverage_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "thresholds": {
            "min_win_rate_pct": min_win_rate_pct,
            "max_safe_drawdown_pct": max_safe_drawdown_pct,
            "min_review_coverage_pct": min_review_coverage_pct,
            "min_reviewed_samples": min_reviewed_samples,
        },
        "checks": {
            "win_rate": "pass" if win_rate_ok else "fail",
            "drawdown": "safe" if drawdown_ok else "unsafe",
            "review_coverage": "pass" if coverage_ok else "fail",
            "review_backlog": "pass" if backlog_ok else "fail",
        },
        "system_status": status,
    }
