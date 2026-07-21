#!/usr/bin/env python3
"""Label V6 entry-gate decisions with next-bar MFE/MAE outcomes.

The job is intentionally cache-only. V6 already refreshes execution K-lines;
this script reads those files and adds no Binance request load. It records both
allowed and blocked candidates so gate changes can be evaluated against the
same market path without submitting orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.persistence import append_text_line, atomic_write_json, load_json  # noqa: E402


HORIZON_BARS = {"1h": 4, "4h": 16, "24h": 96, "48h": 192, "168h": 672}
ROUND_TRIP_COST_PCT = 0.16  # taker 5 bps + slippage 3 bps, each side


def parse_iso_ms(value: str) -> int:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return 0
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_gate_events(path: Path, max_events: int) -> list[dict[str, Any]]:
    rows = load_jsonl(path.with_suffix(path.suffix + ".1")) + load_jsonl(path)
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows[-max(1, int(max_events)):]:
        if str(row.get("event") or "") != "entry_gate_decision":
            continue
        event_id = str(row.get("event_id") or "")
        if event_id:
            deduped[event_id] = row
    return list(deduped.values())


def load_cached_klines(cache_dir: Path, symbol: str) -> tuple[list[list[Any]], dict[str, Any]]:
    path = cache_dir / "15m" / f"{str(symbol or '').upper()}.json"
    payload = load_json(path, {})
    rows = payload.get("klines") if isinstance(payload, dict) else []
    normalized = [row for row in rows if isinstance(row, list) and len(row) >= 7]
    normalized.sort(key=lambda row: int(row[0]))
    return normalized, {
        "path": str(path),
        "updated_at": payload.get("updated_at") if isinstance(payload, dict) else None,
        "source": payload.get("source") if isinstance(payload, dict) else None,
    }


def next_bar_index(rows: list[list[Any]], event_ms: int) -> int | None:
    for index, row in enumerate(rows):
        try:
            if int(row[0]) > event_ms:
                return index
        except (TypeError, ValueError):
            continue
    return None


def _number(row: list[Any], index: int) -> float:
    try:
        return float(row[index])
    except (IndexError, TypeError, ValueError):
        return 0.0


def label_horizon(
    rows: list[list[Any]],
    *,
    start_index: int,
    bar_count: int,
    side: str,
    now_ms: int,
) -> dict[str, Any] | None:
    window = rows[start_index:start_index + int(bar_count)]
    if len(window) < int(bar_count):
        return None
    try:
        last_close_ms = int(window[-1][6])
    except (IndexError, TypeError, ValueError):
        return None
    if last_close_ms > now_ms:
        return None
    entry = _number(window[0], 1)
    exit_price = _number(window[-1], 4)
    highs = [_number(row, 2) for row in window]
    lows = [_number(row, 3) for row in window]
    if entry <= 0 or exit_price <= 0 or not highs or not lows:
        return None
    if side == "short":
        gross_return = (entry - exit_price) / entry * 100.0
        mfe = (entry - min(lows)) / entry * 100.0
        mae = (entry - max(highs)) / entry * 100.0
    else:
        gross_return = (exit_price - entry) / entry * 100.0
        mfe = (max(highs) - entry) / entry * 100.0
        mae = (min(lows) - entry) / entry * 100.0
    return {
        "bars": int(bar_count),
        "entry_time": datetime.fromtimestamp(int(window[0][0]) / 1000, tz=timezone.utc).isoformat(),
        "exit_time": datetime.fromtimestamp(last_close_ms / 1000, tz=timezone.utc).isoformat(),
        "entry_price": round(entry, 8),
        "exit_price": round(exit_price, 8),
        "gross_return_pct": round(gross_return, 6),
        "net_return_pct": round(gross_return - ROUND_TRIP_COST_PCT, 6),
        "mfe_pct": round(max(0.0, mfe), 6),
        "mae_pct": round(min(0.0, mae), 6),
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "funding_omitted": True,
    }


def label_event(event: dict[str, Any], rows: list[list[Any]], now_ms: int) -> dict[str, Any] | None:
    event_ms = parse_iso_ms(str(event.get("computed_at") or event.get("recorded_at") or ""))
    if event_ms <= 0:
        return None
    start_index = next_bar_index(rows, event_ms)
    if start_index is None:
        return None
    action = str(event.get("action") or "")
    side = "short" if action == "open_short" else "long" if action == "open_long" else ""
    if not side:
        return None
    labels: dict[str, dict[str, Any]] = {}
    for name, bar_count in HORIZON_BARS.items():
        label = label_horizon(rows, start_index=start_index, bar_count=bar_count, side=side, now_ms=now_ms)
        if label is not None:
            labels[name] = label
    if not labels:
        return None
    decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
    complete = len(labels) == len(HORIZON_BARS)
    primary = labels.get("24h") or labels.get("4h") or labels.get("1h") or {}
    favorable = bool(
        float(primary.get("mfe_pct") or 0.0) >= 3.0
        and float(primary.get("mfe_pct") or 0.0) >= abs(float(primary.get("mae_pct") or 0.0)) * 1.25
    )
    harmful = bool(
        float(primary.get("mae_pct") or 0.0) <= -3.0
        and abs(float(primary.get("mae_pct") or 0.0)) > float(primary.get("mfe_pct") or 0.0)
    )
    return {
        "event": "entry_gate_shadow_outcome",
        "event_id": event.get("event_id"),
        "computed_at": event.get("computed_at"),
        "labeled_at": datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).isoformat(),
        "symbol": str(event.get("symbol") or "").upper(),
        "action": action,
        "side": side,
        "signal_price": event.get("price"),
        "selected_entry_level": dict(event.get("selected_entry_level") or {}),
        "gate_state": decision.get("state"),
        "gate_allowed": bool(decision.get("allowed")),
        "notional_multiplier": decision.get("notional_multiplier"),
        "blocking_reason": decision.get("blocking_reason"),
        "hard_reasons": list(decision.get("hard_reasons") or []),
        "watch_reasons": list(decision.get("watch_reasons") or []),
        "soft_reasons": list(decision.get("soft_reasons") or []),
        "support_reasons": list(decision.get("support_reasons") or []),
        "transition_probe": bool(decision.get("transition_probe")),
        "outcome_status": "complete" if complete else "partial",
        "favorable_counterfactual": favorable,
        "harmful_counterfactual": harmful,
        "horizons": labels,
    }


def outcome_signature(outcome: dict[str, Any]) -> str:
    stable = {
        "event_id": outcome.get("event_id"),
        "outcome_status": outcome.get("outcome_status"),
        "horizons": outcome.get("horizons") or {},
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()


def build_summary(outcomes: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allowed_count = 0
    blocked_count = 0
    for row in outcomes:
        allowed_count += int(bool(row.get("gate_allowed")))
        blocked_count += int(not bool(row.get("gate_allowed")))
        reasons = [
            *list(row.get("hard_reasons") or []),
            *list(row.get("watch_reasons") or []),
            *list(row.get("soft_reasons") or []),
        ]
        if not reasons:
            reasons = ["allowed_without_gate_reason"]
        for reason in sorted(set(str(item) for item in reasons if item)):
            by_reason[reason].append(row)

    reason_rows: list[dict[str, Any]] = []
    recommendations: dict[str, Any] = {}
    for reason, rows in sorted(by_reason.items()):
        qualified_rows = [
            row
            for row in rows
            if isinstance(row.get("horizons"), dict) and isinstance(row["horizons"].get("24h"), dict)
        ]
        if not qualified_rows:
            continue
        horizon_rows = [row["horizons"]["24h"] for row in qualified_rows]

        def favorable_24h(row: dict[str, Any]) -> bool:
            outcome = row["horizons"]["24h"]
            mfe = float(outcome.get("mfe_pct") or 0.0)
            mae = abs(float(outcome.get("mae_pct") or 0.0))
            return bool(mfe >= 3.0 and mfe >= mae * 1.25)

        def harmful_24h(row: dict[str, Any]) -> bool:
            outcome = row["horizons"]["24h"]
            mfe = float(outcome.get("mfe_pct") or 0.0)
            mae = abs(float(outcome.get("mae_pct") or 0.0))
            return bool(mae >= 3.0 and mae > mfe)

        favorable_rate = sum(favorable_24h(row) for row in qualified_rows) / len(qualified_rows)
        harmful_rate = sum(harmful_24h(row) for row in qualified_rows) / len(qualified_rows)
        avg = lambda key: sum(float(item.get(key) or 0.0) for item in horizon_rows) / len(horizon_rows)
        summary_row = {
            "reason": reason,
            "samples": len(horizon_rows),
            "favorable_rate": round(favorable_rate, 6),
            "harmful_rate": round(harmful_rate, 6),
            "avg_net_return_pct_24h": round(avg("net_return_pct"), 6),
            "avg_mfe_pct_24h": round(avg("mfe_pct"), 6),
            "avg_mae_pct_24h": round(avg("mae_pct"), 6),
        }
        reason_rows.append(summary_row)
        recommendation = "insufficient_samples"
        if len(horizon_rows) >= 30:
            if favorable_rate >= 0.58 and summary_row["avg_net_return_pct_24h"] > 0:
                recommendation = "downgrade_to_soft_or_sizing"
            elif harmful_rate >= 0.58 and summary_row["avg_net_return_pct_24h"] < 0:
                recommendation = "retain_hard_or_watch"
            else:
                recommendation = "retain_watch_and_collect_more"
        recommendations[reason] = {**summary_row, "recommendation": recommendation}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "next_15m_bar_open_cache_only",
        "cost_model": {"round_trip_pct": ROUND_TRIP_COST_PCT, "funding_omitted": True},
        "events_labeled": len(outcomes),
        "allowed_events": allowed_count,
        "blocked_events": blocked_count,
        "complete_events": sum(str(row.get("outcome_status")) == "complete" for row in outcomes),
        "reasons": reason_rows,
    }
    return summary, {
        "generated_at": summary["generated_at"],
        "auto_apply": False,
        "minimum_samples": 30,
        "recommendations": recommendations,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-file", type=Path, default=ROOT / "data/trading_system/v6_execution/entry_gate_events.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/trading_system/cache/execution_klines")
    parser.add_argument("--outcomes-file", type=Path, default=ROOT / "data/trading_system/v6_execution/entry_gate_outcomes.jsonl")
    parser.add_argument("--state-file", type=Path, default=ROOT / "data/trading_system/v6_execution/entry_gate_outcome_state.json")
    parser.add_argument("--summary-file", type=Path, default=ROOT / "data/trading_system/v6_execution/entry_gate_outcome_summary.json")
    parser.add_argument(
        "--recommendations-file",
        type=Path,
        default=ROOT / "data/trading_system/v6_execution/entry_gate_policy_recommendations.json",
    )
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--now", default="", help="Optional ISO UTC timestamp for deterministic tests.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now_ms = parse_iso_ms(args.now) if args.now else int(datetime.now(timezone.utc).timestamp() * 1000)
    if now_ms <= 0:
        raise SystemExit("invalid --now timestamp")
    events = load_gate_events(args.events_file, args.max_events)
    state = load_json(args.state_file, {"events": {}})
    state_events = state.get("events") if isinstance(state.get("events"), dict) else {}
    cache_by_symbol: dict[str, tuple[list[list[Any]], dict[str, Any]]] = {}
    latest_outcomes: list[dict[str, Any]] = []
    appended = 0

    for event in events:
        symbol = str(event.get("symbol") or "").upper()
        if symbol not in cache_by_symbol:
            cache_by_symbol[symbol] = load_cached_klines(args.cache_dir, symbol)
        klines, cache_meta = cache_by_symbol[symbol]
        outcome = label_event(event, klines, now_ms)
        if outcome is None:
            continue
        outcome["kline_cache"] = cache_meta
        event_id = str(outcome.get("event_id") or "")
        signature = outcome_signature(outcome)
        previous = state_events.get(event_id) if isinstance(state_events.get(event_id), dict) else {}
        if previous.get("signature") != signature and not args.dry_run:
            append_text_line(args.outcomes_file, json.dumps(outcome, ensure_ascii=False, separators=(",", ":")))
            appended += 1
        state_events[event_id] = {
            "signature": signature,
            "outcome_status": outcome.get("outcome_status"),
            "horizons": sorted((outcome.get("horizons") or {}).keys()),
        }
        latest_outcomes.append(outcome)

    summary, recommendations = build_summary(latest_outcomes)
    summary["events_seen"] = len(events)
    summary["outcomes_appended"] = appended
    summary["symbols_cached"] = sorted(cache_by_symbol)
    if not args.dry_run:
        active_ids = {str(row.get("event_id") or "") for row in events}
        compact_state = {key: value for key, value in state_events.items() if key in active_ids}
        atomic_write_json(args.state_file, {"updated_at": summary["generated_at"], "events": compact_state})
        atomic_write_json(args.summary_file, summary)
        atomic_write_json(args.recommendations_file, recommendations)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
