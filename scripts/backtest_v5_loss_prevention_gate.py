#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.v5_loss_prevention import (  # noqa: E402
    LABEL_DEFAULT_ACTIONS,
    LABEL_PHASES,
    RISK_LABEL_DESCRIPTIONS,
    classify_v5_loss_risk,
    selected_level_name,
)
from scripts.backtest_v5_loss_order_15m_paths import (  # noqa: E402
    find_entry_bar,
    indicator_snapshot,
    load_klines,
    parse_dt_ms,
    pct_diff,
)


ALL_LABELS = list(RISK_LABEL_DESCRIPTIONS)


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


def iso_or_empty(value: Any) -> str:
    parsed = parse_dt(value)
    return parsed.isoformat() if parsed else ""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            item["_line"] = line_no
            rows.append(item)
    return rows


def side_from_action(action: str, side: Any = "") -> str:
    text = str(action or "").lower()
    if "long" in text:
        return "long"
    if "short" in text:
        return "short"
    raw = str(side or "").lower()
    if raw in {"long", "short"}:
        return raw
    if raw == "buy":
        return "long"
    if raw == "sell":
        return "short"
    return raw


def pnl_pct(side: str, entry: float | None, close: float | None) -> float | None:
    if not entry or not close or entry <= 0 or close <= 0:
        return None
    if side == "long":
        return (close - entry) / entry * 100.0
    if side == "short":
        return (entry - close) / entry * 100.0
    return None


def gross_profit(rows: Iterable[Mapping[str, Any]]) -> float:
    return sum(max(0.0, finite_float(row.get("pnl_usdt"), 0.0) or 0.0) for row in rows)


def gross_loss(rows: Iterable[Mapping[str, Any]]) -> float:
    return abs(sum(min(0.0, finite_float(row.get("pnl_usdt"), 0.0) or 0.0) for row in rows))


def profit_factor(rows: list[Mapping[str, Any]]) -> float | None:
    loss = gross_loss(rows)
    profit = gross_profit(rows)
    if loss <= 0:
        return None
    return profit / loss


def max_drawdown(rows: list[Mapping[str, Any]]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for row in sorted(rows, key=lambda item: str(item.get("closed_at") or "")):
        equity += finite_float(row.get("pnl_usdt"), 0.0) or 0.0
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def get_path(value: Mapping[str, Any], *parts: str) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def open_record_time(order: Mapping[str, Any]) -> datetime | None:
    intent = order.get("intent") if isinstance(order.get("intent"), Mapping) else {}
    return parse_dt(intent.get("generated_at") or order.get("submitted_at"))


def build_open_history(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for order in orders:
        intent = order.get("intent") if isinstance(order.get("intent"), Mapping) else {}
        action = str(intent.get("action") or "")
        if order.get("status") != "submitted" or action not in {"open_long", "open_short"}:
            continue
        history.append(order)
    return history


def previous_open(open_history: list[dict[str, Any]], symbol: str, side: str, line_no: int) -> dict[str, Any] | None:
    matched: dict[str, Any] | None = None
    for order in open_history:
        if int(order.get("_line") or 0) >= line_no:
            break
        intent = order.get("intent") if isinstance(order.get("intent"), Mapping) else {}
        if intent.get("symbol") == symbol and side_from_action(str(intent.get("action") or ""), intent.get("side")) == side:
            matched = order
    return matched


def matching_open_for_queue(open_history: list[dict[str, Any]], item: Mapping[str, Any]) -> dict[str, Any] | None:
    symbol = str(item.get("symbol") or "")
    side = str(item.get("side") or "")
    opened_at = parse_dt(item.get("opened_at"))
    best: tuple[float, dict[str, Any]] | None = None
    for order in open_history:
        intent = order.get("intent") if isinstance(order.get("intent"), Mapping) else {}
        if intent.get("symbol") != symbol or side_from_action(str(intent.get("action") or ""), intent.get("side")) != side:
            continue
        opened = parse_dt(order.get("submitted_at")) or open_record_time(order)
        if opened_at and opened:
            delta = abs((opened - opened_at).total_seconds())
            if best is None or delta < best[0]:
                best = (delta, order)
        elif best is None:
            best = (999999.0, order)
    if best and best[0] <= 300.0:
        return best[1]
    return best[1] if best else None


def feature_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    intent = record.get("intent") if isinstance(record.get("intent"), Mapping) else record
    risk_plan = intent.get("risk_plan") if isinstance(intent.get("risk_plan"), Mapping) else record.get("risk_plan") or {}
    entry_context = risk_plan.get("entry_context") if isinstance(risk_plan.get("entry_context"), Mapping) else {}
    selected = entry_context.get("selected_entry_level") if isinstance(entry_context.get("selected_entry_level"), Mapping) else {}
    h4 = entry_context.get("h4_right_side") if isinstance(entry_context.get("h4_right_side"), Mapping) else {}
    daily = entry_context.get("daily_ma7_trend") if isinstance(entry_context.get("daily_ma7_trend"), Mapping) else {}
    ma20 = get_path(risk_plan, "entry_indicators", "ma20_trend")
    if not isinstance(ma20, Mapping):
        ma20 = entry_context.get("ma20_trend") if isinstance(entry_context.get("ma20_trend"), Mapping) else {}
    return {
        "horizon": risk_plan.get("horizon") or "",
        "entry_timeframe": risk_plan.get("entry_timeframe") or "",
        "selected_level": selected.get("name") or selected_level_name(record),
        "selected_tf": selected.get("timeframe") or "",
        "selected_signal": selected.get("structure_signal") or "",
        "ma20_state": ma20.get("state") or "",
        "h4_state": h4.get("state") or "",
        "h4_price_distance_pct": finite_float(h4.get("price_distance_pct"), None),
        "daily_bearish_stack": daily.get("bearish_stack"),
        "daily_ma7_reclaimed": daily.get("ma7_reclaimed"),
        "daily_ma20_reclaimed": daily.get("ma20_reclaimed"),
    }


def classifier_record_from_queue(item: Mapping[str, Any], matched_open: Mapping[str, Any] | None) -> dict[str, Any]:
    if matched_open and isinstance(matched_open.get("intent"), Mapping):
        return {"intent": matched_open["intent"], "opened_at": item.get("opened_at")}
    side = str(item.get("side") or "")
    return {
        "symbol": item.get("symbol"),
        "side": side,
        "action": f"open_{side}" if side in {"long", "short"} else "",
        "opened_at": item.get("opened_at"),
        "entry_price": item.get("entry_price"),
        "risk_plan": item.get("risk_plan") if isinstance(item.get("risk_plan"), Mapping) else {},
        "rationale": "",
    }


def extract_closed_outcomes(input_dir: Path) -> list[dict[str, Any]]:
    orders = load_jsonl(input_dir / "orders.jsonl")
    closed_queue = load_jsonl(input_dir / "closed_trade_review_queue.jsonl")
    open_history = build_open_history(orders)
    outcomes: list[dict[str, Any]] = []

    for order in orders:
        intent = order.get("intent") if isinstance(order.get("intent"), Mapping) else {}
        action = str(intent.get("action") or "")
        if order.get("status") != "submitted" or action not in {"close_long", "close_short"}:
            continue
        symbol = str(intent.get("symbol") or "")
        side = side_from_action(action, intent.get("side"))
        reconcile = order.get("position_reconcile") if isinstance(order.get("position_reconcile"), Mapping) else {}
        before = reconcile.get("before") if isinstance(reconcile.get("before"), Mapping) else {}
        entry = finite_float(before.get("entry_price"), None)
        close = finite_float(before.get("mark_price"), None)
        qty = finite_float(before.get("abs_position_qty"), None)
        pnl_usdt = finite_float(before.get("unrealized_profit"), None)
        if not close or close <= 0:
            continue
        notional = entry * qty if entry and qty else None
        pct = (pnl_usdt / notional * 100.0) if pnl_usdt is not None and notional else pnl_pct(side, entry, close)
        if pct is None:
            continue
        matched_open = previous_open(open_history, symbol, side, int(order.get("_line") or 0))
        classifier_record = {"intent": matched_open.get("intent"), "opened_at": (matched_open or {}).get("submitted_at")} if matched_open else {"intent": intent}
        summary = feature_summary(classifier_record)
        opened_at = (
            ((matched_open or {}).get("intent") or {}).get("generated_at")
            if matched_open and isinstance((matched_open or {}).get("intent"), Mapping)
            else intent.get("generated_at")
        )
        outcomes.append(
            {
                "record_id": f"orders:{order.get('_line')}",
                "source": "explicit_close",
                "symbol": symbol,
                "side": side,
                "opened_at": iso_or_empty(opened_at or (matched_open or {}).get("submitted_at")),
                "closed_at": iso_or_empty(order.get("submitted_at") or intent.get("generated_at")),
                "entry_price": entry,
                "close_price": close,
                "quantity": qty,
                "notional_usdt": notional,
                "pnl_pct": pct,
                "pnl_usdt": pnl_usdt if pnl_usdt is not None else (pct / 100.0 * notional if notional else None),
                "close_reason": intent.get("rationale") or "",
                "close_fraction": intent.get("close_fraction"),
                "classifier_record": classifier_record,
                **summary,
            }
        )

    for item in closed_queue:
        symbol = str(item.get("symbol") or "")
        side = str(item.get("side") or "")
        entry = finite_float(item.get("entry_price"), None)
        close = finite_float(item.get("close_price"), None)
        pct = pnl_pct(side, entry, close)
        if pct is None:
            continue
        notional = finite_float(item.get("original_notional_usdt"), None)
        matched_open = matching_open_for_queue(open_history, item)
        classifier_record = classifier_record_from_queue(item, matched_open)
        summary = feature_summary(classifier_record)
        outcomes.append(
            {
                "record_id": f"closed_queue:{item.get('_line')}",
                "source": "closed_queue",
                "symbol": symbol,
                "side": side,
                "opened_at": iso_or_empty(item.get("opened_at")),
                "closed_at": iso_or_empty(item.get("closed_at")),
                "entry_price": entry,
                "close_price": close,
                "quantity": (notional / entry) if notional and entry else None,
                "notional_usdt": notional,
                "pnl_pct": pct,
                "pnl_usdt": (pct / 100.0 * notional) if notional else None,
                "close_reason": item.get("close_reason") or "",
                "close_fraction": 1.0,
                "classifier_record": classifier_record,
                **summary,
            }
        )

    return sorted(outcomes, key=lambda row: (row.get("opened_at") or "", row.get("closed_at") or "", row.get("record_id") or ""))


def build_15m_contexts(outcomes: list[dict[str, Any]], kline_dir: Path) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    kline_cache: dict[str, list[dict[str, float]]] = {}
    for outcome in outcomes:
        record_id = str(outcome.get("record_id") or "")
        symbol = str(outcome.get("symbol") or "")
        side = str(outcome.get("side") or "")
        opened_ms = parse_dt_ms(outcome.get("opened_at"))
        entry = finite_float(outcome.get("entry_price"), None)
        if not record_id or not symbol or opened_ms is None or entry is None or entry <= 0:
            contexts[record_id] = {
                "entry_indicator_available": False,
                "early_15m_path_available": False,
            }
            continue
        if symbol not in kline_cache:
            kline_cache[symbol] = load_klines(kline_dir, symbol)
        klines = kline_cache[symbol]
        entry_idx = find_entry_bar(klines, opened_ms)
        if entry_idx is None:
            contexts[record_id] = {
                "entry_indicator_available": False,
                "early_15m_path_available": False,
            }
            continue
        context: dict[str, Any] = {
            "entry_indicator_available": True,
            "entry_indicator": indicator_snapshot(klines, entry_idx, side),
            "entry_bar_open_time": int(klines[entry_idx]["open_time"]),
        }
        early_path: dict[str, Any] = {}
        for n in (1, 2):
            future_idx = entry_idx + n
            if future_idx >= len(klines):
                continue
            raw_return = pct_diff(klines[future_idx]["close"], entry)
            favorable_return = raw_return if side == "long" else -raw_return
            early_path[f"n{n}_raw_return_pct"] = raw_return
            early_path[f"n{n}_favorable_return_pct"] = favorable_return
            early_path[f"n{n}_future_close"] = klines[future_idx]["close"]
            early_path[f"n{n}_future_open_time"] = int(klines[future_idx]["open_time"])
        context["early_15m_path_available"] = all(f"n{n}_favorable_return_pct" in early_path for n in (1, 2))
        context["early_15m_path"] = early_path
        contexts[record_id] = context
    return contexts


def label_outcomes(outcomes: list[dict[str, Any]], kline_contexts: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    kline_contexts = kline_contexts or {}
    for outcome in outcomes:
        opened_at = outcome.get("opened_at")
        recent_failures = [
            failure for failure in failures if parse_dt(failure.get("closed_at")) and parse_dt(opened_at) and parse_dt(failure.get("closed_at")) < parse_dt(opened_at)
        ]
        context: dict[str, Any] = {
            "opened_at": opened_at,
            "recent_failures": recent_failures,
            "early_health_available": False,
        }
        context.update(dict(kline_contexts.get(str(outcome.get("record_id") or ""), {})))
        result = classify_v5_loss_risk(
            outcome.get("classifier_record") if isinstance(outcome.get("classifier_record"), Mapping) else outcome,
            context,
        )
        row = dict(outcome)
        row.pop("classifier_record", None)
        row.update(
            {
                "risk_labels": result.risk_labels,
                "risk_reasons": result.reasons,
                "risk_tags": result.tags,
                "risk_details": result.details,
                "unavailable_labels": result.unavailable_labels,
                "entry_indicator_available": context.get("entry_indicator_available"),
                "early_15m_path_available": context.get("early_15m_path_available"),
            }
        )
        entry_indicator = context.get("entry_indicator") if isinstance(context.get("entry_indicator"), Mapping) else {}
        early_15m_path = context.get("early_15m_path") if isinstance(context.get("early_15m_path"), Mapping) else {}
        for key in ("upper_wick_pct", "candle_body_pct", "ma20_slope_4_pct", "close_vs_ma20_pct", "ma7_vs_ma20_pct"):
            if key in entry_indicator:
                row[key] = entry_indicator[key]
        for key in ("n1_favorable_return_pct", "n2_favorable_return_pct"):
            if key in early_15m_path:
                row[key] = early_15m_path[key]
        labeled.append(row)
        if (finite_float(outcome.get("pnl_usdt"), 0.0) or 0.0) < 0:
            failures.append(
                {
                    "symbol": outcome.get("symbol"),
                    "side": outcome.get("side"),
                    "selected_level": outcome.get("selected_level"),
                    "closed_at": outcome.get("closed_at"),
                    "pnl_usdt": outcome.get("pnl_usdt"),
                }
            )
    return labeled


def summarize_label(rows: list[dict[str, Any]], label: str, group_key: str | None = None, group_value: str | None = None) -> dict[str, Any]:
    scoped = [row for row in rows if group_key is None or str(row.get(group_key) or "") == str(group_value or "")]
    blocked = [row for row in scoped if label in (row.get("risk_labels") or [])]
    losers = [row for row in scoped if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) < 0]
    winners = [row for row in scoped if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) > 0]
    blocked_losers = [row for row in blocked if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) < 0]
    blocked_winners = [row for row in blocked if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) > 0]
    avoided_loss = gross_loss(blocked_losers)
    missed_profit = gross_profit(blocked_winners)
    after_rows = [row for row in scoped if label not in (row.get("risk_labels") or [])]
    by_symbol_avoided = defaultdict(float)
    for row in blocked_losers:
        by_symbol_avoided[str(row.get("symbol") or "")] += abs(finite_float(row.get("pnl_usdt"), 0.0) or 0.0)
    max_symbol_concentration = (max(by_symbol_avoided.values()) / avoided_loss) if avoided_loss > 0 and by_symbol_avoided else 0.0
    return {
        "label": label,
        "description": RISK_LABEL_DESCRIPTIONS[label],
        "default_action": LABEL_DEFAULT_ACTIONS[label],
        "phase": LABEL_PHASES[label],
        "group_key": group_key or "all",
        "group_value": group_value or "all",
        "records_count": len(scoped),
        "winners_count": len(winners),
        "losers_count": len(losers),
        "blocked_records_count": len(blocked),
        "blocked_losers_count": len(blocked_losers),
        "blocked_winners_count": len(blocked_winners),
        "avoided_loss_usdt": round(avoided_loss, 6),
        "missed_profit_usdt": round(missed_profit, 6),
        "net_effect_usdt": round(avoided_loss - missed_profit, 6),
        "missed_profit_to_avoided_loss": round(missed_profit / avoided_loss, 6) if avoided_loss > 0 else "",
        "profit_factor_before": _round_or_empty(profit_factor(scoped)),
        "profit_factor_after": _round_or_empty(profit_factor(after_rows)),
        "max_drawdown_before_usdt": round(max_drawdown(scoped), 6),
        "max_drawdown_after_usdt": round(max_drawdown(after_rows), 6),
        "max_avoided_loss_symbol_concentration": round(max_symbol_concentration, 6),
        "rule_grade": "",
    }


def grade_rule(row: Mapping[str, Any]) -> str:
    label = str(row.get("label") or "")
    blocked = int(row.get("blocked_records_count") or 0)
    blocked_losers = int(row.get("blocked_losers_count") or 0)
    blocked_winners = int(row.get("blocked_winners_count") or 0)
    net = finite_float(row.get("net_effect_usdt"), 0.0) or 0.0
    ratio = finite_float(row.get("missed_profit_to_avoided_loss"), None)
    concentration = finite_float(row.get("max_avoided_loss_symbol_concentration"), 0.0) or 0.0
    if label in {"one_m_upper_pressure_active", "early_health_deterioration"}:
        return "C_report_only"
    if blocked < 10 or net <= 0:
        return "C_report_only"
    if label == "m15_scalp_no_follow_1_2":
        return "B_tighten_watch_or_no_add"
    if blocked_losers <= blocked_winners:
        return "B_reduce_or_more_shadow"
    if concentration > 0.65:
        return "B_reduce_or_more_shadow"
    if ratio is not None and ratio > 0.4:
        return "B_reduce_or_more_shadow"
    if label in {"h4_ma20_extension_moderate", "daily_ma7_only_reclaim"}:
        return "B_reduce_or_observe"
    return "A_candidate_for_live_report_only_shadow"


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
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _round_or_empty(value: float | None) -> str | float:
    return "" if value is None else round(value, 6)


def build_group_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    values = sorted({str(row.get(group_key) or "") for row in rows})
    out: list[dict[str, Any]] = []
    for label in ALL_LABELS:
        for value in values:
            summary = summarize_label(rows, label, group_key, value)
            summary["rule_grade"] = grade_rule(summary)
            out.append(summary)
    return out


def write_report(path: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], output_dir: Path, input_dir: Path) -> None:
    winners = [row for row in rows if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) > 0]
    losers = [row for row in rows if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) < 0]
    neutral = len(rows) - len(winners) - len(losers)
    lines = [
        "# V5 Loss Prevention One-Pass Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"Input dir: `{input_dir}`",
        f"Output dir: `{output_dir}`",
        "",
        "## Safety Boundary",
        "",
        "- Live execution changed: **No**.",
        "- Order placement, stop loss, take profit, position sync, and reconcile changed: **No**.",
        "- This pass only labels historical closed outcome records and runs report-only ablation.",
        "- First live action, if any later pass approves it, is limited to reduce-size / no-add / tighten-watch. No hard blocking is enabled here.",
        "",
        "## Added Files",
        "",
        "- `trading_system/v5_loss_prevention.py`",
        "- `scripts/backtest_v5_loss_prevention_gate.py`",
        "- `tests/test_v5_loss_prevention.py`",
        "- `docs/V5_LOSS_PREVENTION_ONE_PASS_REPORT.md`",
        "- `data/trading_system/v5_loss_prevention/risk_labeled_closed_trades.jsonl`",
        "- `data/trading_system/v5_loss_prevention/ablation_summary.csv`",
        "- `data/trading_system/v5_loss_prevention/ablation_by_horizon.csv`",
        "- `data/trading_system/v5_loss_prevention/ablation_by_entry_timeframe.csv`",
        "- `data/trading_system/v5_loss_prevention/ablation_by_selected_level.csv`",
        "- `data/trading_system/v5_loss_prevention/ablation_by_symbol.csv`",
        "",
        "## Sample Size",
        "",
        f"- Closed outcome records: {len(rows)}",
        f"- Winners: {len(winners)}",
        f"- Losers: {len(losers)}",
        f"- Neutral/unknown: {neutral}",
        f"- Gross profit: {gross_profit(rows):.6f} USDT",
        f"- Gross loss: {gross_loss(rows):.6f} USDT",
        f"- Net PnL: {sum((finite_float(row.get('pnl_usdt'), 0.0) or 0.0) for row in rows):.6f} USDT",
        f"- Profit factor before filters: {_round_or_empty(profit_factor(rows))}",
        f"- Max drawdown before filters: {max_drawdown(rows):.6f} USDT",
        "",
        "## Label Definitions",
        "",
    ]
    for label, description in RISK_LABEL_DESCRIPTIONS.items():
        lines.append(f"- `{label}`: {description} Default action: `{LABEL_DEFAULT_ACTIONS[label]}`. Phase: `{LABEL_PHASES[label]}`.")
    lines.extend(["", "## Ablation Summary", "", "| Label | Grade | Blocked losers | Blocked winners | Avoided loss | Missed profit | Net effect | PF before | PF after |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    str(row["rule_grade"]),
                    str(row["blocked_losers_count"]),
                    str(row["blocked_winners_count"]),
                    f"{float(row['avoided_loss_usdt']):.2f}",
                    f"{float(row['missed_profit_usdt']):.2f}",
                    f"{float(row['net_effect_usdt']):.2f}",
                    str(row["profit_factor_before"]),
                    str(row["profit_factor_after"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Layered Ablation Highlights", ""])
    for group_key, title in [
        ("horizon", "By Horizon"),
        ("entry_timeframe", "By Entry Timeframe"),
        ("selected_level", "By Selected Level"),
        ("symbol", "By Symbol"),
    ]:
        group_rows = [
            row
            for row in build_group_rows(rows, group_key)
            if int(row.get("blocked_records_count") or 0) > 0 and (finite_float(row.get("net_effect_usdt"), 0.0) or 0.0) > 0
        ]
        group_rows = sorted(group_rows, key=lambda item: finite_float(item.get("net_effect_usdt"), 0.0) or 0.0, reverse=True)[:8]
        lines.extend(
            [
                f"### {title}",
                "",
                "| Label | Group | Blocked losers | Blocked winners | Avoided loss | Missed profit | Net effect | Grade |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        if not group_rows:
            lines.append("| -- | -- | 0 | 0 | 0.00 | 0.00 | 0.00 | C_report_only |")
        for row in group_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['label']}`",
                        f"`{row['group_value']}`",
                        str(row["blocked_losers_count"]),
                        str(row["blocked_winners_count"]),
                        f"{float(row['avoided_loss_usdt']):.2f}",
                        f"{float(row['missed_profit_usdt']):.2f}",
                        f"{float(row['net_effect_usdt']):.2f}",
                        str(row["rule_grade"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "## Rule Grades",
            "",
            "- A: candidate for a later live report-only/shadow phase. This still does **not** mean live blocking is enabled.",
            "- B: risk feature is useful, but should remain reduce-size/observe-only or needs more shadow data.",
            "- C: report-only only, because sample size is too small, net effect is weak/negative, or the rule is intentionally deferred.",
            "- Hard filter eligibility requires blocked_losers > blocked_winners and controlled missed_profit. No rule in this pass is promoted to hard filtering.",
            "",
            "## Rules Not Ready For Live Blocking",
            "",
            "- `one_m_upper_pressure_active`: intentionally phase 3; can easily kill valid 4H continuation trades.",
            "- `m15_scalp_no_follow_1_2`: post-entry health check only; it can tighten observation or forbid add-ons, but it is not an entry hard block.",
            "- `early_health_deterioration`: first pass keeps it as a future live context label; historical 15m no-follow is tracked separately.",
            "- Any label with fewer than 10 blocked records remains report-only.",
            "- Any label whose net effect depends heavily on one symbol should remain report-only or reduce-size only.",
            "",
            "## Next Allowed Step",
            "",
            "The maximum next step is live report-only/shadow logging for A-grade labels. Do not connect these labels to order blocking, sizing, stops, or reconcile until a separate review confirms live shadow results.",
            "",
            "## Verification Commands",
            "",
            "```bash",
            "python -m pytest -q tests/test_v5_loss_prevention.py",
            "python scripts/backtest_v5_loss_prevention_gate.py --closed-trades all --report-only",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input)
    output_dir = Path(args.out)
    docs_out = Path(args.docs_out)
    outcomes = extract_closed_outcomes(input_dir)
    kline_contexts = build_15m_contexts(outcomes, Path(args.kline_dir))
    labeled = label_outcomes(outcomes, kline_contexts)

    summary_rows = [summarize_label(labeled, label) for label in ALL_LABELS]
    for row in summary_rows:
        row["rule_grade"] = grade_rule(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "risk_labeled_closed_trades.jsonl", labeled)
    write_csv(output_dir / "ablation_summary.csv", summary_rows)
    write_csv(output_dir / "ablation_by_horizon.csv", build_group_rows(labeled, "horizon"))
    write_csv(output_dir / "ablation_by_entry_timeframe.csv", build_group_rows(labeled, "entry_timeframe"))
    write_csv(output_dir / "ablation_by_selected_level.csv", build_group_rows(labeled, "selected_level"))
    write_csv(output_dir / "ablation_by_symbol.csv", build_group_rows(labeled, "symbol"))
    write_report(docs_out, labeled, summary_rows, output_dir, input_dir)

    winners = sum(1 for row in labeled if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) > 0)
    losers = sum(1 for row in labeled if (finite_float(row.get("pnl_usdt"), 0.0) or 0.0) < 0)
    return {
        "closed_outcome_records": len(labeled),
        "winners": winners,
        "losers": losers,
        "records_with_15m_context": sum(1 for value in kline_contexts.values() if value.get("entry_indicator_available")),
        "output_dir": str(output_dir),
        "report": str(docs_out),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report-only V5 loss prevention gate ablation.")
    parser.add_argument("--input", default="data/trading_system/v5_v6_execution", help="V5 execution state directory.")
    parser.add_argument("--kline-dir", default="data/trading_system/cache/backtest_klines/15m", help="15m kline cache for report-only path labels.")
    parser.add_argument("--out", default="data/trading_system/v5_loss_prevention", help="Output data directory.")
    parser.add_argument("--docs-out", default="docs/V5_LOSS_PREVENTION_ONE_PASS_REPORT.md", help="Markdown report path.")
    parser.add_argument("--closed-trades", default="all", choices=["all"], help="Closed trade sample selector. Only 'all' is supported.")
    parser.add_argument("--report-only", action="store_true", help="No live behavior is changed; kept for explicit command semantics.")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
