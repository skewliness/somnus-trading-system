from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


RISK_LABEL_DESCRIPTIONS: dict[str, str] = {
    "mixed_structure_conflict": "BB/MA entry scan is mixed or has both long and short candidates.",
    "m15_guard_broken_or_mixed": "15M MA20 trend guard is broken or mixed at entry.",
    "repeat_selected_level_failure": "Same symbol, side, and selected level failed recently.",
    "h4_ma20_extension_extreme": "Long entry is more than 20% away from 4H MA20.",
    "h4_ma20_extension_moderate": "Long entry is 12%-20% away from 4H MA20.",
    "daily_ma7_only_reclaim": "Daily trend is still bearish and only MA7 has been reclaimed.",
    "one_m_upper_pressure_active": "Long entry has active 1M upper pressure.",
    "long_large_upper_wick": "Long entry confirmation candle has a large upper wick.",
    "m15_scalp_no_follow_1_2": "15M scalp-style entry has no favorable follow-through after 1-2 candles.",
    "early_health_deterioration": "Post-entry early health context reports deterioration.",
}

LABEL_PHASES: dict[str, str] = {
    "mixed_structure_conflict": "phase_1",
    "m15_guard_broken_or_mixed": "phase_1",
    "repeat_selected_level_failure": "phase_1",
    "h4_ma20_extension_extreme": "phase_2",
    "h4_ma20_extension_moderate": "phase_2",
    "daily_ma7_only_reclaim": "phase_2",
    "one_m_upper_pressure_active": "phase_3",
    "long_large_upper_wick": "phase_1",
    "m15_scalp_no_follow_1_2": "phase_1",
    "early_health_deterioration": "phase_3",
}

LABEL_DEFAULT_ACTIONS: dict[str, str] = {
    "mixed_structure_conflict": "tighten_watch_candidate",
    "m15_guard_broken_or_mixed": "tighten_watch_candidate",
    "repeat_selected_level_failure": "no_add_candidate",
    "h4_ma20_extension_extreme": "reduce_size_candidate",
    "h4_ma20_extension_moderate": "reduce_size_candidate",
    "daily_ma7_only_reclaim": "observe_or_reduce_candidate",
    "one_m_upper_pressure_active": "report_only",
    "long_large_upper_wick": "no_add_or_reduce_candidate",
    "m15_scalp_no_follow_1_2": "tighten_watch_or_no_add_candidate",
    "early_health_deterioration": "report_only",
}


@dataclass(frozen=True)
class LossRiskResult:
    risk_labels: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    unavailable_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_labels": list(self.risk_labels),
            "reasons": list(self.reasons),
            "tags": list(self.tags),
            "details": dict(self.details),
            "unavailable_labels": list(self.unavailable_labels),
        }


def classify_v5_loss_risk(record: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> LossRiskResult:
    """Classify report-only V5 loss-prevention labels for an entry record.

    This function is intentionally pure: it reads only the supplied record/context
    and returns labels. It must not place orders, mutate state, or change live
    execution behavior.
    """

    context = context or {}
    view = _RecordView(record)
    labels: list[str] = []
    reasons: list[str] = []
    tags: list[str] = []
    details: dict[str, Any] = {}
    unavailable_labels: list[str] = []

    long_count = _finite_float(view.structure_scan.get("long_candidate_count"), 0.0)
    short_count = _finite_float(view.structure_scan.get("short_candidate_count"), 0.0)
    rationale_lower = view.rationale.lower()
    mixed_by_text = "entry scan is mixed" in rationale_lower or "structure scan is mixed" in rationale_lower
    mixed_by_candidates = long_count > 0 and short_count > 0
    if mixed_by_text or mixed_by_candidates:
        _add(
            labels,
            reasons,
            "mixed_structure_conflict",
            "entry structure is mixed or has candidates on both sides",
        )
        details["mixed_structure_conflict"] = {
            "long_candidate_count": int(long_count),
            "short_candidate_count": int(short_count),
            "matched_by_text": mixed_by_text,
        }

    ma20_state = str(view.ma20_trend.get("state") or view.flat.get("ma20_state") or "").lower()
    ma20_guard_text = "15m ma20 trend guard is broken or mixed" in rationale_lower
    if ma20_guard_text or "broken" in ma20_state or "mixed" in ma20_state:
        _add(labels, reasons, "m15_guard_broken_or_mixed", "15M MA20 guard is broken or mixed")
        details["m15_guard_broken_or_mixed"] = {"ma20_state": ma20_state or None, "matched_by_text": ma20_guard_text}

    repeat_detail = _repeat_failure_detail(view, context)
    if repeat_detail["repeat_selected_level_failure"]:
        _add(labels, reasons, "repeat_selected_level_failure", "same selected level failed recently")
        details["repeat_selected_level_failure"] = repeat_detail

    h4_extension_pct = view.h4_extension_pct()
    if view.side == "long" and h4_extension_pct is not None:
        details["h4_ma20_extension_pct"] = round(h4_extension_pct, 6)
        if h4_extension_pct > 20.0:
            _add(labels, reasons, "h4_ma20_extension_extreme", "long entry is more than 20% away from 4H MA20")
        elif h4_extension_pct >= 12.0:
            _add(labels, reasons, "h4_ma20_extension_moderate", "long entry is 12%-20% away from 4H MA20")

    daily_bearish_stack = _boolish(view.daily_trend.get("bearish_stack"), view.flat.get("daily_bearish_stack"))
    daily_ma7_reclaimed = _boolish(view.daily_trend.get("ma7_reclaimed"), view.flat.get("daily_ma7_reclaimed"))
    daily_ma20_reclaimed = _boolish(view.daily_trend.get("ma20_reclaimed"), view.flat.get("daily_ma20_reclaimed"))
    if daily_bearish_stack and daily_ma7_reclaimed and not daily_ma20_reclaimed:
        _add(labels, reasons, "daily_ma7_only_reclaim", "daily trend is bearish and only MA7 is reclaimed")
        details["daily_ma7_only_reclaim"] = {
            "daily_bearish_stack": daily_bearish_stack,
            "daily_ma7_reclaimed": daily_ma7_reclaimed,
            "daily_ma20_reclaimed": daily_ma20_reclaimed,
        }

    one_m_pressure = bool(view.flat.get("one_m_upper_pressure_active")) or "1m upper pressure is active" in rationale_lower
    if view.side == "long" and one_m_pressure:
        _add(labels, reasons, "one_m_upper_pressure_active", "1M upper pressure is active for long entry")

    entry_indicator = _mapping(context.get("entry_indicator") or context.get("indicator_snapshot"))
    upper_wick_pct = _finite_float(entry_indicator.get("upper_wick_pct") or view.flat.get("upper_wick_pct"), None)
    candle_body_pct = _finite_float(entry_indicator.get("candle_body_pct") or view.flat.get("candle_body_pct"), 0.0)
    if view.side == "long" and upper_wick_pct is not None:
        body_threshold = max(candle_body_pct or 0.0, 0.01)
        if upper_wick_pct >= 0.2 and upper_wick_pct > body_threshold:
            _add(labels, reasons, "long_large_upper_wick", "long entry candle has obvious upper-wick rejection")
            details["long_large_upper_wick"] = {
                "upper_wick_pct": round(upper_wick_pct, 6),
                "candle_body_pct": round(candle_body_pct or 0.0, 6),
                "threshold_pct": 0.2,
            }
    elif view.side == "long" and context.get("entry_indicator_available") is False:
        unavailable_labels.append("long_large_upper_wick")

    early_15m_path = _mapping(context.get("early_15m_path"))
    if view.is_15m_scalp_style():
        n1_return = _finite_float(early_15m_path.get("n1_favorable_return_pct"), None)
        n2_return = _finite_float(early_15m_path.get("n2_favorable_return_pct"), None)
        if n1_return is not None and n2_return is not None:
            max_follow = max(n1_return, n2_return)
            if max_follow <= 0.0:
                _add(
                    labels,
                    reasons,
                    "m15_scalp_no_follow_1_2",
                    "15M scalp-style entry had no favorable close after the first 1-2 candles",
                )
                details["m15_scalp_no_follow_1_2"] = {
                    "n1_favorable_return_pct": round(n1_return, 6),
                    "n2_favorable_return_pct": round(n2_return, 6),
                    "max_follow_pct": round(max_follow, 6),
                    "definition": "max(n1,n2) <= 0 by side-normalized close return",
                }
        elif context.get("early_15m_path_available") is False:
            unavailable_labels.append("m15_scalp_no_follow_1_2")

    early_health = context.get("early_health") if isinstance(context.get("early_health"), Mapping) else {}
    if early_health.get("deteriorated"):
        _add(labels, reasons, "early_health_deterioration", "post-entry early health deteriorated")
        details["early_health_deterioration"] = dict(early_health)
    elif context.get("early_health_available") is False:
        unavailable_labels.append("early_health_deterioration")

    for label in labels:
        tags.append(LABEL_DEFAULT_ACTIONS[label])
        tags.append(LABEL_PHASES[label])

    return LossRiskResult(
        risk_labels=labels,
        reasons=reasons,
        tags=_dedupe(tags),
        details=details,
        unavailable_labels=_dedupe(unavailable_labels),
    )


def selected_level_name(record: Mapping[str, Any]) -> str:
    return _RecordView(record).selected_level


def entry_side(record: Mapping[str, Any]) -> str:
    return _RecordView(record).side


class _RecordView:
    def __init__(self, record: Mapping[str, Any]) -> None:
        self.record = record
        self.intent = _mapping(record.get("intent")) or record
        self.flat = record
        self.risk_plan = _mapping(self.intent.get("risk_plan")) or _mapping(record.get("risk_plan"))
        self.entry_context = _mapping(self.risk_plan.get("entry_context"))
        self.structure_scan = _mapping(self.risk_plan.get("structure_entry_scan")) or _mapping(self.entry_context.get("structure_entry_scan"))
        self.ma20_trend = _mapping(_path(self.risk_plan, "entry_indicators", "ma20_trend")) or _mapping(self.entry_context.get("ma20_trend"))
        self.h4_right_side = _mapping(self.entry_context.get("h4_right_side"))
        self.daily_trend = _mapping(self.entry_context.get("daily_ma7_trend"))
        self.rationale = str(self.intent.get("rationale") or record.get("rationale") or "")
        self.action = str(self.intent.get("action") or record.get("action") or "")
        self.symbol = str(self.intent.get("symbol") or record.get("symbol") or "")
        self.side = _resolve_side(self.action, self.intent.get("side") or record.get("side"))
        self.selected_level = self._selected_level_name()
        self.selected_tf = self._selected_level_tf()
        self.horizon = str(self.risk_plan.get("horizon") or record.get("horizon") or "")
        self.entry_timeframe = str(self.risk_plan.get("entry_timeframe") or record.get("entry_timeframe") or "")

    def h4_extension_pct(self) -> float | None:
        flat_value = _finite_float(self.flat.get("h4_price_distance_pct"), None)
        if flat_value is not None:
            return abs(flat_value)
        h4_distance = _finite_float(self.h4_right_side.get("price_distance_pct"), None)
        if h4_distance is not None:
            return abs(h4_distance)
        price = _finite_float(self.intent.get("entry_price") or self.flat.get("entry_price"), None)
        ma20 = _finite_float(self.h4_right_side.get("ma20"), None)
        if price is None or ma20 is None or ma20 <= 0:
            return None
        return abs(price - ma20) / ma20 * 100.0

    def _selected_level_name(self) -> str:
        candidates = [
            _mapping(self.entry_context.get("selected_entry_level")).get("name"),
            _mapping(self.risk_plan.get("stop_anchor")).get("name"),
            self.flat.get("selected_level"),
        ]
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return ""

    def _selected_level_tf(self) -> str:
        candidates = [
            _mapping(self.entry_context.get("selected_entry_level")).get("timeframe"),
            self.flat.get("selected_tf"),
        ]
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return ""

    def is_15m_scalp_style(self) -> bool:
        tokens = {
            self.entry_timeframe.lower(),
            self.selected_tf.lower(),
            self.horizon.lower(),
        }
        return "15m" in tokens or any("15m" in token or "scalp" in token for token in tokens)


def _repeat_failure_detail(view: _RecordView, context: Mapping[str, Any]) -> dict[str, Any]:
    current_opened_at = _parse_dt(
        context.get("opened_at")
        or view.flat.get("opened_at")
        or view.intent.get("generated_at")
        or view.flat.get("generated_at")
        or view.flat.get("submitted_at")
    )
    selected_level = view.selected_level
    counts = {"12h": 0, "24h": 0, "48h": 0}
    recent_matches: list[dict[str, Any]] = []
    for failure in context.get("recent_failures") or []:
        if not isinstance(failure, Mapping):
            continue
        if str(failure.get("symbol") or "") != view.symbol:
            continue
        if str(failure.get("side") or "") != view.side:
            continue
        failure_time = _parse_dt(failure.get("closed_at") or failure.get("timestamp"))
        if current_opened_at is None or failure_time is None or failure_time > current_opened_at:
            continue
        hours = (current_opened_at - failure_time).total_seconds() / 3600.0
        if hours < 0 or hours > 48.0:
            continue
        same_level = selected_level and str(failure.get("selected_level") or "") == selected_level
        if hours <= 12.0:
            counts["12h"] += 1
        if hours <= 24.0:
            counts["24h"] += 1
        counts["48h"] += 1
        if same_level:
            recent_matches.append(
                {
                    "closed_at": failure.get("closed_at"),
                    "hours_ago": round(hours, 4),
                    "selected_level": selected_level,
                    "pnl_usdt": failure.get("pnl_usdt"),
                }
            )
    return {
        "repeat_selected_level_failure": bool(recent_matches),
        "symbol_side_failures": counts,
        "recent_same_level_failures": recent_matches,
    }


def _add(labels: list[str], reasons: list[str], label: str, reason: str) -> None:
    if label not in labels:
        labels.append(label)
    if reason not in reasons:
        reasons.append(reason)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _path(value: Mapping[str, Any], *parts: str) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _finite_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        output = float(value)
    except Exception:
        return default
    if output != output or output in (float("inf"), float("-inf")):
        return default
    return output


def _boolish(*values: Any) -> bool:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return False


def _resolve_side(action: str, side: Any) -> str:
    action = str(action or "").lower()
    if "long" in action:
        return "long"
    if "short" in action:
        return "short"
    raw = str(side or "").lower()
    if raw in {"long", "short"}:
        return raw
    if raw == "buy":
        return "long"
    if raw == "sell":
        return "short"
    return raw


def _parse_dt(value: Any) -> datetime | None:
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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
