"""Bounded, causal features derived from Hyperdash Telegram liquidation flow.

This module is intentionally independent from the execution client. It only
reads local bridge/event files and returns normalized intelligence features.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BRIDGE_FILE = Path("data/trading_system/skill_bridges/telegram_whale_crypto.json")
DEFAULT_EVENT_DIR = Path("data/trading_system/telegram_whale/events")
IMMINENT_FLOW_WEIGHT = 0.25
ACTIVITY_NOTIONAL_SCALE_USDT = 10_000_000.0

TELEGRAM_WHALE_FEATURE_NAMES = (
    "telegram_whale_actual_score_60m",
    "telegram_whale_imminent_score_60m",
    "telegram_whale_combined_score_60m",
    "telegram_whale_combined_score_24h",
    "telegram_whale_activity_score_24h",
    "telegram_whale_freshness_score",
)


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper().replace("_", "")
    if symbol.endswith("USDT"):
        symbol = symbol[:-4]
    return symbol


def _bounded_ratio(signed_value: float, total_value: float) -> float:
    if total_value <= 0:
        return 0.0
    return max(-1.0, min(1.0, signed_value / total_value))


def _event_amount(event: dict[str, Any]) -> float:
    try:
        return max(0.0, float(event.get("amount_usd") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _summarize_window(
    events: Iterable[dict[str, Any]],
    *,
    symbol: str,
    as_of: datetime,
    minutes: int,
) -> dict[str, Any]:
    cutoff = as_of - timedelta(minutes=max(1, int(minutes)))
    selected: list[tuple[datetime, dict[str, Any]]] = []
    for event in events:
        if normalize_symbol(event.get("symbol")) != symbol:
            continue
        posted_at = parse_datetime(event.get("posted_at"))
        if posted_at is None or posted_at > as_of or posted_at < cutoff:
            continue
        selected.append((posted_at, event))
    selected.sort(key=lambda item: item[0])

    liquidated_long = 0.0
    liquidated_short = 0.0
    latest_imminent: dict[str, tuple[datetime, float]] = {}
    for posted_at, event in selected:
        side = str(event.get("position_side") or "").lower()
        if side not in {"long", "short"}:
            continue
        amount = _event_amount(event)
        event_type = str(event.get("event_type") or "")
        if event_type == "liquidation":
            if side == "long":
                liquidated_long += amount
            else:
                liquidated_short += amount
        elif event_type == "imminent_liquidation":
            latest_imminent[side] = (posted_at, amount)

    imminent_long = latest_imminent.get("long", (as_of, 0.0))[1]
    imminent_short = latest_imminent.get("short", (as_of, 0.0))[1]
    actual_total = liquidated_long + liquidated_short
    actual_signed = liquidated_short - liquidated_long
    imminent_total = imminent_long + imminent_short
    imminent_signed = imminent_short - imminent_long
    effective_total = actual_total + IMMINENT_FLOW_WEIGHT * imminent_total
    effective_signed = actual_signed + IMMINENT_FLOW_WEIGHT * imminent_signed
    latest_at = selected[-1][0] if selected else None
    return {
        "minutes": int(minutes),
        "event_count": len(selected),
        "liquidated_long_usd": round(liquidated_long, 8),
        "liquidated_short_usd": round(liquidated_short, 8),
        "imminent_long_usd": round(imminent_long, 8),
        "imminent_short_usd": round(imminent_short, 8),
        "actual_notional_usdt": round(actual_total, 8),
        "imminent_notional_usdt": round(imminent_total, 8),
        "effective_notional_usdt": round(effective_total, 8),
        "actual_score": round(_bounded_ratio(actual_signed, actual_total), 8),
        "imminent_score": round(_bounded_ratio(imminent_signed, imminent_total), 8),
        "combined_score": round(_bounded_ratio(effective_signed, effective_total), 8),
        "latest_event_at": latest_at.isoformat() if latest_at else "",
    }


def build_feature_snapshot(
    events: Iterable[dict[str, Any]],
    *,
    symbol: str,
    as_of: datetime | None = None,
    max_event_age_minutes: float = 360.0,
) -> dict[str, Any]:
    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    base_symbol = normalize_symbol(symbol)
    event_rows = [dict(event) for event in events if isinstance(event, dict)]
    window_60m = _summarize_window(event_rows, symbol=base_symbol, as_of=now, minutes=60)
    window_24h = _summarize_window(event_rows, symbol=base_symbol, as_of=now, minutes=1440)

    latest_at = parse_datetime(window_24h.get("latest_event_at"))
    age_minutes = (now - latest_at).total_seconds() / 60.0 if latest_at else math.inf
    max_age = max(1.0, float(max_event_age_minutes))
    freshness = max(0.0, 1.0 - age_minutes / max_age) if math.isfinite(age_minutes) else 0.0
    activity = min(
        1.0,
        float(window_24h.get("effective_notional_usdt") or 0.0) / ACTIVITY_NOTIONAL_SCALE_USDT,
    )
    fallback_window = window_60m if int(window_60m.get("event_count") or 0) > 0 else window_24h
    fallback_score = float(fallback_window.get("combined_score") or 0.0)
    fallback_notional = float(fallback_window.get("effective_notional_usdt") or 0.0)
    direction = "long" if fallback_score > 0 else "short" if fallback_score < 0 else "neutral"
    feature_values = {
        "telegram_whale_actual_score_60m": float(window_60m.get("actual_score") or 0.0),
        "telegram_whale_imminent_score_60m": float(window_60m.get("imminent_score") or 0.0),
        "telegram_whale_combined_score_60m": float(window_60m.get("combined_score") or 0.0),
        "telegram_whale_combined_score_24h": float(window_24h.get("combined_score") or 0.0),
        "telegram_whale_activity_score_24h": round(activity, 8),
        "telegram_whale_freshness_score": round(freshness, 8),
    }
    return {
        "available": bool(base_symbol and int(window_24h.get("event_count") or 0) > 0),
        "symbol": base_symbol,
        "as_of": now.isoformat(),
        "latest_event_at": latest_at.isoformat() if latest_at else "",
        "event_age_minutes": round(age_minutes, 4) if math.isfinite(age_minutes) else None,
        "fresh": bool(latest_at and age_minutes <= max_age),
        "direction": direction,
        "fallback_score": round(fallback_score, 8),
        "fallback_notional_usdt": round(fallback_notional, 8),
        "fallback_window_minutes": int(fallback_window.get("minutes") or 0),
        "windows": {"60m": window_60m, "24h": window_24h},
        "features": feature_values,
        "policy": "auxiliary_training_and_near_tie_fallback_only",
    }


def apply_snapshot_to_features(features: Any, snapshot: dict[str, Any]) -> None:
    values = snapshot.get("features") if isinstance(snapshot, dict) else {}
    for name in TELEGRAM_WHALE_FEATURE_NAMES:
        if hasattr(features, name):
            try:
                setattr(features, name, float((values or {}).get(name) or 0.0))
            except (TypeError, ValueError):
                setattr(features, name, 0.0)


def compact_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep execution summaries small while preserving every fallback input."""
    if not isinstance(snapshot, dict):
        return {}
    keys = (
        "available",
        "symbol",
        "as_of",
        "latest_event_at",
        "event_age_minutes",
        "fresh",
        "direction",
        "fallback_score",
        "fallback_notional_usdt",
        "fallback_window_minutes",
        "bridge_generated_at",
        "bridge_age_seconds",
        "bridge_fresh",
        "unavailable_reason",
        "policy",
    )
    compact = {key: snapshot.get(key) for key in keys if key in snapshot}
    compact["features"] = {
        name: float((snapshot.get("features") or {}).get(name) or 0.0)
        for name in TELEGRAM_WHALE_FEATURE_NAMES
    }
    return compact


@dataclass(slots=True)
class TelegramWhaleEventHistory:
    events: list[dict[str, Any]]

    @classmethod
    def from_event_dir(cls, event_dir: Path, *, lookback_days: int = 45) -> "TelegramWhaleEventHistory":
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max(1, int(lookback_days)))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        if event_dir.exists():
            for path in sorted(event_dir.glob("*.jsonl")):
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                for line in lines:
                    try:
                        event = json.loads(line)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    posted_at = parse_datetime(event.get("posted_at"))
                    if posted_at is None or posted_at < cutoff:
                        continue
                    event_id = str(event.get("event_id") or f"{path.name}:{len(rows)}")
                    if event_id in seen:
                        continue
                    seen.add(event_id)
                    rows.append(event)
        rows.sort(key=lambda item: parse_datetime(item.get("posted_at")) or datetime.min.replace(tzinfo=timezone.utc))
        return cls(rows)

    def snapshot(
        self,
        symbol: str,
        *,
        as_of: datetime | None = None,
        max_event_age_minutes: float = 360.0,
    ) -> dict[str, Any]:
        return build_feature_snapshot(
            self.events,
            symbol=symbol,
            as_of=as_of,
            max_event_age_minutes=max_event_age_minutes,
        )


class TelegramWhaleBridgeReader:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._mtime_ns = -1
        self._payload: dict[str, Any] = {}

    def _load(self) -> dict[str, Any]:
        try:
            stat = self.path.stat()
        except OSError:
            self._payload = {}
            self._mtime_ns = -1
            return {}
        if stat.st_mtime_ns == self._mtime_ns:
            return self._payload
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
        self._payload = payload if isinstance(payload, dict) else {}
        self._mtime_ns = stat.st_mtime_ns
        return self._payload

    def snapshot(
        self,
        symbol: str,
        *,
        as_of: datetime | None = None,
        max_bridge_age_seconds: float = 180.0,
        max_event_age_minutes: float = 360.0,
    ) -> dict[str, Any]:
        now = as_of or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)
        payload = self._load()
        generated_at = parse_datetime(payload.get("generated_at"))
        bridge_age = (now - generated_at).total_seconds() if generated_at else math.inf
        snapshot = build_feature_snapshot(
            payload.get("recent_events") or [],
            symbol=symbol,
            as_of=now,
            max_event_age_minutes=max_event_age_minutes,
        )
        bridge_fresh = bool(generated_at and bridge_age <= max(1.0, float(max_bridge_age_seconds)))
        snapshot.update(
            {
                "bridge_generated_at": generated_at.isoformat() if generated_at else "",
                "bridge_age_seconds": round(bridge_age, 4) if math.isfinite(bridge_age) else None,
                "bridge_fresh": bridge_fresh,
                "available": bool(snapshot.get("available") and bridge_fresh),
            }
        )
        if not bridge_fresh:
            snapshot["unavailable_reason"] = "bridge_stale_or_missing"
        return snapshot
