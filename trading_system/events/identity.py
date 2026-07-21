from __future__ import annotations

import hashlib
from typing import Any, Mapping


def trade_side(row: Mapping[str, Any]) -> str:
    action = str(row.get("action") or "").strip().lower()
    if action in {"open_long", "close_long"}:
        return "long"
    if action in {"open_short", "close_short"}:
        return "short"
    side = str(row.get("side") or row.get("position_side") or "").strip().lower()
    return side if side in {"long", "short"} else "unknown"


def extract_trade_id(row: Mapping[str, Any] | None) -> str:
    if not isinstance(row, Mapping):
        return ""
    direct = str(row.get("trade_id") or "").strip()
    if direct:
        return direct[:160]
    risk_plan = row.get("risk_plan") if isinstance(row.get("risk_plan"), Mapping) else {}
    nested = str(risk_plan.get("trade_id") or "").strip()
    if nested:
        return nested[:160]
    intent = row.get("intent") if isinstance(row.get("intent"), Mapping) else None
    return extract_trade_id(intent)


def derive_trade_id(
    *,
    account_id: str,
    symbol: str,
    side: str,
    seed: str,
) -> str:
    identity = "|".join(
        (
            str(account_id or "default").strip().lower(),
            str(symbol or "UNKNOWN").strip().upper(),
            str(side or "unknown").strip().lower(),
            str(seed or "unknown").strip(),
        )
    )
    return "trd_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
