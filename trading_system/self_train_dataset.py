from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_H4_DEFAULTS = {
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
    "h4_price_location": "unknown",
    "h4_pattern_type": "unknown",
}

_HF_QUALITY_KEYS = (
    "hf_signal_count_total",
    "hf_win_rate_3bar",
    "hf_avg_return_3bar",
    "hf_profit_factor_3bar",
    "hf_max_adverse_excursion_pct",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _extract_h4_structure(source: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source, dict):
        return dict(_H4_DEFAULTS)
    result = dict(_H4_DEFAULTS)
    for key, default in _H4_DEFAULTS.items():
        if key not in source:
            continue
        value = source.get(key)
        if isinstance(default, bool):
            result[key] = _bool(value)
        elif isinstance(default, (int, float)):
            result[key] = _float(value, float(default))
        else:
            result[key] = str(value or default)
    return result


def _extract_hf_quality(source: dict[str, Any] | None, *, side: str = "") -> dict[str, float]:
    if not isinstance(source, dict):
        return {key: 0.0 for key in _HF_QUALITY_KEYS}
    by_side = source.get("by_side")
    if isinstance(by_side, dict):
        side_payload = by_side.get(str(side or "").lower())
        if not isinstance(side_payload, dict):
            return {key: 0.0 for key in _HF_QUALITY_KEYS}
        source = side_payload
    return {key: _float(source.get(key)) for key in _HF_QUALITY_KEYS}


def _load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            key = str(payload.get("sample_key") or "")
            if key:
                keys.add(key)
    return keys


def append_new_samples(path: str | Path, samples: list[dict[str, Any]]) -> int:
    path = Path(path)
    if not samples:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_keys(path)
    added = 0
    with path.open("a", encoding="utf-8") as fh:
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            key = str(sample.get("sample_key") or "")
            if key and key in existing:
                continue
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if key:
                existing.add(key)
            added += 1
    return added


def collect_decision_samples(
    report: dict[str, Any],
    *,
    captured_at: str | None = None,
    hf_quality_lookup: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    captured_at = captured_at or _now_iso()
    basket = report.get("candidate_basket") if isinstance(report, dict) else []
    if not isinstance(basket, list):
        return []
    entries = report.get("paper_rebalance", {}).get("entries", []) if isinstance(report.get("paper_rebalance"), dict) else []
    entered = {str(item.get("symbol") or "").upper() for item in entries if isinstance(item, dict)}
    hf_quality_lookup = hf_quality_lookup if isinstance(hf_quality_lookup, dict) else {}
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(item for item in basket if isinstance(item, dict)):
        symbol = str(row.get("symbol") or "").upper()
        selected = symbol in entered or str(row.get("execution_status") or "") in {"entered", "executable_now"}
        h4 = _extract_h4_structure(row)
        hf = _extract_hf_quality(hf_quality_lookup.get(symbol), side="long")
        sample = {
            "sample_key": f"decision|{captured_at}|{index}|{symbol}",
            "captured_at": captured_at,
            "sample_type": "decision",
            "symbol": symbol,
            "rank": index + 1,
            "score": _float(row.get("score")),
            "setup_type": row.get("setup_type") or "",
            "execution_status": row.get("execution_status") or "",
            "decision_label": "entered" if selected else "blocked",
            "selected_for_entry": bool(selected),
            "sentiment_score": _float(row.get("sentiment_score")),
            "sentiment_mentions": _float(row.get("sentiment_mentions")),
            "taker_buy_sell_ratio": _float(row.get("taker_buy_sell_ratio")),
            "oi_change_pct_4h": _float(row.get("oi_change_pct_4h")),
            "price_change_pct_24h": _float(row.get("price_change_pct_24h")),
        }
        sample.update(h4)
        sample.update(hf)
        samples.append(sample)
    return samples


def collect_closed_trade_samples(
    state: dict[str, Any],
    report: dict[str, Any],
    *,
    captured_at: str | None = None,
    hf_quality_lookup: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    captured_at = captured_at or _now_iso()
    exits = report.get("exits") if isinstance(report, dict) else []
    if not isinstance(exits, list):
        exits = []
    hf_quality_lookup = hf_quality_lookup if isinstance(hf_quality_lookup, dict) else {}
    journal = state.get("journal") if isinstance(state, dict) else []
    setup_by_symbol: dict[str, str] = {}
    if isinstance(journal, list):
        for item in journal:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").upper()
            setup_type = str(item.get("setup_type") or "")
            if symbol and setup_type:
                setup_by_symbol[symbol] = setup_type
    basket = report.get("candidate_basket") if isinstance(report, dict) else []
    if isinstance(basket, list):
        for item in basket:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").upper()
            setup_type = str(item.get("setup_type") or "")
            if symbol and setup_type:
                setup_by_symbol.setdefault(symbol, setup_type)
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(item for item in exits if isinstance(item, dict)):
        symbol = str(row.get("symbol") or "").upper()
        pnl_pct = _float(row.get("pnl_pct") or row.get("return_pct") or row.get("realized_pnl_pct"))
        entry_h4 = row.get("entry_h4_structure") if isinstance(row.get("entry_h4_structure"), dict) else None
        setup_type = str(row.get("setup_type") or "")
        if (not setup_type or setup_type == "trend") and setup_by_symbol.get(symbol):
            setup_type = setup_by_symbol[symbol]
        sample = {
            "sample_key": f"closed|{captured_at}|{index}|{symbol}|{row.get('rationale') or row.get('reason') or ''}",
            "captured_at": captured_at,
            "sample_type": "closed_trade",
            "symbol": symbol,
            "is_win": bool(row.get("is_win")) or pnl_pct > 0,
            "pnl_pct": pnl_pct,
            "pnl_usdt": _float(row.get("pnl_usdt")),
            "reason_family": row.get("reason_family") or row.get("rationale") or row.get("reason") or "",
            "setup_type": setup_type,
        }
        sample.update(_extract_h4_structure(entry_h4))
        sample.update(_extract_hf_quality(hf_quality_lookup.get(symbol), side=str(row.get("side") or "long")))
        samples.append(sample)
    return samples, {"closed_exit_count": len(exits)}


def collect_distill_process_samples(
    state: dict[str, Any],
    report: dict[str, Any],
    *,
    captured_at: str | None = None,
) -> list[dict[str, Any]]:
    return []


def build_analysis_maturity_summary(
    decision_dataset_file: str | Path,
    closed_trade_dataset_file: str | Path,
    state_file: str | Path,
) -> dict[str, Any]:
    decision_file = Path(decision_dataset_file)
    closed_file = Path(closed_trade_dataset_file)
    return {
        "status": "ok",
        "mode": "lightweight_snapshot_fallback",
        "decision_dataset_file": str(decision_file),
        "closed_trade_dataset_file": str(closed_file),
        "state_file": str(state_file),
        "decision_rows": sum(1 for line in decision_file.read_text(encoding="utf-8").splitlines() if line.strip()) if decision_file.exists() else 0,
        "closed_rows": sum(1 for line in closed_file.read_text(encoding="utf-8").splitlines() if line.strip()) if closed_file.exists() else 0,
    }
