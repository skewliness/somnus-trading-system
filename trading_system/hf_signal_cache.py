from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean


_cache: dict = {"data": {}, "mtime": 0.0, "loaded_at": 0.0, "path": ""}


def load_hf_symbol_quality_cache(
    entries_file: Path,
    report_file: Path | None = None,
    ttl_seconds: int = 10800,
) -> dict[str, dict]:
    """Load and aggregate HF backtest entries into per-symbol quality metrics.

    Uses in-memory cache with TTL. Only reloads when file mtime changes or TTL expires.
    Returns empty dict if file doesn't exist or is empty.
    """
    if not entries_file.exists():
        return {}

    now = time.time()
    file_mtime = entries_file.stat().st_mtime
    resolved = str(entries_file.resolve())
    if (
        _cache["data"]
        and _cache["path"] == resolved
        and _cache["mtime"] == file_mtime
        and now - _cache["loaded_at"] < ttl_seconds
    ):
        return _cache["data"]

    entries = _load_entries(entries_file)
    if not entries:
        return {}

    result = _aggregate_per_symbol(entries)
    _cache["data"] = result
    _cache["mtime"] = file_mtime
    _cache["loaded_at"] = now
    _cache["path"] = resolved
    return result


def _load_entries(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _aggregate_per_symbol(entries: list[dict]) -> dict[str, dict]:
    """Aggregate HF entries into per-symbol quality metrics.

    Returns per-symbol dict with overall metrics plus by_side breakdown for
    directional filtering (long/short may have very different win rates).
    """
    by_symbol: dict[str, list[dict]] = {}
    for e in entries:
        symbol = str(e.get("symbol") or "")
        if symbol:
            by_symbol.setdefault(symbol, []).append(e)

    result: dict[str, dict] = {}
    for symbol, rows in by_symbol.items():
        n = len(rows)
        if n == 0:
            continue

        metrics = _compute_quality_metrics(rows)

        # Per-side breakdown
        by_side: dict[str, dict] = {}
        side_groups: dict[str, list[dict]] = {}
        for r in rows:
            side = str(r.get("side") or "").lower()
            if side in ("long", "short"):
                side_groups.setdefault(side, []).append(r)
        for side, side_rows in side_groups.items():
            by_side[side] = _compute_quality_metrics(side_rows)

        metrics["by_side"] = by_side
        result[symbol] = metrics

    return result


def _compute_quality_metrics(rows: list[dict]) -> dict:
    """Compute HF quality metrics for a group of entries."""
    n = len(rows)
    if n == 0:
        return {"hf_signal_count_total": 0, "hf_win_rate_3bar": 0.0, "hf_avg_mfe_12bar": 0.0, "hf_avg_risk_reward": 0.0, "hf_best_signal_win_rate": 0.0, "hf_momentum_edge": 0.0, "hf_oi_surge_edge": 0.0}

    win_3 = [1 for r in rows if r.get("is_profitable_3bar")]
    mfes = [float(r.get("mfe_12bar") or 0) for r in rows]
    rrs = [float(r.get("risk_reward_12bar") or 0) for r in rows]

    signal_edges: dict[str, list[float]] = {}
    signal_wins: dict[str, list[int]] = {}
    for r in rows:
        ret3 = float(r.get("return_3bar") or 0)
        profitable = 1 if r.get("is_profitable_3bar") else 0
        for sig in (r.get("triggered_signals") or []):
            signal_edges.setdefault(sig, []).append(ret3)
            signal_wins.setdefault(sig, []).append(profitable)

    per_signal_win_rates = {
        sig: sum(wins) / len(wins) for sig, wins in signal_wins.items() if wins
    }
    best_signal_wr = max(per_signal_win_rates.values()) if per_signal_win_rates else 0.0

    return {
        "hf_signal_count_total": n,
        "hf_win_rate_3bar": round(len(win_3) / n, 4),
        "hf_avg_mfe_12bar": round(mean(mfes), 6) if mfes else 0.0,
        "hf_avg_risk_reward": round(mean(rrs), 4) if rrs else 0.0,
        "hf_best_signal_win_rate": round(best_signal_wr, 4),
        "hf_momentum_edge": round(
            mean(signal_edges.get("momentum", [0.0])), 6
        ),
        "hf_oi_surge_edge": round(
            mean(signal_edges.get("oi_surge", [0.0])), 6
        ),
    }
