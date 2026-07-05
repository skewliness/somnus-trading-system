from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


BINANCE_SIGNAL_SEQUENCES = {"binance_first", "binance_only", "single_source"}
OKX_SIGNAL_SEQUENCES = {"okx_first", "okx_only", "onchain_first"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
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


def _row_float(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row.get(key) is not None:
            try:
                return float(row.get(key) or 0)
            except (TypeError, ValueError):
                return default
    return default


def _row_int(row: dict, *keys: str, default: int = 0) -> int:
    return int(_row_float(row, *keys, default=float(default)))


def _signal_sequence(row: dict) -> str:
    return str(
        row.get("signal_sequence")
        or row.get("distill_signal_sequence")
        or row.get("sentiment_signal_sequence")
        or "single_source"
    ).strip() or "single_source"


def _engine_for_row(row: dict) -> str:
    explicit = str(row.get("engine") or row.get("venue") or row.get("exchange") or "").lower()
    if explicit in {"binance", "okx"}:
        return explicit
    sequence = _signal_sequence(row).lower()
    if sequence in OKX_SIGNAL_SEQUENCES or sequence.startswith("okx") or sequence.startswith("onchain"):
        return "okx"
    return "binance"


def _is_win(row: dict) -> bool:
    if "is_win" in row:
        return bool(row.get("is_win"))
    return _row_float(row, "pnl_pct") > 0


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return abs(max_dd)


def _trial_metrics(selected: list[dict]) -> dict[str, Any]:
    pnls = [_row_float(row, "pnl_pct") for row in selected]
    wins = sum(1 for row in selected if _is_win(row))
    total = len(selected)
    avg_pnl = mean(pnls) if pnls else 0.0
    win_rate = wins / total if total else 0.0
    sharpe_like = (avg_pnl / pstdev(pnls)) if len(pnls) > 1 and pstdev(pnls) > 0 else 0.0
    max_dd = _max_drawdown(pnls)
    return {
        "sample_count": total,
        "wins": wins,
        "win_rate": round(win_rate, 6),
        "win_rate_pct": round(win_rate * 100.0, 2),
        "avg_pnl_pct": round(avg_pnl, 6),
        "total_pnl_pct": round(sum(pnls), 6),
        "max_drawdown_pct": round(max_dd, 6),
        "sharpe_like": round(sharpe_like, 6),
    }


def _select_rows(rows: list[dict], params: dict[str, Any], engine: str | None = None) -> list[dict]:
    selected = []
    for row in rows:
        if engine is not None and _engine_for_row(row) != engine:
            continue
        if _row_int(row, "mentions", "sentiment_mentions") < int(params["mention_threshold"]):
            continue
        if _row_float(row, "hotness_score", "sentiment_score") < float(params["hotness_threshold"]):
            continue
        if _row_float(row, "score") and _row_float(row, "score") < float(params["entry_threshold"]):
            continue
        setup_type = str(row.get("setup_type") or "trend")
        if setup_type == "oi_divergence_long_candidate":
            if _row_float(row, "oi_price_divergence_score") < float(params.get("oi_divergence_min_divergence_score", 0)):
                continue
            if _row_int(row, "mentions", "sentiment_mentions") < int(params.get("oi_divergence_min_sentiment_mentions", 0)):
                continue
        selected.append(row)
    return selected


def _score_trial(rows: list[dict], entry_threshold: float, mention_threshold: int, hotness_threshold: float) -> float:
    selected = _select_rows(
        rows,
        {
            "entry_threshold": entry_threshold,
            "mention_threshold": mention_threshold,
            "hotness_threshold": hotness_threshold,
            "oi_divergence_min_divergence_score": 0,
            "oi_divergence_min_sentiment_mentions": 0,
        },
    )
    if not selected:
        return -1e9
    metrics = _trial_metrics(selected)
    threshold_penalty = abs(entry_threshold - 60.0) * 0.002
    return round(metrics["win_rate_pct"] + (metrics["avg_pnl_pct"] * 100.0) - threshold_penalty, 6)


def _candidate_grid(engine: str | None = None) -> list[dict[str, Any]]:
    if engine == "okx":
        return [
            {"entry_threshold": 54.0, "mention_threshold": 1, "hotness_threshold": 45.0, "oi_divergence_min_divergence_score": 0.0, "oi_divergence_min_sentiment_mentions": 1, "probe_size_multiplier": 0.35, "cooldown_minutes": 90},
            {"entry_threshold": 58.0, "mention_threshold": 2, "hotness_threshold": 52.0, "oi_divergence_min_divergence_score": 90.0, "oi_divergence_min_sentiment_mentions": 2, "probe_size_multiplier": 0.30, "cooldown_minutes": 120},
            {"entry_threshold": 62.0, "mention_threshold": 3, "hotness_threshold": 58.0, "oi_divergence_min_divergence_score": 120.0, "oi_divergence_min_sentiment_mentions": 3, "probe_size_multiplier": 0.25, "cooldown_minutes": 180},
        ]
    if engine == "binance":
        return [
            {"entry_threshold": 60.0, "mention_threshold": 8, "hotness_threshold": 58.0, "oi_divergence_min_divergence_score": 160.0, "oi_divergence_min_sentiment_mentions": 15, "probe_size_multiplier": 0.40, "cooldown_minutes": 180},
            {"entry_threshold": 64.0, "mention_threshold": 12, "hotness_threshold": 62.0, "oi_divergence_min_divergence_score": 180.0, "oi_divergence_min_sentiment_mentions": 18, "probe_size_multiplier": 0.35, "cooldown_minutes": 240},
            {"entry_threshold": 68.0, "mention_threshold": 18, "hotness_threshold": 68.0, "oi_divergence_min_divergence_score": 200.0, "oi_divergence_min_sentiment_mentions": 20, "probe_size_multiplier": 0.30, "cooldown_minutes": 360},
        ]
    return [
        {"entry_threshold": 52.0, "mention_threshold": 2, "hotness_threshold": 45.0},
        {"entry_threshold": 56.0, "mention_threshold": 3, "hotness_threshold": 50.0},
        {"entry_threshold": 60.0, "mention_threshold": 4, "hotness_threshold": 55.0},
        {"entry_threshold": 64.0, "mention_threshold": 5, "hotness_threshold": 60.0},
        {"entry_threshold": 68.0, "mention_threshold": 6, "hotness_threshold": 65.0},
        {"entry_threshold": 72.0, "mention_threshold": 8, "hotness_threshold": 70.0},
    ]


def _run_search_for_engine(rows: list[dict], engine: str, iterations: int) -> dict[str, Any]:
    engine_rows = [row for row in rows if _engine_for_row(row) == engine]
    trials = []
    for idx in range(max(1, iterations)):
        params = _candidate_grid(engine)[idx % len(_candidate_grid(engine))]
        selected = _select_rows(engine_rows, params, engine=engine)
        metrics = _trial_metrics(selected)
        objective = -1e9 if not selected else round(
            metrics["win_rate_pct"]
            + (metrics["avg_pnl_pct"] * 100.0)
            + (metrics["sharpe_like"] * 2.0)
            - (metrics["max_drawdown_pct"] * 50.0)
            - (0.05 * abs(float(params["entry_threshold"]) - (62.0 if engine == "binance" else 58.0))),
            6,
        )
        trials.append({"trial": idx + 1, "engine": engine, "params": params, "objective": objective, "metrics": metrics})
    best_trial = max(trials, key=lambda item: item["objective"])
    return {
        "engine": engine,
        "status": "ok" if engine_rows else "missing",
        "sample_count": len(engine_rows),
        "signal_sequences": sorted({_signal_sequence(row) for row in engine_rows}),
        "trials": trials,
        "best_trial": best_trial,
        "best_params": best_trial["params"] if engine_rows else {},
    }


def run_dual_engine_parameter_search(dataset_path: Path, iterations: int = 12, output_path: Path | None = None) -> dict:
    rows = _load_jsonl(dataset_path)
    if not rows:
        payload = {"status": "missing", "dataset_file": str(dataset_path), "engines": {}, "best_params": {}}
    else:
        engines = {
            "binance": _run_search_for_engine(rows, "binance", iterations),
            "okx": _run_search_for_engine(rows, "okx", iterations),
        }
        payload = {
            "status": "ok",
            "dataset_file": str(dataset_path),
            "optimizer": "dual-engine-grid-v1",
            "engines": engines,
            "best_params": {engine: result["best_params"] for engine, result in engines.items()},
        }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_bayesian_parameter_search(dataset_path: Path, iterations: int = 12, output_path: Path | None = None) -> dict:
    rows = _load_jsonl(dataset_path)
    if not rows:
        payload = {"status": "missing", "dataset_file": str(dataset_path), "trials": [], "best_params": {}, "best_trial": {}}
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    trials = []
    candidate_grid = _candidate_grid()
    for idx in range(max(1, iterations)):
        params = candidate_grid[idx % len(candidate_grid)]
        objective = _score_trial(rows, **params)
        trials.append({"trial": idx + 1, "params": params, "objective": objective})
    best_trial = max(trials, key=lambda item: item["objective"])
    dual_engine = run_dual_engine_parameter_search(dataset_path, iterations=iterations)
    payload = {
        "status": "ok",
        "dataset_file": str(dataset_path),
        "trials": trials,
        "best_trial": best_trial,
        "best_params": best_trial["params"],
        "dual_engine_best_params": dual_engine.get("best_params", {}),
        "engine_summaries": {
            engine: {
                "status": result.get("status"),
                "sample_count": result.get("sample_count"),
                "best_params": result.get("best_params"),
                "best_metrics": result.get("best_trial", {}).get("metrics", {}),
            }
            for engine, result in dual_engine.get("engines", {}).items()
        },
        "optimizer": "bayesian-lite-grid",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
