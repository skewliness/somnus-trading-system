from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


VALID_USDT_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,30}USDT$")

FEATURE_NAMES = [
    "score",
    "sentiment_score",
    "sentiment_mentions",
    "taker_buy_sell_ratio",
    "oi_change_pct_4h",
    "price_change_pct_24h",
    "price_change_pct_5m",
    "funding_rate",
    "hf_win_rate_3bar",
    "hf_signal_count_total",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def build_realtime_features(snapshot: Any, hf_quality: dict[str, Any] | None = None, pdf_context: dict[str, Any] | None = None) -> list[float]:
    hf_quality = hf_quality if isinstance(hf_quality, dict) else {}
    pdf_context = pdf_context if isinstance(pdf_context, dict) else {}
    return [
        _float(getattr(snapshot, "score", 0.0)),
        _float(getattr(snapshot, "sentiment_score", 0.0)),
        _float(getattr(snapshot, "sentiment_mentions", 0.0)),
        _float(getattr(snapshot, "taker_buy_sell_ratio", 0.0)),
        _float(getattr(snapshot, "oi_change_pct_4h", 0.0)),
        _float(getattr(snapshot, "price_change_pct_24h", 0.0)),
        _float(getattr(snapshot, "price_change_pct_5m", 0.0)),
        _float(getattr(snapshot, "funding_rate", 0.0)),
        _float(hf_quality.get("hf_win_rate_3bar")),
        _float(hf_quality.get("hf_signal_count_total")),
        _float(pdf_context.get("news_score")),
    ]


def _feature_value(row: dict[str, Any], feature: str) -> float:
    return _float(row.get(feature))


def _symbol_stats(closed_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in closed_rows:
        symbol = str(row.get("symbol") or "").upper()
        if VALID_USDT_SYMBOL_RE.fullmatch(symbol):
            grouped.setdefault(symbol, []).append(row)
    stats: dict[str, dict[str, Any]] = {}
    for symbol, rows in grouped.items():
        wins = [1.0 if bool(row.get("is_win")) or _float(row.get("pnl_pct")) > 0 else 0.0 for row in rows]
        pnls = [_float(row.get("pnl_pct")) for row in rows]
        stats[symbol] = {
            "sample_count": len(rows),
            "win_rate": mean(wins) if wins else 0.0,
            "avg_pnl_pct": mean(pnls) if pnls else 0.0,
        }
    return stats


def _feature_weights(decision_rows: list[dict[str, Any]], closed_rows: list[dict[str, Any]]) -> dict[str, float]:
    rows = list(closed_rows) + list(decision_rows)
    if not rows:
        return {name: 0.0 for name in FEATURE_NAMES}
    weights: dict[str, float] = {}
    for feature in FEATURE_NAMES:
        positive = [_feature_value(row, feature) for row in rows if bool(row.get("is_win")) or _float(row.get("pnl_pct")) > 0 or row.get("decision_label") == "entered"]
        negative = [_feature_value(row, feature) for row in rows if not (bool(row.get("is_win")) or _float(row.get("pnl_pct")) > 0 or row.get("decision_label") == "entered")]
        if positive and negative:
            weights[feature] = mean(positive) - mean(negative)
        elif positive:
            weights[feature] = abs(mean(positive))
        else:
            weights[feature] = 0.0
    return weights


def train_predictive_model(
    decision_dataset_path: str | Path,
    closed_trade_dataset_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    decision_rows = _load_jsonl(Path(decision_dataset_path))
    closed_rows = _load_jsonl(Path(closed_trade_dataset_path))
    weights = _feature_weights(decision_rows, closed_rows)
    symbol_stats = _symbol_stats(closed_rows)

    def win_predict_fn(features: list[float]) -> float:
        score = 0.0
        for idx, value in enumerate(features[: len(FEATURE_NAMES)]):
            score += _float(value) * _float(weights.get(FEATURE_NAMES[idx])) / 100.0
        return _sigmoid(score)

    def sl_predict_fn(features: list[float]) -> float:
        return 1.0 - win_predict_fn(features)

    summary = {
        "status": "ok",
        "mode": "lightweight_snapshot_fallback",
        "decision_rows": len(decision_rows),
        "closed_rows": len(closed_rows),
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES) + 1,
        "feature_weights": weights,
        "symbol_stats": symbol_stats,
        "predictions": [
            {
                "symbol": symbol,
                "predicted_win_probability": stats["win_rate"],
                "predicted_stop_loss_risk": 1.0 - stats["win_rate"],
            }
            for symbol, stats in sorted(symbol_stats.items())
        ],
        "_win_predict_fn": win_predict_fn,
        "_sl_predict_fn": sl_predict_fn,
    }
    if output_path:
        serializable = {key: value for key, value in summary.items() if not key.startswith("_")}
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
