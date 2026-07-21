from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean


VALID_USDT_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,30}USDT$")


def _is_valid_usdt_symbol(symbol: object) -> bool:
    return bool(VALID_USDT_SYMBOL_RE.fullmatch(str(symbol or "")))


# Numeric features with proven information value
FEATURES_NUMERIC = (
    "mentions",
    "sentiment_score",
    "sentiment_mentions",
    "candidate_score",
    "oi_divergence_score",
    "oi_change_pct_4h",
    "taker_buy_sell_ratio",
    "quote_volume_usdt_24h",
    "price_change_pct_5m",
    "price_change_pct_24h",
    "hotness_score",
    "priority_score",
    "whale_total_notional_usdt",
    "whale_long_notional_usdt",
    "whale_short_notional_usdt",
    "whale_total_count",
    "whale_long_count",
    "whale_short_count",
    "whale_long_short_ratio",
    "whale_long_avg_entry_price",
    "whale_short_avg_entry_price",
    "whale_long_pnl_usdt",
    "whale_short_pnl_usdt",
    "whale_long_profit_ratio",
    "whale_short_profit_ratio",
    "whale_pressure_score",
    "whale_okx_sold_ratio",
    "whale_okx_hold_ratio",
    "whale_okx_hotness_score",
    "whale_okx_mention_count",
    "whale_market_oi_change_pct_4h",
    "whale_market_taker_buy_sell_ratio",
    # Structure indicators (already computed but previously unused)
    "price_position_7d",
    "price_position_14d",
    "price_position_20d",
    "breakout_distance_7d",
    "compression_score",
    # Kline statistics
    "kline_lower_shadow_pct",
    "kline_upper_shadow_pct",
    "kline_body_pct",
    "kline_direction",
    "kline_volume_vs_ma5",
    "kline_range_vs_ma5_range",
    "kline_close_position",
    "kline_consecutive_direction",
    "kline_atr_pct",
    "kline_ma20_distance_pct",
    "kline_volume_trend",
    "kline_max_drawdown_5d",
    # Multi-timeframe BB context, available before entry and at live prediction time.
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
    "bb_upper_heat_score",
    "bb_extension_max",
    "atr_regime_high",
    "false_breakout_4h",
    "bb_touch_upper",
    # Kline pattern recognition
    "kline_pattern_bullish_count",
    "kline_pattern_bearish_count",
    # HF backtest signal quality (per-symbol, from 2h cache)
    "hf_win_rate_3bar",
    "hf_avg_mfe_12bar",
    "hf_avg_risk_reward",
    "hf_best_signal_win_rate",
    "hf_momentum_edge",
    "hf_oi_surge_edge",
    "hf_signal_count_total",
    # PDF strategy context features (pdf_entry_score excluded — it participates in label correction)
    "pdf_risk_score",
    "pdf_revenge_entry_flag",
    "daily_realized_pnl_usdt",
    "consecutive_sl_count",
)

# Categorical features that get one-hot encoded (PRE-TRADE only)
# reason_family is deliberately excluded — it's a post-hoc outcome, not a pre-trade signal
FEATURES_CATEGORICAL = (
    "setup_type",
    "risk_band",
    "execution_status",
)

# All known setup_type values (for consistent one-hot encoding)
KNOWN_SETUP_TYPES = (
    "trend",
    "oi_divergence_long_candidate",
    "base_breakout_confirmed",
    "mania_followthrough",
    "squeeze_anomaly",
)

KNOWN_RISK_BANDS = (
    "high",
    "medium",
    "low",
    "oi_divergence",
    "base_breakout",
    "trend",
)

KNOWN_EXECUTION_STATUSES = (
    "executable_now",
    "confirm_needed",
    "watch_only",
    "blocked",
)

KNOWN_REASON_FAMILIES = (
    "stop_loss_hit",
    "take_profit_hit",
    "score_faded",
    "oi_divergence_signal_fade",
    "market_snapshot_missing_timeout",
    "max_hold_reached",
)


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


def _f(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


_BB_HEAT_FIELDS = (
    "bb_price_position_96",
    "bb_price_position_210",
    "bb_price_position_672_15m",
    "bb_price_position_105_4h",
)

_BB_EXTENSION_FIELDS = (
    "bb_price_position_96",
    "bb_price_position_210",
    "bb_price_position_672_15m",
    "bb_price_position_105_4h",
    "bb_price_position_672_4h",
)

_MODEL_POSITION_FIELDS = {
    "bb_price_position_96",
    "bb_price_position_210",
    "bb_price_position_672_15m",
    "bb_price_position_105_4h",
    "bb_price_position_672_4h",
    "bb_price_position_1440_4h",
    "bb_price_position_105_1m",
    "bb_price_position_300_1m",
}

_MODEL_DISTANCE_FIELDS = {
    "bb105_mid_4h_distance_pct",
    "bb672_mid_4h_distance_pct",
    "bb1440_mid_4h_distance_pct",
    "bb105_upper_1m_distance_pct",
    "bb300_upper_1m_distance_pct",
    "bb96_upper_15m_distance_pct",
    "short_4h_mid_retest_distance_min_pct",
}


def _model_position(value: object, default: float = 0.5) -> float:
    return max(0.0, min(1.0, _f(value, default)))


def _distance_pct(price: object, reference: object) -> float:
    # Keep distance fields available for schema compatibility, but neutral for
    # model input. Raw cross-timeframe distances produced 1000%+ outliers.
    return 0.0


def _source_get(source: object, name: str, default: object = None) -> object:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _bb_touch_upper_value(zone: object) -> float:
    text = str(zone or "").lower()
    return 1.0 if "upper" in text else 0.0


def _derived_numeric_feature(source: object, name: str) -> float | None:
    if name in _MODEL_DISTANCE_FIELDS:
        return 0.0
    if name == "short_1m_upper_pressure_max":
        if _source_get(source, name) is not None:
            return _model_position(_source_get(source, name), 0.5)
        return max(
            _model_position(_source_get(source, "bb_price_position_105_1m"), 0.5),
            _model_position(_source_get(source, "bb_price_position_300_1m"), 0.5),
        )
    if name == "short_15m_bb96_position":
        if _source_get(source, name) is not None:
            return _model_position(_source_get(source, name), 0.5)
        return _model_position(_source_get(source, "bb_price_position_96"), 0.5)
    if name in _MODEL_POSITION_FIELDS:
        return _model_position(_source_get(source, name), 0.5)
    if name == "bb_upper_heat_score":
        return float(sum(1 for field in _BB_HEAT_FIELDS if _model_position(_source_get(source, field), 0.5) >= 0.85))
    if name == "bb_extension_max":
        return max(_model_position(_source_get(source, field), 0.5) for field in _BB_EXTENSION_FIELDS)
    if name == "atr_regime_high":
        return 1.0 if _f(_source_get(source, "kline_atr_pct")) >= 0.10 else 0.0
    if name == "false_breakout_4h":
        bb_pos = _f(_source_get(source, "bb_price_position_105_4h"), 0.5)
        slope = _f(_source_get(source, "h4_ma20_slope_pct"))
        return 1.0 if bb_pos >= 0.70 and slope < 0.0 else 0.0
    if name == "bb_touch_upper":
        return _bb_touch_upper_value(_source_get(source, "bb_touch_zone"))
    if name == "bb105_mid_4h_distance_pct":
        return _distance_pct(_source_get(source, "mark_price"), _source_get(source, "bb105_mid_4h"))
    if name == "bb672_mid_4h_distance_pct":
        return _distance_pct(_source_get(source, "mark_price"), _source_get(source, "bb672_mid_4h"))
    if name == "bb1440_mid_4h_distance_pct":
        return _distance_pct(_source_get(source, "mark_price"), _source_get(source, "bb1440_mid_4h"))
    if name == "bb105_upper_1m_distance_pct":
        return _distance_pct(_source_get(source, "mark_price"), _source_get(source, "bb105_upper_1m"))
    if name == "bb300_upper_1m_distance_pct":
        return _distance_pct(_source_get(source, "mark_price"), _source_get(source, "bb300_upper_1m"))
    if name == "bb96_upper_15m_distance_pct":
        return _distance_pct(_source_get(source, "mark_price"), _source_get(source, "bb96_upper_15m"))
    if name == "short_4h_mid_retest_distance_min_pct":
        distances = [
            -_distance_pct(_source_get(source, "mark_price"), _source_get(source, "bb105_mid_4h")),
            -_distance_pct(_source_get(source, "mark_price"), _source_get(source, "bb672_mid_4h")),
            -_distance_pct(_source_get(source, "mark_price"), _source_get(source, "bb1440_mid_4h")),
        ]
        return min([value for value in distances if value > 0], default=0.0)
    return None


def _build_features(row: dict) -> list[float]:
    """Build feature vector from a closed trade row."""
    features = []
    # Numeric features
    for fn in FEATURES_NUMERIC:
        derived = _derived_numeric_feature(row, fn)
        features.append(_f(row.get(fn)) if derived is None else derived)

    # One-hot categorical features
    setup = str(row.get("setup_type") or "").lower()
    for st in KNOWN_SETUP_TYPES:
        features.append(1.0 if setup == st else 0.0)

    risk = str(row.get("risk_band") or "").lower()
    for rb in KNOWN_RISK_BANDS:
        features.append(1.0 if risk == rb else 0.0)

    exec_status = str(row.get("execution_status") or "").lower()
    for es in KNOWN_EXECUTION_STATUSES:
        features.append(1.0 if exec_status == es else 0.0)

    return features


def get_feature_names() -> list[str]:
    """Return ordered list of all feature names."""
    names = list(FEATURES_NUMERIC)
    for st in KNOWN_SETUP_TYPES:
        names.append(f"setup_{st}")
    for rb in KNOWN_RISK_BANDS:
        names.append(f"risk_{rb}")
    for es in KNOWN_EXECUTION_STATUSES:
        names.append(f"exec_{es}")
    return names


_SNAPSHOT_FIELD_MAP: dict[str, str] = {
    "mentions": "sentiment_mentions",
    "sentiment_score": "sentiment_score",
    "sentiment_mentions": "sentiment_mentions",
    "candidate_score": "score",
    "oi_divergence_score": "oi_price_divergence_score",
    "oi_change_pct_4h": "oi_change_pct_4h",
    "taker_buy_sell_ratio": "taker_buy_sell_ratio",
    "quote_volume_usdt_24h": "quote_volume_usdt_24h",
    "price_change_pct_5m": "price_change_pct_5m",
    "price_change_pct_24h": "price_change_pct_24h",
    "hotness_score": "sentiment_score",
    "priority_score": "hotness_priority_score",
    "price_position_7d": "price_position_7d",
    "price_position_14d": "price_position_14d",
    "price_position_20d": "price_position_20d",
    "breakout_distance_7d": "breakout_distance_7d",
    "compression_score": "compression_score",
    "kline_lower_shadow_pct": "kline_lower_shadow_pct",
    "kline_upper_shadow_pct": "kline_upper_shadow_pct",
    "kline_body_pct": "kline_body_pct",
    "kline_direction": "kline_direction",
    "kline_volume_vs_ma5": "kline_volume_vs_ma5",
    "kline_range_vs_ma5_range": "kline_range_vs_ma5_range",
    "kline_close_position": "kline_close_position",
    "kline_consecutive_direction": "kline_consecutive_direction",
    "kline_atr_pct": "kline_atr_pct",
    "kline_ma20_distance_pct": "kline_ma20_distance_pct",
    "kline_volume_trend": "kline_volume_trend",
    "kline_max_drawdown_5d": "kline_max_drawdown_5d",
    "daily_price_vs_ma7_pct": "daily_price_vs_ma7_pct",
    "daily_price_vs_ma20_pct": "daily_price_vs_ma20_pct",
    "daily_price_position_365": "daily_price_position_365",
    "bb_price_position_96": "bb_price_position_96",
    "bb_price_position_210": "bb_price_position_210",
    "bb_price_position_672_15m": "bb_price_position_672_15m",
    "bb_price_position_105_4h": "bb_price_position_105_4h",
    "bb_price_position_672_4h": "bb_price_position_672_4h",
    "williams_r_15m": "williams_r_15m",
    "williams_r_15m_120": "williams_r_15m_120",
    "williams_r_1m": "williams_r_1m",
    "kline_pattern_bullish_count": "kline_pattern_bullish_count",
    "kline_pattern_bearish_count": "kline_pattern_bearish_count",
}

_HF_FIELDS = (
    "hf_win_rate_3bar",
    "hf_avg_mfe_12bar",
    "hf_avg_risk_reward",
    "hf_best_signal_win_rate",
    "hf_momentum_edge",
    "hf_oi_surge_edge",
    "hf_signal_count_total",
)

_PDF_FIELDS = (
    "pdf_risk_score",
    "pdf_revenge_entry_flag",
    "daily_realized_pnl_usdt",
    "consecutive_sl_count",
)


def build_realtime_features(
    snapshot: object,
    hf_quality: dict | None = None,
    pdf_context: dict | None = None,
) -> list[float]:
    """Build 62+15 dim feature vector from a live MarketSnapshot for real-time predict_proba."""
    hf = hf_quality or {}
    pdf = pdf_context or {}
    features: list[float] = []

    for fn in FEATURES_NUMERIC:
        derived = _derived_numeric_feature(snapshot, fn)
        if derived is not None:
            features.append(_f(derived))
        elif fn in _SNAPSHOT_FIELD_MAP:
            val = getattr(snapshot, _SNAPSHOT_FIELD_MAP[fn], None)
            features.append(_f(val))
        elif fn in _HF_FIELDS:
            features.append(_f(hf.get(fn)))
        elif fn in _PDF_FIELDS:
            features.append(_f(pdf.get(fn)))
        else:
            features.append(0.0)

    setup = str(getattr(snapshot, "setup_type", "") or "").lower()
    for st in KNOWN_SETUP_TYPES:
        features.append(1.0 if setup == st else 0.0)

    risk = ""
    if setup in ("oi_divergence_long_candidate",):
        risk = "oi_divergence"
    elif setup in ("base_breakout", "base_breakout_confirmed"):
        risk = "base_breakout"
    elif setup == "trend":
        risk = "trend"
    else:
        risk = "medium"
    for rb in KNOWN_RISK_BANDS:
        features.append(1.0 if risk == rb else 0.0)

    for es in KNOWN_EXECUTION_STATUSES:
        features.append(1.0 if es == "executable_now" else 0.0)

    return features


def _train_gb(X, y, validation_ratio: float = 0.2, sample_weights: list[float] | None = None) -> tuple[object, dict, list[float], str, object]:
    """Train GradientBoostingClassifier."""
    from sklearn.ensemble import GradientBoostingClassifier

    n = len(X)
    split_idx = max(1, int(n * (1.0 - validation_ratio)))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    w_train = sample_weights[:split_idx] if sample_weights else None

    model = GradientBoostingClassifier(
        # 降复杂度修过拟合: 517样本上 n150/d5/leaf2 使 trainAUC 0.997 vs valAUC 0.5986(背答案)。
        # 实测 n40/d3/lr0.05/leaf25 → valAUC 0.6216, trainAUC 0.767(健康泛化)。
        n_estimators=40,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        min_samples_leaf=25,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_correct = sum(1 for pred, actual in zip(val_preds, y_val) if pred == actual)
    val_accuracy = val_correct / len(y_val) if y_val else 0.0

    # 真实判别力指标 (holdout, 非样本内). AUC=0.5 即无技能, 与基础胜率无关
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    try:
        val_auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.0
    except Exception:
        val_auc = 0.0
    val_precision = precision_score(y_val, val_preds, zero_division=0)
    val_recall = recall_score(y_val, val_preds, zero_division=0)

    importance = list(model.feature_importances_)

    def predict_fn(feature_vec: list[float]) -> float:
        prob = model.predict_proba([feature_vec])[0, 1]
        return round(float(prob), 6)

    metrics = {
        "validation_samples": len(y_val),
        "validation_accuracy": round(val_accuracy, 4),
        "validation_auc": round(float(val_auc), 4),
        "validation_precision": round(float(val_precision), 4),
        "validation_recall": round(float(val_recall), 4),
        "validation_wins": sum(y_val),
        # 基础胜率: 验证集赢样本占比 (基线, 非模型表现). 旧名 validation_win_rate 易误读为模型能力
        "validation_base_rate": round(sum(y_val) / len(y_val), 4) if y_val else 0.0,
    }

    return model, metrics, importance, "sklearn_gradient_boosting", predict_fn


def _train_logistic(X, y, validation_ratio: float = 0.2) -> tuple[object, dict, list[float], str, object]:
    """Fallback: logistic regression via simple weight optimization."""
    n = len(X)
    n_features = len(X[0]) if X else 0
    split_idx = max(1, int(n * (1.0 - validation_ratio)))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    weights = []
    for fi in range(n_features):
        pos_vals = [X_train[i][fi] for i in range(len(X_train)) if y_train[i] == 1]
        neg_vals = [X_train[i][fi] for i in range(len(X_train)) if y_train[i] == 0]
        pos_mean = mean(pos_vals) if pos_vals else 0.0
        neg_mean = mean(neg_vals) if neg_vals else 0.0
        weights.append(pos_mean - neg_mean)

    def stable_sigmoid(raw: float) -> float:
        z = max(-60.0, min(60.0, raw / 100.0))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_logistic(feature_vec: list[float]) -> float:
        raw = 0.0
        for idx, w in enumerate(weights):
            raw += feature_vec[idx] * w
        return round(stable_sigmoid(raw), 6) if raw != 0 else 0.5

    val_correct = 0
    val_probs: list[float] = []
    val_preds: list[int] = []
    for vec, actual in zip(X_val, y_val):
        pred = predict_logistic(vec)
        val_probs.append(pred)
        val_preds.append(1 if pred >= 0.5 else 0)
        if (pred >= 0.5 and actual == 1) or (pred < 0.5 and actual == 0):
            val_correct += 1
    val_accuracy = val_correct / len(y_val) if y_val else 0.0

    # 真实判别力指标 (holdout). 与 GB 路径对齐, 保持报告口径一致
    try:
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        val_auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.0
        val_precision = precision_score(y_val, val_preds, zero_division=0)
        val_recall = recall_score(y_val, val_preds, zero_division=0)
    except Exception:
        val_auc = val_precision = val_recall = 0.0

    metrics = {
        "validation_samples": len(y_val),
        "validation_accuracy": round(val_accuracy, 4),
        "validation_auc": round(float(val_auc), 4),
        "validation_precision": round(float(val_precision), 4),
        "validation_recall": round(float(val_recall), 4),
        "validation_wins": sum(y_val),
        # 基础胜率: 验证集赢样本占比 (基线, 非模型表现)
        "validation_base_rate": round(sum(y_val) / len(y_val), 4) if y_val else 0.0,
    }

    return weights, metrics, weights, "logistic", predict_logistic


def _train_r_multiple_regressor(X, y_r, sample_weights: list[float] | None = None, validation_ratio: float = 0.2) -> tuple[object, dict, object]:
    """Train GradientBoostingRegressor to predict R-multiple."""
    from sklearn.ensemble import GradientBoostingRegressor

    n = len(X)
    split_idx = max(1, int(n * (1.0 - validation_ratio)))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_r[:split_idx], y_r[split_idx:]
    w_train = sample_weights[:split_idx] if sample_weights else None

    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    val_preds = model.predict(X_val) if len(X_val) > 0 else []
    mae = sum(abs(p - a) for p, a in zip(val_preds, y_val)) / len(y_val) if y_val else 0.0
    mean_pred = sum(val_preds) / len(val_preds) if len(val_preds) > 0 else 0.0

    def predict_fn(feature_vec: list[float]) -> float:
        return round(float(model.predict([feature_vec])[0]), 4)

    metrics = {
        "val_mae": round(mae, 4),
        "val_samples": len(y_val),
        "val_mean_predicted_r": round(mean_pred, 4),
    }
    return model, metrics, predict_fn


def train_predictive_model(
    decision_dataset_path: Path,
    closed_trade_dataset_path: Path,
    output_path: Path | None = None,
) -> dict:
    """Train a predictive model on closed trade samples with rich feature engineering.

    Features include: numeric signals + one-hot encoded setup_type, risk_band,
    execution_status, and reason_family.

    Uses sklearn GradientBoostingClassifier with 80/20 split.
    Outputs feature importance, validation metrics, and per-symbol stats.
    """
    closed_rows = _load_jsonl(closed_trade_dataset_path)
    raw_closed_rows = list(closed_rows)
    closed_rows = [row for row in raw_closed_rows if not row.get("exclude_from_training")]
    if not closed_rows:
        payload = {
            "status": "missing",
            "closed_rows": 0,
            "raw_closed_rows": len(raw_closed_rows),
            "excluded_rows": len(raw_closed_rows),
            "feature_importance": {},
            "validation_metrics": {},
            "symbol_stats": {},
            "model_type": "none",
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    # Build feature matrix
    X: list[list[float]] = []
    y_is_win: list[int] = []
    y_stop_loss: list[int] = []
    y_r_multiple: list[float] = []
    sample_weights: list[float] = []
    joined_rows = []

    for row in closed_rows:
        if not _is_valid_usdt_symbol(row.get("symbol")):
            continue
        try:
            features = _build_features(row)
            target_win = 1 if bool(row.get("is_win")) else 0
            reason = str(row.get("reason_family") or "").lower()
            target_sl = 1 if reason == "stop_loss_hit" else 0
            initial_risk = _f(row.get("initial_risk_pct"))
            r_mult = max(-2.0, min(5.0, _f(row.get("r_multiple")))) if initial_risk > 0 else None
            weight = _f(row.get("sample_weight"), 1.0)
            X.append(features)
            y_is_win.append(target_win)
            y_stop_loss.append(target_sl)
            y_r_multiple.append(r_mult)
            sample_weights.append(weight)
            joined_rows.append({
                "symbol": str(row.get("symbol") or ""),
                "features": dict(zip(get_feature_names(), features)),
                "target_win": target_win,
                "target_stop_loss": target_sl,
                "r_multiple": r_mult,
                "pnl_pct": _f(row.get("pnl_pct")),
                "reason_family": reason,
            })
        except Exception:
            continue

    if len(X) < 1:
        payload = {
            "status": "insufficient",
            "closed_rows": len(closed_rows),
            "raw_closed_rows": len(raw_closed_rows),
            "excluded_rows": len(raw_closed_rows) - len(closed_rows),
            "usable_rows": len(X),
            "feature_importance": {},
            "validation_metrics": {},
            "symbol_stats": {},
            "model_type": "none",
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    # Compute symbol-level stats
    symbol_stats: dict[str, dict[str, float]] = {}
    for row in closed_rows:
        symbol = str(row.get("symbol") or "").upper()
        if not _is_valid_usdt_symbol(symbol):
            continue
        slot = symbol_stats.setdefault(symbol, {"wins": 0.0, "count": 0.0, "avg_pnl": 0.0, "stop_losses": 0.0})
        pnl = _f(row.get("pnl_pct"))
        slot["count"] += 1.0
        slot["avg_pnl"] += pnl
        if bool(row.get("is_win")) or pnl > 0:
            slot["wins"] += 1.0
        reason = str(row.get("reason_family") or "").lower()
        if reason == "stop_loss_hit":
            slot["stop_losses"] += 1.0
    for slot in symbol_stats.values():
        cnt = max(slot["count"], 1.0)
        slot["win_rate"] = slot["wins"] / cnt
        slot["avg_pnl"] = slot["avg_pnl"] / cnt
        slot["stop_loss_rate"] = slot["stop_losses"] / cnt

    # Train win/loss predictor. Prefer sklearn GradientBoosting when available,
    # but keep the training path portable on machines without sklearn installed.
    try:
        if len(X) < 8 or len(set(y_is_win)) < 2:
            raise ValueError("too few rows/classes for gradient boosting")
        model, val_metrics, importance, model_type, predict_fn = _train_gb(X, y_is_win, sample_weights=sample_weights)
    except (ModuleNotFoundError, ValueError):
        model, val_metrics, importance, model_type, predict_fn = _train_logistic(X, y_is_win)

    # Train stop-loss risk predictor (separate model) with the same portable fallback.
    try:
        if len(X) < 8 or len(set(y_stop_loss)) < 2:
            raise ValueError("too few rows/classes for gradient boosting")
        sl_model, sl_val_metrics, sl_importance, _, sl_predict_fn = _train_gb(X, y_stop_loss, sample_weights=sample_weights)
    except (ModuleNotFoundError, ValueError):
        sl_model, sl_val_metrics, sl_importance, _, sl_predict_fn = _train_logistic(X, y_stop_loss)

    # Train R-multiple regressor (predicts expected risk-adjusted return).
    # Only use samples with explicit initial_risk_pct (r_multiple != None).
    r_model = None
    r_metrics: dict = {}
    r_predict_fn = None
    r_importance: list[float] = []
    X_r = [X[i] for i in range(len(X)) if y_r_multiple[i] is not None]
    y_r = [y_r_multiple[i] for i in range(len(y_r_multiple)) if y_r_multiple[i] is not None]
    w_r = [sample_weights[i] for i in range(len(sample_weights)) if y_r_multiple[i] is not None]
    try:
        if len(X_r) < 8:
            raise ValueError("insufficient R-multiple data")
        r_model, r_metrics, r_predict_fn = _train_r_multiple_regressor(X_r, y_r, sample_weights=w_r)
        r_importance = list(r_model.feature_importances_)
        r_metrics["r_training_rows"] = len(X_r)
    except (ModuleNotFoundError, ValueError):
        r_metrics["r_training_rows"] = len(X_r)

    # Build feature importance
    feature_names = get_feature_names()
    importance_dict = {}
    for name, val in zip(feature_names, importance):
        if abs(val) > 0.0001:  # only include non-trivial features
            importance_dict[name] = round(float(val), 6)

    sl_importance_dict = {}
    for name, val in zip(feature_names, sl_importance):
        if abs(val) > 0.0001:
            sl_importance_dict[name] = round(float(val), 6)

    r_importance_dict = {}
    for name, val in zip(feature_names, r_importance):
        if abs(val) > 0.0001:
            r_importance_dict[name] = round(float(val), 6)

    # Generate predictions
    predictions = []
    for row in joined_rows:
        feature_vec = [_f(row["features"].get(fn)) for fn in feature_names]
        win_prob = predict_fn(feature_vec)
        sl_risk = sl_predict_fn(feature_vec)
        predicted_r = r_predict_fn(feature_vec) if r_predict_fn else 0.0
        predictions.append({
            "symbol": row["symbol"],
            "predicted_win_probability": win_prob,
            "predicted_stop_loss_risk": sl_risk,
            "predicted_r_multiple": predicted_r,
            "target_win": row["target_win"],
            "target_stop_loss": row["target_stop_loss"],
            "actual_r_multiple": row["r_multiple"],
            "pnl_pct": round(row["pnl_pct"], 6),
            "reason_family": row["reason_family"],
        })

    # Post-training analysis
    # 仅在 holdout (最后 20%, 与 _train_gb 时间序切分一致) 上分桶, 避免样本内"背答案"幻觉.
    # 注意: 上面的 predictions 列表保持全量, scoring.py 依赖它做按 symbol 的实时回退查询.
    holdout_split = max(1, int(len(predictions) * (1.0 - 0.2)))
    holdout_preds = predictions[holdout_split:]

    high_conf_wins = [p for p in holdout_preds if p["predicted_win_probability"] >= 0.6]
    low_conf_wins = [p for p in holdout_preds if p["predicted_win_probability"] < 0.4]
    high_sl_risk = [p for p in holdout_preds if p["predicted_stop_loss_risk"] >= 0.5]

    high_win_actual = sum(1 for p in high_conf_wins if p["target_win"] == 1) if high_conf_wins else 0
    high_win_rate = high_win_actual / len(high_conf_wins) if high_conf_wins else 0.0
    low_win_actual = sum(1 for p in low_conf_wins if p["target_win"] == 1) if low_conf_wins else 0
    low_win_rate = low_win_actual / len(low_conf_wins) if low_conf_wins else 0.0

    high_sl_actual = sum(1 for p in high_sl_risk if p["target_stop_loss"] == 1) if high_sl_risk else 0
    high_sl_rate = high_sl_actual / len(high_sl_risk) if high_sl_risk else 0.0

    post_training = {
        "holdout_samples": len(holdout_preds),
        "high_confidence_win_rate": round(high_win_rate, 4),
        "high_confidence_count": len(high_conf_wins),
        "low_confidence_win_rate": round(low_win_rate, 4),
        "low_confidence_count": len(low_conf_wins),
        "high_stop_loss_risk_rate": round(high_sl_rate, 4),
        "high_stop_loss_risk_count": len(high_sl_risk),
        "win_rate_lift_high_vs_low": round(high_win_rate - low_win_rate, 4) if high_conf_wins and low_conf_wins else 0.0,
    }

    payload = {
        "status": "ok",
        "closed_rows": len(closed_rows),
        "raw_closed_rows": len(raw_closed_rows),
        "excluded_rows": len(raw_closed_rows) - len(closed_rows),
        "usable_rows": len(X),
        "feature_count": len(feature_names),
        "model_type": model_type,
        "feature_importance": importance_dict,
        "feature_weights": {
            **importance_dict,
            "sentiment_score": float(importance_dict.get("hotness_score", 0.000001) or 0.000001),
            "taker_buy_sell_ratio": float(importance_dict.get("priority_score", 0.000001) or 0.000001),
        },
        "stop_loss_risk_feature_importance": sl_importance_dict,
        "validation_metrics": val_metrics,
        "stop_loss_validation_metrics": sl_val_metrics,
        "post_training_analysis": post_training,
        "symbol_stats": symbol_stats,
        "predictions": predictions,
        "train_win_rate": round(sum(y_is_win) / len(y_is_win), 4) if y_is_win else 0.0,
        "train_stop_loss_rate": round(sum(y_stop_loss) / len(y_stop_loss), 4) if y_stop_loss else 0.0,
        "r_multiple_model_type": "sklearn_gradient_boosting_regressor" if r_model else "none",
        "r_multiple_metrics": r_metrics,
        "r_multiple_feature_importance": r_importance_dict,
        "trained_until": max((str(row.get("timestamp") or "") for row in closed_rows), default=""),
        "_win_predict_fn": predict_fn,
        "_sl_predict_fn": sl_predict_fn,
        "_r_predict_fn": r_predict_fn,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_payload = {k: v for k, v in payload.items() if not k.startswith("_")}
        output_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload
