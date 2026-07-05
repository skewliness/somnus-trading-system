"""K-line pattern exam system: generate questions, evaluate model, score results.

Flow:
1. Generate exam questions from historical kline data (rule-based ground truth)
2. Train a lightweight GBM model on pattern features
3. Model predicts patterns/direction for each question
4. Auto-score: compare predictions vs ground truth
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

from trading_system.kline_pattern import (
    detect_patterns, compute_pattern_features, PATTERN_NAMES, PatternResult,
)
from trading_system.kline_pattern_dataset import _load_and_group, _row_to_kline


def generate_exam(
    market_data_path: Path,
    n_questions: int = 200,
    window_size: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Generate exam questions from market data.

    Each question: a kline window + correct answers (patterns + direction).
    Stratified sampling ensures each pattern type gets representation.
    """
    rows_by_symbol = _load_and_group(market_data_path)
    all_candidates: list[dict] = []

    for symbol, rows in rows_by_symbol.items():
        if len(rows) < window_size + 12:
            continue
        indices = list(range(window_size, len(rows) - 12))
        for i in indices:
            window = [_row_to_kline(r) for r in rows[i - window_size: i + 1]]
            patterns = detect_patterns(window)
            if not patterns:
                continue
            close_now = float(rows[i].get("close") or 0)
            future_close = float(rows[i + 3].get("close") or 0)
            if close_now <= 0:
                continue

            all_candidates.append({
                "symbol": symbol,
                "timestamp": int(rows[i].get("timestamp") or 0),
                "window": window,
                "correct_patterns": [p.name for p in patterns],
                "correct_direction": _majority_direction(patterns),
                "forward_return_3bar": round((future_close - close_now) / close_now, 6) if future_close > 0 else 0.0,
            })

    rng = random.Random(seed)
    # Stratified: try to get at least some of each pattern
    selected = _stratified_sample(all_candidates, n_questions, rng)
    return selected


def _majority_direction(patterns: list[PatternResult]) -> str:
    bullish = sum(1 for p in patterns if p.direction == "bullish")
    bearish = sum(1 for p in patterns if p.direction == "bearish")
    if bullish > bearish:
        return "bullish"
    elif bearish > bullish:
        return "bearish"
    return "neutral"


def _stratified_sample(candidates: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample ensuring pattern diversity."""
    by_pattern: dict[str, list[dict]] = {}
    for c in candidates:
        for p in c["correct_patterns"]:
            by_pattern.setdefault(p, []).append(c)

    selected_set: set[int] = set()
    selected: list[dict] = []

    # First pass: ensure each pattern gets at least a few samples
    per_pattern_min = max(2, n // len(PATTERN_NAMES)) if PATTERN_NAMES else 2
    for pattern_name in PATTERN_NAMES:
        pool = by_pattern.get(pattern_name, [])
        rng.shuffle(pool)
        for c in pool[:per_pattern_min]:
            cid = id(c)
            if cid not in selected_set:
                selected_set.add(cid)
                selected.append(c)
            if len(selected) >= n:
                break

    # Fill remaining with random samples
    remaining = [c for c in candidates if id(c) not in selected_set]
    rng.shuffle(remaining)
    for c in remaining:
        if len(selected) >= n:
            break
        selected.append(c)

    return selected[:n]


def evaluate_model_pattern_detection(
    exam: list[dict],
) -> dict:
    """Evaluate pattern detection accuracy using rule-based detector as both teacher and student.

    This tests the feature pipeline: can compute_pattern_features correctly encode
    what detect_patterns finds? Useful as a sanity check before training a GBM.
    """
    total = len(exam)
    if total == 0:
        return {"total": 0, "accuracy": 0.0}

    pattern_tp: dict[str, int] = {p: 0 for p in PATTERN_NAMES}
    pattern_fn: dict[str, int] = {p: 0 for p in PATTERN_NAMES}
    pattern_fp: dict[str, int] = {p: 0 for p in PATTERN_NAMES}
    direction_correct = 0

    for q in exam:
        features = compute_pattern_features(q["window"])
        predicted_patterns = {name for name in PATTERN_NAMES if features.get(f"pat_{name}", 0) > 0.5}
        correct_patterns = set(q["correct_patterns"])

        for p in PATTERN_NAMES:
            if p in correct_patterns and p in predicted_patterns:
                pattern_tp[p] += 1
            elif p in correct_patterns and p not in predicted_patterns:
                pattern_fn[p] += 1
            elif p not in correct_patterns and p in predicted_patterns:
                pattern_fp[p] += 1

        # Direction from feature counts
        bc = features.get("pattern_bullish_count", 0)
        brc = features.get("pattern_bearish_count", 0)
        if bc > brc:
            pred_dir = "bullish"
        elif brc > bc:
            pred_dir = "bearish"
        else:
            pred_dir = "neutral"
        if pred_dir == q["correct_direction"]:
            direction_correct += 1

    per_pattern: dict[str, dict] = {}
    for p in PATTERN_NAMES:
        tp = pattern_tp[p]
        fn = pattern_fn[p]
        fp = pattern_fp[p]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_pattern[p] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    overall_tp = sum(pattern_tp.values())
    overall_fn = sum(pattern_fn.values())
    overall_fp = sum(pattern_fp.values())
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0

    return {
        "total_questions": total,
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "direction_accuracy": round(direction_correct / total, 4),
        "per_pattern": per_pattern,
    }


def run_exam_report(
    market_data_path: Path,
    n_questions: int = 200,
    output_path: Path | None = None,
) -> dict:
    """Full exam pipeline: generate questions → detect → score → report."""
    exam = generate_exam(market_data_path, n_questions=n_questions)
    if not exam:
        return {"error": "no exam questions generated (insufficient data)"}

    results = evaluate_model_pattern_detection(exam)

    # Pattern frequency in exam
    pattern_freq: dict[str, int] = {}
    for q in exam:
        for p in q["correct_patterns"]:
            pattern_freq[p] = pattern_freq.get(p, 0) + 1
    results["pattern_frequency"] = pattern_freq

    # Forward return by direction
    dir_returns: dict[str, list[float]] = {"bullish": [], "bearish": [], "neutral": []}
    for q in exam:
        dir_returns[q["correct_direction"]].append(q["forward_return_3bar"])
    results["avg_return_by_direction"] = {
        d: round(mean(rets), 6) if rets else 0.0 for d, rets in dir_returns.items()
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    return results


def run_gbm_exam(
    market_data_path: Path,
    n_questions: int = 200,
    test_ratio: float = 0.3,
    output_path: Path | None = None,
) -> dict:
    """Train GBM on sequence features → predict pattern labels → score.

    This is the real exam: can a GBM learn to identify patterns from raw
    bar-level features (body_ratio, shadow_ratio, range_vs_prev, volume_vs_prev)?

    Split: train on (1-test_ratio) of data, test on test_ratio.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np

    from trading_system.kline_pattern_dataset import _load_and_group, _row_to_kline
    from trading_system.kline_pattern import SEQUENCE_FEATURE_KEYS, compute_pattern_features

    rows_by_symbol = _load_and_group(market_data_path)
    all_samples: list[dict] = []
    window_size = 20

    for symbol, rows in rows_by_symbol.items():
        if len(rows) < window_size + 12:
            continue
        for i in range(window_size, len(rows) - 12):
            window = [_row_to_kline(r) for r in rows[i - window_size: i + 1]]
            features = compute_pattern_features(window)
            all_samples.append(features)

    if len(all_samples) < 50:
        return {"error": "insufficient data for GBM exam"}

    # Split train/test
    rng = random.Random(42)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]

    # Build feature matrix (sequence features only — no pattern labels)
    X_train = np.array([[all_samples[i][k] for k in SEQUENCE_FEATURE_KEYS] for i in train_idx])
    X_test = np.array([[all_samples[i][k] for k in SEQUENCE_FEATURE_KEYS] for i in test_idx])

    # Train one classifier per pattern + one for direction
    results: dict[str, dict] = {}
    pattern_results: dict[str, dict] = {}

    # Direction classifier
    y_train_dir = np.array([
        1 if all_samples[i]["pattern_bullish_count"] > all_samples[i]["pattern_bearish_count"]
        else (-1 if all_samples[i]["pattern_bearish_count"] > all_samples[i]["pattern_bullish_count"] else 0)
        for i in train_idx
    ])
    y_test_dir = np.array([
        1 if all_samples[i]["pattern_bullish_count"] > all_samples[i]["pattern_bearish_count"]
        else (-1 if all_samples[i]["pattern_bearish_count"] > all_samples[i]["pattern_bullish_count"] else 0)
        for i in test_idx
    ])

    if len(set(y_train_dir)) > 1:
        clf_dir = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf_dir.fit(X_train, y_train_dir)
        pred_dir = clf_dir.predict(X_test)
        dir_acc = float(np.mean(pred_dir == y_test_dir))
    else:
        dir_acc = 0.0

    # Per-pattern classifiers (only for patterns with enough positive samples)
    for pname in PATTERN_NAMES:
        key = f"pat_{pname}"
        y_train = np.array([int(all_samples[i].get(key, 0) > 0.5) for i in train_idx])
        y_test = np.array([int(all_samples[i].get(key, 0) > 0.5) for i in test_idx])

        pos_train = int(y_train.sum())
        pos_test = int(y_test.sum())
        if pos_train < 5 or pos_test < 2:
            pattern_results[pname] = {"skipped": True, "reason": "insufficient_positives", "train_pos": pos_train, "test_pos": pos_test}
            continue

        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = float(np.mean(pred == y_test))

        pattern_results[pname] = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support_train": pos_train,
            "support_test": pos_test,
        }

    results = {
        "total_samples": len(all_samples),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "direction_accuracy": round(dir_acc, 4),
        "per_pattern": pattern_results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    return results
