#!/usr/bin/env python3
"""
Pipeline 回测验证：比较模型过滤 vs 无模型 baseline

用已关闭的 paper trades (123 条) 做回测：
1. 从 trade 记录中的特征字段构建 CoinFeatures（缺失的填 0）
2. 用训练好的模型预测每笔交易的分类
3. 按预测分类统计 win rate / avg PnL
4. 对比：全量入场 vs 模型筛选后入场

用法:
    python3 scripts/pipeline_backtest_comparison.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_system.coin_scout_model import CoinFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLOSED_TRADES_PATH = ROOT / "data/trading_system/model_training/paper_replay_autolearn/paper_closed_trades.jsonl"

MODEL_DIRS = {
    "baseline": ROOT / "data/trading_system/model_training/ablation_v2/baseline",
    "class_weighted": ROOT / "data/trading_system/model_training/ablation_v2/class_weighted",
    "undersampled": ROOT / "data/trading_system/model_training/ablation_v2/undersampled",
}

FEAT_NAMES = [f.name for f in fields(CoinFeatures) if f.name not in ("symbol", "timestamp")]
LABEL_NAMES = {0: "非妖币", 1: "弱妖币", 2: "强妖币"}

# trade 记录中直接有的特征字段 → CoinFeatures 字段名映射
TRADE_FEAT_MAP = {
    "oi_change_pct_4h": "oi_change_pct_4h",
    "price_change_pct_24h": "price_change_pct_24h",
    "price_change_pct_5m": "price_change_pct_5m",
    "taker_buy_sell_ratio": "taker_buy_sell_ratio",
    "quote_volume_usdt_24h": "quote_volume_usdt_24h",
    "sentiment_score": "sentiment_score",
    "sentiment_mentions": "sentiment_mentions",
    "oi_divergence_score": "oi_price_divergence_score",
    "hotness_score": "lanaai_hotness_score",
    "mentions": "sentiment_mentions",  # 备用
}


def load_jsonl(path: Path) -> list[dict]:
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


def load_model(model_dir: Path):
    pkl_path = model_dir / "coin_scout_model.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data.get("model", data)
    return data


def trade_to_feature_vector(trade: dict) -> list[float]:
    """从 trade 记录构建特征向量（缺失的 CoinFeatures 字段填 0）"""
    feat_dict = {}
    for trade_key, feat_key in TRADE_FEAT_MAP.items():
        val = trade.get(trade_key, 0)
        if val is not None:
            try:
                feat_dict[feat_key] = float(val)
            except (TypeError, ValueError):
                pass

    return [feat_dict.get(name, 0.0) for name in FEAT_NAMES]


def evaluate_model(model, trades: list[dict], label: str) -> dict:
    results = []
    for trade in trades:
        x = np.array([trade_to_feature_vector(trade)])
        try:
            pred = int(model.predict(x)[0])
        except Exception:
            continue
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(x)[0]
            except Exception:
                pass

        pnl_pct = float(trade.get("pnl_pct", 0))
        is_win = bool(trade.get("is_win")) or pnl_pct > 0

        results.append({
            "symbol": trade.get("symbol", ""),
            "pred_label": pred,
            "pred_name": LABEL_NAMES.get(pred, f"unknown_{pred}"),
            "pnl_pct": pnl_pct,
            "is_win": is_win,
            "proba": proba.tolist() if proba is not None else None,
            "reason_family": trade.get("reason_family", ""),
        })

    if not results:
        return {"status": "no_data", "count": 0}

    total = len(results)
    wins = sum(1 for r in results if r["is_win"])
    pnls = [r["pnl_pct"] for r in results]

    # 按预测分类
    by_pred = {}
    for pred_label in sorted(set(r["pred_label"] for r in results)):
        pred_results = [r for r in results if r["pred_label"] == pred_label]
        pred_wins = sum(1 for r in pred_results if r["is_win"])
        pred_pnls = [r["pnl_pct"] for r in pred_results]
        by_pred[pred_label] = {
            "name": LABEL_NAMES.get(pred_label, f"unknown_{pred_label}"),
            "count": len(pred_results),
            "win_rate": pred_wins / len(pred_results),
            "avg_pnl_pct": sum(pred_pnls) / len(pred_pnls),
            "total_pnl_pct": sum(pred_pnls),
        }

    # 筛选策略
    strong = [r for r in results if r["pred_label"] == 2]
    weak = [r for r in results if r["pred_label"] == 1]
    any_coin = [r for r in results if r["pred_label"] >= 1]

    def _stats(subset):
        if not subset:
            return {"count": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0}
        w = sum(1 for r in subset if r["is_win"])
        p = [r["pnl_pct"] for r in subset]
        return {"count": len(subset), "win_rate": w / len(subset), "avg_pnl_pct": sum(p) / len(p), "total_pnl_pct": sum(p)}

    return {
        "status": "ok",
        "model": label,
        "total": total,
        "overall": {"count": total, "win_rate": wins / total, "avg_pnl_pct": sum(pnls) / len(pnls), "total_pnl_pct": sum(pnls)},
        "by_prediction": {str(k): v for k, v in by_pred.items()},
        "filter_strong": _stats(strong),
        "filter_weak": _stats(weak),
        "filter_any_coin": _stats(any_coin),
    }


def main():
    logger.info("=" * 60)
    logger.info("Pipeline 回测验证：模型 vs 无模型")
    logger.info("=" * 60)

    trades = load_jsonl(CLOSED_TRADES_PATH)
    logger.info("加载 closed trades: %d 笔", len(trades))

    # 全量 baseline
    all_pnls = [float(t.get("pnl_pct", 0)) for t in trades]
    all_wins = sum(1 for t in trades if bool(t.get("is_win")) or float(t.get("pnl_pct", 0)) > 0)
    all_wr = all_wins / len(trades)
    all_avg = sum(all_pnls) / len(all_pnls)

    logger.info("全量入场（无模型）: %d 笔, win_rate=%.4f, avg_pnl=%.4f%%", len(trades), all_wr, all_avg)

    # 按 reason_family 统计
    by_reason = {}
    for t in trades:
        rf = t.get("reason_family", "unknown")
        if rf not in by_reason:
            by_reason[rf] = []
        by_reason[rf].append(float(t.get("pnl_pct", 0)))

    logger.info("\n按 reason_family 分布:")
    for rf, pnls in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        logger.info(f"  {rf:<40} n={len(pnls):>4} wr={wr:.4f} avg_pnl={sum(pnls)/len(pnls):.4f}%")

    # 加载模型并评估
    all_results = {
        "no_model": {
            "count": len(trades), "win_rate": all_wr,
            "avg_pnl_pct": all_avg, "total_pnl_pct": sum(all_pnls),
        },
        "models": {},
    }

    for name, model_dir in MODEL_DIRS.items():
        model = load_model(model_dir)
        if model is None:
            logger.warning("跳过 %s: 模型未找到", name)
            continue
        result = evaluate_model(model, trades, name)
        all_results["models"][name] = result

        if result["status"] == "ok":
            logger.info(f"\n--- {name} ---")
            for pred_label, stats in result["by_prediction"].items():
                logger.info(f"  预测={stats['name']:<6} n={stats['count']:>4} wr={stats['win_rate']:.4f} avg_pnl={stats['avg_pnl_pct']:.4f}% total={stats['total_pnl_pct']:.4f}%")
            f = result["filter_strong"]
            logger.info(f"  [强妖币筛选]     n={f['count']:>4} wr={f['win_rate']:.4f} avg_pnl={f['avg_pnl_pct']:.4f}%")
            f = result["filter_any_coin"]
            logger.info(f"  [弱+强妖币筛选]  n={f['count']:>4} wr={f['win_rate']:.4f} avg_pnl={f['avg_pnl_pct']:.4f}%")

    # 保存
    out_path = ROOT / "data/trading_system/model_training/paper_replay_autolearn/pipeline_backtest_comparison.json"

    def _to_native(obj):
        if isinstance(obj, dict):
            return {str(k): _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path.write_text(json.dumps(_to_native(all_results), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("\n结果已保存: %s", out_path)

    # 最终对比表
    logger.info("\n" + "=" * 70)
    logger.info("最终对比")
    logger.info("=" * 70)
    logger.info(f"{'策略':<30} {'笔数':>6} {'WinRate':>8} {'AvgPnL%':>10} {'TotalPnL%':>12}")
    logger.info(f"{'-' * 70}")
    logger.info(f"{'无模型(全量)':<30} {all_results['no_model']['count']:>6} {all_results['no_model']['win_rate']:>8.4f} {all_results['no_model']['avg_pnl_pct']:>10.4f} {all_results['no_model']['total_pnl_pct']:>12.4f}")

    for name, result in all_results["models"].items():
        if result["status"] != "ok":
            continue
        for suffix, key in [(" [强妖币]", "filter_strong"), (" [弱+强妖币]", "filter_any_coin")]:
            f = result[key]
            if f["count"] > 0:
                logger.info(f"{name + suffix:<30} {f['count']:>6} {f['win_rate']:>8.4f} {f['avg_pnl_pct']:>10.4f} {f['total_pnl_pct']:>12.4f}")


if __name__ == "__main__":
    main()
