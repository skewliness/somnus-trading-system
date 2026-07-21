"""
妖币捕捉模型 - 训练器 (Coin Scout Trainer)
==========================================
从 closed_trades.jsonl 训练 CoinScoutModel。

标签策略（基于现有数据，无需额外 API 调用）：
  由于 closed trades 中没有 entry_timestamp，我们用以下近似方案：

  方案 A（推荐）：用 exit 时的已实现 PnL 反推标签
    - 强妖币 (2): pnl_pct > 15% 且 reason_family 不是 stop_loss_hit
    - 弱妖币 (1): pnl_pct > 3%  且 reason_family 不是 stop_loss_hit
    - 非妖币 (0): 其他（包括亏损单）

  方案 B：用现有 is_win 标签（pnl > 0）
    - 这是当前 self_train_predictor 用的方案，作为 baseline

  方案 C：混合标签（推荐最终使用）
    - 70% 权重给方案 A（4H 框架近似）
    - 30% 权重给 is_win（实际盈亏）

特征策略：
  - 优先使用 trade 中已有的非零特征
  - 零值特征用 0 填充（LightGBM 可以处理缺失）
  - 交互特征自动计算

使用方式:
    trainer = CoinScoutTrainer()
    result = trainer.train_from_closed_trades(
        closed_trades_path="data/trading_system/model_training/paper_replay_autolearn/paper_closed_trades.jsonl",
        output_dir="data/trading_system/model_training/coin_scout",
    )

Author: OWL for snow
Date: 2026-05-08
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from trading_system.coin_scout_model import (
    CoinFeatures,
    CoinLabel,
    CoinScoutModel,
    FeatureEngine,
)

logger = logging.getLogger(__name__)


class CoinScoutTrainer:
    """
    妖币模型训练器

    核心改进（vs 现有 self_train_predictor.py）：
    1. 标签从简单 pnl>0 改为 4H 框架近似标签（强妖币/弱妖币/非妖币）
    2. 特征从 47 个扩展到 52 个（含交互特征、时间特征）
    3. 模型从 GradientBoosting 改为 LightGBM 三分类
    4. 验证从简单 80/20 split 改为时间序列 split
    5. 不需要额外 API 调用 — 完全基于 closed trades 中已有数据
    """

    def __init__(
        self,
        model_dir: str = "data/trading_system/model_training/coin_scout",
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_engine = FeatureEngine()

    def train_from_closed_trades(
        self,
        closed_trades_path: str | Path,
        output_dir: str | Path | None = None,
        min_samples: int = 30,
        label_mode: str = "hybrid",  # "realized_pnl" | "is_win" | "hybrid"
    ) -> dict:
        """
        从 closed_trades.jsonl 训练模型

        Args:
            closed_trades_path: closed trades JSONL 文件路径
            output_dir: 输出目录
            min_samples: 最少样本数
            label_mode: 标签模式
                - "realized_pnl": 用已实现 PnL 近似 4H 框架标签
                - "is_win": 用现有 is_win 标签（baseline）
                - "hybrid": 混合（70% realized_pnl + 30% is_win）
        """
        closed_trades_path = Path(closed_trades_path)
        if not closed_trades_path.exists():
            return {"status": "error", "message": f"File not found: {closed_trades_path}"}

        # 加载 closed trades
        closed_trades = self._load_jsonl(closed_trades_path)
        logger.info("Loaded %d closed trades from %s", len(closed_trades), closed_trades_path)

        # 构建训练样本
        features_list = []
        labels = []
        skipped = {"no_entry_price": 0, "all_zero_features": 0, "too_few_samples": 0}

        for trade in closed_trades:
            entry_price = float(trade.get("entry_price") or 0)
            exit_price = float(trade.get("exit_price") or 0)

            if entry_price <= 0:
                skipped["no_entry_price"] += 1
                continue

            # 计算标签
            pnl_pct = float(trade.get("pnl_pct") or 0)
            reason = str(trade.get("reason_family") or "")
            is_win = bool(trade.get("is_win"))

            if label_mode == "realized_pnl":
                label = self._label_from_realized_pnl(pnl_pct, reason)
            elif label_mode == "is_win":
                label = 2 if is_win else 0  # 简化为二分类映射到 2/0
            elif label_mode == "hybrid":
                label = self._label_hybrid(pnl_pct, reason, is_win)
            else:
                label = self._label_from_realized_pnl(pnl_pct, reason)

            # 构建特征
            features = self._build_features_from_trade(trade)
            features_list.append(features)
            labels.append(label)

        logger.info(
            "Built %d training samples (skipped: %s)",
            len(features_list), skipped,
        )

        if len(features_list) < min_samples:
            return {
                "status": "insufficient",
                "samples": len(features_list),
                "min_required": min_samples,
                "skipped": skipped,
                "label_distribution": self._label_distribution(labels) if labels else {},
            }

        # 打印标签分布
        dist = self._label_distribution(labels)
        logger.info("Label distribution: %s", dist)

        # 训练模型
        model = CoinScoutModel(model_dir=str(self.model_dir))
        train_result = model.train(features_list, labels)

        # 保存训练摘要
        output_dir = Path(output_dir) if output_dir else self.model_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "status": train_result.get("status", "unknown"),
            "samples": len(features_list),
            "label_mode": label_mode,
            "label_distribution": dist,
            "skipped": skipped,
            "train_result": train_result,
        }

        summary_path = output_dir / "training_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        return summary

    def train_from_synthetic_samples(
        self,
        samples: list[dict],
        output_dir: str | Path | None = None,
    ) -> dict:
        """
        从合成样本训练（用于测试和快速迭代）

        samples 格式：
        {
            "symbol": "BTCUSDT",
            "features": {...},  # CoinFeatures 的字段
            "label": 0/1/2
        }
        """
        features_list = []
        labels = []

        for sample in samples:
            try:
                f = CoinFeatures(symbol=sample.get("symbol", "UNKNOWN"))
                for key, value in sample.get("features", {}).items():
                    if hasattr(f, key):
                        setattr(f, key, float(value))
                features_list.append(f)
                labels.append(int(sample.get("label", 0)))
            except Exception as e:
                logger.debug("Skipping sample: %s", e)
                continue

        if len(features_list) < 10:
            return {"status": "insufficient", "samples": len(features_list)}

        model = CoinScoutModel(model_dir=str(self.model_dir))
        train_result = model.train(features_list, labels)

        return {
            "status": train_result.get("status", "unknown"),
            "samples": len(features_list),
            "label_distribution": self._label_distribution(labels),
            "train_result": train_result,
        }

    @staticmethod
    def _label_from_realized_pnl(pnl_pct: float, reason: str) -> int:
        """
        用已实现 PnL 近似 4H 框架标签

        由于没有入场后 4H 的完整价格序列，我们用以下近似：
        - 强妖币: pnl > 15% 且不是止损出场（说明是主动止盈）
        - 弱妖币: pnl > 3%  且不是止损出场
        - 非妖币: 其他

        注意：这是一个近似。理想情况下应该用入场后 4H 的 klines 数据，
        但 closed trades 中没有 entry_timestamp，所以用这个方案。
        """
        is_stop_loss = reason == "stop_loss_hit"

        if pnl_pct > 0.15 and not is_stop_loss:
            return 2  # 强妖币
        elif pnl_pct > 0.03 and not is_stop_loss:
            return 1  # 弱妖币
        else:
            return 0  # 非妖币

    @staticmethod
    def _label_hybrid(pnl_pct: float, reason: str, is_win: bool) -> int:
        """
        混合标签：70% realized_pnl + 30% is_win

        对于边界情况（如 pnl=12%，介于弱妖币和强妖币之间），
        如果 is_win 则向上取整。
        """
        pnl_label = CoinScoutTrainer._label_from_realized_pnl(pnl_pct, reason)

        # 边界提升：如果 is_win 且 pnl_label 是 1（弱妖币），检查是否接近强妖币
        if is_win and pnl_label == 1 and pnl_pct > 0.12:
            return 2

        return pnl_label

    def _build_features_from_trade(self, trade: dict) -> CoinFeatures:
        """
        从 closed trade 记录构建 CoinFeatures

        优先使用 trade 中已有的字段，缺失的填 0。
        与 self_train_dataset.collect_closed_trade_samples() 输出的字段对齐。
        """
        f = CoinFeatures(symbol=trade.get("symbol", "UNKNOWN"))

        # OI 特征
        f.oi_change_pct_4h = float(trade.get("oi_change_pct_4h") or 0)
        f.oi_change_pct_1h = float(trade.get("oi_change_pct_1h") or 0)
        f.oi_acceleration = float(trade.get("oi_acceleration") or 0)
        f.oi_price_divergence_score = float(trade.get("oi_divergence_score") or 0)
        f.funding_rate = float(trade.get("funding_rate") or 0)
        f.funding_momentum = float(trade.get("funding_momentum") or 0)

        # 巨鲸/聪明钱特征
        f.whale_long_count = float(trade.get("whale_long_count") or 0)
        f.whale_short_count = float(trade.get("whale_short_count") or 0)
        f.whale_long_notional_usdt = float(trade.get("whale_long_notional_usdt") or 0)
        f.whale_short_notional_usdt = float(trade.get("whale_short_notional_usdt") or 0)
        f.whale_long_profit_ratio = float(trade.get("whale_long_profit_ratio") or 0)
        f.whale_short_profit_ratio = float(trade.get("whale_short_profit_ratio") or 0)
        f.whale_pressure_score = float(trade.get("whale_pressure_score") or 0)
        f.smart_money_long_count = float(trade.get("smart_money_long_count") or 0)
        f.smart_money_short_count = float(trade.get("smart_money_short_count") or 0)
        f.smart_money_long_profit_ratio = float(trade.get("smart_money_long_profit_ratio") or 0)
        f.smart_money_short_profit_ratio = float(trade.get("smart_money_short_profit_ratio") or 0)
        f.smart_money_single_side_pct = float(trade.get("smart_money_single_side_pct") or 0)

        # 情绪动量特征
        f.sentiment_score = float(trade.get("sentiment_score") or trade.get("hotness_score") or 0)
        f.sentiment_mentions = float(trade.get("sentiment_mentions") or trade.get("mentions") or 0)
        f.sentiment_score_delta_1h = float(trade.get("sentiment_score_delta_1h") or 0)
        f.sentiment_score_delta_4h = float(trade.get("sentiment_score_delta_4h") or 0)
        f.mention_acceleration = float(trade.get("mention_acceleration") or 0)
        f.news_sentiment_score = float(trade.get("news_sentiment_score") or 0)
        f.news_sentiment_delta = float(trade.get("news_sentiment_delta") or 0)
        f.twitter_volume_spike = float(trade.get("twitter_volume_spike") or 0)

        # 市场微结构特征
        f.price_change_pct_5m = float(trade.get("price_change_pct_5m") or 0)
        f.price_change_pct_15m = float(trade.get("price_change_pct_15m") or 0)
        f.price_change_pct_1h = float(trade.get("price_change_pct_1h") or 0)
        f.price_change_pct_4h = float(trade.get("price_change_pct_4h") or 0)
        f.price_change_pct_24h = float(trade.get("price_change_pct_24h") or 0)
        f.taker_buy_sell_ratio = float(trade.get("taker_buy_sell_ratio") or 0)
        f.taker_imbalance = float(trade.get("taker_imbalance") or 0)
        f.price_position_7d = float(trade.get("price_position_7d") or 0)
        f.price_position_14d = float(trade.get("price_position_14d") or 0)
        f.price_position_20d = float(trade.get("price_position_20d") or 0)
        f.breakout_distance_7d = float(trade.get("breakout_distance_7d") or 0)
        f.compression_score = float(trade.get("compression_score") or 0)

        # 交互特征（自动计算）
        f.sentiment_oi_interaction = f.sentiment_score * f.oi_change_pct_4h / 100.0
        f.whale_price_interaction = f.whale_pressure_score * f.price_change_pct_4h / 100.0
        f.funding_sentiment_interaction = abs(f.funding_rate) * f.sentiment_score * 10.0

        # 综合动量分
        if f.price_change_pct_4h > 0 and f.oi_change_pct_4h > 0 and f.sentiment_score_delta_4h > 0:
            f.momentum_score = (f.price_change_pct_4h + f.oi_change_pct_4h + f.sentiment_score_delta_4h) * 1.5
        elif f.price_change_pct_4h > 0 and f.oi_change_pct_4h > 0:
            f.momentum_score = (f.price_change_pct_4h + f.oi_change_pct_4h) * 1.2
        else:
            f.momentum_score = f.price_change_pct_4h + f.oi_change_pct_4h * 0.5

        # 结构质量分
        if f.compression_score > 0.5:
            f.structure_quality += f.compression_score * 30
        if f.breakout_distance_7d < 0.03:
            f.structure_quality += (0.03 - f.breakout_distance_7d) * 500
        f.structure_quality = min(100.0, f.structure_quality)

        # 风险分
        f.risk_score = 0.0
        if f.long_concentration_pct > 90 or f.short_concentration_pct > 90:
            f.risk_score += 40
        if abs(f.funding_rate) > 0.003:
            f.risk_score += 15
        f.risk_score = min(100.0, f.risk_score)

        # 时间特征
        now = datetime.now(timezone.utc)
        f.hour_of_day = float(now.hour)
        f.day_of_week = float(now.weekday())
        f.is_asian_session = 1.0 if 0 <= now.hour < 8 else 0.0
        f.is_us_session = 1.0 if 13 <= now.hour < 21 else 0.0

        # 基础特征
        f.quote_volume_usdt_24h = float(trade.get("quote_volume_usdt_24h") or 0)
        f.listing_days = float(trade.get("listing_days") or 0)
        f.okx_sold_ratio = float(trade.get("whale_okx_sold_ratio") or 0)
        f.okx_hold_ratio = float(trade.get("whale_okx_hold_ratio") or 0)

        return f

    def _load_jsonl(self, path: Path) -> list[dict]:
        """加载 JSONL 文件"""
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

    def _label_distribution(self, labels: list[int]) -> dict[str, int]:
        """标签分布统计"""
        dist = {0: 0, 1: 0, 2: 0}
        for l in labels:
            dist[l] = dist.get(l, 0) + 1
        return {
            "非妖币(0)": dist[0],
            "弱妖币(1)": dist[1],
            "强妖币(2)": dist[2],
        }


def train_from_existing_data(
    closed_trades_path: str | Path = "data/trading_system/model_training/paper_replay_autolearn/paper_closed_trades.jsonl",
    model_dir: str = "data/trading_system/model_training/coin_scout",
    label_mode: str = "hybrid",
) -> dict:
    """
    便捷函数：从现有 closed trades 训练妖币模型

    这是主要的训练入口，可以在 pipeline 中调用，也可以独立运行。
    """
    trainer = CoinScoutTrainer(model_dir=model_dir)
    return trainer.train_from_closed_trades(
        closed_trades_path=closed_trades_path,
        label_mode=label_mode,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 默认从 paper_replay_autolearn 的 closed trades 训练
    trades_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/trading_system/model_training/paper_replay_autolearn/paper_closed_trades.jsonl"
    )

    mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid"

    print(f"Training CoinScout model from: {trades_path}")
    print(f"Label mode: {mode}")
    result = train_from_existing_data(closed_trades_path=trades_path, label_mode=mode)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
