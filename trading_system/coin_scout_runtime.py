"""
妖币捕捉模型 - 实时特征构建器 (Coin Scout Runtime)
===================================================
将现有数据源（MarketSnapshot + WhaleOverview）转换为 CoinFeatures，
供 CoinScoutModel 实时评分。

使用方式:
    runtime = CoinScoutRuntime(model_dir="data/trading_system/model_training/coin_scout")
    features = runtime.build_features_from_snapshot(snapshot, whale_overview)
    decision = runtime.evaluate(snapshot, whale_overview, current_position)

Author: OWL for snow
Date: 2026-05-08
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.coin_scout_model import (
    CoinFeatures,
    CoinLabel,
    CoinScoutModel,
    FeatureEngine,
    OIMomentumData,
    MarketMicrostructureData,
    ScoutDecision,
    ScoutDecisionEngine,
    SentimentMomentumData,
    SquareWhaleOpinionData,
    WhaleSmartMoneyData,
)
from trading_system.models import MarketSnapshot, Position

logger = logging.getLogger(__name__)


class CoinScoutRuntime:
    """
    实时特征构建 + 模型推理运行时

    将 MarketSnapshot（来自 BinanceMarketDataProvider.collect()）
    和 whale_overview（来自 build_whale_overview()）统一转换为 CoinFeatures。
    """

    # 拉哪广场蒸馏 bridge 文件路径
    LANAII_SQUARE_DISTILL_PATH = Path(
        "data/trading_system/skill_bridges/lanaai-square-distill.json"
    )

    def __init__(
        self,
        model_dir: str = "data/trading_system/model_training/coin_scout",
        sentiment_history: dict | None = None,
        lanaai_distill_path: str | Path | None = None,
    ):
        self.model = CoinScoutModel(model_dir=model_dir)
        self.decision_engine = ScoutDecisionEngine()
        self.feature_engine = FeatureEngine()
        self.sentiment_history = sentiment_history or {}
        self.lanaai_distill_path = Path(lanaai_distill_path) if lanaai_distill_path else self.LANAII_SQUARE_DISTILL_PATH
        self._lanaai_cache: dict[str, SquareWhaleOpinionData] = {}
        self._lanaai_cache_loaded = False

        # 尝试加载已训练模型
        if self.model.load_model():
            logger.info("CoinScout model loaded from %s", model_dir)
        else:
            logger.warning("No trained CoinScout model found at %s — predictions will fail until trained", model_dir)

    def build_features_from_snapshot(
        self,
        snapshot: MarketSnapshot,
        whale_overview: dict | None = None,
        sentiment_history: dict | None = None,
    ) -> CoinFeatures:
        """
        从 MarketSnapshot + whale_overview + 拉哪广场蒸馏 构建 CoinFeatures

        Args:
            snapshot: 来自 BinanceMarketDataProvider 的市场快照
            whale_overview: 来自 build_whale_overview() 的鲸鱼数据（可选）
            sentiment_history: 历史情绪数据，用于计算动量（可选）
        """
        whale_data = self._parse_whale_data(whale_overview)
        oi_data = self._parse_oi_data(snapshot, whale_overview)
        sentiment_data = self._parse_sentiment_data(snapshot, sentiment_history)
        market_data = self._parse_market_data(snapshot)
        square_data = self._get_lanaai_data(snapshot.symbol)

        return self.feature_engine.build_features(
            symbol=snapshot.symbol,
            oi_data=oi_data,
            whale_data=whale_data,
            sentiment_data=sentiment_data,
            market_data=market_data,
            square_data=square_data,
        )

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        whale_overview: dict | None = None,
        current_position: dict | None = None,
        sentiment_history: dict | None = None,
    ) -> ScoutDecision:
        """
        完整评估流程：特征构建 → 模型预测 → 交易决策

        Args:
            snapshot: 市场快照
            whale_overview: 鲸鱼数据
            current_position: 当前持仓信息（None = 无持仓）
            sentiment_history: 历史情绪数据

        Returns:
            ScoutDecision 交易决策
        """
        if not self.model.is_trained:
            return ScoutDecision(
                symbol=snapshot.symbol,
                action="skip",
                reasons=["模型未训练"],
            )

        # 构建特征
        features = self.build_features_from_snapshot(
            snapshot, whale_overview, sentiment_history
        )

        # 模型预测
        prediction = self.model.predict(features)
        if prediction.get("status") == "error":
            return ScoutDecision(
                symbol=snapshot.symbol,
                action="skip",
                reasons=[f"预测错误: {prediction.get('message', '')}"],
            )

        # 交易决策
        return self.decision_engine.decide(
            symbol=snapshot.symbol,
            prediction=prediction,
            features=features,
            current_position=current_position,
        )

    def _parse_whale_data(self, whale_overview: dict | None) -> WhaleSmartMoneyData:
        """解析 whale_overview 为 WhaleSmartMoneyData"""
        data = WhaleSmartMoneyData()
        if not whale_overview:
            return data

        # 解析 smart_money_context
        smart = whale_overview.get("smart_money_context") or {}
        if isinstance(smart, dict):
            data.smart_money_long_count = int(smart.get("smart_money_long_count") or 0)
            data.smart_money_short_count = int(smart.get("smart_money_short_count") or 0)
            data.smart_money_long_profit_ratio = float(smart.get("smart_money_long_profit_ratio") or 0)
            data.smart_money_short_profit_ratio = float(smart.get("smart_money_short_profit_ratio") or 0)
            data.smart_money_single_side_pct = float(smart.get("single_side_pct") or 0)
            data.smart_money_oi_composition = str(smart.get("oi_composition") or "")
            data.smart_money_symbol_class = str(smart.get("symbol_class") or "")
            data.smart_money_monitor_action = str(smart.get("monitor_action") or "")

        # 解析 long_group / short_group
        long_group = whale_overview.get("long_group") or {}
        short_group = whale_overview.get("short_group") or {}
        if isinstance(long_group, dict):
            data.whale_long_count = int(long_group.get("whale_count") or 0)
            data.whale_long_notional_usdt = float(long_group.get("current_position_usdt") or 0)
            data.whale_long_avg_entry_price = float(long_group.get("average_entry_price") or 0)
            data.whale_long_pnl_usdt = float(
                long_group.get("unrealized_pnl_usdt") or long_group.get("estimated_pnl_usdt") or 0
            )
            data.whale_long_profit_ratio = float(long_group.get("profit_ratio_pct") or 0)
        if isinstance(short_group, dict):
            data.whale_short_count = int(short_group.get("whale_count") or 0)
            data.whale_short_notional_usdt = float(short_group.get("current_position_usdt") or 0)
            data.whale_short_avg_entry_price = float(short_group.get("average_entry_price") or 0)
            data.whale_short_pnl_usdt = float(
                short_group.get("unrealized_pnl_usdt") or short_group.get("estimated_pnl_usdt") or 0
            )
            data.whale_short_profit_ratio = float(short_group.get("profit_ratio_pct") or 0)

        # 解析 summary
        summary = whale_overview.get("summary") or {}
        if isinstance(summary, dict):
            data.whale_pressure_score = float(summary.get("pressure_score") or 0)

        # 解析 okx_signal_context
        okx = whale_overview.get("okx_signal_context") or {}
        if isinstance(okx, dict):
            data.okx_sold_ratio = float(okx.get("avg_sold_ratio") or 0)
            data.okx_hold_ratio = float(okx.get("avg_hold_ratio") or 0)
            data.okx_hotness_score = float(okx.get("hotness_score") or 0)
            data.okx_mention_count = int(okx.get("mention_count") or 0)

        # 解析 market 聚合
        market = whale_overview.get("market") or {}
        if isinstance(market, dict):
            data.global_long_short_ratio = float(market.get("global_long_short_ratio") or 0)
            data.top_long_short_ratio = float(market.get("top_long_short_ratio") or 0)

        return data

    def _parse_oi_data(self, snapshot: MarketSnapshot, whale_overview: dict | None) -> OIMomentumData:
        """解析 OI 数据"""
        data = OIMomentumData()
        data.current_oi_usdt = snapshot.open_interest_usdt
        data.oi_change_pct_4h = snapshot.oi_change_pct_4h
        data.oi_price_divergence_score = snapshot.oi_price_divergence_score
        data.funding_rate = snapshot.funding_rate

        # 从 whale_overview 获取额外 OI 数据
        if whale_overview:
            market = whale_overview.get("market") or {}
            if isinstance(market, dict):
                data.oi_change_pct_1h = float(market.get("oi_change_pct_1h") or 0)
                data.oi_acceleration = float(market.get("oi_acceleration") or 0)

            # 单边集中度
            smart = whale_overview.get("smart_money_context") or {}
            if isinstance(smart, dict):
                oi_comp = smart.get("oi_composition") or {}
                if isinstance(oi_comp, dict):
                    data.long_concentration_pct = float(oi_comp.get("long_pct") or 0)
                    data.short_concentration_pct = float(oi_comp.get("short_pct") or 0)

            # 资金费率动量
            funding_hist = whale_overview.get("funding_history") or []
            if len(funding_hist) >= 2:
                current = float(funding_hist[-1].get("funding_rate") or 0)
                previous = float(funding_hist[0].get("funding_rate") or 0)
                data.funding_momentum = current - previous

        # 集中度风险判断
        if data.long_concentration_pct > 90 or data.short_concentration_pct > 90:
            data.concentration_risk = "danger"
        elif data.long_concentration_pct > 70 or data.short_concentration_pct > 70:
            data.concentration_risk = "warning"
        else:
            data.concentration_risk = "normal"

        return data

    def _parse_sentiment_data(
        self, snapshot: MarketSnapshot, sentiment_history: dict | None
    ) -> SentimentMomentumData:
        """解析情绪数据"""
        data = SentimentMomentumData()
        data.current_score = snapshot.sentiment_score
        data.mentions = snapshot.sentiment_mentions
        data.source = snapshot.sentiment_source

        # 从历史数据计算动量
        if sentiment_history:
            symbol_history = sentiment_history.get(snapshot.symbol, [])
            if len(symbol_history) >= 2:
                # 计算 1h 和 4h 情绪变化
                now = datetime.now(timezone.utc)
                for entry in reversed(symbol_history):
                    entry_time = entry.get("timestamp")
                    if entry_time:
                        delta_hours = (now - entry_time).total_seconds() / 3600
                        if delta_hours >= 4 and data.score_delta_4h == 0:
                            data.score_delta_4h = snapshot.sentiment_score - entry.get("score", 0)
                            data.mentions_delta_4h = snapshot.sentiment_mentions - entry.get("mentions", 0)
                        elif delta_hours >= 1 and data.score_delta_1h == 0:
                            data.score_delta_1h = snapshot.sentiment_score - entry.get("score", 0)
                            data.mentions_delta_1h = snapshot.sentiment_mentions - entry.get("mentions", 0)

                # 提及量加速度
                if len(symbol_history) >= 3:
                    recent = symbol_history[-1].get("mentions", 0)
                    mid = symbol_history[-2].get("mentions", 0)
                    old = symbol_history[-3].get("mentions", 0)
                    data.mention_acceleration = (recent - mid) - (mid - old)

        return data

    def _parse_market_data(self, snapshot: MarketSnapshot) -> MarketMicrostructureData:
        """解析市场微结构数据"""
        data = MarketMicrostructureData()
        data.mark_price = snapshot.mark_price
        data.price_change_pct_5m = snapshot.price_change_pct_5m
        data.price_change_pct_4h = snapshot.price_change_pct_4h
        data.price_change_pct_24h = snapshot.price_change_pct_24h
        data.quote_volume_usdt_24h = snapshot.quote_volume_usdt_24h
        data.taker_buy_sell_ratio = snapshot.taker_buy_sell_ratio
        data.price_position_7d = snapshot.price_position_7d or 0
        data.price_position_14d = snapshot.price_position_14d or 0
        data.price_position_20d = snapshot.price_position_20d or 0
        data.breakout_distance_7d = snapshot.breakout_distance_7d or 0
        data.compression_score = snapshot.compression_score or 0

        # 计算 taker_imbalance
        if data.taker_buy_sell_ratio > 0:
            data.taker_imbalance = (data.taker_buy_sell_ratio - 1.0) / (data.taker_buy_sell_ratio + 1.0)

        return data

    def _get_lanaai_data(self, symbol: str) -> SquareWhaleOpinionData | None:
        """
        从拉哪广场蒸馏 bridge 读取该 symbol 的广场鲸鱼意见领袖数据。

        缓存机制：首次调用时加载整个 bridge 文件，后续调用从缓存读取。
        如果 bridge 文件不存在或符号不在其中，返回 None（特征全填 0）。
        """
        if not self._lanaai_cache_loaded:
            self._load_lanaai_cache()

        return self._lanaai_cache.get(symbol)

    def _load_lanaai_cache(self) -> None:
        """加载拉哪广场蒸馏 bridge 文件到内存缓存"""
        self._lanaai_cache = {}
        self._lanaai_cache_loaded = True

        if not self.lanaai_distill_path.exists():
            logger.debug("Lanaai distill file not found: %s", self.lanaai_distill_path)
            return

        try:
            import json
            data = json.loads(self.lanaai_distill_path.read_text(encoding="utf-8"))

            # 从 metadata 获取作者信息
            metadata = data.get("metadata", {})
            author_followers = int(metadata.get("followers", 0))
            author_post_count = int(metadata.get("post_count", 0))

            # 从 signals 构建 symbol → SquareWhaleOpinionData 映射
            # 取每个 symbol 的最高热度信号
            signals = data.get("signals", [])

            # 也从 selected_posts 获取更详细的互动数据
            posts_by_symbol: dict[str, list[dict]] = {}
            for post in data.get("selected_posts", []):
                sym = post.get("symbol", "")
                if sym:
                    posts_by_symbol.setdefault(sym, []).append(post)

            for signal in signals:
                sym = signal.get("symbol", "")
                if not sym:
                    continue

                sq = SquareWhaleOpinionData()
                sq.hotness_score = float(signal.get("hotness_score", 0))
                sq.mention_count = int(signal.get("mention_count", 0))

                # 从 selected_posts 汇总互动数据
                posts = posts_by_symbol.get(sym, [])
                sq.views = sum(int(p.get("views", 0)) for p in posts)
                sq.likes = sum(int(p.get("likes", 0)) for p in posts)
                sq.comments = sum(int(p.get("comments", 0)) for p in posts)
                sq.engagement_total = float(sq.likes + sq.comments)
                sq.engagement_view_ratio = sq.engagement_total / sq.views if sq.views > 0 else 0.0

                # 作者权重 = 粉丝数 × 互动率
                sq.author_followers = author_followers
                sq.author_post_count = author_post_count
                sq.author_weight = float(author_followers) * sq.engagement_view_ratio / 1000.0

                # 帖子新鲜度（小时）
                captured_at = data.get("captured_at") or metadata.get("captured_at")
                if captured_at:
                    try:
                        from datetime import datetime, timezone
                        cap_dt = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
                        age_hours = (datetime.now(timezone.utc) - cap_dt).total_seconds() / 3600.0
                        sq.post_recency_hours = max(0.0, age_hours)
                    except Exception:
                        sq.post_recency_hours = 0.0

                # 简单情绪分析：根据 hotness_score 和 notes 判断看涨程度
                notes = signal.get("notes", "")
                # 从 notes 提取情绪关键词
                bullish_keywords = ["爆发", "猛", "妖", "拉升", "强势", "看好", "最稳", "复刻", "狂潮", "新周期"]
                bearish_keywords = ["弱", "回调", "结束", "波动太大", "风险"]
                bull_count = sum(1 for kw in bullish_keywords if kw in notes)
                bear_count = sum(1 for kw in bearish_keywords if kw in notes)
                sq.bullish_score = float((bull_count - bear_count) * 20)  # -100 ~ +100

                # 是否实盘（从帖子文本判断）
                sq.is_live_trade = any(
                    kw in notes for kw in ["实盘", "仓位", "买入", "卖出", "止损", "止盈", "杠杆", "合约"]
                )

                self._lanaai_cache[sym] = sq

            logger.info(
                "Loaded lanaai distill data: %d symbols from %s",
                len(self._lanaai_cache), self.lanaai_distill_path,
            )

        except Exception as e:
            logger.warning("Failed to load lanaai distill: %s", e)


def evaluate_snapshot(
    snapshot: MarketSnapshot,
    whale_overview: dict | None = None,
    current_position: dict | None = None,
    model_dir: str = "data/trading_system/model_training/coin_scout",
) -> ScoutDecision:
    """
    便捷函数：单次评估

    用于 pipeline 中快速调用，无需显式创建 runtime。
    """
    runtime = CoinScoutRuntime(model_dir=model_dir)
    return runtime.evaluate(
        snapshot=snapshot,
        whale_overview=whale_overview,
        current_position=current_position,
    )


# ============================================================
# Whale Scout Runtime（聪明钱/巨鲸模型推理）
# ============================================================

class WhaleScoutRuntime:
    """
    聪明钱/巨鲸模型实时推理运行时

    使用从 FIFO 配对平仓记录训练的 CoinScoutModel（whale_scout 版本），
    对候选币种进行妖币概率预测。

    与 CoinScoutRuntime 的区别：
    - 加载 whale_scout 目录下的模型（用聪明钱实际盈亏训练）
    - 输出格式相同（ScoutDecision），但来源标注为 whale_scout
    - 可作为 coin_scout 的补充/对比信号

    使用方式:
        runtime = WhaleScoutRuntime(model_dir="data/trading_system/model_training/whale_scout")
        decision = runtime.evaluate(snapshot, whale_overview)
    """

    def __init__(
        self,
        model_dir: str = "data/trading_system/model_training/whale_scout",
    ):
        self.model = CoinScoutModel(model_dir=model_dir)
        self.decision_engine = ScoutDecisionEngine()
        self.feature_engine = FeatureEngine()

        # 尝试加载已训练模型
        if self.model.load_model():
            logger.info("WhaleScout model loaded from %s", model_dir)
        else:
            logger.warning("No trained WhaleScout model found at %s — predictions will fail until trained", model_dir)

    def build_features_from_snapshot(
        self,
        snapshot: MarketSnapshot,
        whale_overview: dict | None = None,
    ) -> CoinFeatures:
        """
        从 MarketSnapshot + whale_overview 构建 CoinFeatures
        （复用 CoinScoutRuntime 的特征构建逻辑）
        """
        whale_data = self._parse_whale_data(whale_overview)
        oi_data = self._parse_oi_data(snapshot, whale_overview)
        sentiment_data = self._parse_sentiment_data(snapshot)
        market_data = self._parse_market_data(snapshot)

        return self.feature_engine.build_features(
            symbol=snapshot.symbol,
            oi_data=oi_data,
            whale_data=whale_data,
            sentiment_data=sentiment_data,
            market_data=market_data,
            square_data=None,
        )

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        whale_overview: dict | None = None,
        current_position: dict | None = None,
    ) -> ScoutDecision:
        """
        完整评估流程：特征构建 → 模型预测 → 交易决策
        """
        if not self.model.is_trained:
            return ScoutDecision(
                symbol=snapshot.symbol,
                action="skip",
                reasons=["WhaleScout 模型未训练"],
            )

        features = self.build_features_from_snapshot(snapshot, whale_overview)
        prediction = self.model.predict(features)
        if prediction.get("status") == "error":
            return ScoutDecision(
                symbol=snapshot.symbol,
                action="skip",
                reasons=[f"WhaleScout 预测错误: {prediction.get('message', '')}"],
            )

        decision = self.decision_engine.decide(
            symbol=snapshot.symbol,
            prediction=prediction,
            features=features,
            current_position=current_position,
        )
        # 标记来源
        decision.reasons = [f"[whale_scout] {r}" for r in decision.reasons]
        return decision

    def _parse_whale_data(self, whale_overview: dict | None):
        """解析 whale_overview 为 WhaleSmartMoneyData"""
        from trading_system.coin_scout_model import WhaleSmartMoneyData
        data = WhaleSmartMoneyData()
        if not whale_overview:
            return data

        smart = whale_overview.get("smart_money_context") or {}
        if isinstance(smart, dict):
            data.smart_money_long_count = int(smart.get("smart_money_long_count") or 0)
            data.smart_money_short_count = int(smart.get("smart_money_short_count") or 0)
            data.smart_money_long_profit_ratio = float(smart.get("smart_money_long_profit_ratio") or 0)
            data.smart_money_short_profit_ratio = float(smart.get("smart_money_short_profit_ratio") or 0)
            data.smart_money_single_side_pct = float(smart.get("single_side_pct") or 0)

        long_group = whale_overview.get("long_group") or {}
        short_group = whale_overview.get("short_group") or {}
        if isinstance(long_group, dict):
            data.whale_long_count = int(long_group.get("whale_count") or 0)
            data.whale_long_notional_usdt = float(long_group.get("current_position_usdt") or 0)
            data.whale_long_avg_entry_price = float(long_group.get("average_entry_price") or 0)
            data.whale_long_pnl_usdt = float(
                long_group.get("unrealized_pnl_usdt") or long_group.get("estimated_pnl_usdt") or 0
            )
            data.whale_long_profit_ratio = float(long_group.get("profit_ratio_pct") or 0)
        if isinstance(short_group, dict):
            data.whale_short_count = int(short_group.get("whale_count") or 0)
            data.whale_short_notional_usdt = float(short_group.get("current_position_usdt") or 0)
            data.whale_short_avg_entry_price = float(short_group.get("average_entry_price") or 0)
            data.whale_short_pnl_usdt = float(
                short_group.get("unrealized_pnl_usdt") or short_group.get("estimated_pnl_usdt") or 0
            )
            data.whale_short_profit_ratio = float(short_group.get("profit_ratio_pct") or 0)

        summary = whale_overview.get("summary") or {}
        if isinstance(summary, dict):
            data.whale_pressure_score = float(summary.get("pressure_score") or 0)

        okx = whale_overview.get("okx_signal_context") or {}
        if isinstance(okx, dict):
            data.okx_sold_ratio = float(okx.get("avg_sold_ratio") or 0)
            data.okx_hold_ratio = float(okx.get("avg_hold_ratio") or 0)

        market = whale_overview.get("market") or {}
        if isinstance(market, dict):
            data.global_long_short_ratio = float(market.get("global_long_short_ratio") or 0)
            data.top_long_short_ratio = float(market.get("top_long_short_ratio") or 0)

        return data

    def _parse_oi_data(self, snapshot: MarketSnapshot, whale_overview: dict | None):
        """解析 OI 数据"""
        from trading_system.coin_scout_model import OIMomentumData
        data = OIMomentumData()
        data.current_oi_usdt = snapshot.open_interest_usdt
        data.oi_change_pct_4h = snapshot.oi_change_pct_4h
        data.oi_price_divergence_score = snapshot.oi_price_divergence_score
        data.funding_rate = snapshot.funding_rate

        if whale_overview:
            market = whale_overview.get("market") or {}
            if isinstance(market, dict):
                data.oi_change_pct_1h = float(market.get("oi_change_pct_1h") or 0)
                data.oi_acceleration = float(market.get("oi_acceleration") or 0)

            smart = whale_overview.get("smart_money_context") or {}
            if isinstance(smart, dict):
                oi_comp = smart.get("oi_composition") or {}
                if isinstance(oi_comp, dict):
                    data.long_concentration_pct = float(oi_comp.get("long_pct") or 0)
                    data.short_concentration_pct = float(oi_comp.get("short_pct") or 0)

            funding_hist = whale_overview.get("funding_history") or []
            if len(funding_hist) >= 2:
                current = float(funding_hist[-1].get("funding_rate") or 0)
                previous = float(funding_hist[0].get("funding_rate") or 0)
                data.funding_momentum = current - previous

        if data.long_concentration_pct > 90 or data.short_concentration_pct > 90:
            data.concentration_risk = "danger"
        elif data.long_concentration_pct > 70 or data.short_concentration_pct > 70:
            data.concentration_risk = "warning"
        else:
            data.concentration_risk = "normal"

        return data

    def _parse_sentiment_data(self, snapshot: MarketSnapshot):
        """解析情绪数据"""
        from trading_system.coin_scout_model import SentimentMomentumData
        data = SentimentMomentumData()
        data.current_score = snapshot.sentiment_score
        data.mentions = snapshot.sentiment_mentions
        data.source = snapshot.sentiment_source
        return data

    def _parse_market_data(self, snapshot: MarketSnapshot):
        """解析市场微结构数据"""
        from trading_system.coin_scout_model import MarketMicrostructureData
        data = MarketMicrostructureData()
        data.mark_price = snapshot.mark_price
        data.price_change_pct_5m = snapshot.price_change_pct_5m
        data.price_change_pct_4h = snapshot.price_change_pct_4h
        data.price_change_pct_24h = snapshot.price_change_pct_24h
        data.quote_volume_usdt_24h = snapshot.quote_volume_usdt_24h
        data.taker_buy_sell_ratio = snapshot.taker_buy_sell_ratio
        data.price_position_7d = snapshot.price_position_7d or 0
        data.price_position_14d = snapshot.price_position_14d or 0
        data.price_position_20d = snapshot.price_position_20d or 0
        data.breakout_distance_7d = snapshot.breakout_distance_7d or 0
        data.compression_score = snapshot.compression_score or 0

        if data.taker_buy_sell_ratio > 0:
            data.taker_imbalance = (data.taker_buy_sell_ratio - 1.0) / (data.taker_buy_sell_ratio + 1.0)

        return data
