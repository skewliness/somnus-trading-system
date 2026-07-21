"""
妖币捕捉模型 (Coin Scout Model)
================================
核心哲学：
  早发现，小仓进，4H 拿，回踩加，破位走。

  4H 定方向，15m 找触发。
  不等完美确认，第一次强信号可小仓。
  多次信号且结构没破，继续做，不封死。
  不用 24H 固定平仓。
  不因一次 OI/taker 变弱就跑。
  OI 只辅助，看单边集中度：<70% 健康慢涨，90%+ 是庄盘风险。
  浮亏不急止损，4H 破位才退。
  不做 SOL / SOL meme。

模型架构：
  所有数据源（OI、巨鲸/聪明钱、新闻情绪、市场微结构）→ 特征工程 → LightGBM → 妖币概率

标签定义（4H 框架，side-aware）：
  V6 smart-money 训练目标改为二分类：
  方向有效 = 入场后 4H/24H 窗口内顺方向走势有效且回撤可接受
  方向无效 = 其他

Author: OWL for snow
Date: 2026-05-08
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ============================================================
# 1. 数据结构
# ============================================================

# Feature source documentation: which features are real vs proxy in each path.
# This is the authoritative reference for train/inference distribution alignment.
FEATURE_SOURCE_MAP = {
    "runtime (CoinScoutRuntime)": {
        "oi": "real (Binance OI API)",
        "whale": "real (whale panel live)",
        "sentiment": "real (aggregated sentiment)",
        "lanaai": "real (square distill)",
        "microstructure": "real (Binance ticker/klines)",
        "interaction": "computed from real inputs",
    },
    "decision_samples (coin_scout_4h_labeler)": {
        "oi": "partial (oi_change_pct_4h from decision snapshot, no history)",
        "whale": "from decision snapshot if available, else 0",
        "sentiment": "from decision snapshot if available, else 0",
        "lanaai": "always 0 (not captured at decision time)",
        "microstructure": "partial (price changes from snapshot)",
        "interaction": "computed from available inputs (degraded)",
    },
    "smart_money (smart_money_trainer)": {
        "oi": "real (fetched from Binance API at build time)",
        "whale": "PROXY (derived from long/short ratio, not real whale data)",
        "sentiment": "PROXY (derived from trader_roi/winRate)",
        "lanaai": "always 0",
        "microstructure": "real (fetched from Binance API at build time)",
        "interaction": "computed from proxy inputs",
    },
}


@dataclass(slots=True)
class WhaleSmartMoneyData:
    """巨鲸/聪明钱面板数据"""
    # 聪明钱信号
    smart_money_long_count: int = 0
    smart_money_short_count: int = 0
    smart_money_long_profit_ratio: float = 0.0
    smart_money_short_profit_ratio: float = 0.0
    smart_money_oi_composition: str = ""
    smart_money_symbol_class: str = ""
    smart_money_monitor_action: str = ""
    smart_money_single_side_pct: float = 0.0

    # 巨鲸数据
    whale_long_count: int = 0
    whale_short_count: int = 0
    whale_long_notional_usdt: float = 0.0
    whale_short_notional_usdt: float = 0.0
    whale_long_avg_entry_price: float = 0.0
    whale_short_avg_entry_price: float = 0.0
    whale_long_pnl_usdt: float = 0.0
    whale_short_pnl_usdt: float = 0.0
    whale_long_profit_ratio: float = 0.0
    whale_short_profit_ratio: float = 0.0
    whale_pressure_score: float = 0.0  # -100 ~ +100

    # OKX 信号
    okx_sold_ratio: float = 0.0
    okx_hold_ratio: float = 0.0
    okx_hotness_score: float = 0.0
    okx_mention_count: int = 0

    # 市场聚合
    global_long_account_ratio: float = 0.0
    global_short_account_ratio: float = 0.0
    global_long_short_ratio: float = 0.0
    top_long_account_ratio: float = 0.0
    top_short_account_ratio: float = 0.0
    top_long_short_ratio: float = 0.0


@dataclass(slots=True)
class SentimentMomentumData:
    """情绪动量数据"""
    current_score: float = 0.0
    mentions: int = 0
    source: str = ""

    # 动量（需要历史数据计算）
    score_delta_1h: float = 0.0      # 1h 情绪分变化
    score_delta_4h: float = 0.0      # 4h 情绪分变化
    mentions_delta_1h: int = 0       # 1h 提及量变化
    mentions_delta_4h: int = 0       # 4h 提及量变化
    mention_acceleration: float = 0.0  # 提及量加速度

    # 新闻情绪
    news_sentiment_score: float = 0.0
    news_sentiment_delta: float = 0.0
    news_volume: int = 0

    # Twitter 情绪
    twitter_sentiment_score: float = 0.0
    twitter_volume_spike: float = 0.0  # Twitter 量突增倍数


@dataclass(slots=True)
class OIMomentumData:
    """OI 动量数据"""
    current_oi_usdt: float = 0.0
    oi_change_pct_4h: float = 0.0
    oi_change_pct_1h: float = 0.0     # 需要额外采集
    oi_acceleration: float = 0.0       # OI 变化加速度 (当前4h - 上一个4h)
    oi_price_divergence_score: float = 0.0

    # 单边集中度（关键指标）
    long_concentration_pct: float = 0.0   # 多头集中度
    short_concentration_pct: float = 0.0  # 空头集中度
    concentration_risk: str = "normal"     # normal / warning / danger

    # 资金费率
    funding_rate: float = 0.0
    funding_momentum: float = 0.0      # 资金费率变化率


@dataclass(slots=True)
class SquareWhaleOpinionData:
    """广场鲸鱼意见领袖数据（拉哪等 KOL 的广场发言蒸馏）"""
    # 核心信号
    hotness_score: float = 0.0          # 广场热度分 (0-100)
    mention_count: int = 0              # 该 KOL 对该币的提及次数
    engagement_total: float = 0.0       # 互动总量 (like + comment + quote + bookmark)
    engagement_view_ratio: float = 0.0  # 互动/观看比（高 = 内容质量好）
    views: int = 0                      # 帖子观看量
    likes: int = 0
    comments: int = 0

    # 情绪倾向（从帖子文本分析）
    bullish_score: float = 0.0          # 看涨程度 (-100 ~ +100)
    is_live_trade: bool = False         # 是否实盘发言（vs 泛泛而谈）
    post_recency_hours: float = 0.0     # 帖子新鲜度（小时，越小越新）

    # 作者权重
    author_followers: int = 0           # 作者粉丝数
    author_post_count: int = 0          # 作者总帖子数
    author_weight: float = 0.0          # 作者权重（粉丝 × 互动率）


@dataclass(slots=True)
class MarketMicrostructureData:
    """市场微结构数据"""
    mark_price: float = 0.0
    price_change_pct_5m: float = 0.0
    price_change_pct_15m: float = 0.0
    price_change_pct_1h: float = 0.0
    price_change_pct_4h: float = 0.0
    price_change_pct_24h: float = 0.0

    quote_volume_usdt_24h: float = 0.0
    taker_buy_sell_ratio: float = 0.0
    taker_imbalance: float = 0.0       # (buy_volume - sell_volume) / total

    # 价格位置
    price_position_7d: float = 0.0
    price_position_14d: float = 0.0
    price_position_20d: float = 0.0
    breakout_distance_7d: float = 0.0
    compression_score: float = 0.0

    # 4H 结构关键位
    structure_high_4h: float = 0.0
    structure_low_4h: float = 0.0
    structure_mid_4h: float = 0.0


@dataclass(slots=True)
class CoinFeatures:
    """单个币的完整特征向量"""
    symbol: str
    timestamp: str = ""

    # === OI 特征 (8) ===
    oi_change_pct_4h: float = 0.0
    oi_change_pct_1h: float = 0.0
    oi_acceleration: float = 0.0
    oi_price_divergence_score: float = 0.0
    long_concentration_pct: float = 0.0
    short_concentration_pct: float = 0.0
    funding_rate: float = 0.0
    funding_momentum: float = 0.0

    # === 巨鲸/聪明钱特征 (12) ===
    whale_long_count: float = 0.0
    whale_short_count: float = 0.0
    whale_long_notional_usdt: float = 0.0
    whale_short_notional_usdt: float = 0.0
    whale_long_profit_ratio: float = 0.0
    whale_short_profit_ratio: float = 0.0
    whale_pressure_score: float = 0.0
    smart_money_long_count: float = 0.0
    smart_money_short_count: float = 0.0
    smart_money_long_profit_ratio: float = 0.0
    smart_money_short_profit_ratio: float = 0.0
    smart_money_single_side_pct: float = 0.0

    # === 情绪动量特征 (8) ===
    sentiment_score: float = 0.0
    sentiment_mentions: float = 0.0
    sentiment_score_delta_1h: float = 0.0
    sentiment_score_delta_4h: float = 0.0
    mention_acceleration: float = 0.0
    news_sentiment_score: float = 0.0
    news_sentiment_delta: float = 0.0
    twitter_volume_spike: float = 0.0

    # === 广场鲸鱼意见领袖特征 (10) ===
    lanaai_hotness_score: float = 0.0          # 拉哪热度分 (0-100)
    lanaai_mention_count: float = 0.0           # 拉哪对该币提及次数
    lanaai_engagement_total: float = 0.0        # 互动总量
    lanaai_engagement_view_ratio: float = 0.0   # 互动/观看比
    lanaai_views: float = 0.0                   # 帖子观看量
    lanaai_likes: float = 0.0                   # 点赞数
    lanaai_comments: float = 0.0                # 评论数
    lanaai_bullish_score: float = 0.0           # 看涨程度 (-100 ~ +100)
    lanaai_is_live_trade: float = 0.0           # 是否实盘 (1/0)
    lanaai_post_recency_hours: float = 0.0      # 帖子新鲜度（小时）
    lanaai_author_weight: float = 0.0           # 作者权重（粉丝 × 互动率）

    # === 市场微结构特征 (12) ===
    price_change_pct_5m: float = 0.0
    price_change_pct_15m: float = 0.0
    price_change_pct_1h: float = 0.0
    price_change_pct_4h: float = 0.0
    price_change_pct_24h: float = 0.0
    taker_buy_sell_ratio: float = 0.0
    taker_imbalance: float = 0.0
    price_position_7d: float = 0.0
    price_position_14d: float = 0.0
    price_position_20d: float = 0.0
    breakout_distance_7d: float = 0.0
    compression_score: float = 0.0

    # === 交互特征 (6) ===
    sentiment_oi_interaction: float = 0.0     # sentiment × oi_change
    whale_price_interaction: float = 0.0       # whale_pressure × price_change
    funding_sentiment_interaction: float = 0.0 # funding × sentiment
    momentum_score: float = 0.0                # 综合动量分
    structure_quality: float = 0.0             # 结构质量分
    risk_score: float = 0.0                    # 风险分

    # === 时间特征 (4) ===
    hour_of_day: float = 0.0
    day_of_week: float = 0.0
    is_asian_session: float = 0.0
    is_us_session: float = 0.0

    # === 基础特征 (4) ===
    quote_volume_usdt_24h: float = 0.0
    listing_days: float = 0.0
    okx_sold_ratio: float = 0.0
    okx_hold_ratio: float = 0.0

    # === K 线统计特征 (12) ===
    kline_lower_shadow_pct: float = 0.0
    kline_upper_shadow_pct: float = 0.0
    kline_body_pct: float = 0.0
    kline_direction: float = 0.0
    kline_volume_vs_ma5: float = 0.0
    kline_range_vs_ma5_range: float = 0.0
    kline_close_position: float = 0.0
    kline_consecutive_direction: float = 0.0
    kline_atr_pct: float = 0.0
    kline_ma20_distance_pct: float = 0.0
    kline_volume_trend: float = 0.0
    kline_max_drawdown_5d: float = 0.0

    # === 多周期做空结构特征 ===
    bb_price_position_96: float = 0.5
    bb_price_position_105_4h: float = 0.5
    bb_price_position_672_4h: float = 0.5
    bb_price_position_1440_4h: float = 0.5
    bb_price_position_105_1m: float = 0.5
    bb_price_position_300_1m: float = 0.5
    bb105_mid_4h_distance_pct: float = 0.0
    bb672_mid_4h_distance_pct: float = 0.0
    bb1440_mid_4h_distance_pct: float = 0.0
    bb105_upper_1m_distance_pct: float = 0.0
    bb300_upper_1m_distance_pct: float = 0.0
    bb96_upper_15m_distance_pct: float = 0.0
    short_4h_mid_retest_distance_min_pct: float = 0.0
    short_1m_upper_pressure_max: float = 0.5
    short_15m_bb96_position: float = 0.5

    # === Telegram Hyperdash liquidation flow (bounded auxiliary features) ===
    telegram_whale_actual_score_60m: float = 0.0
    telegram_whale_imminent_score_60m: float = 0.0
    telegram_whale_combined_score_60m: float = 0.0
    telegram_whale_combined_score_24h: float = 0.0
    telegram_whale_activity_score_24h: float = 0.0
    telegram_whale_freshness_score: float = 0.0

    def to_feature_vector(self) -> list[float]:
        """转换为模型输入向量"""
        return [
            # OI 特征
            self.oi_change_pct_4h,
            self.oi_change_pct_1h,
            self.oi_acceleration,
            self.oi_price_divergence_score,
            self.long_concentration_pct,
            self.short_concentration_pct,
            self.funding_rate,
            self.funding_momentum,
            # 巨鲸/聪明钱
            self.whale_long_count,
            self.whale_short_count,
            self.whale_long_notional_usdt,
            self.whale_short_notional_usdt,
            self.whale_long_profit_ratio,
            self.whale_short_profit_ratio,
            self.whale_pressure_score,
            self.smart_money_long_count,
            self.smart_money_short_count,
            self.smart_money_long_profit_ratio,
            self.smart_money_short_profit_ratio,
            self.smart_money_single_side_pct,
            # 情绪动量
            self.sentiment_score,
            self.sentiment_mentions,
            self.sentiment_score_delta_1h,
            self.sentiment_score_delta_4h,
            self.mention_acceleration,
            self.news_sentiment_score,
            self.news_sentiment_delta,
            self.twitter_volume_spike,
            # 广场鲸鱼意见领袖
            self.lanaai_hotness_score,
            self.lanaai_mention_count,
            self.lanaai_engagement_total,
            self.lanaai_engagement_view_ratio,
            self.lanaai_views,
            self.lanaai_likes,
            self.lanaai_comments,
            self.lanaai_bullish_score,
            self.lanaai_is_live_trade,
            self.lanaai_post_recency_hours,
            self.lanaai_author_weight,
            # 市场微结构
            self.price_change_pct_5m,
            self.price_change_pct_15m,
            self.price_change_pct_1h,
            self.price_change_pct_4h,
            self.price_change_pct_24h,
            self.taker_buy_sell_ratio,
            self.taker_imbalance,
            self.price_position_7d,
            self.price_position_14d,
            self.price_position_20d,
            self.breakout_distance_7d,
            self.compression_score,
            # 交互特征
            self.sentiment_oi_interaction,
            self.whale_price_interaction,
            self.funding_sentiment_interaction,
            self.momentum_score,
            self.structure_quality,
            self.risk_score,
            # 时间特征
            self.hour_of_day,
            self.day_of_week,
            self.is_asian_session,
            self.is_us_session,
            # 基础特征
            self.quote_volume_usdt_24h,
            self.listing_days,
            self.okx_sold_ratio,
            self.okx_hold_ratio,
            # 多周期做空结构
            self.bb_price_position_96,
            self.bb_price_position_105_4h,
            self.bb_price_position_672_4h,
            self.bb_price_position_1440_4h,
            self.bb_price_position_105_1m,
            self.bb_price_position_300_1m,
            self.bb105_mid_4h_distance_pct,
            self.bb672_mid_4h_distance_pct,
            self.bb1440_mid_4h_distance_pct,
            self.bb105_upper_1m_distance_pct,
            self.bb300_upper_1m_distance_pct,
            self.bb96_upper_15m_distance_pct,
            self.short_4h_mid_retest_distance_min_pct,
            self.short_1m_upper_pressure_max,
            self.short_15m_bb96_position,
            # Telegram Hyperdash liquidation flow
            self.telegram_whale_actual_score_60m,
            self.telegram_whale_imminent_score_60m,
            self.telegram_whale_combined_score_60m,
            self.telegram_whale_combined_score_24h,
            self.telegram_whale_activity_score_24h,
            self.telegram_whale_freshness_score,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            # OI
            "oi_change_pct_4h", "oi_change_pct_1h", "oi_acceleration",
            "oi_price_divergence_score", "long_concentration_pct", "short_concentration_pct",
            "funding_rate", "funding_momentum",
            # 巨鲸/聪明钱
            "whale_long_count", "whale_short_count", "whale_long_notional_usdt",
            "whale_short_notional_usdt", "whale_long_profit_ratio", "whale_short_profit_ratio",
            "whale_pressure_score", "smart_money_long_count", "smart_money_short_count",
            "smart_money_long_profit_ratio", "smart_money_short_profit_ratio",
            "smart_money_single_side_pct",
            # 情绪动量
            "sentiment_score", "sentiment_mentions", "sentiment_score_delta_1h",
            "sentiment_score_delta_4h", "mention_acceleration", "news_sentiment_score",
            "news_sentiment_delta", "twitter_volume_spike",
            # 广场鲸鱼意见领袖
            "lanaai_hotness_score", "lanaai_mention_count", "lanaai_engagement_total",
            "lanaai_engagement_view_ratio", "lanaai_views", "lanaai_likes",
            "lanaai_comments", "lanaai_bullish_score", "lanaai_is_live_trade",
            "lanaai_post_recency_hours", "lanaai_author_weight",
            # 市场微结构
            "price_change_pct_5m", "price_change_pct_15m", "price_change_pct_1h",
            "price_change_pct_4h", "price_change_pct_24h", "taker_buy_sell_ratio",
            "taker_imbalance", "price_position_7d", "price_position_14d", "price_position_20d",
            "breakout_distance_7d", "compression_score",
            # 交互特征
            "sentiment_oi_interaction", "whale_price_interaction", "funding_sentiment_interaction",
            "momentum_score", "structure_quality", "risk_score",
            # 时间特征
            "hour_of_day", "day_of_week", "is_asian_session", "is_us_session",
            # 基础
            "quote_volume_usdt_24h", "listing_days", "okx_sold_ratio", "okx_hold_ratio",
            # 多周期做空结构
            "bb_price_position_96", "bb_price_position_105_4h", "bb_price_position_672_4h",
            "bb_price_position_1440_4h", "bb_price_position_105_1m", "bb_price_position_300_1m",
            "bb105_mid_4h_distance_pct", "bb672_mid_4h_distance_pct", "bb1440_mid_4h_distance_pct",
            "bb105_upper_1m_distance_pct", "bb300_upper_1m_distance_pct", "bb96_upper_15m_distance_pct",
            "short_4h_mid_retest_distance_min_pct", "short_1m_upper_pressure_max", "short_15m_bb96_position",
            # Telegram Hyperdash liquidation flow
            "telegram_whale_actual_score_60m", "telegram_whale_imminent_score_60m",
            "telegram_whale_combined_score_60m", "telegram_whale_combined_score_24h",
            "telegram_whale_activity_score_24h", "telegram_whale_freshness_score",
        ]


# ============================================================
# 2. 特征工程
# ============================================================

class FeatureEngine:
    """特征工程：将所有原始数据转换为模型特征"""

    # 排除的币种
    EXCLUDED_SYMBOLS = {"SOLUSDT", "SOLMEMEUSDT"}

    def build_features(
        self,
        symbol: str,
        oi_data: OIMomentumData,
        whale_data: WhaleSmartMoneyData,
        sentiment_data: SentimentMomentumData,
        market_data: MarketMicrostructureData,
        square_data: SquareWhaleOpinionData | None = None,
    ) -> CoinFeatures:
        """构建完整特征向量

        Args:
            square_data: 广场鲸鱼意见领袖数据（拉哪等 KOL），可选
        """

        features = CoinFeatures(symbol=symbol)
        now = datetime.now(timezone.utc)

        # === OI 特征 ===
        features.oi_change_pct_4h = oi_data.oi_change_pct_4h
        features.oi_change_pct_1h = oi_data.oi_change_pct_1h
        features.oi_acceleration = oi_data.oi_acceleration
        features.oi_price_divergence_score = oi_data.oi_price_divergence_score
        features.long_concentration_pct = oi_data.long_concentration_pct
        features.short_concentration_pct = oi_data.short_concentration_pct
        features.funding_rate = oi_data.funding_rate
        features.funding_momentum = oi_data.funding_momentum

        # === 巨鲸/聪明钱特征 ===
        features.whale_long_count = float(whale_data.whale_long_count)
        features.whale_short_count = float(whale_data.whale_short_count)
        features.whale_long_notional_usdt = whale_data.whale_long_notional_usdt
        features.whale_short_notional_usdt = whale_data.whale_short_notional_usdt
        features.whale_long_profit_ratio = whale_data.whale_long_profit_ratio
        features.whale_short_profit_ratio = whale_data.whale_short_profit_ratio
        features.whale_pressure_score = whale_data.whale_pressure_score
        features.smart_money_long_count = float(whale_data.smart_money_long_count)
        features.smart_money_short_count = float(whale_data.smart_money_short_count)
        features.smart_money_long_profit_ratio = whale_data.smart_money_long_profit_ratio
        features.smart_money_short_profit_ratio = whale_data.smart_money_short_profit_ratio
        features.smart_money_single_side_pct = whale_data.smart_money_single_side_pct

        # === 情绪动量特征 ===
        features.sentiment_score = sentiment_data.current_score
        features.sentiment_mentions = float(sentiment_data.mentions)
        features.sentiment_score_delta_1h = sentiment_data.score_delta_1h
        features.sentiment_score_delta_4h = sentiment_data.score_delta_4h
        features.mention_acceleration = sentiment_data.mention_acceleration
        features.news_sentiment_score = sentiment_data.news_sentiment_score
        features.news_sentiment_delta = sentiment_data.news_sentiment_delta
        features.twitter_volume_spike = sentiment_data.twitter_volume_spike

        # === 广场鲸鱼意见领袖特征 ===
        if square_data is not None:
            features.lanaai_hotness_score = square_data.hotness_score
            features.lanaai_mention_count = float(square_data.mention_count)
            features.lanaai_engagement_total = square_data.engagement_total
            features.lanaai_engagement_view_ratio = square_data.engagement_view_ratio
            features.lanaai_views = float(square_data.views)
            features.lanaai_likes = float(square_data.likes)
            features.lanaai_comments = float(square_data.comments)
            features.lanaai_bullish_score = square_data.bullish_score
            features.lanaai_is_live_trade = 1.0 if square_data.is_live_trade else 0.0
            features.lanaai_post_recency_hours = square_data.post_recency_hours
            features.lanaai_author_weight = square_data.author_weight
        # else: 全部保持 0.0（LightGBM/sklearn 都能处理全零特征）

        # === 市场微结构特征 ===
        features.price_change_pct_5m = market_data.price_change_pct_5m
        features.price_change_pct_15m = market_data.price_change_pct_15m
        features.price_change_pct_1h = market_data.price_change_pct_1h
        features.price_change_pct_4h = market_data.price_change_pct_4h
        features.price_change_pct_24h = market_data.price_change_pct_24h
        features.taker_buy_sell_ratio = market_data.taker_buy_sell_ratio
        features.taker_imbalance = market_data.taker_imbalance
        features.price_position_7d = market_data.price_position_7d
        features.price_position_14d = market_data.price_position_14d
        features.price_position_20d = market_data.price_position_20d
        features.breakout_distance_7d = market_data.breakout_distance_7d
        features.compression_score = market_data.compression_score

        # === 交互特征（核心） ===
        features.sentiment_oi_interaction = self._calc_sentiment_oi_interaction(
            sentiment_data.current_score, oi_data.oi_change_pct_4h
        )
        features.whale_price_interaction = self._calc_whale_price_interaction(
            whale_data.whale_pressure_score, market_data.price_change_pct_4h
        )
        features.funding_sentiment_interaction = self._calc_funding_sentiment_interaction(
            oi_data.funding_rate, sentiment_data.current_score
        )
        features.momentum_score = self._calc_momentum_score(
            market_data, oi_data, sentiment_data
        )
        features.structure_quality = self._calc_structure_quality(market_data)
        features.risk_score = self._calc_risk_score(oi_data, whale_data, market_data)

        # === 时间特征 ===
        features.hour_of_day = float(now.hour)
        features.day_of_week = float(now.weekday())
        features.is_asian_session = 1.0 if 0 <= now.hour < 8 else 0.0
        features.is_us_session = 1.0 if 13 <= now.hour < 21 else 0.0

        # === 基础特征 ===
        features.quote_volume_usdt_24h = market_data.quote_volume_usdt_24h
        features.okx_sold_ratio = whale_data.okx_sold_ratio
        features.okx_hold_ratio = whale_data.okx_hold_ratio

        return features

    def _calc_sentiment_oi_interaction(self, sentiment_score: float, oi_change: float) -> float:
        """情绪 × OI 交互：情绪高 + OI 扩张 = 强信号"""
        return sentiment_score * oi_change / 100.0

    def _calc_whale_price_interaction(self, whale_pressure: float, price_change: float) -> float:
        """巨鲸压力 × 价格变化：巨鲸做多 + 价格上涨 = 强信号"""
        return whale_pressure * price_change / 100.0

    def _calc_funding_sentiment_interaction(self, funding_rate: float, sentiment: float) -> float:
        """资金费率 × 情绪：负费率 + 高情绪 = 空头挤压潜力"""
        return abs(funding_rate) * sentiment * 10.0

    def _calc_momentum_score(
        self,
        market: MarketMicrostructureData,
        oi: OIMomentumData,
        sentiment: SentimentMomentumData,
    ) -> float:
        """综合动量分：价格 + OI + 情绪的三维动量"""
        price_momentum = market.price_change_pct_4h
        oi_momentum = oi.oi_change_pct_4h
        sentiment_momentum = sentiment.score_delta_4h

        # 三维同向 = 最强信号
        if price_momentum > 0 and oi_momentum > 0 and sentiment_momentum > 0:
            return (price_momentum + oi_momentum + sentiment_momentum) * 1.5
        elif price_momentum > 0 and oi_momentum > 0:
            return (price_momentum + oi_momentum) * 1.2
        else:
            return price_momentum + oi_momentum * 0.5 + sentiment_momentum * 0.3

    def _calc_structure_quality(self, market: MarketMicrostructureData) -> float:
        """结构质量分：压缩 + 突破距离 + 价格位置"""
        score = 0.0
        # 压缩越充分越好
        if market.compression_score > 0.5:
            score += market.compression_score * 30
        # 突破距离越小越好（刚突破）
        if market.breakout_distance_7d < 0.03:
            score += (0.03 - market.breakout_distance_7d) * 500
        # 7d 位置越高越好
        if market.price_position_7d > 0.85:
            score += (market.price_position_7d - 0.85) * 200
        return min(100.0, score)

    def _calc_risk_score(
        self,
        oi: OIMomentumData,
        whale: WhaleSmartMoneyData,
        market: MarketMicrostructureData,
    ) -> float:
        """风险分：越高越危险"""
        risk = 0.0

        # OI 单边集中度风险
        if oi.long_concentration_pct > 90 or oi.short_concentration_pct > 90:
            risk += 40  # 庄盘风险
        elif oi.long_concentration_pct > 70 or oi.short_concentration_pct > 70:
            risk += 20  # 偏高

        # 资金费率极端
        if abs(oi.funding_rate) > 0.003:
            risk += 15

        # 巨鲸亏损中
        if whale.whale_long_profit_ratio < 30 and whale.whale_short_profit_ratio < 30:
            risk += 10

        # 成交量过低
        if market.quote_volume_usdt_24h < 20_000_000:
            risk += 15

        return min(100.0, risk)


# ============================================================
# 3. 妖币标签定义（4H 框架）
# ============================================================

class CoinLabel:
    """
    妖币标签定义（基于你的交易哲学）

    方向有效 (1): 入场后 4H/24H 窗口顺方向走势有效且回撤可接受
    方向无效 (0): 其他

    注意：需要采集入场后 4H 的价格数据才能打标签
    """

    DIRECTION_GAIN_THRESHOLD = 0.025
    DIRECTION_CLOSE_RETURN_THRESHOLD = 0.015
    MAX_DRAWDOWN_DIRECTION = 0.055

    @staticmethod
    def label(peak_gain_4h: float, max_drawdown_4h: float) -> int:
        """根据 4H 内的顺方向波动和最大回撤打二分类标签。"""
        if peak_gain_4h >= CoinLabel.DIRECTION_GAIN_THRESHOLD and max_drawdown_4h <= CoinLabel.MAX_DRAWDOWN_DIRECTION:
            return 1
        return 0

    @staticmethod
    def label_name(label: int) -> str:
        return {0: "方向无效", 1: "方向有效"}.get(label, "未知")


# ============================================================
# 4. 模型训练
# ============================================================

class CoinScoutModel:
    """
    妖币捕捉模型
    ============
    使用 LightGBM/GradientBoosting 分类方向是否有效。
    """

    def __init__(self, model_dir: str = "data/trading_system/model_training/coin_scout"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_engine = FeatureEngine()
        self.is_trained = False

    def _compatible_model_input(self, features: CoinFeatures) -> np.ndarray:
        values = [float(value) for value in features.to_feature_vector()]
        expected = getattr(self.model, "n_features_in_", None)
        if expected is None:
            num_feature = getattr(self.model, "num_feature", None)
            if callable(num_feature):
                try:
                    expected = int(num_feature())
                except (TypeError, ValueError):
                    expected = None
        try:
            expected_count = int(expected) if expected is not None else len(values)
        except (TypeError, ValueError):
            expected_count = len(values)
        if expected_count < len(values):
            values = values[:expected_count]
        elif expected_count > len(values):
            values.extend([0.0] * (expected_count - len(values)))
        return np.asarray([values], dtype=float)

    def train(
        self,
        features_list: list[CoinFeatures],
        labels: list[int],
        validation_ratio: float = 0.2,
    ) -> dict:
        """训练模型"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix

        # 准备数据
        X = np.array([f.to_feature_vector() for f in features_list])
        y = np.array(labels)
        feature_names = CoinFeatures.feature_names()
        all_labels = sorted(int(label) for label in np.unique(y))

        if not 0.0 <= validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in [0, 1)")

        # Ablation/replay callers already own the out-of-sample split. A zero
        # ratio means fit every supplied row instead of silently dropping the
        # newest 20% into a second internal validation set.
        has_validation = validation_ratio > 0.0
        if has_validation:
            split_idx = int(len(X) * (1 - validation_ratio))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val = np.empty((0, X.shape[1]), dtype=X.dtype)
            y_val = np.empty((0,), dtype=y.dtype)

        # 处理类别不平衡
        unique, counts = np.unique(y_train, return_counts=True)
        class_weight = {}
        total = len(y_train)
        for u, c in zip(unique, counts):
            class_weight[u] = total / (len(unique) * c)

        # 优先尝试 LightGBM，失败则用 sklearn GradientBoosting
        model_type = "lightgbm"
        try:
            import lightgbm as lgb

            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_data = (
                lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
                if has_validation
                else None
            )

            params = {
                'objective': 'binary' if len(all_labels) == 2 and set(all_labels) <= {0, 1} else 'multiclass',
                'metric': 'binary_logloss' if len(all_labels) == 2 and set(all_labels) <= {0, 1} else 'multi_logloss',
                'n_estimators': 500,
                'max_depth': 7,
                'learning_rate': 0.05,
                'num_leaves': 63,
                'min_child_samples': 10,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'class_weight': 'balanced',
                'random_state': 42,
                'verbose': -1,
            }
            if params["objective"] == "multiclass":
                params["num_class"] = len(all_labels)

            if has_validation:
                self.model = lgb.train(
                    params, train_data, valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
                raw_pred = self.model.predict(X_val)
                if params["objective"] == "binary":
                    y_pred = (np.asarray(raw_pred) >= 0.5).astype(int)
                else:
                    y_pred = np.argmax(raw_pred, axis=1)
            else:
                self.model = lgb.train(
                    params,
                    train_data,
                    callbacks=[lgb.log_evaluation(0)],
                )
                y_pred = np.asarray([], dtype=int)
            importance = dict(zip(feature_names, self.model.feature_importance().tolist()))
            model_path = self.model_dir / "coin_scout_model.txt"
            self.model.save_model(str(model_path))

        except (ImportError, OSError):
            # Fallback: sklearn GradientBoosting
            from sklearn.ensemble import GradientBoostingClassifier
            model_type = "sklearn_gradient_boosting"

            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                subsample=0.85, min_samples_leaf=3, random_state=42,
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val) if has_validation else np.asarray([], dtype=int)
            importance = dict(zip(feature_names, self.model.feature_importances_.tolist()))

            import pickle
            model_path = self.model_dir / "coin_scout_model.pkl"
            with open(model_path, "wb") as fh:
                pickle.dump(self.model, fh)

        # 评估（labels 参数确保即使 val 中缺少某些类别也不报错）
        if has_validation:
            present_labels = sorted(set(y_val) | set(y_pred))
            target_names = [CoinLabel.label_name(int(label)) for label in present_labels]
            report = classification_report(
                y_val, y_pred,
                labels=present_labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )
        else:
            report = {}

        self.is_trained = True

        return {
            "status": "ok",
            "model_type": model_type,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "classification_report": report,
            "feature_importance": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]),
            "model_path": str(model_path),
        }

    def predict(self, features: CoinFeatures) -> dict:
        """预测单个币（自动适配 LightGBM / sklearn 模型）"""
        if not self.is_trained or self.model is None:
            return {"status": "error", "message": "model not trained"}

        try:
            X = self._compatible_model_input(features)

            # 判断模型类型并做预测
            model_type = type(self.model).__name__

            if model_type == "Booster":
                # LightGBM 原生模型
                raw = self.model.predict(X)
                if np.asarray(raw).ndim == 1:
                    positive = float(np.asarray(raw)[0])
                    probs = np.array([1.0 - positive, positive])
                else:
                    probs = raw[0]
            elif hasattr(self.model, "predict_proba"):
                # sklearn 模型（GradientBoostingClassifier 等）
                probs = self.model.predict_proba(X)[0]
            else:
                # 没有概率输出，用 predict + 伪概率
                pred = self.model.predict(X)[0]
                size = max(2, int(pred) + 1)
                probs = [0.0] * size
                probs[int(pred)] = 1.0

            predicted_class = int(np.argmax(probs))
            confidence = float(max(probs))

            return {
                "status": "ok",
                "symbol": features.symbol,
                "predicted_label": predicted_class,
                "label_name": CoinLabel.label_name(predicted_class),
                "confidence": confidence,
                "probabilities": {
                    CoinLabel.label_name(index): float(value)
                    for index, value in enumerate(probs)
                },
                "key_signals": self._explain_prediction(features),
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _explain_prediction(self, features: CoinFeatures) -> dict:
        """可解释性：输出关键信号"""
        signals = {}

        # OI 信号
        if features.oi_change_pct_4h > 15:
            signals["oi_expansion"] = f"OI 4h 扩张 {features.oi_change_pct_4h:.1f}%"
        if features.long_concentration_pct > 70:
            signals["concentration_warning"] = f"多头集中度 {features.long_concentration_pct:.0f}%"

        # 巨鲸信号
        if features.whale_pressure_score > 30:
            signals["whale_bullish"] = f"巨鲸做多压力 {features.whale_pressure_score:.0f}"
        elif features.whale_pressure_score < -30:
            signals["whale_bearish"] = f"巨鲸做空压力 {features.whale_pressure_score:.0f}"

        # 情绪信号
        if features.sentiment_score_delta_4h > 10:
            signals["sentiment_rising"] = f"情绪 4h 上升 {features.sentiment_score_delta_4h:.0f}"
        if features.mention_acceleration > 5:
            signals["mention_spike"] = f"提及量加速 {features.mention_acceleration:.1f}"

        # 动量信号
        if features.momentum_score > 20:
            signals["strong_momentum"] = f"综合动量 {features.momentum_score:.1f}"

        # 结构信号
        if features.structure_quality > 50:
            signals["good_structure"] = f"结构质量 {features.structure_quality:.0f}"

        # 风险信号
        if features.risk_score > 40:
            signals["high_risk"] = f"风险分 {features.risk_score:.0f}"

        return signals

    def load_model(self, path: str | None = None) -> bool:
        """加载已训练模型（支持 LightGBM 和 sklearn pickle 格式）"""
        model_path = Path(path) if path else None

        # 尝试 LightGBM 格式
        if model_path is None:
            lgb_path = self.model_dir / "coin_scout_model.txt"
            if lgb_path.exists():
                try:
                    import lightgbm as lgb
                    self.model = lgb.Booster(model_file=str(lgb_path))
                    self.is_trained = True
                    return True
                except Exception:
                    pass

        # 尝试 sklearn pickle 格式
        if model_path is None:
            pkl_path = self.model_dir / "coin_scout_model.pkl"
        else:
            pkl_path = model_path

        if pkl_path.exists():
            try:
                import pickle
                with open(pkl_path, "rb") as fh:
                    self.model = pickle.load(fh)
                self.is_trained = True
                return True
            except Exception:
                pass

        return False


# ============================================================
# 5. 决策层（匹配你的交易规则）
# ============================================================

@dataclass
class ScoutDecision:
    """模型决策输出"""
    symbol: str
    action: str                    # "skip" / "probe_small" / "add_position" / "hold" / "exit"
    confidence: float = 0.0
    predicted_label: int = 0
    position_scale: float = 0.0    # 仓位比例
    reasons: list[str] = field(default_factory=list)
    key_signals: dict = field(default_factory=dict)


class ScoutDecisionEngine:
    """
    决策引擎：将模型预测转换为交易决策

    核心规则（匹配你的交易哲学）：
    - 早发现，小仓进：第一次强信号 → probe_small (0.25x)
    - 4H 拿：不因短期波动退出
    - 回踩加：多次信号且结构没破 → add_position (0.4x)
    - 破位走：4H 结构破坏 → exit
    - 不做 SOL / SOL meme
    - OI 单边集中度 >90% 是庄盘风险
    """

    # 排除的币种
    EXCLUDED_SYMBOLS = {"SOLUSDT"}

    # 仓位配置
    POSITION_SCALES = {
        "probe_small": 0.25,      # 首次试探
        "add_position": 0.40,     # 加仓
        "full_position": 0.60,     # 满仓（极少用）
    }

    def decide(
        self,
        symbol: str,
        prediction: dict,
        features: CoinFeatures,
        current_position: dict | None = None,
    ) -> ScoutDecision:
        """根据模型预测和当前持仓做决策"""

        # 排除币种
        if symbol in self.EXCLUDED_SYMBOLS or "SOL" in symbol:
            return ScoutDecision(symbol=symbol, action="skip", reasons=["排除币种"])

        # 庄盘风险检查
        if features.long_concentration_pct > 90 or features.short_concentration_pct > 90:
            return ScoutDecision(
                symbol=symbol, action="skip",
                reasons=[f"庄盘风险：多头集中度 {features.long_concentration_pct:.0f}%"],
                key_signals=prediction.get("key_signals", {}),
            )

        # 风险分过高
        if features.risk_score > 60:
            return ScoutDecision(
                symbol=symbol, action="skip",
                reasons=[f"风险分过高：{features.risk_score:.0f}"],
                key_signals=prediction.get("key_signals", {}),
            )

        predicted_label = prediction.get("predicted_label", 0)
        confidence = prediction.get("confidence", 0.0)
        key_signals = prediction.get("key_signals", {})

        # === 无持仓：考虑入场 ===
        if current_position is None:
            return self._decide_entry(symbol, predicted_label, confidence, features, key_signals)

        # === 有持仓：考虑持有/加仓/退出 ===
        return self._decide_position_management(
            symbol, predicted_label, confidence, features, current_position, key_signals
        )

    def _decide_entry(
        self, symbol: str, predicted_label: int, confidence: float,
        features: CoinFeatures, key_signals: dict,
    ) -> ScoutDecision:
        """入场决策"""

        # 方向有效 + 高置信度 → 小仓试探
        if predicted_label == 1 and confidence > 0.55:
            return ScoutDecision(
                symbol=symbol,
                action="probe_small",
                confidence=confidence,
                predicted_label=predicted_label,
                position_scale=self.POSITION_SCALES["probe_small"],
                reasons=["方向有效信号，小仓试探"],
                key_signals=key_signals,
            )

        # 方向无效或置信度不足 → 跳过
        else:
            return ScoutDecision(
                symbol=symbol, action="skip",
                confidence=confidence, predicted_label=predicted_label,
                reasons=["方向无效或置信度不足"],
                key_signals=key_signals,
            )

    def _decide_position_management(
        self, symbol: str, predicted_label: int, confidence: float,
        features: CoinFeatures, current_position: dict, key_signals: dict,
    ) -> ScoutDecision:
        """持仓管理决策"""

        entry_price = current_position.get("entry_price", 0)
        current_price = features.mark_price
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        hold_hours = current_position.get("hold_hours", 0)

        # === 退出条件（按优先级） ===

        # 1. 4H 结构破位（最重要）
        if self._is_4h_structure_broken(features, current_position):
            return ScoutDecision(
                symbol=symbol, action="exit",
                reasons=["4H 结构破位，退出"],
                key_signals=key_signals,
            )

        # 2. 浮亏超过 8%（硬止损）
        if pnl_pct < -0.08:
            return ScoutDecision(
                symbol=symbol, action="exit",
                reasons=[f"浮亏 {pnl_pct:.1%}，硬止损"],
                key_signals=key_signals,
            )

        # 3. 盈利回撤保护（浮盈 >10% 后回撤 5%）
        peak_pnl = current_position.get("peak_pnl_pct", 0)
        if peak_pnl > 0.10 and (peak_pnl - pnl_pct) > 0.05:
            return ScoutDecision(
                symbol=symbol, action="exit",
                reasons=[f"盈利回撤保护：峰值 {peak_pnl:.1%} → 当前 {pnl_pct:.1%}"],
                key_signals=key_signals,
            )

        # === 加仓条件 ===

        # 多次信号 + 结构没破 + 浮盈中
        if (predicted_label >= 1 and pnl_pct > 0.02 and
                not self._is_4h_structure_broken(features, current_position) and
                features.risk_score < 40):
            return ScoutDecision(
                symbol=symbol, action="add_position",
                confidence=confidence, predicted_label=predicted_label,
                position_scale=self.POSITION_SCALES["add_position"],
                reasons=["多次信号 + 结构健康 + 浮盈中，加仓"],
                key_signals=key_signals,
            )

        # === 持有 ===
        return ScoutDecision(
            symbol=symbol, action="hold",
            confidence=confidence, predicted_label=predicted_label,
            reasons=[f"持有中，浮盈 {pnl_pct:.1%}，持仓 {hold_hours:.1f}h"],
            key_signals=key_signals,
        )

    def _is_4h_structure_broken(self, features: CoinFeatures, position: dict) -> bool:
        """
        判断 4H 结构是否破位

        破位条件（满足任一）：
        1. 价格跌破 4H 结构低点
        2. 4H 跌幅超过 8%
        3. OI 大幅萎缩（>20%）+ 价格下跌
        """
        entry_price = position.get("entry_price", 0)
        current_price = features.mark_price

        # 价格跌破 4H 结构低点
        if features.structure_low_4h > 0 and current_price < features.structure_low_4h:
            return True

        # 4H 跌幅超过 8%
        if features.price_change_pct_4h < -8:
            return True

        # OI 大幅萎缩 + 价格下跌
        if features.oi_change_pct_4h < -20 and features.price_change_pct_4h < -3:
            return True

        return False


# ============================================================
# 6. 便捷函数
# ============================================================

def create_model(model_dir: str = "data/trading_system/model_training/coin_scout") -> CoinScoutModel:
    """创建模型实例"""
    return CoinScoutModel(model_dir=model_dir)


def load_model(model_dir: str = "data/trading_system/model_training/coin_scout") -> CoinScoutModel | None:
    """加载已训练模型"""
    model = CoinScoutModel(model_dir=model_dir)
    if model.load_model():
        return model
    return None


if __name__ == "__main__":
    print("妖币捕捉模型 (Coin Scout Model)")
    print("=" * 50)
    print(f"特征维度: {len(CoinFeatures.feature_names())}")
    print(f"标签: 方向无效(0) / 方向有效(1)")
    print(f"核心规则: 早发现小仓进，4H拿，回踩加，破位走")
    print()
    print("使用方式:")
    print("  model = create_model()")
    print("  result = model.train(features_list, labels)")
    print("  prediction = model.predict(features)")
