from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class SentimentSignal:
    symbol: str
    hotness_score: float
    mention_count: int
    source: str
    updated_at: datetime
    notes: str = ""
    earliest_strong_source: str = ""
    earliest_strong_timestamp: datetime | None = None
    okx_confirmed_at: datetime | None = None
    binance_confirmed_at: datetime | None = None
    signal_sequence: str = "single_source"
    lead_time_minutes: float | None = None


@dataclass(slots=True)
class MarketSnapshot:
    symbol: str
    mark_price: float
    price_change_pct_5m: float = 0.0
    price_change_pct_4h: float = 0.0
    price_change_pct_24h: float = 0.0
    quote_volume_usdt_24h: float = 0.0
    funding_rate: float = 0.0
    open_interest_usdt: float = 0.0
    oi_change_pct_4h: float = 0.0
    taker_buy_sell_ratio: float = 0.0
    listing_days: int | None = None
    sentiment_score: float = 0.0
    sentiment_mentions: int = 0
    sentiment_source: str = ""
    price_position_7d: float | None = None
    price_position_14d: float | None = None
    price_position_20d: float | None = None
    breakout_distance_7d: float | None = None
    compression_score: float | None = None
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
    # K-line pattern flags (observation-only, not used in scoring)
    kline_pattern_bullish_count: float = 0.0
    kline_pattern_bearish_count: float = 0.0
    kline_pattern_flags: str = ""
    # 4H structure observation (Phase 1: record only)
    h4_close_above_ma20: bool = True
    h4_ma20_slope_pct: float = 0.0
    h4_price_vs_ma20_pct: float = 0.0
    h4_close_position_20bars: float = 0.5
    h4_low_break_recent_20bars: bool = False
    h4_high_break_recent_20bars: bool = False
    h4_atr_compression: float = 1.0
    h4_lower_highs_count: int = 0
    h4_higher_lows_count: int = 0
    h4_structure_state: str = "unknown"
    h4_price_location: str = "middle"
    h4_pattern_type: str = "unknown"
    h4_range_low: float = 0.0
    h4_recent_swing_low: float = 0.0
    momo_15m_bars: int = 0
    momo_1m_bars: int = 0
    momo_4h_bars: int = 0
    momo_1d_bars: int = 0
    ma7_15m: float = 0.0
    ma20_15m: float = 0.0
    ma60_15m: float = 0.0
    ma96_15m: float = 0.0
    ma105_15m: float = 0.0
    ma120_15m: float = 0.0
    volume_ma5_15m: float = 0.0
    volume_ma10_15m: float = 0.0
    bb96_upper_15m: float = 0.0
    bb96_mid_15m: float = 0.0
    bb96_lower_15m: float = 0.0
    bb105_upper_15m: float = 0.0
    bb105_mid_15m: float = 0.0
    bb105_lower_15m: float = 0.0
    bb120_upper_15m: float = 0.0
    bb120_mid_15m: float = 0.0
    bb120_lower_15m: float = 0.0
    bb210_upper_15m: float = 0.0
    bb210_mid_15m: float = 0.0
    bb210_lower_15m: float = 0.0
    bb300_upper_15m: float = 0.0
    bb300_mid_15m: float = 0.0
    bb300_lower_15m: float = 0.0
    bb672_upper_15m: float = 0.0
    bb672_mid_15m: float = 0.0
    bb672_lower_15m: float = 0.0
    bb1440_upper_15m: float = 0.0
    bb1440_mid_15m: float = 0.0
    bb1440_lower_15m: float = 0.0
    bb1920_upper_15m: float = 0.0
    bb1920_mid_15m: float = 0.0
    bb1920_lower_15m: float = 0.0
    bb2880_upper_15m: float = 0.0
    bb2880_mid_15m: float = 0.0
    bb2880_lower_15m: float = 0.0
    bb4404_upper_15m: float = 0.0
    bb4404_mid_15m: float = 0.0
    bb4404_lower_15m: float = 0.0
    bb4460_upper_15m: float = 0.0
    bb4460_mid_15m: float = 0.0
    bb4460_lower_15m: float = 0.0
    bb4707_upper_15m: float = 0.0
    bb4707_mid_15m: float = 0.0
    bb4707_lower_15m: float = 0.0
    ma7_4h: float = 0.0
    ma20_4h: float = 0.0
    ma60_4h: float = 0.0
    ma96_4h: float = 0.0
    ma105_4h: float = 0.0
    ma120_4h: float = 0.0
    volume_ma5_4h: float = 0.0
    volume_ma10_4h: float = 0.0
    bb96_upper_4h: float = 0.0
    bb96_mid_4h: float = 0.0
    bb96_lower_4h: float = 0.0
    bb105_upper_4h: float = 0.0
    bb105_mid_4h: float = 0.0
    bb105_lower_4h: float = 0.0
    bb120_upper_4h: float = 0.0
    bb120_mid_4h: float = 0.0
    bb120_lower_4h: float = 0.0
    bb210_upper_4h: float = 0.0
    bb210_mid_4h: float = 0.0
    bb210_lower_4h: float = 0.0
    bb300_upper_4h: float = 0.0
    bb300_mid_4h: float = 0.0
    bb300_lower_4h: float = 0.0
    bb672_upper_4h: float = 0.0
    bb672_mid_4h: float = 0.0
    bb672_lower_4h: float = 0.0
    bb1440_upper_4h: float = 0.0
    bb1440_mid_4h: float = 0.0
    bb1440_lower_4h: float = 0.0
    bb1920_upper_4h: float = 0.0
    bb1920_mid_4h: float = 0.0
    bb1920_lower_4h: float = 0.0
    bb2880_upper_4h: float = 0.0
    bb2880_mid_4h: float = 0.0
    bb2880_lower_4h: float = 0.0
    bb4404_upper_4h: float = 0.0
    bb4404_mid_4h: float = 0.0
    bb4404_lower_4h: float = 0.0
    bb4460_upper_4h: float = 0.0
    bb4460_mid_4h: float = 0.0
    bb4460_lower_4h: float = 0.0
    bb4707_upper_4h: float = 0.0
    bb4707_mid_4h: float = 0.0
    bb4707_lower_4h: float = 0.0
    bb105_upper_1m: float = 0.0
    bb105_mid_1m: float = 0.0
    bb105_lower_1m: float = 0.0
    sma210_1m: float = 0.0
    bb300_upper_1m: float = 0.0
    bb300_mid_1m: float = 0.0
    bb300_lower_1m: float = 0.0
    bb672_upper_1m: float = 0.0
    bb672_mid_1m: float = 0.0
    bb672_lower_1m: float = 0.0
    bb1920_upper_1m: float = 0.0
    bb1920_mid_1m: float = 0.0
    bb1920_lower_1m: float = 0.0
    ma7_1d: float = 0.0
    ma20_1d: float = 0.0
    ma60_1d: float = 0.0
    ma96_1d: float = 0.0
    ma105_1d: float = 0.0
    ma120_1d: float = 0.0
    volume_ma5_1d: float = 0.0
    volume_ma10_1d: float = 0.0
    daily_ma7: float = 0.0
    daily_ma20: float = 0.0
    daily_ema21: float = 0.0
    daily_bb365_upper: float = 0.0
    daily_bb365_mid: float = 0.0
    daily_bb365_lower: float = 0.0
    bb96_upper_1d: float = 0.0
    bb96_mid_1d: float = 0.0
    bb96_lower_1d: float = 0.0
    bb105_upper_1d: float = 0.0
    bb105_mid_1d: float = 0.0
    bb105_lower_1d: float = 0.0
    bb120_upper_1d: float = 0.0
    bb120_mid_1d: float = 0.0
    bb120_lower_1d: float = 0.0
    bb210_upper_1d: float = 0.0
    bb210_mid_1d: float = 0.0
    bb210_lower_1d: float = 0.0
    bb300_upper_1d: float = 0.0
    bb300_mid_1d: float = 0.0
    bb300_lower_1d: float = 0.0
    bb672_upper_1d: float = 0.0
    bb672_mid_1d: float = 0.0
    bb672_lower_1d: float = 0.0
    bb1440_upper_1d: float = 0.0
    bb1440_mid_1d: float = 0.0
    bb1440_lower_1d: float = 0.0
    bb1920_upper_1d: float = 0.0
    bb1920_mid_1d: float = 0.0
    bb1920_lower_1d: float = 0.0
    bb2880_upper_1d: float = 0.0
    bb2880_mid_1d: float = 0.0
    bb2880_lower_1d: float = 0.0
    bb4404_upper_1d: float = 0.0
    bb4404_mid_1d: float = 0.0
    bb4404_lower_1d: float = 0.0
    bb4460_upper_1d: float = 0.0
    bb4460_mid_1d: float = 0.0
    bb4460_lower_1d: float = 0.0
    bb4707_upper_1d: float = 0.0
    bb4707_mid_1d: float = 0.0
    bb4707_lower_1d: float = 0.0
    daily_price_vs_ma7_pct: float = 0.0
    daily_price_vs_ma20_pct: float = 0.0
    daily_price_position_365: float = 0.5
    bb_price_position_96_1d: float = 0.5
    bb_price_position_120_1d: float = 0.5
    bb_price_position_210_1d: float = 0.5
    bb_price_position_300_1d: float = 0.5
    bb_price_position_672_1d: float = 0.5
    bb_price_position_1440_1d: float = 0.5
    bb_price_position_1920_1d: float = 0.5
    bb_price_position_2880_1d: float = 0.5
    bb_price_position_4404_1d: float = 0.5
    bb_price_position_4460_1d: float = 0.5
    bb_price_position_4707_1d: float = 0.5
    bb_price_position_105_15m: float = 0.5
    bb_price_position_96: float = 0.5
    bb_price_position_120: float = 0.5
    bb_price_position_210: float = 0.5
    bb_price_position_300: float = 0.5
    bb_price_position_672_15m: float = 0.5
    bb_price_position_1440_15m: float = 0.5
    bb_price_position_1920_15m: float = 0.5
    bb_price_position_2880_15m: float = 0.5
    bb_price_position_4404_15m: float = 0.5
    bb_price_position_4460_15m: float = 0.5
    bb_price_position_4707_15m: float = 0.5
    bb_price_position_105_1d: float = 0.5
    bb_price_position_105_4h: float = 0.5
    bb_price_position_96_4h: float = 0.5
    bb_price_position_120_4h: float = 0.5
    bb_price_position_210_4h: float = 0.5
    bb_price_position_300_4h: float = 0.5
    bb_price_position_672_4h: float = 0.5
    bb_price_position_1440_4h: float = 0.5
    bb_price_position_1920_4h: float = 0.5
    bb_price_position_2880_4h: float = 0.5
    bb_price_position_4404_4h: float = 0.5
    bb_price_position_4460_4h: float = 0.5
    bb_price_position_4707_4h: float = 0.5
    bb_price_position_105_1m: float = 0.5
    bb_price_position_300_1m: float = 0.5
    bb_price_position_672_1m: float = 0.5
    bb_price_position_1920_1m: float = 0.5
    williams_r_15m: float = -50.0
    williams_r_15m_120: float = -50.0
    williams_r_1m: float = -50.0
    williams_r_15m_zone: str = "upper_mid"
    williams_r_15m_120_zone: str = "upper_mid"
    williams_r_1m_zone: str = "upper_mid"
    bb_touch_zone: str = "none"
    intraday_price_position_15m: float | None = None
    intraday_breakout_distance_15m: float | None = None
    intraday_compression_score_15m: float | None = None
    structure_timeframe: str | None = None
    setup_type: str = "trend"
    setup_quality: float = 0.0
    manipulation_stage: str = "trend"
    manipulation_quality: float = 0.0
    manipulation_flags: list[str] = field(default_factory=list)
    oi_price_divergence_score: float = 0.0
    derivatives_flow_regime: str = "neutral"
    derivatives_flow_tags: list[str] = field(default_factory=list)
    derivatives_flow_confidence_adjustment: float = 0.0
    derivatives_flow_long_risk_score: float = 0.0
    derivatives_flow_short_risk_score: float = 0.0
    derivatives_flow_spot_confirmation: str = "unknown"
    derivatives_flow_explanation: str = ""
    hotness_priority_score: float = 0.0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TradePlan:
    action: str
    symbol: str
    score: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    notional_usdt: float
    rationale: str
    size_fraction: float = 1.0
    setup_type: str = "trend"
    sentiment_mentions: int = 0
    sentiment_score: float = 0.0
    oi_change_pct_4h: float = 0.0
    taker_buy_sell_ratio: float = 0.0
    smart_money_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Position:
    symbol: str
    opened_at: str
    entry_price: float
    current_price: float
    stop_loss_price: float
    take_profit_price: float
    notional_usdt: float
    score: float
    rationale: str
    setup_type: str = "trend"
    take_profit_stage: int = 0
    initial_risk_pct: float = 0.0
    structure_timeframe: str = "daily"
    last_sentiment_score: float = 0.0
    last_sentiment_mentions: int = 0
    last_oi_change_pct_4h: float = 0.0
    last_taker_buy_sell_ratio: float = 0.0
    # 入场时完整 CoinFeatures（用于模型回测）
    entry_coin_features: dict[str, Any] | None = None
    # K-line pattern observation at entry time
    entry_kline_pattern_bullish_count: float = 0.0
    entry_kline_pattern_bearish_count: float = 0.0
    entry_kline_pattern_flags: str = ""
    entry_h4_structure: dict[str, Any] | None = None
    state_warning: str = ""
    pyramid_count: int = 0
    predictive_snapshot: dict[str, Any] | None = None
    high_watermark_pnl_pct: float = 0.0

    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pnl_pct"] = round(self.pnl_pct(), 6)
        return data
