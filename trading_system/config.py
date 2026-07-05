from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_token_pool(*names: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for name in names:
        token = os.getenv(name, "").strip()
        if token and token not in tokens:
            tokens.append(token)
    return tuple(tokens)


def _first_env_token(*names: str) -> str:
    tokens = _env_token_pool(*names)
    return tokens[0] if tokens else ""


def _env_csv_tuple(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    values: list[str] = []
    for item in raw.replace(";", ",").split(","):
        value = item.strip().upper()
        if value and value not in values:
            values.append(value)
    return tuple(values)


@dataclass(slots=True)
class StrategyConfig:
    base_url: str = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", os.getenv("BINANCE_SECRET_KEY", ""))
    binance_testnet_base_url: str = os.getenv("BINANCE_TESTNET_FUTURES_BASE_URL", "https://demo-fapi.binance.com")
    binance_testnet_api_key: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    binance_testnet_api_secret: str = os.getenv(
        "BINANCE_TESTNET_API_SECRET",
        os.getenv("BINANCE_TESTNET_SECRET_KEY", ""),
    )
    binance_recv_window_ms: int = int(os.getenv("BINANCE_RECV_WINDOW_MS", "30000"))
    binance_position_mode: str = os.getenv("TS_BINANCE_POSITION_MODE", os.getenv("BINANCE_POSITION_MODE", "auto"))
    min_quote_volume_usdt: float = float(os.getenv("TS_MIN_QUOTE_VOLUME_USDT", "20000000"))
    min_price_change_pct: float = float(os.getenv("TS_MIN_PRICE_CHANGE_PCT", "3"))
    min_sentiment_score: float = float(os.getenv("TS_MIN_SENTIMENT_SCORE", "40"))
    news_confirmation_bonus_enabled: bool = os.getenv("TS_NEWS_CONFIRMATION_BONUS_ENABLED", "true").lower() == "true"
    news_confirmation_bonus_max: float = float(os.getenv("TS_NEWS_CONFIRMATION_BONUS_MAX", "8"))
    news_confirmation_min_score: float = float(os.getenv("TS_NEWS_CONFIRMATION_MIN_SCORE", "60"))
    news_confirmation_twitter_min_mentions: int = int(os.getenv("TS_NEWS_CONFIRMATION_TWITTER_MIN_MENTIONS", "2"))
    min_oi_change_pct: float = float(os.getenv("TS_MIN_OI_CHANGE_PCT", "5"))
    min_positive_funding_rate: float = float(os.getenv("TS_MIN_POSITIVE_FUNDING_RATE", "0"))
    min_taker_ratio: float = float(os.getenv("TS_MIN_TAKER_RATIO", "1.02"))
    min_listing_days: int = int(os.getenv("TS_MIN_LISTING_DAYS", "7"))
    max_listing_days: int = int(os.getenv("TS_MAX_LISTING_DAYS", "365"))
    max_universe_size: int = int(os.getenv("TS_MAX_UNIVERSE_SIZE", "30"))
    short_scan_enabled: bool = os.getenv("TS_SHORT_SCAN_ENABLED", "true").lower() == "true"
    short_scan_min_quote_volume_usdt: float = float(
        os.getenv("TS_SHORT_SCAN_MIN_QUOTE_VOLUME_USDT", os.getenv("TS_MIN_QUOTE_VOLUME_USDT", "20000000"))
    )
    short_scan_min_abs_price_change_pct: float = float(
        os.getenv("TS_SHORT_SCAN_MIN_ABS_PRICE_CHANGE_PCT", os.getenv("TS_MIN_PRICE_CHANGE_PCT", "3"))
    )
    short_scan_max_universe_size: int = int(os.getenv("TS_SHORT_SCAN_MAX_UNIVERSE_SIZE", "30"))
    short_scan_squeeze_min_price_change_pct: float = float(
        os.getenv("TS_SHORT_SCAN_SQUEEZE_MIN_PRICE_CHANGE_PCT", os.getenv("TS_ANOMALY_PRICE_CHANGE_PCT", "80"))
    )
    entry_threshold: float = float(os.getenv("TS_ENTRY_THRESHOLD", "60"))
    exit_threshold: float = float(os.getenv("TS_EXIT_THRESHOLD", "45"))
    max_positions: int = int(os.getenv("TS_MAX_POSITIONS", "10"))
    # --- 聪明钱面板融入打分 (置信度浮动权重, 默认关) ---
    panel_blend_enabled: bool = os.getenv("TS_PANEL_BLEND_ENABLED", "false").lower() == "true"
    panel_blend_max_weight: float = float(os.getenv("TS_PANEL_BLEND_MAX_WEIGHT", "0.30"))
    panel_blend_conf_floor: float = float(os.getenv("TS_PANEL_BLEND_CONF_FLOOR", "70"))
    panel_blend_conf_ceil: float = float(os.getenv("TS_PANEL_BLEND_CONF_CEIL", "90"))
    panel_blend_candidate_multiplier: int = int(os.getenv("TS_PANEL_BLEND_CANDIDATE_MULTIPLIER", "2"))
    shadow_paper_enabled: bool = os.getenv("TS_SHADOW_PAPER_ENABLED", "false").lower() == "true"
    shadow_paper_prob_moon_threshold: float = float(os.getenv("TS_SHADOW_PAPER_PROB_MOON_THRESHOLD", "0.6"))
    shadow_paper_max_positions: int = int(os.getenv("TS_SHADOW_PAPER_MAX_POSITIONS", "5"))
    shadow_paper_max_hold_hours: int = int(os.getenv("TS_SHADOW_PAPER_MAX_HOLD_HOURS", "24"))
    shadow_paper_journal_limit: int = int(os.getenv("TS_SHADOW_PAPER_JOURNAL_LIMIT", "2000"))
    shadow_paper_reentry_cooldown_minutes: int = int(os.getenv("TS_SHADOW_PAPER_REENTRY_COOLDOWN_MINUTES", "120"))
    shadow_paper_reentry_min_price_move_pct: float = float(os.getenv("TS_SHADOW_PAPER_REENTRY_MIN_PRICE_MOVE_PCT", "0.03"))
    shadow_paper_partial_tp_fraction: float = float(os.getenv("TS_SHADOW_PAPER_PARTIAL_TP_FRACTION", "0.333"))
    shadow_paper_trailing_pct: float = float(os.getenv("TS_SHADOW_PAPER_TRAILING_PCT", "0.07"))
    shadow_paper_max_tp_multiplier: float = float(os.getenv("TS_SHADOW_PAPER_MAX_TP_MULTIPLIER", "2.0"))
    shadow_paper_tp_stage1_r: float = float(os.getenv("TS_SHADOW_PAPER_TP_STAGE1_R", "1.5"))
    shadow_paper_tp_stage2_r: float = float(os.getenv("TS_SHADOW_PAPER_TP_STAGE2_R", "3.0"))
    shadow_paper_tp_stage3_r: float = float(os.getenv("TS_SHADOW_PAPER_TP_STAGE3_R", "5.0"))
    shadow_paper_tp_stage1_fraction: float = float(os.getenv("TS_SHADOW_PAPER_TP_STAGE1_FRACTION", "0.333"))
    shadow_paper_tp_stage2_fraction: float = float(os.getenv("TS_SHADOW_PAPER_TP_STAGE2_FRACTION", "0.5"))
    shadow_paper_breakeven_after_stage1: bool = os.getenv("TS_SHADOW_PAPER_BREAKEVEN_AFTER_STAGE1", "true").lower() == "true"
    block_trend_entry: bool = os.getenv("TS_BLOCK_TREND_ENTRY", "true").lower() == "true"
    macro_trend_enabled: bool = os.getenv("TS_MACRO_TREND_ENABLED", "true").lower() == "true"
    macro_trend_ma_period: int = int(os.getenv("TS_MACRO_TREND_MA_PERIOD", "7"))
    macro_trend_ma20_period: int = int(os.getenv("TS_MACRO_TREND_MA20_PERIOD", "20"))
    macro_trend_neutral_band_pct: float = float(os.getenv("TS_MACRO_TREND_NEUTRAL_BAND_PCT", "0.75"))
    macro_trend_slope_neutral_threshold_pct: float = float(os.getenv("TS_MACRO_TREND_SLOPE_NEUTRAL_THRESHOLD_PCT", "1.0"))
    macro_trend_cache_ttl_seconds: int = int(os.getenv("TS_MACRO_TREND_CACHE_TTL_SECONDS", str(6 * 60 * 60)))
    macro_trend_cache_file: Path = Path(os.getenv("TS_MACRO_TREND_CACHE_FILE", "data/trading_system/macro_trend_cache.json"))
    absolute_min_sentiment_mentions: int = int(os.getenv("TS_ABSOLUTE_MIN_SENTIMENT_MENTIONS", "10"))
    min_entry_sentiment_mentions: int = int(os.getenv("TS_MIN_ENTRY_SENTIMENT_MENTIONS", "3"))
    low_info_min_sentiment_mentions: int = int(os.getenv("TS_LOW_INFO_MIN_SENTIMENT_MENTIONS", "5"))
    # Historical paper trades with hotness_priority_score/priority_score below 5 have been
    # low-information churn (13.95% win rate vs 35.38% above 5). Keep trusted/manual ideas
    # eligible, but require untrusted entries to clear a minimal priority signal.
    low_info_min_hotness_priority_score: float = float(os.getenv("TS_LOW_INFO_MIN_HOTNESS_PRIORITY_SCORE", "5"))
    untrusted_entry_min_sentiment_mentions: int = int(os.getenv("TS_UNTRUSTED_ENTRY_MIN_SENTIMENT_MENTIONS", "50"))
    untrusted_entry_min_source_count: int = int(os.getenv("TS_UNTRUSTED_ENTRY_MIN_SOURCE_COUNT", "3"))
    untrusted_entry_high_mentions_override: int = int(os.getenv("TS_UNTRUSTED_ENTRY_HIGH_MENTIONS_OVERRIDE", "50"))
    untrusted_entry_min_score: float = float(os.getenv("TS_UNTRUSTED_ENTRY_MIN_SCORE", "85"))
    manual_hard_block_symbols: str = os.getenv("TS_MANUAL_HARD_BLOCK_SYMBOLS", "")
    symbol_block_zero_win_min_closed_trades: int = int(os.getenv("TS_SYMBOL_BLOCK_ZERO_WIN_MIN_CLOSED_TRADES", "2"))
    symbol_block_zero_win_apply_to_confirmed: bool = os.getenv(
        "TS_SYMBOL_BLOCK_ZERO_WIN_APPLY_TO_CONFIRMED", "true"
    ).lower() == "true"
    symbol_block_min_closed_trades: int = int(os.getenv("TS_SYMBOL_BLOCK_MIN_CLOSED_TRADES", "3"))
    symbol_block_low_win_rate_min_closed_trades: int = int(os.getenv("TS_SYMBOL_BLOCK_LOW_WIN_RATE_MIN_CLOSED_TRADES", "4"))
    symbol_block_low_win_rate_max_win_rate: float = float(os.getenv("TS_SYMBOL_BLOCK_LOW_WIN_RATE_MAX_WIN_RATE", "0.25"))
    symbol_hard_block_underperform_min_closed_trades: int = int(os.getenv("TS_SYMBOL_HARD_BLOCK_UNDERPERFORM_MIN_CLOSED_TRADES", "6"))
    symbol_hard_block_underperform_max_win_rate: float = float(os.getenv("TS_SYMBOL_HARD_BLOCK_UNDERPERFORM_MAX_WIN_RATE", "0.25"))
    symbol_hard_block_underperform_max_avg_pnl_pct: float = float(os.getenv("TS_SYMBOL_HARD_BLOCK_UNDERPERFORM_MAX_AVG_PNL_PCT", "-0.005"))
    setup_block_min_closed_trades: int = int(os.getenv("TS_SETUP_BLOCK_MIN_CLOSED_TRADES", "2"))
    setup_type_block_min_closed_trades: int = int(os.getenv("TS_SETUP_TYPE_BLOCK_MIN_CLOSED_TRADES", "6"))
    setup_type_block_max_win_rate: float = float(os.getenv("TS_SETUP_TYPE_BLOCK_MAX_WIN_RATE", "0.15"))
    low_info_setup_type_block_min_closed_trades: int = int(os.getenv("TS_LOW_INFO_SETUP_TYPE_BLOCK_MIN_CLOSED_TRADES", "3"))
    low_info_setup_type_block_max_win_rate: float = float(os.getenv("TS_LOW_INFO_SETUP_TYPE_BLOCK_MAX_WIN_RATE", "0.10"))
    low_info_symbol_block_min_closed_trades: int = int(os.getenv("TS_LOW_INFO_SYMBOL_BLOCK_MIN_CLOSED_TRADES", "2"))
    low_info_symbol_block_max_win_rate: float = float(os.getenv("TS_LOW_INFO_SYMBOL_BLOCK_MAX_WIN_RATE", "0.0"))
    oi_divergence_setup_block_min_closed_trades: int = int(os.getenv("TS_OI_DIVERGENCE_SETUP_BLOCK_MIN_CLOSED_TRADES", "4"))
    oi_divergence_setup_block_max_win_rate: float = float(os.getenv("TS_OI_DIVERGENCE_SETUP_BLOCK_MAX_WIN_RATE", "0.25"))
    base_breakout_setup_block_min_closed_trades: int = int(os.getenv("TS_BASE_BREAKOUT_SETUP_BLOCK_MIN_CLOSED_TRADES", "3"))
    base_breakout_setup_block_max_win_rate: float = float(os.getenv("TS_BASE_BREAKOUT_SETUP_BLOCK_MAX_WIN_RATE", "0.1"))
    symbol_block_lookback: int = int(os.getenv("TS_SYMBOL_BLOCK_LOOKBACK", "12"))
    recent_symbol_loss_streak_block: int = int(os.getenv("TS_RECENT_SYMBOL_LOSS_STREAK_BLOCK", "3"))
    recent_setup_loss_streak_block: int = int(os.getenv("TS_RECENT_SETUP_LOSS_STREAK_BLOCK", "2"))
    stop_loss_symbol_block_min_closed_trades: int = int(os.getenv("TS_STOP_LOSS_SYMBOL_BLOCK_MIN_CLOSED_TRADES", "4"))
    stop_loss_setup_block_min_closed_trades: int = int(os.getenv("TS_STOP_LOSS_SETUP_BLOCK_MIN_CLOSED_TRADES", "1"))
    stop_loss_block_apply_to_confirmed: bool = os.getenv(
        "TS_STOP_LOSS_BLOCK_APPLY_TO_CONFIRMED", "true"
    ).lower() == "true"
    # Optional consecutive stop-loss blockers. Production can disable these and use half-sizing instead.
    stop_loss_consecutive_symbol_block: int = int(os.getenv("TS_STOP_LOSS_CONSECUTIVE_SYMBOL_BLOCK", "3"))
    stop_loss_consecutive_setup_block: int = int(os.getenv("TS_STOP_LOSS_CONSECUTIVE_SETUP_BLOCK", "2"))
    flat_score_fade_block_min_closed_trades: int = int(os.getenv("TS_FLAT_SCORE_FADE_BLOCK_MIN_CLOSED_TRADES", "2"))
    # Treat repeated score-faded exits as low-information churn up to a wider 5% loss band.
    # The live paper dataset shows score_faded non-winners are mostly zero/bleed samples;
    # blocking their repeat symbol/setup combinations improves the effective training win rate.
    flat_score_fade_max_abs_pnl_pct: float = float(os.getenv("TS_FLAT_SCORE_FADE_MAX_ABS_PNL_PCT", "0.05"))
    non_progress_symbol_block_min_closed_trades: int = int(os.getenv("TS_NON_PROGRESS_SYMBOL_BLOCK_MIN_CLOSED_TRADES", "2"))
    non_progress_setup_block_min_closed_trades: int = int(os.getenv("TS_NON_PROGRESS_SETUP_BLOCK_MIN_CLOSED_TRADES", "2"))
    recent_reentry_cooldown_minutes: int = int(os.getenv("TS_RECENT_REENTRY_COOLDOWN_MINUTES", "120"))
    recent_reentry_min_price_move_pct: float = float(os.getenv("TS_RECENT_REENTRY_MIN_PRICE_MOVE_PCT", "0.05"))
    recent_reentry_min_score_delta: float = float(os.getenv("TS_RECENT_REENTRY_MIN_SCORE_DELTA", "8"))
    non_progress_reentry_cooldown_minutes: int = int(os.getenv("TS_NON_PROGRESS_REENTRY_COOLDOWN_MINUTES", "720"))
    non_progress_reentry_min_sentiment_mentions: int = int(os.getenv("TS_NON_PROGRESS_REENTRY_MIN_SENTIMENT_MENTIONS", "20"))
    non_progress_reentry_min_score_delta: float = float(os.getenv("TS_NON_PROGRESS_REENTRY_MIN_SCORE_DELTA", "15"))
    position_notional_usdt: float = float(os.getenv("TS_POSITION_NOTIONAL_USDT", "200"))
    min_entry_notional_usdt: float = float(os.getenv("TS_MIN_ENTRY_NOTIONAL_USDT", "0"))
    stop_loss_pct: float = float(os.getenv("TS_STOP_LOSS_PCT", "0.03"))
    # Live paper replay is dragged mainly by outsized stop-loss exits (avg stop loss >8%).
    # Cap volatility-expanded stops at 4% so high-volatility memes still train, but cannot
    # donate repeated 8-12% loss samples before the symbol/setup blockers engage.
    max_dynamic_stop_loss_pct: float = float(os.getenv("TS_MAX_DYNAMIC_STOP_LOSS_PCT", "0.04"))
    # === 双层止损改造 (2026-05-30) ===
    # 灾难硬止损: 入场即锚定的不可逾越底线, 防 -94% 灾难。按币种波动率(ATR%)动态。
    # hard_stop_pct = clamp(ATR_ratio * mult, floor, ceil); ATR_ratio 为小数比例(0.18=18%)。
    disaster_hard_stop_enabled: bool = os.getenv("TS_DISASTER_HARD_STOP_ENABLED", "true").lower() == "true"
    disaster_hard_stop_atr_mult: float = float(os.getenv("TS_DISASTER_HARD_STOP_ATR_MULT", "0.6"))
    disaster_hard_stop_floor_pct: float = float(os.getenv("TS_DISASTER_HARD_STOP_FLOOR_PCT", "0.06"))
    disaster_hard_stop_ceil_pct: float = float(os.getenv("TS_DISASTER_HARD_STOP_CEIL_PCT", "0.10"))
    # 结构止损: 入场即锚定 range_low 下方 buffer 处, 不等破位确认(原逻辑滞后 8-15%)。
    structural_stop_buffer_pct: float = float(os.getenv("TS_STRUCTURAL_STOP_BUFFER_PCT", "0.015"))
    take_profit_pct: float = float(os.getenv("TS_TAKE_PROFIT_PCT", "0.08"))
    # Staged take-profit for main positions (replaces fixed take_profit_pct)
    staged_tp_enabled: bool = os.getenv("TS_STAGED_TP_ENABLED", "true").lower() == "true"
    staged_tp_stage1_r: float = float(os.getenv("TS_STAGED_TP_STAGE1_R", "1.5"))
    staged_tp_stage2_r: float = float(os.getenv("TS_STAGED_TP_STAGE2_R", "3.0"))
    staged_tp_stage3_r: float = float(os.getenv("TS_STAGED_TP_STAGE3_R", "5.0"))
    staged_tp_stage1_fraction: float = float(os.getenv("TS_STAGED_TP_STAGE1_FRACTION", "0.333"))
    staged_tp_stage2_fraction: float = float(os.getenv("TS_STAGED_TP_STAGE2_FRACTION", "0.5"))
    staged_tp_trailing_pct: float = float(os.getenv("TS_STAGED_TP_TRAILING_PCT", "0.07"))
    staged_tp_breakeven_after_stage1: bool = os.getenv("TS_STAGED_TP_BREAKEVEN_AFTER_STAGE1", "true").lower() == "true"
    breakout_stop_loss_floor_pct: float = float(os.getenv("TS_BREAKOUT_STOP_LOSS_FLOOR_PCT", "0.02"))
    breakout_stop_loss_cap_pct: float = float(os.getenv("TS_BREAKOUT_STOP_LOSS_CAP_PCT", "0.04"))
    breakout_risk_multiple_from_distance: float = float(os.getenv("TS_BREAKOUT_RISK_MULTIPLE_FROM_DISTANCE", "2.0"))
    breakout_first_take_profit_r: float = float(os.getenv("TS_BREAKOUT_FIRST_TAKE_PROFIT_R", "1.0"))
    breakout_second_take_profit_r: float = float(os.getenv("TS_BREAKOUT_SECOND_TAKE_PROFIT_R", "2.0"))
    breakout_runner_take_profit_r: float = float(os.getenv("TS_BREAKOUT_RUNNER_TAKE_PROFIT_R", "4.0"))
    breakout_first_exit_fraction: float = float(os.getenv("TS_BREAKOUT_FIRST_EXIT_FRACTION", "0.25"))
    breakout_second_exit_fraction: float = float(os.getenv("TS_BREAKOUT_SECOND_EXIT_FRACTION", "0.25"))
    breakout_retain_min_sentiment_score: float = float(os.getenv("TS_BREAKOUT_RETAIN_MIN_SENTIMENT_SCORE", "70"))
    breakout_retain_min_mentions: int = int(os.getenv("TS_BREAKOUT_RETAIN_MIN_MENTIONS", "25"))
    breakout_retain_min_oi_change_pct: float = float(os.getenv("TS_BREAKOUT_RETAIN_MIN_OI_CHANGE_PCT", "8"))
    breakout_retain_min_taker_ratio: float = float(os.getenv("TS_BREAKOUT_RETAIN_MIN_TAKER_RATIO", "1.08"))
    max_hold_hours: int = int(os.getenv("TS_MAX_HOLD_HOURS", "24"))
    # High-conviction breakout/continuation setups are allowed to run longer before time-based exit.
    long_hold_hours: int = int(os.getenv("TS_LONG_HOLD_HOURS", "72"))
    trend_followthrough_max_hold_hours: int = int(os.getenv("TS_TREND_FOLLOWTHROUGH_MAX_HOLD_HOURS", str(24)))
    mania_followthrough_max_hold_hours: int = int(os.getenv("TS_MANIA_FOLLOWTHROUGH_MAX_HOLD_HOURS", str(48)))
    trend_up_exit_threshold: float = float(os.getenv("TS_TREND_UP_EXIT_THRESHOLD", "30"))
    trend_up_tp_extension_multiplier: float = float(os.getenv("TS_TREND_UP_TP_EXTENSION_MULTIPLIER", "1.5"))
    trend_up_max_hold_hours: int = int(os.getenv("TS_TREND_UP_MAX_HOLD_HOURS", "48"))
    hold_extension_min_sentiment_score: float = float(os.getenv("TS_HOLD_EXTENSION_MIN_SENTIMENT_SCORE", "65"))
    hold_extension_min_mentions: int = int(os.getenv("TS_HOLD_EXTENSION_MIN_MENTIONS", "20"))
    hold_extension_min_oi_change_pct: float = float(os.getenv("TS_HOLD_EXTENSION_MIN_OI_CHANGE_PCT", "5"))
    hold_extension_min_taker_ratio: float = float(os.getenv("TS_HOLD_EXTENSION_MIN_TAKER_RATIO", "1.03"))
    hold_extension_min_quote_volume_usdt: float = float(os.getenv("TS_HOLD_EXTENSION_MIN_QUOTE_VOLUME_USDT", "20000000"))
    hold_extension_max_adverse_pnl_pct: float = float(os.getenv("TS_HOLD_EXTENSION_MAX_ADVERSE_PNL_PCT", "-0.01"))
    repeated_signal_4h_enabled: bool = os.getenv("TS_REPEATED_SIGNAL_4H_ENABLED", "true").lower() == "true"
    repeated_signal_4h_min_score: float = float(os.getenv("TS_REPEATED_SIGNAL_4H_MIN_SCORE", "60"))
    repeated_signal_4h_min_mentions: int = int(os.getenv("TS_REPEATED_SIGNAL_4H_MIN_MENTIONS", "20"))
    repeated_signal_4h_min_source_count: int = int(os.getenv("TS_REPEATED_SIGNAL_4H_MIN_SOURCE_COUNT", "2"))
    repeated_signal_4h_min_oi_change_pct: float = float(os.getenv("TS_REPEATED_SIGNAL_4H_MIN_OI_CHANGE_PCT", "-8"))
    repeated_signal_4h_min_taker_ratio: float = float(os.getenv("TS_REPEATED_SIGNAL_4H_MIN_TAKER_RATIO", "0.98"))
    repeated_signal_4h_min_price_position: float = float(os.getenv("TS_REPEATED_SIGNAL_4H_MIN_PRICE_POSITION", "0.72"))
    repeated_signal_4h_max_breakout_distance: float = float(os.getenv("TS_REPEATED_SIGNAL_4H_MAX_BREAKOUT_DISTANCE", "0.45"))
    repeated_signal_4h_max_adverse_pnl_pct: float = float(os.getenv("TS_REPEATED_SIGNAL_4H_MAX_ADVERSE_PNL_PCT", "-0.12"))
    repeated_signal_4h_position_scale: float = float(os.getenv("TS_REPEATED_SIGNAL_4H_POSITION_SCALE", "0.6"))
    first_signal_probe_enabled: bool = os.getenv("TS_FIRST_SIGNAL_PROBE_ENABLED", "true").lower() == "true"
    first_signal_probe_min_score: float = float(os.getenv("TS_FIRST_SIGNAL_PROBE_MIN_SCORE", "35"))
    first_signal_probe_min_sentiment_score: float = float(os.getenv("TS_FIRST_SIGNAL_PROBE_MIN_SENTIMENT_SCORE", "70"))
    first_signal_probe_min_mentions: int = int(os.getenv("TS_FIRST_SIGNAL_PROBE_MIN_MENTIONS", "10"))
    first_signal_probe_min_taker_ratio: float = float(os.getenv("TS_FIRST_SIGNAL_PROBE_MIN_TAKER_RATIO", "1.0"))
    first_signal_probe_min_quote_volume_usdt: float = float(os.getenv("TS_FIRST_SIGNAL_PROBE_MIN_QUOTE_VOLUME_USDT", "20000000"))
    first_signal_probe_max_breakout_distance: float = float(os.getenv("TS_FIRST_SIGNAL_PROBE_MAX_BREAKOUT_DISTANCE", "0.12"))
    first_signal_probe_position_scale: float = float(os.getenv("TS_FIRST_SIGNAL_PROBE_POSITION_SCALE", "0.25"))
    retest_add_enabled: bool = os.getenv("TS_RETEST_ADD_ENABLED", "true").lower() == "true"
    retest_add_min_price_position: float = float(os.getenv("TS_RETEST_ADD_MIN_PRICE_POSITION", "0.88"))
    retest_add_max_breakout_distance: float = float(os.getenv("TS_RETEST_ADD_MAX_BREAKOUT_DISTANCE", "0.20"))
    retest_add_position_scale: float = float(os.getenv("TS_RETEST_ADD_POSITION_SCALE", "0.4"))
    execution_mode: str = os.getenv("TS_EXECUTION_MODE", "paper")
    execution_order_mode: str = os.getenv("TS_EXECUTION_ORDER_MODE", "auto")
    execution_signal_source: str = os.getenv("TS_EXECUTION_SIGNAL_SOURCE", "paper")
    state_schema_version: int = int(os.getenv("TS_STATE_SCHEMA_VERSION", "1"))
    live_execution_requires_confirmation: bool = os.getenv(
        "TS_LIVE_EXECUTION_REQUIRES_CONFIRMATION", "true"
    ).lower() == "true"
    live_execution_confirmation_phrase: str = os.getenv(
        "TS_LIVE_EXECUTION_CONFIRMATION_PHRASE",
        "CONFIRM_LIVE_BINANCE_FUTURES",
    )
    live_execution_confirmation: str = os.getenv("TS_LIVE_EXECUTION_CONFIRMATION", "")
    live_max_request_loss_usdt: float = float(os.getenv("TS_LIVE_MAX_REQUEST_LOSS_USDT", "50"))
    live_max_order_notional_usdt: float = float(os.getenv("TS_LIVE_MAX_ORDER_NOTIONAL_USDT", "1000"))
    live_max_intents_per_request: int = int(os.getenv("TS_LIVE_MAX_INTENTS_PER_REQUEST", "5"))
    log_level: str = os.getenv("TS_LOG_LEVEL", "INFO")
    log_format: str = os.getenv("TS_LOG_FORMAT", "json")
    execution_skill_name: str = os.getenv(
        "TS_EXECUTION_SKILL_NAME",
        "derivatives-trading-usds-futures",
    )
    execution_test_order: bool = os.getenv("TS_EXECUTION_TEST_ORDER", "true").lower() == "true"
    default_leverage: int = int(os.getenv("TS_DEFAULT_LEVERAGE", "8"))
    dynamic_leverage_enabled: bool = os.getenv("TS_DYNAMIC_LEVERAGE_ENABLED", "true").lower() == "true"
    min_leverage: int = int(os.getenv("TS_MIN_LEVERAGE", "4"))
    max_leverage: int = int(os.getenv("TS_MAX_LEVERAGE", "12"))
    leverage_score_baseline: float = float(os.getenv("TS_LEVERAGE_SCORE_BASELINE", "80"))
    sentiment_file: Path = Path(os.getenv("TS_SENTIMENT_FILE", "data/trading_system/sentiment_signals.csv"))
    replay_file: Path = Path(os.getenv("TS_REPLAY_FILE", "data/trading_system/replay_market_data.json"))
    state_file: Path = Path(os.getenv("TS_STATE_FILE", "data/trading_system/paper_state.json"))
    data_dir: Path = Path(os.getenv("TS_DATA_DIR", "data/trading_system"))
    whale_overview_enabled: bool = os.getenv("TS_WHALE_OVERVIEW_ENABLED", "true").lower() == "true"
    oi_divergence_state_file: Path = Path(
        os.getenv("TS_OI_DIVERGENCE_STATE_FILE", "data/trading_system/oi_divergence_paper_state.json")
    )
    report_file: Path = Path(os.getenv("TS_REPORT_FILE", "data/trading_system/last_scan_report.json"))
    square_feed_report_file: Path = Path(
        os.getenv("TS_SQUARE_FEED_REPORT_FILE", "data/trading_system/square_feed_analysis.json")
    )
    square_enriched_report_file: Path = Path(
        os.getenv("TS_SQUARE_ENRICHED_REPORT_FILE", "data/trading_system/square_feed_enriched.json")
    )
    square_feed_max_age_minutes: int = int(os.getenv("TS_SQUARE_FEED_MAX_AGE_MINUTES", "30"))
    square_feed_stale_hotness_multiplier: float = float(os.getenv("TS_SQUARE_FEED_STALE_HOTNESS_MULTIPLIER", "0.25"))
    meme_radar_archive_dir: Path = Path(
        os.getenv("TS_MEME_RADAR_ARCHIVE_DIR", "data/trading_system/meme_radar_daily")
    )
    merged_sentiment_file: Path = Path(
        os.getenv("TS_MERGED_SENTIMENT_FILE", "data/trading_system/merged_sentiment_signals.csv")
    )
    issues_json_file: Path = Path(os.getenv("TS_ISSUES_JSON_FILE", "data/trading_system/open_issues.json"))
    issues_md_file: Path = Path(os.getenv("TS_ISSUES_MD_FILE", "data/trading_system/open_issues.md"))
    pipeline_report_file: Path = Path(
        os.getenv("TS_PIPELINE_REPORT_FILE", "data/trading_system/pipeline_run_report.json")
    )
    pipeline_report_md_file: Path = Path(
        os.getenv("TS_PIPELINE_REPORT_MD_FILE", "data/trading_system/pipeline_run_report.md")
    )
    opennews_report_file: Path = Path(
        os.getenv("TS_OPENNEWS_REPORT_FILE", "data/trading_system/opennews_report.json")
    )
    options_delivery_report_file: Path = Path(
        os.getenv("TS_OPTIONS_DELIVERY_REPORT_FILE", "data/trading_system/options_delivery_report.json")
    )
    options_delivery_enabled: bool = os.getenv("TS_OPTIONS_DELIVERY_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    options_delivery_underlyings: tuple[str, ...] = _env_csv_tuple("TS_OPTIONS_DELIVERY_UNDERLYINGS", "BTCUSDT,ETHUSDT")
    options_delivery_limit: int = int(os.getenv("TS_OPTIONS_DELIVERY_LIMIT", "5"))
    opentwitter_report_file: Path = Path(
        os.getenv("TS_OPENTWITTER_REPORT_FILE", "data/trading_system/opentwitter_report.json")
    )
    skill_bridge_dir: Path = Path(os.getenv("TS_SKILL_BRIDGE_DIR", "data/trading_system/skill_bridges"))
    skill_raw_dir: Path = Path(os.getenv("TS_SKILL_RAW_DIR", "data/trading_system/skill_raw"))
    execution_request_dir: Path = Path(
        os.getenv("TS_EXECUTION_REQUEST_DIR", "data/trading_system/execution_requests")
    )
    execution_result_dir: Path = Path(
        os.getenv("TS_EXECUTION_RESULT_DIR", "data/trading_system/execution_results")
    )
    execution_archive_dir: Path = Path(
        os.getenv("TS_EXECUTION_ARCHIVE_DIR", "data/trading_system/execution_archive")
    )
    execution_wal_file: Path = Path(
        os.getenv("TS_EXECUTION_WAL_FILE", "data/trading_system/execution_wal.jsonl")
    )
    opennews_tokens: tuple[str, ...] = _env_token_pool("OPENNEWS_PRIMARY_TOKEN", "OPENNEWS_TOKEN", "OPEN_TOKEN")
    twitter_tokens: tuple[str, ...] = _env_token_pool(
        "OPENTWITTER_PRIMARY_TOKEN",
        "TWITTER_PRIMARY_TOKEN",
        "TWITTER_TOKEN",
        "OPENTWITTER_TOKEN",
        "OPEN_TOKEN",
    )
    opennews_token: str = _first_env_token("OPENNEWS_PRIMARY_TOKEN", "OPENNEWS_TOKEN", "OPEN_TOKEN")
    twitter_token: str = _first_env_token(
        "OPENTWITTER_PRIMARY_TOKEN",
        "TWITTER_PRIMARY_TOKEN",
        "TWITTER_TOKEN",
        "OPENTWITTER_TOKEN",
        "OPEN_TOKEN",
    )
    disable_open_token_bridges: bool = os.getenv("TS_DISABLE_OPEN_TOKEN_BRIDGES", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    opennews_lookback_hours: int = int(os.getenv("TS_OPENNEWS_LOOKBACK_HOURS", "24"))
    opennews_page_limit: int = int(os.getenv("TS_OPENNEWS_PAGE_LIMIT", "20"))
    opennews_max_pages: int = int(os.getenv("TS_OPENNEWS_MAX_PAGES", "2"))
    targeted_opennews_page_limit: int = int(os.getenv("TS_TARGETED_OPENNEWS_PAGE_LIMIT", "8"))
    targeted_opennews_max_pages: int = int(os.getenv("TS_TARGETED_OPENNEWS_MAX_PAGES", "1"))
    targeted_twitter_max_results: int = int(os.getenv("TS_TARGETED_TWITTER_MAX_RESULTS", "100"))
    candidate_confirm_limit: int = int(os.getenv("TS_CANDIDATE_CONFIRM_LIMIT", "8"))
    anomaly_price_change_pct: float = float(os.getenv("TS_ANOMALY_PRICE_CHANGE_PCT", "80"))
    anomaly_price_change_pct_5m: float = float(os.getenv("TS_ANOMALY_PRICE_CHANGE_PCT_5M", "30"))
    anomaly_negative_funding_rate: float = float(os.getenv("TS_ANOMALY_NEGATIVE_FUNDING_RATE", "-0.002"))
    anomaly_positive_funding_rate: float = float(os.getenv("TS_ANOMALY_POSITIVE_FUNDING_RATE", "0.003"))
    anomaly_oi_change_pct_4h: float = float(os.getenv("TS_ANOMALY_OI_CHANGE_PCT_4H", "20"))
    anomaly_quote_volume_usdt_24h: float = float(os.getenv("TS_ANOMALY_QUOTE_VOLUME_USDT_24H", "200000000"))
    market_flash_crash_drop_pct_5m: float = float(os.getenv("TS_MARKET_FLASH_CRASH_DROP_PCT_5M", "8"))
    market_flash_crash_symbol_count: int = int(os.getenv("TS_MARKET_FLASH_CRASH_SYMBOL_COUNT", "3"))
    liquidity_fuse_min_taker_ratio: float = float(os.getenv("TS_LIQUIDITY_FUSE_MIN_TAKER_RATIO", "0.92"))
    liquidity_fuse_max_abs_price_change_pct_5m: float = float(os.getenv("TS_LIQUIDITY_FUSE_MAX_ABS_PRICE_CHANGE_PCT_5M", "12"))
    predictive_hard_block_enabled: bool = os.getenv("TS_PREDICTIVE_HARD_BLOCK_ENABLED", "true").lower() == "true"
    predictive_hard_block_max_win_prob: float = float(os.getenv("TS_PREDICTIVE_HARD_BLOCK_MAX_WIN_PROB", "0.35"))
    predictive_hard_block_min_stop_loss_risk: float = float(os.getenv("TS_PREDICTIVE_HARD_BLOCK_MIN_STOP_LOSS_RISK", "0.60"))
    predictive_soft_block_max_win_prob: float = float(os.getenv("TS_PREDICTIVE_SOFT_BLOCK_MAX_WIN_PROB", "0.28"))
    predictive_soft_block_min_stop_loss_risk: float = float(os.getenv("TS_PREDICTIVE_SOFT_BLOCK_MIN_STOP_LOSS_RISK", "0.75"))
    h4_weakening_score_penalty: float = float(os.getenv("TS_H4_WEAKENING_SCORE_PENALTY", "5"))
    h4_structure_filter_enabled: bool = os.getenv("TS_H4_STRUCTURE_FILTER_ENABLED", "true").lower() == "true"
    decision_dataset_file: Path = Path(
        os.getenv("TS_DECISION_DATASET_FILE", "data/trading_system/model_training/paper_replay_autolearn/paper_decision_samples.jsonl")
    )
    closed_trade_dataset_file: Path = Path(
        os.getenv("TS_CLOSED_TRADE_DATASET_FILE", "data/trading_system/model_training/paper_replay_autolearn/paper_closed_trades.jsonl")
    )
    oi_divergence_decision_dataset_file: Path = Path(
        os.getenv("TS_OI_DIVERGENCE_DECISION_DATASET_FILE", "data/trading_system/model_training/oi_divergence_independent/decision_samples.jsonl")
    )
    oi_divergence_closed_trade_dataset_file: Path = Path(
        os.getenv("TS_OI_DIVERGENCE_CLOSED_TRADE_DATASET_FILE", "data/trading_system/model_training/oi_divergence_independent/closed_trades.jsonl")
    )
    predictive_model_file: Path = Path(
        os.getenv("TS_PREDICTIVE_MODEL_FILE", "data/trading_system/model_training/paper_replay_autolearn/predictive_model_summary.json")
    )
    factor_backtest_file: Path = Path(
        os.getenv("TS_FACTOR_BACKTEST_FILE", "data/trading_system/model_training/paper_replay_autolearn/factor_backtest_report.json")
    )
    hf_backtest_entries_file: Path = Path(
        os.getenv("TS_HF_BACKTEST_ENTRIES_FILE", "data/trading_system/hf_backtest_v2_entries.jsonl")
    )
    hf_backtest_report_file: Path = Path(
        os.getenv("TS_HF_BACKTEST_REPORT_FILE", "data/trading_system/hf_backtest_v2_report.json")
    )
    hf_cache_ttl_seconds: int = int(os.getenv("TS_HF_CACHE_TTL_SECONDS", str(3 * 60 * 60)))
    hf_entry_boost_min_win_rate: float = float(os.getenv("TS_HF_ENTRY_BOOST_MIN_WIN_RATE", "0.55"))
    hf_entry_block_max_win_rate: float = float(os.getenv("TS_HF_ENTRY_BLOCK_MAX_WIN_RATE", "0.40"))
    hf_entry_block_min_samples: int = int(os.getenv("TS_HF_ENTRY_BLOCK_MIN_SAMPLES", "20"))
    factor_distilled_training_file: Path = Path(
        os.getenv(
            "TS_FACTOR_DISTILLED_TRAINING_FILE",
            "data/trading_system/model_training/paper_replay_autolearn/factor_distilled_training_samples.jsonl",
        )
    )
    bayes_opt_output_file: Path = Path(
        os.getenv("TS_BAYES_OPT_OUTPUT_FILE", "data/trading_system/model_training/bayes_opt_report.json")
    )
    # Secondary path for training improvement report (where the 12H optimizer writes)
    training_improvement_report_file: Path = Path(
        os.getenv("TS_TRAINING_IMPROVEMENT_REPORT_FILE", "data/trading_system/model_training/paper_replay_autolearn/training_improvement_report.json")
    )
    # Replay snapshot for live-training continuity
    replay_snapshot_file: Path = Path(
        os.getenv("TS_REPLAY_SNAPSHOT_FILE", "data/trading_system/model_training/paper_replay_autolearn/latest_replay_snapshot.json")
    )
    mania_followthrough_min_price_change_pct_24h: float = float(os.getenv("TS_MANIA_MIN_PRICE_CHANGE_PCT_24H", "18"))
    mania_followthrough_max_price_change_pct_24h: float = float(os.getenv("TS_MANIA_MAX_PRICE_CHANGE_PCT_24H", "120"))
    mania_followthrough_min_price_change_pct_5m: float = float(os.getenv("TS_MANIA_MIN_PRICE_CHANGE_PCT_5M", "2.5"))
    mania_followthrough_min_sentiment_score: float = float(os.getenv("TS_MANIA_MIN_SENTIMENT_SCORE", "65"))
    mania_followthrough_min_sentiment_mentions: int = int(os.getenv("TS_MANIA_MIN_SENTIMENT_MENTIONS", "3"))
    mania_followthrough_min_quote_volume_usdt: float = float(os.getenv("TS_MANIA_MIN_QUOTE_VOLUME_USDT", "50000000"))
    mania_followthrough_min_oi_change_pct_4h: float = float(os.getenv("TS_MANIA_MIN_OI_CHANGE_PCT_4H", "-2"))
    mania_followthrough_min_taker_ratio: float = float(os.getenv("TS_MANIA_MIN_TAKER_RATIO", "0.98"))
    mania_followthrough_position_scale: float = float(os.getenv("TS_MANIA_POSITION_SCALE", "0.5"))
    # Strong-trend/high-hotness entries are allowed only as a reduced-size probe.
    trend_followthrough_position_scale: float = float(os.getenv("TS_TREND_POSITION_SCALE", "0.25"))
    trend_followthrough_min_score: float = float(os.getenv("TS_TREND_MIN_SCORE", "85"))
    trend_followthrough_min_hotness_priority_score: float = float(os.getenv("TS_TREND_MIN_HOTNESS_PRIORITY_SCORE", "5"))
    trend_followthrough_min_sentiment_mentions: int = int(os.getenv("TS_TREND_MIN_SENTIMENT_MENTIONS", "3"))
    trend_followthrough_min_price_change_pct_24h: float = float(os.getenv("TS_TREND_MIN_PRICE_CHANGE_PCT_24H", "12"))
    mania_followthrough_stop_loss_pct: float = float(os.getenv("TS_MANIA_STOP_LOSS_PCT", "0.025"))
    mania_followthrough_take_profit_pct: float = float(os.getenv("TS_MANIA_TAKE_PROFIT_PCT", "0.06"))
    price_sanity_max_deviation_pct: float = float(os.getenv("TS_PRICE_SANITY_MAX_DEVIATION_PCT", "20"))
    oi_divergence_min_oi_change_pct_4h: float = float(os.getenv("TS_OI_DIVERGENCE_MIN_OI_CHANGE_PCT_4H", "12"))
    oi_divergence_max_abs_price_change_pct_4h: float = float(os.getenv("TS_OI_DIVERGENCE_MAX_ABS_PRICE_CHANGE_PCT_4H", "2.5"))
    oi_divergence_min_sentiment_score: float = float(os.getenv("TS_OI_DIVERGENCE_MIN_SENTIMENT_SCORE", "55"))
    oi_divergence_min_sentiment_mentions: int = int(os.getenv("TS_OI_DIVERGENCE_MIN_SENTIMENT_MENTIONS", "20"))
    oi_divergence_min_quote_volume_usdt: float = float(os.getenv("TS_OI_DIVERGENCE_MIN_QUOTE_VOLUME_USDT", "40000000"))
    oi_divergence_min_taker_ratio: float = float(os.getenv("TS_OI_DIVERGENCE_MIN_TAKER_RATIO", "1.01"))
    # _oi_price_divergence_score ~= oi_change_pct_4h * (1 - price_damping) * 10.
    # With the default OI floor (12%) and max flat-price band (2.5%), a valid setup scores
    # roughly 90-120. A 180 default made the OI-divergence branch practically unreachable.
    oi_divergence_min_divergence_score: float = float(os.getenv("TS_OI_DIVERGENCE_MIN_DIVERGENCE_SCORE", "90"))
    oi_divergence_min_source_count: int = int(os.getenv("TS_OI_DIVERGENCE_MIN_SOURCE_COUNT", "2"))
    oi_divergence_require_trusted_confirmation: bool = os.getenv(
        "TS_OI_DIVERGENCE_REQUIRE_TRUSTED_CONFIRMATION", "true"
    ).lower() == "true"
    oi_divergence_position_scale: float = float(os.getenv("TS_OI_DIVERGENCE_POSITION_SCALE", "0.35"))
    oi_divergence_stop_loss_pct: float = float(os.getenv("TS_OI_DIVERGENCE_STOP_LOSS_PCT", "0.022"))
    oi_divergence_take_profit_pct: float = float(os.getenv("TS_OI_DIVERGENCE_TAKE_PROFIT_PCT", "0.055"))
    oi_divergence_break_even_trigger_pct: float = float(os.getenv("TS_OI_DIVERGENCE_BREAK_EVEN_TRIGGER_PCT", "0.03"))
    oi_divergence_hold_min_oi_change_pct: float = float(os.getenv("TS_OI_DIVERGENCE_HOLD_MIN_OI_CHANGE_PCT", "6"))
    oi_divergence_hold_min_taker_ratio: float = float(os.getenv("TS_OI_DIVERGENCE_HOLD_MIN_TAKER_RATIO", "0.99"))
    oi_divergence_hold_min_sentiment_score: float = float(os.getenv("TS_OI_DIVERGENCE_HOLD_MIN_SENTIMENT_SCORE", "45"))
    base_breakout_min_price_change_pct: float = float(os.getenv("TS_BASE_BREAKOUT_MIN_PRICE_CHANGE_PCT", "4"))
    base_breakout_max_price_change_pct: float = float(os.getenv("TS_BASE_BREAKOUT_MAX_PRICE_CHANGE_PCT", "35"))
    base_breakout_min_price_position_7d: float = float(os.getenv("TS_BASE_BREAKOUT_MIN_PRICE_POSITION_7D", "0.88"))
    base_breakout_max_price_position_20d: float = float(os.getenv("TS_BASE_BREAKOUT_MAX_PRICE_POSITION_20D", "0.72"))
    base_breakout_max_breakout_distance_7d: float = float(os.getenv("TS_BASE_BREAKOUT_MAX_BREAKOUT_DISTANCE_7D", "0.035"))
    base_breakout_min_compression_score: float = float(os.getenv("TS_BASE_BREAKOUT_MIN_COMPRESSION_SCORE", "0.55"))
    base_breakout_require_trusted_confirmation: bool = os.getenv(
        "TS_BASE_BREAKOUT_REQUIRE_TRUSTED_CONFIRMATION", "true"
    ).lower() == "true"
    base_breakout_min_sentiment_mentions: int = int(os.getenv("TS_BASE_BREAKOUT_MIN_SENTIMENT_MENTIONS", "10"))
    base_breakout_min_source_count: int = int(os.getenv("TS_BASE_BREAKOUT_MIN_SOURCE_COUNT", "2"))

    # PDF Strategy Hard Blocks
    pdf_cooldown_after_sl_minutes: int = int(os.getenv("TS_PDF_COOLDOWN_AFTER_SL_MINUTES", "120"))
    pdf_protect_profit_pct: float = float(os.getenv("TS_PDF_PROTECT_PROFIT_PCT", "0.02"))
    score_faded_profit_protect_threshold: float = float(os.getenv("TS_SCORE_FADED_PROFIT_PROTECT_THRESHOLD", "0.03"))
    score_faded_profit_drawdown_exit: float = float(os.getenv("TS_SCORE_FADED_PROFIT_DRAWDOWN_EXIT", "0.5"))
    score_faded_structure_guard_enabled: bool = os.getenv(
        "TS_SCORE_FADED_STRUCTURE_GUARD_ENABLED", "true"
    ).lower() == "true"
    score_faded_structure_guard_max_adverse_pnl_pct: float = float(
        os.getenv("TS_SCORE_FADED_STRUCTURE_GUARD_MAX_ADVERSE_PNL_PCT", "-0.06")
    )
    score_faded_structure_guard_max_breakout_distance: float = float(
        os.getenv("TS_SCORE_FADED_STRUCTURE_GUARD_MAX_BREAKOUT_DISTANCE", "0.55")
    )
    pdf_trend_min_ma20_distance: float = float(os.getenv("TS_PDF_TREND_MIN_MA20_DISTANCE", "-1.5"))
    pdf_trend_min_consecutive: int = int(os.getenv("TS_PDF_TREND_MIN_CONSECUTIVE", "-1"))
    pdf_daily_max_loss_usdt: float = float(os.getenv("TS_PDF_DAILY_MAX_LOSS_USDT", "100"))
    pdf_consecutive_loss_scale_1: float = float(os.getenv("TS_PDF_CONSECUTIVE_LOSS_SCALE_1", "0.5"))
    pdf_consecutive_loss_scale_2: float = float(os.getenv("TS_PDF_CONSECUTIVE_LOSS_SCALE_2", "0.25"))
    consecutive_stop_loss_half_size_after: int = int(os.getenv("TS_CONSECUTIVE_STOP_LOSS_HALF_SIZE_AFTER", "3"))
    consecutive_stop_loss_half_size_scale: float = float(os.getenv("TS_CONSECUTIVE_STOP_LOSS_HALF_SIZE_SCALE", "0.5"))
    # 小数比例单位, 与 kline_atr_pct 对齐 (2026-05-30 修单位bug: 原3.0误把百分数当小数, 致缩放永久卡顶)
    pdf_atr_base_pct: float = float(os.getenv("TS_PDF_ATR_BASE_PCT", "0.08"))
    pdf_atr_max_scale: float = float(os.getenv("TS_PDF_ATR_MAX_SCALE", "1.3"))
    pdf_atr_min_scale: float = float(os.getenv("TS_PDF_ATR_MIN_SCALE", "0.4"))
    pdf_close_confirmation_enabled: bool = os.getenv("TS_PDF_CLOSE_CONFIRMATION_ENABLED", "true").lower() == "true"
    pdf_close_confirmation_min: float = float(os.getenv("TS_PDF_CLOSE_CONFIRMATION_MIN", "0.6"))
    pdf_pyramid_enabled: bool = os.getenv("TS_PDF_PYRAMID_ENABLED", "true").lower() == "true"
    pdf_pyramid_min_pnl_pct: float = float(os.getenv("TS_PDF_PYRAMID_MIN_PNL_PCT", "0.02"))
    pdf_pyramid_max_count: int = int(os.getenv("TS_PDF_PYRAMID_MAX_COUNT", "1"))
    pdf_pyramid_size_fraction: float = float(os.getenv("TS_PDF_PYRAMID_SIZE_FRACTION", "0.5"))
    pdf_pyramid_min_score: float = float(os.getenv("TS_PDF_PYRAMID_MIN_SCORE", "65"))
    pdf_max_same_setup_positions: int = int(os.getenv("TS_PDF_MAX_SAME_SETUP_POSITIONS", "3"))
    pdf_max_correlated_positions: int = int(os.getenv("TS_PDF_MAX_CORRELATED_POSITIONS", "2"))
    pdf_atr_stop_multiplier: float = float(os.getenv("TS_PDF_ATR_STOP_MULTIPLIER", "1.5"))
    pdf_trailing_stop_enabled: bool = os.getenv("TS_PDF_TRAILING_STOP_ENABLED", "true").lower() == "true"
    trend_runner_score_fade_hold_enabled: bool = os.getenv("TS_TREND_RUNNER_SCORE_FADE_HOLD_ENABLED", "true").lower() == "true"
    trend_runner_min_score_floor: float = float(os.getenv("TS_TREND_RUNNER_MIN_SCORE_FLOOR", "25"))
    trend_runner_min_price_change_pct_24h: float = float(os.getenv("TS_TREND_RUNNER_MIN_PRICE_CHANGE_PCT_24H", "20"))
    trend_runner_min_price_change_pct_4h: float = float(os.getenv("TS_TREND_RUNNER_MIN_PRICE_CHANGE_PCT_4H", "4"))
    trend_runner_min_oi_change_pct_4h: float = float(os.getenv("TS_TREND_RUNNER_MIN_OI_CHANGE_PCT_4H", "10"))
    trend_runner_min_taker_ratio: float = float(os.getenv("TS_TREND_RUNNER_MIN_TAKER_RATIO", "0.85"))
    trend_runner_min_sentiment_score: float = float(os.getenv("TS_TREND_RUNNER_MIN_SENTIMENT_SCORE", "40"))
    trend_runner_min_quote_volume_usdt: float = float(os.getenv("TS_TREND_RUNNER_MIN_QUOTE_VOLUME_USDT", "20000000"))
    trend_runner_max_breakout_distance: float = float(os.getenv("TS_TREND_RUNNER_MAX_BREAKOUT_DISTANCE", "0.50"))
    trend_runner_max_adverse_pnl_pct: float = float(os.getenv("TS_TREND_RUNNER_MAX_ADVERSE_PNL_PCT", "-0.01"))
    trend_runner_min_momo_15m_bars: int = int(os.getenv("TS_TREND_RUNNER_MIN_MOMO_15M_BARS", "300"))
    trend_runner_min_bb672_15m_bars: int = int(os.getenv("TS_TREND_RUNNER_MIN_BB672_15M_BARS", "672"))
    trend_runner_bb210_min_position: float = float(os.getenv("TS_TREND_RUNNER_BB210_MIN_POSITION", "0.20"))
    trend_runner_bb672_15m_min_position: float = float(os.getenv("TS_TREND_RUNNER_BB672_15M_MIN_POSITION", "0.20"))

    # Behavioral Trap Detection
    behavioral_sunk_cost_window_minutes: int = int(os.getenv("TS_BEHAVIORAL_SUNK_COST_WINDOW_MINUTES", "120"))
    behavioral_tilt_daily_loss_pct: float = float(os.getenv("TS_BEHAVIORAL_TILT_DAILY_LOSS_PCT", "0.5"))

    # BB Structure (Phase 1: data collection)
    bb_structure_enabled: bool = os.getenv("TS_BB_STRUCTURE_ENABLED", "true").lower() == "true"
    bb_touch_threshold_pct: float = float(os.getenv("TS_BB_TOUCH_THRESHOLD_PCT", "0.3"))
    momo_structure_enabled: bool = os.getenv("TS_MOMO_STRUCTURE_ENABLED", "true").lower() == "true"
    momo_structure_top_n: int = int(os.getenv("TS_MOMO_STRUCTURE_TOP_N", "30"))
    momo_structure_backfill_symbols_per_run: int = int(os.getenv("TS_MOMO_STRUCTURE_BACKFILL_SYMBOLS_PER_RUN", "3"))
    momo_structure_cache_enabled: bool = os.getenv("TS_MOMO_STRUCTURE_CACHE_ENABLED", "true").lower() == "true"
    momo_structure_cache_dir: Path = Path(
        os.getenv("TS_MOMO_STRUCTURE_CACHE_DIR", "data/trading_system/cache/binance_klines")
    )
    momo_15m_kline_limit: int = int(os.getenv("TS_MOMO_15M_KLINE_LIMIT", "4707"))
    momo_1m_kline_limit: int = int(os.getenv("TS_MOMO_1M_KLINE_LIMIT", "4707"))
    momo_4h_kline_limit: int = int(os.getenv("TS_MOMO_4H_KLINE_LIMIT", "4707"))
    momo_daily_kline_limit: int = int(os.getenv("TS_MOMO_DAILY_KLINE_LIMIT", "4707"))
    momo_1m_refresh_seconds: int = int(os.getenv("TS_MOMO_1M_REFRESH_SECONDS", "120"))
    momo_15m_refresh_seconds: int = int(os.getenv("TS_MOMO_15M_REFRESH_SECONDS", "300"))
    momo_4h_refresh_seconds: int = int(os.getenv("TS_MOMO_4H_REFRESH_SECONDS", "1800"))
    momo_daily_refresh_seconds: int = int(os.getenv("TS_MOMO_DAILY_REFRESH_SECONDS", "21600"))

    # BB Touch Count (Phase 2: entry filter)
    bb_touch_skip_enabled: bool = os.getenv("TS_BB_TOUCH_SKIP_ENABLED", "false").lower() == "true"
    bb_touch_2nd_position_scale: float = float(os.getenv("TS_BB_TOUCH_2ND_POSITION_SCALE", "0.5"))
    bb_touch_max_count: int = int(os.getenv("TS_BB_TOUCH_MAX_COUNT", "3"))
    bb_touch_reset_distance_pct: float = float(os.getenv("TS_BB_TOUCH_RESET_DISTANCE_PCT", "2.0"))
    bb_touch_reset_hours: int = int(os.getenv("TS_BB_TOUCH_RESET_HOURS", "24"))
    bb_touch_min_gap_minutes: int = int(os.getenv("TS_BB_TOUCH_MIN_GAP_MINUTES", "30"))

    # Consecutive Entry Limit (Phase 3)
    consecutive_entry_skip_enabled: bool = os.getenv("TS_CONSECUTIVE_ENTRY_SKIP_ENABLED", "false").lower() == "true"
    consecutive_entry_max_count: int = int(os.getenv("TS_CONSECUTIVE_ENTRY_MAX_COUNT", "3"))
    consecutive_entry_reset_hours: int = int(os.getenv("TS_CONSECUTIVE_ENTRY_RESET_HOURS", "4"))
