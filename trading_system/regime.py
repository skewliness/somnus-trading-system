from __future__ import annotations

from statistics import mean, median

from trading_system.config import StrategyConfig
from trading_system.models import MarketSnapshot


def detect_market_regime(snapshots: list[MarketSnapshot], config: StrategyConfig) -> dict:
    if not snapshots:
        return {
            "regime": "unknown",
            "allow_new_entries": True,
            "entry_threshold_multiplier": 1.0,
            "stop_loss_multiplier": 1.0,
            "take_profit_multiplier": 1.0,
            "flash_crash_count": 0,
        }

    flash_crash_count = sum(1 for item in snapshots if item.price_change_pct_5m <= -abs(config.market_flash_crash_drop_pct_5m))
    avg_taker = mean(item.taker_buy_sell_ratio for item in snapshots)
    median_oi = median(item.oi_change_pct_4h for item in snapshots)
    avg_24h = mean(item.price_change_pct_24h for item in snapshots)

    flash_crash_trigger_count = min(max(2, len(snapshots) // 2 + len(snapshots) % 2), max(2, config.market_flash_crash_symbol_count))
    if flash_crash_count >= flash_crash_trigger_count and avg_taker <= 0.95:
        regime = "flash_crash"
        return {
            "regime": regime,
            "allow_new_entries": False,
            "entry_threshold_multiplier": 1.25,
            "stop_loss_multiplier": 0.85,
            "take_profit_multiplier": 0.9,
            "flash_crash_count": flash_crash_count,
            "avg_taker_ratio": round(avg_taker, 4),
            "median_oi_change_pct_4h": round(median_oi, 4),
            "avg_price_change_pct_24h": round(avg_24h, 4),
        }
    if avg_24h <= -6.0 or avg_taker <= 0.98:
        regime = "risk_off"
        return {
            "regime": regime,
            "allow_new_entries": True,
            "entry_threshold_multiplier": 1.1,
            "stop_loss_multiplier": 0.95,
            "take_profit_multiplier": 0.95,
            "flash_crash_count": flash_crash_count,
            "avg_taker_ratio": round(avg_taker, 4),
            "median_oi_change_pct_4h": round(median_oi, 4),
            "avg_price_change_pct_24h": round(avg_24h, 4),
        }
    if avg_24h >= 8.0 and avg_taker >= 1.05 and median_oi >= 5.0:
        regime = "trend_up"
        return {
            "regime": regime,
            "allow_new_entries": True,
            "entry_threshold_multiplier": 0.96,
            "stop_loss_multiplier": 1.05,
            "take_profit_multiplier": 1.1,
            "flash_crash_count": flash_crash_count,
            "avg_taker_ratio": round(avg_taker, 4),
            "median_oi_change_pct_4h": round(median_oi, 4),
            "avg_price_change_pct_24h": round(avg_24h, 4),
        }
    return {
        "regime": "range",
        "allow_new_entries": True,
        "entry_threshold_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 1.0,
        "flash_crash_count": flash_crash_count,
        "avg_taker_ratio": round(avg_taker, 4),
        "median_oi_change_pct_4h": round(median_oi, 4),
        "avg_price_change_pct_24h": round(avg_24h, 4),
    }
