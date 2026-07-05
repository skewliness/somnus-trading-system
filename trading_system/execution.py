from __future__ import annotations

import logging
import os
import pickle
import re
from collections import defaultdict
from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_system.config import StrategyConfig
from trading_system.models import MarketSnapshot, Position, TradePlan
from trading_system.persistence import atomic_write_state, load_state
from trading_system.smart_money_panel import get_smart_money_panel_payload
from trading_system.panel_blend import blend_panel_score
from trading_system.coin_scout_model import CoinFeatures, FeatureEngine

# Shadow 模型路径（只打分，不拦截）
_SHADOW_MODEL_PATHS = {
    "mixed_undersampled": Path(os.getenv(
        "TS_SHADOW_MODEL_MIXED_UNDERSAMPLED",
        "data/trading_system/model_training/ablation_v4/mixed_undersampled/coin_scout_model.pkl",
    )),
    "decision_class_weighted": Path(os.getenv(
        "TS_SHADOW_MODEL_DECISION_CLASS_WEIGHTED",
        "data/trading_system/model_training/ablation_v4/decision_class_weighted/coin_scout_model.pkl",
    )),
}

# CoinFeatures 字段名列表（用于从 dict 重建特征向量）
_COIN_FEATURE_NAMES = [f.name for f in fields(CoinFeatures) if f.name not in ("symbol", "timestamp")]


def _shadow_feature_names_for_model(model) -> list[str]:
    expected = getattr(model, "n_features_in_", None)
    if isinstance(expected, int) and expected > 0:
        return _COIN_FEATURE_NAMES[:expected]
    return _COIN_FEATURE_NAMES


def _shadow_feature_row(coin_features_dict: dict, model) -> list[float]:
    names = _shadow_feature_names_for_model(model)
    return [float(coin_features_dict.get(name, 0)) for name in names]


ENTRY_TRAINING_CONTEXT_FIELDS = (
    "daily_price_vs_ma7_pct",
    "daily_price_vs_ma20_pct",
    "daily_price_position_365",
    "bb_price_position_96",
    "bb_price_position_210",
    "bb_price_position_672_15m",
    "bb_price_position_1440_15m",
    "bb_price_position_1920_15m",
    "bb_price_position_2880_15m",
    "bb_price_position_4404_15m",
    "bb_price_position_4460_15m",
    "bb_price_position_105_4h",
    "bb_price_position_672_4h",
    "bb_price_position_1440_4h",
    "bb_price_position_105_1m",
    "bb_price_position_300_1m",
    "williams_r_15m",
    "williams_r_15m_120",
    "williams_r_1m",
    "bb_touch_zone",
)

ENTRY_BB_HEAT_FIELDS = (
    "bb_price_position_96",
    "bb_price_position_210",
    "bb_price_position_672_15m",
    "bb_price_position_105_4h",
)

ENTRY_BB_EXTENSION_FIELDS = (
    "bb_price_position_96",
    "bb_price_position_210",
    "bb_price_position_672_15m",
    "bb_price_position_105_4h",
    "bb_price_position_672_4h",
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _distance_pct(price: object, reference: object) -> float:
    ref = _safe_float(reference)
    if ref <= 0:
        return 0.0
    return (_safe_float(price) - ref) / ref * 100.0


def _model_position(value: object, default: float = 0.5) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _model_distance_pct(price: object, reference: object) -> float:
    # Raw percentage distances can be hundreds of percent for long-period BBs.
    # Keep the feature slot stable but neutral for model training.
    return 0.0


logger = logging.getLogger(__name__)

BREAKOUT_SENTIMENT_SOURCES = (
    "manual_distill",
    "manual_sentiment",
    "crypto-market-rank",
    "meme-rush",
    "trading-signal",
    "lanaaielsa-distill",
    "lanaai-square-distill",
    "gmgn-wallet-distill",
    "square_feed",
    "opentwitter_search",
    "opennews_market",
)
WEAK_DISCOVERY_SOURCES = {"binance_top_movers", "coingecko_trending"}
NON_PROGRESS_EXIT_REASONS = {"score_faded", "market_snapshot_missing_timeout", "max_hold_reached"}
# 所有止损类离场原因, 统一用于连续止损/冷却/缩仓统计 (2026-05-30)
STOP_LOSS_EXIT_REASONS = {"stop_loss_hit", "structural_stop_hit", "disaster_hard_stop"}
BREAKOUT_RUNNER_STAGE = 2
ALLOWED_ENTRY_SETUP_TYPES = {
    "oi_divergence_long_candidate",
    "base_breakout_confirmed",
    "base_breakout_weak",
    "mania_followthrough",
    "trend_followthrough",
    "squeeze_short_candidate",
    "base_breakdown_confirmed",
}


class PaperExecutionEngine:
    def __init__(self, config: StrategyConfig, smart_money_panel_provider=None, mark_price_fetcher=None) -> None:
        self.config = config
        self._last_regime: dict | None = None
        self.smart_money_panel_provider = smart_money_panel_provider or get_smart_money_panel_payload
        self._mark_price_fetcher = mark_price_fetcher
        self.state_file = config.state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        # 加载 shadow 模型（只打分，不拦截交易）
        self._shadow_models = {}
        self._shadow_model_meta = {}
        import hashlib, json as _json
        for name, path in _SHADOW_MODEL_PATHS.items():
            if not path.exists():
                logger.info("Shadow model not found: %s (%s)", name, path)
                self._shadow_model_meta[name] = {"status": "not_found", "model_path": str(path)}
                continue
            try:
                with open(path, "rb") as f:
                    content = f.read()
                model = pickle.loads(content)
                # 兼容直接 GBC 或 dict 包装
                if isinstance(model, dict):
                    model = model.get("model", model)
                self._shadow_models[name] = model
                # 收集版本元数据
                stat = path.stat()
                md5 = hashlib.md5(content).hexdigest()[:12]
                meta = {
                    "status": "loaded",
                    "model_path": str(path),
                    "file_size_bytes": stat.st_size,
                    "file_mtime": stat.st_mtime,
                    "file_md5_prefix": md5,
                    "feature_schema_version": "v1_65dim",
                    "feature_names_count": len(_shadow_feature_names_for_model(model)),
                    "available_feature_names_count": len(_COIN_FEATURE_NAMES),
                    "model_expected_features": getattr(model, "n_features_in_", None),
                }
                # 尝试读同目录 summary
                summary_path = path.parent / "ablation_result.json"
                if summary_path.exists():
                    try:
                        summary = _json.loads(summary_path.read_text())
                        meta["trained_at"] = summary.get("trained_at", summary.get("timestamp", ""))
                        meta["train_samples"] = summary.get("train_samples", 0)
                        meta["val_report"] = {
                            "accuracy": summary.get("val_report", {}).get("accuracy"),
                            "macro_f1": summary.get("val_report", {}).get("macro avg", {}).get("f1-score"),
                        }
                    except Exception:
                        pass
                self._shadow_model_meta[name] = meta
                logger.info("Shadow model loaded: %s (md5=%s, size=%d)", name, md5, stat.st_size)
            except Exception as e:
                logger.warning("Failed to load shadow model %s: %s", name, e)
                self._shadow_model_meta[name] = {"status": "load_error", "model_path": str(path), "error": str(e)}

    def rebalance(self, snapshots: list[MarketSnapshot], regime: dict | None = None, hf_quality_lookup: dict | None = None, predictive_model: dict | None = None) -> dict:
        self._last_regime = regime
        self._hf_quality_lookup = hf_quality_lookup or {}
        self._predictive_model = predictive_model
        state = self._load_state()
        snapshot_map = {snapshot.symbol: snapshot for snapshot in snapshots}
        market_guard = self._assess_market_guard(snapshots)
        bb_touch_counts = dict(state.get("bb_touch_counts") or {})
        consecutive_entries = dict(state.get("consecutive_entries") or {})
        positions = [position for item in state.get("positions", []) if (position := self._position_from_dict(item)) is not None]
        journal = list(state.get("journal", []))
        initial_journal_len = len(journal)
        cycle_realized_pnl_usdt = 0.0
        raw_learning_journal = state.get("learning_journal")
        learning_journal = list(raw_learning_journal) if raw_learning_journal is not None else list(journal)
        symbol_learning, setup_learning, setup_type_learning = self._build_learning_index(learning_journal)
        recent_symbol_loss_streaks, recent_setup_loss_streaks = self._build_recent_loss_streak_index(learning_journal)
        stop_loss_symbols, stop_loss_setups = self._build_stop_loss_index(learning_journal)
        cons_sl_symbols, cons_sl_setups = self._build_consecutive_stop_loss_index(learning_journal)
        flat_score_fade_symbols, flat_score_fade_setups = self._build_flat_score_fade_index(learning_journal)
        non_progress_symbols, non_progress_setups = self._build_non_progress_index(learning_journal)
        recent_exit_index = self._build_recent_exit_index(learning_journal)
        positions = [position for position in (self._sanitize_position(position, snapshot_map, journal) for position in positions) if position is not None]
        # 第2段打分: 对持仓+技术分top候选调面板, 按置信度浮动权重融进 snapshot.score。
        # snapshot_map / ranked 共享同一批 snapshot 对象, 覆写一次 -> 开仓选币与持仓退出两线同时生效。
        self._apply_panel_blend(snapshots, snapshot_map, positions)
        next_positions: list[Position] = []
        exits: list[TradePlan] = []

        for position in positions:
            current = snapshot_map.get(position.symbol)
            if market_guard["triggered"] and current is not None:
                exits.append(
                    TradePlan(
                        action="exit",
                        symbol=position.symbol,
                        score=current.score,
                        entry_price=position.entry_price,
                        stop_loss_price=position.stop_loss_price,
                        take_profit_price=position.take_profit_price,
                        notional_usdt=position.notional_usdt,
                        rationale="market_flash_crash_protection",
                    )
                )
                journal.append(
                    {
                        "timestamp": _utc_now().isoformat(),
                        "action": "exit",
                        "symbol": position.symbol,
                        "reason": "market_flash_crash_protection",
                        "entry_price": position.entry_price,
                        "exit_price": current.mark_price,
                        "notional_usdt": position.notional_usdt,
                        "size_fraction": 1.0,
                        "pnl_pct": round((current.mark_price - position.entry_price) / max(position.entry_price, 1e-9), 6),
                        "setup_type": position.setup_type,
                        # --- 入场时特征快照 ---
                        "entry_timestamp": position.opened_at,
                        "sentiment_score": position.last_sentiment_score,
                        "sentiment_mentions": position.last_sentiment_mentions,
                        "oi_change_pct_4h": position.last_oi_change_pct_4h,
                        "taker_buy_sell_ratio": position.last_taker_buy_sell_ratio,
                        "entry_score": position.score,
                        "entry_mark_price": position.current_price,
                        "coin_features": position.entry_coin_features,
                        "kline_pattern_bullish_count": position.entry_kline_pattern_bullish_count,
                        "kline_pattern_bearish_count": position.entry_kline_pattern_bearish_count,
                        "kline_pattern_flags": position.entry_kline_pattern_flags,
                        "entry_h4_structure": position.entry_h4_structure,
                        "initial_risk_pct": position.initial_risk_pct,
                    }
                )
                cycle_realized_pnl_usdt += position.notional_usdt * (
                    (current.mark_price - position.entry_price) / max(position.entry_price, 1e-9)
                )
                continue
            if current is None:
                stale_exit = self._evaluate_missing_snapshot_position(position)
                if stale_exit is not None:
                    pnl_pct = round((position.current_price - position.entry_price) / position.entry_price, 6)
                    exits.append(stale_exit)
                    journal.append(
                        {
                            "timestamp": _utc_now().isoformat(),
                            "action": stale_exit.action,
                            "symbol": position.symbol,
                            "reason": stale_exit.rationale,
                            "entry_price": position.entry_price,
                            "exit_price": position.current_price,
                            "notional_usdt": stale_exit.notional_usdt,
                            "size_fraction": stale_exit.size_fraction,
                            "pnl_pct": pnl_pct,
                            "setup_type": position.setup_type,
                            # --- 入场时特征快照 ---
                            "entry_timestamp": position.opened_at,
                            "sentiment_score": position.last_sentiment_score,
                            "sentiment_mentions": position.last_sentiment_mentions,
                            "oi_change_pct_4h": position.last_oi_change_pct_4h,
                            "taker_buy_sell_ratio": position.last_taker_buy_sell_ratio,
                            "entry_score": position.score,
                            "entry_mark_price": position.current_price,
                            "coin_features": position.entry_coin_features,
                            "kline_pattern_bullish_count": position.entry_kline_pattern_bullish_count,
                            "kline_pattern_bearish_count": position.entry_kline_pattern_bearish_count,
                            "kline_pattern_flags": position.entry_kline_pattern_flags,
                            "initial_risk_pct": position.initial_risk_pct,
                        }
                    )
                    cycle_realized_pnl_usdt += stale_exit.notional_usdt * stale_exit.size_fraction * pnl_pct
                    continue
                next_positions.append(position)
                continue

            position.current_price = current.mark_price
            if self._is_long_position(position):
                current_pnl = (current.mark_price - position.entry_price) / max(position.entry_price, 1e-9)
            else:
                current_pnl = (position.entry_price - current.mark_price) / max(position.entry_price, 1e-9)
            if current_pnl > position.high_watermark_pnl_pct:
                position.high_watermark_pnl_pct = current_pnl
            exit_plan, updated_position = self._evaluate_position(position, current)
            if exit_plan:
                pnl_pct = round((current.mark_price - position.entry_price) / position.entry_price, 6)
                exits.append(exit_plan)
                journal.append(
                    {
                        "timestamp": _utc_now().isoformat(),
                        "action": exit_plan.action,
                        "symbol": position.symbol,
                        "reason": exit_plan.rationale,
                        "entry_price": position.entry_price,
                        "exit_price": current.mark_price,
                        "notional_usdt": exit_plan.notional_usdt,
                        "size_fraction": exit_plan.size_fraction,
                        "pnl_pct": pnl_pct,
                        "setup_type": position.setup_type,
                        # --- 入场时特征快照（用于模型训练） ---
                        "entry_timestamp": position.opened_at,
                        "sentiment_score": position.last_sentiment_score,
                        "sentiment_mentions": position.last_sentiment_mentions,
                        "oi_change_pct_4h": position.last_oi_change_pct_4h,
                        "taker_buy_sell_ratio": position.last_taker_buy_sell_ratio,
                        "entry_score": position.score,
                        "entry_mark_price": position.current_price,
                        "coin_features": position.entry_coin_features,
                        "kline_pattern_bullish_count": position.entry_kline_pattern_bullish_count,
                        "kline_pattern_bearish_count": position.entry_kline_pattern_bearish_count,
                        "kline_pattern_flags": position.entry_kline_pattern_flags,
                        "entry_h4_structure": position.entry_h4_structure,
                        "initial_risk_pct": position.initial_risk_pct,
                        "predictive_snapshot": position.predictive_snapshot,
                    }
                )
                cycle_realized_pnl_usdt += exit_plan.notional_usdt * exit_plan.size_fraction * pnl_pct
            if updated_position is not None:
                next_positions.append(updated_position)

        if len(journal) > initial_journal_len:
            current_cycle_learning_journal = learning_journal + journal[initial_journal_len:]
            symbol_learning, setup_learning, setup_type_learning = self._build_learning_index(current_cycle_learning_journal)
            recent_symbol_loss_streaks, recent_setup_loss_streaks = self._build_recent_loss_streak_index(current_cycle_learning_journal)
            stop_loss_symbols, stop_loss_setups = self._build_stop_loss_index(current_cycle_learning_journal)
            cons_sl_symbols, cons_sl_setups = self._build_consecutive_stop_loss_index(current_cycle_learning_journal)
            flat_score_fade_symbols, flat_score_fade_setups = self._build_flat_score_fade_index(current_cycle_learning_journal)
            non_progress_symbols, non_progress_setups = self._build_non_progress_index(current_cycle_learning_journal)
            recent_exit_index = self._build_recent_exit_index(current_cycle_learning_journal)

        held_symbols = {position.symbol for position in next_positions}
        ranked = sorted((item for item in snapshots if item.symbol not in held_symbols), key=lambda item: item.score, reverse=True)
        entries: list[TradePlan] = []

        loss_limit = abs(float(self.config.live_max_request_loss_usdt or 0.0))
        cycle_loss_fuse_triggered = bool(loss_limit > 0 and cycle_realized_pnl_usdt <= -loss_limit)
        if cycle_loss_fuse_triggered:
            logger.warning(
                "paper cycle loss fuse triggered: pnl_usdt=%s limit_usdt=%s",
                round(cycle_realized_pnl_usdt, 6),
                loss_limit,
            )

        daily_pnl = self._daily_loss_usdt(journal)
        daily_loss_fuse = daily_pnl <= -self.config.pdf_daily_max_loss_usdt
        if daily_loss_fuse:
            logger.warning("pdf daily loss fuse triggered: daily_pnl=%s limit=%s", round(daily_pnl, 2), self.config.pdf_daily_max_loss_usdt)

        if not market_guard["triggered"] and not cycle_loss_fuse_triggered and not daily_loss_fuse:
            entry_mult = float(regime.get("entry_threshold_multiplier", 1.0)) if regime else 1.0
            effective_entry_threshold = self.config.entry_threshold * entry_mult
            for snapshot in ranked:
                if len(next_positions) >= self.config.max_positions:
                    break
                effective_score = snapshot.score
                if (
                    self.config.h4_structure_filter_enabled
                    and snapshot.h4_structure_state == "weakening"
                    and snapshot.setup_type not in self._SHORT_SETUP_TYPES
                ):
                    effective_score -= self.config.h4_weakening_score_penalty
                if (
                    effective_score < effective_entry_threshold
                    and not self._has_first_signal_probe(snapshot)
                    and not self._has_repeated_signal_4h_support(snapshot)
                    and not self._has_retest_add_support(snapshot)
                ):
                    continue
                if self._should_skip_entry(
                    snapshot,
                    symbol_learning,
                    setup_learning,
                    setup_type_learning,
                    recent_symbol_loss_streaks,
                    recent_setup_loss_streaks,
                    stop_loss_symbols,
                    stop_loss_setups,
                    cons_sl_symbols,
                    cons_sl_setups,
                    flat_score_fade_symbols,
                    flat_score_fade_setups,
                    non_progress_symbols,
                    non_progress_setups,
                    recent_exit_index,
                    bb_touch_counts=bb_touch_counts,
                    consecutive_entries=consecutive_entries,
                ):
                    continue
                if self._exceeds_correlation_limit(snapshot, next_positions):
                    continue
                behavioral_traps = self._detect_behavioral_traps(snapshot, recent_exit_index, daily_pnl)
                if behavioral_traps:
                    logger.info("behavioral trap blocked %s: %s", snapshot.symbol, behavioral_traps)
                    continue
                bb_tc = self._get_bb_touch_count(bb_touch_counts, snapshot)
                plan = self._build_entry(snapshot, cons_sl_count=cons_sl_symbols.get(snapshot.symbol.upper(), 0), bb_touch_count=bb_tc)
                entries.append(plan)

                # 构建完整 CoinFeatures 用于模型回测
                coin_features_dict = PaperExecutionEngine._entry_coin_features_dict(snapshot)

                # Shadow 模型打分（只记录，不拦截）
                shadow_scores = self._score_with_shadow_models(coin_features_dict)
                # 标记 execution 层用的是轻量版特征
                if shadow_scores.get("model_meta"):
                    for name in shadow_scores["model_meta"]:
                        shadow_scores["model_meta"][name]["feature_source"] = "lightweight_execution"

                next_positions.append(
                    Position(
                        symbol=snapshot.symbol,
                        opened_at=_utc_now().isoformat(),
                        entry_price=snapshot.mark_price,
                        current_price=snapshot.mark_price,
                        stop_loss_price=plan.stop_loss_price,
                        take_profit_price=plan.take_profit_price,
                        notional_usdt=plan.notional_usdt,
                        score=snapshot.score,
                        rationale=plan.rationale,
                        setup_type=plan.setup_type,
                        initial_risk_pct=max(0.0, round((snapshot.mark_price - plan.stop_loss_price) / max(snapshot.mark_price, 1e-9), 8)),
                        structure_timeframe=snapshot.structure_timeframe or "daily",
                        last_sentiment_score=snapshot.sentiment_score,
                        last_sentiment_mentions=snapshot.sentiment_mentions,
                        last_oi_change_pct_4h=snapshot.oi_change_pct_4h,
                        last_taker_buy_sell_ratio=snapshot.taker_buy_sell_ratio,
                        entry_coin_features=coin_features_dict,
                        entry_kline_pattern_bullish_count=snapshot.kline_pattern_bullish_count,
                        entry_kline_pattern_bearish_count=snapshot.kline_pattern_bearish_count,
                        entry_kline_pattern_flags=snapshot.kline_pattern_flags,
                        entry_h4_structure={
                            "h4_close_above_ma20": snapshot.h4_close_above_ma20,
                            "h4_ma20_slope_pct": snapshot.h4_ma20_slope_pct,
                            "h4_price_vs_ma20_pct": snapshot.h4_price_vs_ma20_pct,
                            "h4_close_position_20bars": snapshot.h4_close_position_20bars,
                            "h4_low_break_recent_20bars": snapshot.h4_low_break_recent_20bars,
                            "h4_high_break_recent_20bars": snapshot.h4_high_break_recent_20bars,
                            "h4_atr_compression": snapshot.h4_atr_compression,
                            "h4_lower_highs_count": snapshot.h4_lower_highs_count,
                            "h4_higher_lows_count": snapshot.h4_higher_lows_count,
                            "h4_structure_state": snapshot.h4_structure_state,
                            "h4_price_location": snapshot.h4_price_location,
                            "h4_pattern_type": snapshot.h4_pattern_type,
                            # 入场即锚定两条止损线 (2026-05-30)
                            "structural_stop_price": self._structural_stop_price(snapshot),
                            "hard_stop_price": self._disaster_hard_stop_price(snapshot.mark_price, snapshot),
                        },
                        predictive_snapshot=self._build_predictive_snapshot(snapshot.symbol, snapshot.reasons),
                    )
                )
                journal.append(
                    {
                        "timestamp": _utc_now().isoformat(),
                        "action": plan.action,
                        "symbol": snapshot.symbol,
                        "score": snapshot.score,
                        "entry_price": snapshot.mark_price,
                        "notional_usdt": plan.notional_usdt,
                        "reason": plan.rationale,
                        "setup_type": plan.setup_type,
                        "sentiment_mentions": plan.sentiment_mentions,
                        "sentiment_score": plan.sentiment_score,
                        "oi_change_pct_4h": plan.oi_change_pct_4h,
                        "taker_buy_sell_ratio": plan.taker_buy_sell_ratio,
                        "regime": regime.get("regime") if regime else None,
                        "scoring_reasons": snapshot.reasons,
                        "coin_features": coin_features_dict,
                        "kline_pattern_bullish_count": snapshot.kline_pattern_bullish_count,
                        "kline_pattern_bearish_count": snapshot.kline_pattern_bearish_count,
                        "kline_pattern_flags": snapshot.kline_pattern_flags,
                        "shadow_scores": shadow_scores,
                        "hf_quality": self._directional_hf_quality(snapshot),
                        "predictive_snapshot": self._build_predictive_snapshot(snapshot.symbol, snapshot.reasons),
                    }
                )
                self._update_consecutive_entries(consecutive_entries, snapshot.symbol)

        self._update_bb_touch_counts(bb_touch_counts, snapshots)

        shadow_paper = self._rebalance_shadow_paper(
            state=state,
            snapshots=snapshots,
            market_guard=market_guard,
            excluded_symbols={position.symbol for position in next_positions},
        )

        learning_payload = (learning_journal + journal[initial_journal_len:])[-2000:]
        payload = {
            "schema_version": self.config.state_schema_version,
            "positions": [position.to_dict() for position in next_positions],
            "journal": journal[-200:],
            "learning_journal": learning_payload,
            "shadow_positions": shadow_paper.get("positions", []),
            "shadow_journal": shadow_paper.get("journal", []),
            "bb_touch_counts": bb_touch_counts,
            "consecutive_entries": consecutive_entries,
        }
        atomic_write_state(self.state_file, payload)
        return {
            "entries": [plan.to_dict() for plan in entries],
            "exits": [plan.to_dict() for plan in exits],
            "positions": [position.to_dict() for position in next_positions],
            "shadow_paper": {
                key: value
                for key, value in shadow_paper.items()
                if key != "journal"
            },
            "risk_controls": {
                "cycle_realized_pnl_usdt": round(cycle_realized_pnl_usdt, 6),
                "cycle_loss_limit_usdt": loss_limit,
                "loss_fuse_triggered": cycle_loss_fuse_triggered,
                "state_schema_version": self.config.state_schema_version,
            },
        }

    def _score_with_shadow_models(self, coin_features_dict: dict) -> dict:
        """
        用 shadow 模型对入场特征打分。
        返回 {
          "scores": {model_name: {label, prob_0, prob_1, prob_2, prob_moon}},
          "errors": {model_name: error_msg},
          "model_meta": {model_name: {md5_prefix, feature_schema_version, ...}},
        }
        """
        import numpy as np
        result = {"scores": {}, "errors": {}, "model_meta": {}}
        # 先把所有模型的元数据写进去（无论成功与否）
        for name, meta in self._shadow_model_meta.items():
            m = {k: v for k, v in meta.items() if k != "val_report"}
            if "val_report" in meta:
                m["val_accuracy"] = meta["val_report"].get("accuracy")
                m["val_macro_f1"] = meta["val_report"].get("macro_f1")
            result["model_meta"][name] = m
        if not self._shadow_models:
            # 记录为什么没有模型
            for name, meta in self._shadow_model_meta.items():
                if meta.get("status") != "loaded":
                    result["errors"][name] = meta.get("status", "unknown")
            return result
        if not coin_features_dict:
            for name in self._shadow_models:
                result["errors"][name] = "empty_features"
            return result
        try:
            for name, model in self._shadow_models.items():
                try:
                    X = np.array([_shadow_feature_row(coin_features_dict, model)])
                    pred = int(model.predict(X)[0])
                    probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [0, 0, 0]
                    prob_0 = float(probs[0]) if len(probs) > 0 else 0.0
                    prob_1 = float(probs[1]) if len(probs) > 1 else 0.0
                    prob_2 = float(probs[2]) if len(probs) > 2 else 0.0
                    result["scores"][name] = {
                        "model_label": pred,
                        "prob_0": round(prob_0, 4),
                        "prob_1": round(prob_1, 4),
                        "prob_2": round(prob_2, 4),
                        "prob_moon": round(prob_1 + prob_2, 4),
                    }
                except Exception as e:
                    err_msg = f"{type(e).__name__}: {e}"
                    result["errors"][name] = err_msg
                    logger.debug("Shadow model %s scoring failed: %s", name, e)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            for name in self._shadow_models:
                if name not in result["scores"]:
                    result["errors"][name] = err_msg
            logger.debug("Shadow scoring failed: %s", e)
        return result

    @staticmethod
    def _entry_training_context(snapshot: "MarketSnapshot") -> dict:
        context = {
            field: getattr(snapshot, field, 0.0)
            for field in ENTRY_TRAINING_CONTEXT_FIELDS
        }
        context["bb_upper_heat_score"] = float(
            sum(1 for field in ENTRY_BB_HEAT_FIELDS if _safe_float(context.get(field), 0.5) >= 0.85)
        )
        context["bb_extension_max"] = max(
            _safe_float(context.get(field), 0.5) for field in ENTRY_BB_EXTENSION_FIELDS
        )
        context["atr_regime_high"] = 1.0 if _safe_float(snapshot.kline_atr_pct) >= 0.10 else 0.0
        context["false_breakout_4h"] = (
            1.0
            if _safe_float(snapshot.bb_price_position_105_4h, 0.5) >= 0.70
            and _safe_float(snapshot.h4_ma20_slope_pct) < 0.0
            else 0.0
        )
        context["bb_touch_upper"] = 1.0 if "upper" in str(snapshot.bb_touch_zone or "").lower() else 0.0
        context["bb_price_position_96"] = _model_position(context.get("bb_price_position_96"), 0.5)
        context["bb_price_position_105_4h"] = _model_position(context.get("bb_price_position_105_4h"), 0.5)
        context["bb_price_position_672_4h"] = _model_position(context.get("bb_price_position_672_4h"), 0.5)
        context["bb_price_position_1440_4h"] = _model_position(context.get("bb_price_position_1440_4h"), 0.5)
        context["bb_price_position_105_1m"] = _model_position(context.get("bb_price_position_105_1m"), 0.5)
        context["bb_price_position_300_1m"] = _model_position(context.get("bb_price_position_300_1m"), 0.5)
        context["bb105_mid_4h_distance_pct"] = _model_distance_pct(snapshot.mark_price, snapshot.bb105_mid_4h)
        context["bb672_mid_4h_distance_pct"] = _model_distance_pct(snapshot.mark_price, snapshot.bb672_mid_4h)
        context["bb1440_mid_4h_distance_pct"] = _model_distance_pct(snapshot.mark_price, snapshot.bb1440_mid_4h)
        context["bb105_upper_1m_distance_pct"] = _model_distance_pct(snapshot.mark_price, snapshot.bb105_upper_1m)
        context["bb300_upper_1m_distance_pct"] = _model_distance_pct(snapshot.mark_price, snapshot.bb300_upper_1m)
        context["bb96_upper_15m_distance_pct"] = _model_distance_pct(snapshot.mark_price, snapshot.bb96_upper_15m)
        context["short_4h_mid_retest_distance_min_pct"] = 0.0
        context["short_1m_upper_pressure_max"] = max(
            context["bb_price_position_105_1m"],
            context["bb_price_position_300_1m"],
        )
        context["short_15m_bb96_position"] = context["bb_price_position_96"]
        return context

    @staticmethod
    def _entry_coin_features_dict(snapshot: "MarketSnapshot") -> dict | None:
        coin_features = PaperExecutionEngine._snapshot_to_coin_features(snapshot)
        if coin_features is None:
            return None
        coin_features_dict = asdict(coin_features)
        coin_features_dict.update(PaperExecutionEngine._entry_training_context(snapshot))
        return coin_features_dict

    @staticmethod
    def _snapshot_to_coin_features(snapshot: "MarketSnapshot") -> "CoinFeatures | None":
        """
        从 MarketSnapshot 构建 CoinFeatures（轻量版，不依赖 whale_overview）。

        用于在入场/退出 journal 中保存完整特征快照，以便后续模型回试。
        注意：此简化版本没有 whale_overview 数据（OI history、smart money 等），
        对应特征填 0。pipeline 运行时的 CoinScoutRuntime 会用完整数据重新构建。
        """
        try:
            f = CoinFeatures(symbol=snapshot.symbol)
            # OI
            f.oi_change_pct_4h = float(snapshot.oi_change_pct_4h or 0)
            f.oi_price_divergence_score = float(snapshot.oi_price_divergence_score or 0)
            f.funding_rate = float(snapshot.funding_rate or 0)
            # 情绪
            f.sentiment_score = float(snapshot.sentiment_score or 0)
            f.sentiment_mentions = float(snapshot.sentiment_mentions or 0)
            # 市场微结构
            f.price_change_pct_5m = float(snapshot.price_change_pct_5m or 0)
            f.price_change_pct_4h = float(snapshot.price_change_pct_4h or 0)
            f.price_change_pct_24h = float(snapshot.price_change_pct_24h or 0)
            f.taker_buy_sell_ratio = float(snapshot.taker_buy_sell_ratio or 0)
            f.price_position_7d = float(snapshot.price_position_7d or 0)
            f.price_position_14d = float(snapshot.price_position_14d or 0)
            f.price_position_20d = float(snapshot.price_position_20d or 0)
            f.breakout_distance_7d = float(snapshot.breakout_distance_7d or 0)
            f.compression_score = float(snapshot.compression_score or 0)
            f.quote_volume_usdt_24h = float(snapshot.quote_volume_usdt_24h or 0)
            f.listing_days = float(snapshot.listing_days or 0) if snapshot.listing_days is not None else 0
            f.taker_imbalance = (f.taker_buy_sell_ratio - 1.0) * 100 if f.taker_buy_sell_ratio else 0
            # 基础交互特征
            f.sentiment_oi_interaction = f.sentiment_score * f.oi_change_pct_4h / 100.0
            f.momentum_score = f.price_change_pct_4h + f.oi_change_pct_4h * 0.5
            if f.compression_score > 0.5:
                f.structure_quality = min(100.0, f.compression_score * 30)
            # 时间特征（pipeline 运行时用当前时间）
            ts = datetime.now(timezone.utc)
            f.hour_of_day = float(ts.hour)
            f.day_of_week = float(ts.weekday())
            f.is_asian_session = 1.0 if 0 <= ts.hour < 8 else 0.0
            f.is_us_session = 1.0 if 13 <= ts.hour < 21 else 0.0
            # Kline statistics (stored for training dataset extraction)
            f.kline_lower_shadow_pct = float(snapshot.kline_lower_shadow_pct or 0)
            f.kline_upper_shadow_pct = float(snapshot.kline_upper_shadow_pct or 0)
            f.kline_body_pct = float(snapshot.kline_body_pct or 0)
            f.kline_direction = float(snapshot.kline_direction or 0)
            f.kline_volume_vs_ma5 = float(snapshot.kline_volume_vs_ma5 or 0)
            f.kline_range_vs_ma5_range = float(snapshot.kline_range_vs_ma5_range or 0)
            f.kline_close_position = float(snapshot.kline_close_position or 0)
            f.kline_consecutive_direction = float(snapshot.kline_consecutive_direction or 0)
            f.kline_atr_pct = float(snapshot.kline_atr_pct or 0)
            f.kline_ma20_distance_pct = float(snapshot.kline_ma20_distance_pct or 0)
            f.kline_volume_trend = float(snapshot.kline_volume_trend or 0)
            f.kline_max_drawdown_5d = float(snapshot.kline_max_drawdown_5d or 0)
            f.bb_price_position_96 = _model_position(snapshot.bb_price_position_96, 0.5)
            f.bb_price_position_105_4h = _model_position(snapshot.bb_price_position_105_4h, 0.5)
            f.bb_price_position_672_4h = _model_position(snapshot.bb_price_position_672_4h, 0.5)
            f.bb_price_position_1440_4h = _model_position(snapshot.bb_price_position_1440_4h, 0.5)
            f.bb_price_position_105_1m = _model_position(snapshot.bb_price_position_105_1m, 0.5)
            f.bb_price_position_300_1m = _model_position(snapshot.bb_price_position_300_1m, 0.5)
            f.bb105_mid_4h_distance_pct = _model_distance_pct(snapshot.mark_price, snapshot.bb105_mid_4h)
            f.bb672_mid_4h_distance_pct = _model_distance_pct(snapshot.mark_price, snapshot.bb672_mid_4h)
            f.bb1440_mid_4h_distance_pct = _model_distance_pct(snapshot.mark_price, snapshot.bb1440_mid_4h)
            f.bb105_upper_1m_distance_pct = _model_distance_pct(snapshot.mark_price, snapshot.bb105_upper_1m)
            f.bb300_upper_1m_distance_pct = _model_distance_pct(snapshot.mark_price, snapshot.bb300_upper_1m)
            f.bb96_upper_15m_distance_pct = _model_distance_pct(snapshot.mark_price, snapshot.bb96_upper_15m)
            f.short_4h_mid_retest_distance_min_pct = 0.0
            f.short_1m_upper_pressure_max = max(f.bb_price_position_105_1m, f.bb_price_position_300_1m)
            f.short_15m_bb96_position = f.bb_price_position_96
            return f
        except Exception as e:
            logger.debug("Failed to build CoinFeatures for %s: %s", snapshot.symbol, e)
            return None

    def _rebalance_shadow_paper(
        self,
        state: dict,
        snapshots: list[MarketSnapshot],
        market_guard: dict,
        excluded_symbols: set[str] | None = None,
    ) -> dict:
        raw_positions = [dict(item) for item in state.get("shadow_positions", []) if isinstance(item, dict)]
        shadow_journal = list(state.get("shadow_journal", []))
        if not self.config.shadow_paper_enabled:
            return {
                "enabled": False,
                "entries": [],
                "exits": [],
                "positions": raw_positions,
                "journal": shadow_journal[-self.config.shadow_paper_journal_limit:],
                "realized_pnl_usdt": 0.0,
                "unrealized_pnl_usdt": round(sum(self._shadow_position_unrealized_pnl(item) for item in raw_positions), 6),
                "trigger_threshold": self.config.shadow_paper_prob_moon_threshold,
            }

        now = _utc_now()
        snapshot_map = {snapshot.symbol: snapshot for snapshot in snapshots}
        next_positions: list[dict] = []
        entries: list[dict] = []
        exits: list[dict] = []
        exited_symbols: set[str] = set()
        realized_pnl_usdt = 0.0

        for raw in raw_positions:
            symbol = str(raw.get("symbol") or "")
            current = snapshot_map.get(symbol)
            if current is None:
                current = self._fetch_fallback_snapshot(symbol, raw)
            if current is None:
                logger.warning("shadow position %s: price refresh failed, skipping exit check", symbol)
                next_positions.append(raw)
                continue
            updated = dict(raw)
            side = str(updated.get("side") or "long").lower()
            entry_price = float(updated.get("entry_price") or 0.0)
            notional = float(updated.get("notional_usdt") or 0.0)
            current_price = float(current.mark_price)
            updated = self._update_shadow_runner_watermark(updated, current_price)
            pnl_pct = self._shadow_position_pnl_pct(updated, current_price)
            reason = self._shadow_exit_reason(updated, current, market_guard, now)
            updated.update(
                {
                    "current_price": current_price,
                    "last_seen_at": now.isoformat(),
                    "score": current.score,
                    "pnl_pct": round(pnl_pct, 6),
                    "unrealized_pnl_usdt": round(notional * pnl_pct, 6),
                }
            )
            if reason is None:
                next_positions.append(updated)
                continue
            size_fraction = 1.0
            exit_notional = notional
            if reason == "shadow_partial_take_profit":
                current_stage = int(updated.get("take_profit_stage") or 0)
                if current_stage == 0:
                    size_fraction = max(0.0, min(1.0, self.config.shadow_paper_tp_stage1_fraction))
                else:
                    size_fraction = max(0.0, min(1.0, self.config.shadow_paper_tp_stage2_fraction))
                exit_notional = round(notional * size_fraction, 8)
                if size_fraction <= 0 or size_fraction >= 1 or exit_notional <= 0 or notional - exit_notional <= 0:
                    reason = "shadow_take_profit_hit"
                    size_fraction = 1.0
                    exit_notional = notional
            pnl_usdt = round(exit_notional * pnl_pct, 6)
            realized_pnl_usdt += pnl_usdt
            exit_event = {
                "timestamp": now.isoformat(),
                "action": "take_profit_partial" if reason == "shadow_partial_take_profit" else "exit",
                "source": "shadow_paper",
                "symbol": symbol,
                "side": side,
                "reason": reason,
                "entry_price": entry_price,
                "exit_price": current_price,
                "notional_usdt": exit_notional,
                "size_fraction": round(size_fraction, 8),
                "pnl_pct": round(pnl_pct, 6),
                "pnl_usdt": pnl_usdt,
                "setup_type": str(updated.get("setup_type") or ""),
                "entry_timestamp": updated.get("opened_at"),
                "entry_score": float(updated.get("entry_score") or updated.get("score") or 0.0),
                "entry_mark_price": entry_price,
                "coin_features": updated.get("coin_features"),
                "kline_pattern_bullish_count": float(updated.get("kline_pattern_bullish_count") or 0.0),
                "kline_pattern_bearish_count": float(updated.get("kline_pattern_bearish_count") or 0.0),
                "kline_pattern_flags": str(updated.get("kline_pattern_flags") or ""),
                "shadow_scores": updated.get("shadow_scores"),
                "shadow_trigger": updated.get("shadow_trigger"),
                "shadow_position_key": updated.get("shadow_position_key"),
                "binance_mirror_status": str(updated.get("binance_mirror_status") or "pending_submit"),
                "initial_risk_pct": float(updated.get("initial_risk_pct") or 0.0),
                "predictive_snapshot": updated.get("predictive_snapshot"),
            }
            exits.append(exit_event)
            shadow_journal.append(exit_event)
            if reason == "shadow_partial_take_profit":
                remaining_notional = round(notional - exit_notional, 8)
                updated["notional_usdt"] = remaining_notional
                updated["original_notional_usdt"] = float(updated.get("original_notional_usdt") or notional)
                current_stage = int(updated.get("take_profit_stage") or 0)
                next_stage = current_stage + 1
                updated["take_profit_stage"] = next_stage
                initial_risk = float(updated.get("initial_risk_pct") or self.config.stop_loss_pct)
                if next_stage == 1 and self.config.shadow_paper_breakeven_after_stage1:
                    updated["stop_loss_price"] = entry_price
                elif next_stage == 2:
                    if side == "short":
                        updated["stop_loss_price"] = entry_price * (1 - initial_risk)
                    else:
                        updated["stop_loss_price"] = entry_price * (1 + initial_risk)
                if side == "short":
                    updated["low_watermark"] = current_price
                else:
                    updated["high_watermark"] = current_price
                updated["unrealized_pnl_usdt"] = round(remaining_notional * pnl_pct, 6)
                next_positions.append(updated)
                continue
            exited_symbols.add(symbol)

        excluded_symbols = excluded_symbols or set()
        held_symbols = {str(item.get("symbol") or "") for item in next_positions}
        recent_exit_index = self._build_recent_shadow_exit_index(shadow_journal)
        ranked = sorted(snapshots, key=lambda item: item.score, reverse=True)
        max_shadow_positions = int(self.config.shadow_paper_max_positions or 0)
        blocked_entries: list[dict] = []
        for snapshot in ranked:
            if max_shadow_positions > 0 and len(next_positions) >= max_shadow_positions:
                break
            if (
                snapshot.symbol in held_symbols
                or snapshot.symbol in excluded_symbols
                or snapshot.symbol in exited_symbols
                or self._shadow_reentry_blocked(snapshot, recent_exit_index.get(snapshot.symbol), now)
            ):
                continue
            setup_block_reason = self._shadow_entry_setup_block_reason(snapshot)
            if setup_block_reason:
                blocked_entries.append({
                    "symbol": snapshot.symbol,
                    "setup_type": snapshot.setup_type,
                    "setup_quality": snapshot.setup_quality,
                    "score": snapshot.score,
                    "reason": setup_block_reason,
                })
                continue
            coin_features_dict = PaperExecutionEngine._entry_coin_features_dict(snapshot)
            shadow_scores = self._score_with_shadow_models(coin_features_dict)
            if shadow_scores.get("model_meta"):
                for name in shadow_scores["model_meta"]:
                    shadow_scores["model_meta"][name]["feature_source"] = "lightweight_execution"
            trigger = self._shadow_paper_trigger(shadow_scores)
            if trigger is None:
                continue
            plan = self._build_entry(snapshot)
            side = "short" if plan.action == "open_short" else "long"
            initial_risk_pct = self._shadow_initial_risk_pct(
                side=side,
                entry_price=snapshot.mark_price,
                stop_loss_price=plan.stop_loss_price,
            )
            entry = {
                "timestamp": now.isoformat(),
                "action": plan.action,
                "source": "shadow_paper",
                "symbol": snapshot.symbol,
                "shadow_position_key": f"{snapshot.symbol}:{now.isoformat()}",
                "side": side,
                "opened_at": now.isoformat(),
                "entry_price": snapshot.mark_price,
                "current_price": snapshot.mark_price,
                "stop_loss_price": plan.stop_loss_price,
                "take_profit_price": plan.take_profit_price,
                "notional_usdt": plan.notional_usdt,
                "original_notional_usdt": plan.notional_usdt,
                "score": snapshot.score,
                "entry_score": snapshot.score,
                "rationale": plan.rationale,
                "setup_type": plan.setup_type,
                "take_profit_stage": 0,
                "high_watermark": None,
                "low_watermark": None,
                "initial_risk_pct": initial_risk_pct,
                "pnl_pct": 0.0,
                "unrealized_pnl_usdt": 0.0,
                "coin_features": coin_features_dict,
                "kline_pattern_bullish_count": snapshot.kline_pattern_bullish_count,
                "kline_pattern_bearish_count": snapshot.kline_pattern_bearish_count,
                "kline_pattern_flags": snapshot.kline_pattern_flags,
                "entry_h4_structure": {
                    "h4_close_above_ma20": snapshot.h4_close_above_ma20,
                    "h4_ma20_slope_pct": snapshot.h4_ma20_slope_pct,
                    "h4_price_vs_ma20_pct": snapshot.h4_price_vs_ma20_pct,
                    "h4_close_position_20bars": snapshot.h4_close_position_20bars,
                    "h4_low_break_recent_20bars": snapshot.h4_low_break_recent_20bars,
                    "h4_high_break_recent_20bars": snapshot.h4_high_break_recent_20bars,
                    "h4_atr_compression": snapshot.h4_atr_compression,
                    "h4_lower_highs_count": snapshot.h4_lower_highs_count,
                    "h4_higher_lows_count": snapshot.h4_higher_lows_count,
                    "h4_structure_state": snapshot.h4_structure_state,
                    "h4_price_location": snapshot.h4_price_location,
                    "h4_pattern_type": snapshot.h4_pattern_type,
                },
                "shadow_scores": shadow_scores,
                "shadow_trigger": trigger,
                "hf_quality": self._directional_hf_quality(snapshot),
                "sentiment_mentions": plan.sentiment_mentions,
                "sentiment_score": plan.sentiment_score,
                "oi_change_pct_4h": plan.oi_change_pct_4h,
                "taker_buy_sell_ratio": plan.taker_buy_sell_ratio,
                "smart_money_context": plan.smart_money_context,
                "binance_mirror_status": "pending_submit",
                "binance_mirror_error": None,
                "binance_mirror_updated_at": None,
                "binance_request_id": None,
                "predictive_snapshot": self._build_predictive_snapshot(snapshot.symbol, snapshot.reasons),
            }
            next_positions.append(entry)
            held_symbols.add(snapshot.symbol)
            entries.append(entry)
            shadow_journal.append(entry)

        unrealized_pnl_usdt = round(sum(self._shadow_position_unrealized_pnl(item) for item in next_positions), 6)
        return {
            "enabled": True,
            "entries": entries,
            "exits": exits,
            "positions": next_positions,
            "journal": shadow_journal[-self.config.shadow_paper_journal_limit:],
            "realized_pnl_usdt": round(realized_pnl_usdt, 6),
            "unrealized_pnl_usdt": unrealized_pnl_usdt,
            "trigger_threshold": self.config.shadow_paper_prob_moon_threshold,
            "max_positions": max_shadow_positions if max_shadow_positions > 0 else "unlimited",
            "max_hold_hours": self.config.shadow_paper_max_hold_hours,
            "blocked_entry_count": len(blocked_entries),
            "blocked_entries": blocked_entries[:50],
        }

    def _shadow_entry_setup_block_reason(self, snapshot: MarketSnapshot) -> str | None:
        if not self.config.block_trend_entry:
            return None
        setup_type = str(snapshot.setup_type or "").strip()
        if setup_type in ALLOWED_ENTRY_SETUP_TYPES:
            return None
        if setup_type == "trend" and float(snapshot.setup_quality or 0.0) <= 0.0:
            return "setup_type_no_edge"
        return "setup_type_not_allowed"

    @staticmethod
    def _update_shadow_runner_watermark(position: dict, current_price: float) -> dict:
        if int(position.get("take_profit_stage") or 0) < 1:
            return position
        updated = dict(position)
        side = str(updated.get("side") or "long").lower()
        if side == "short":
            existing = float(updated.get("low_watermark") or current_price)
            updated["low_watermark"] = min(existing, current_price)
        else:
            existing = float(updated.get("high_watermark") or current_price)
            updated["high_watermark"] = max(existing, current_price)
        return updated

    @staticmethod
    def _build_recent_shadow_exit_index(shadow_journal: list[dict]) -> dict[str, dict]:
        recent: dict[str, dict] = {}
        for item in reversed(shadow_journal):
            if str(item.get("action") or "") != "exit":
                continue
            if str(item.get("source") or "") != "shadow_paper":
                continue
            if str(item.get("reason") or "") == "shadow_partial_take_profit":
                continue
            symbol = str(item.get("symbol") or "")
            if not symbol or symbol in recent:
                continue
            recent[symbol] = item
        return recent

    def _shadow_reentry_blocked(
        self,
        snapshot: MarketSnapshot,
        recent_exit: dict | None,
        now: datetime,
    ) -> bool:
        if not recent_exit:
            return False
        cooldown_minutes = int(self.config.shadow_paper_reentry_cooldown_minutes or 0)
        if cooldown_minutes <= 0:
            return False
        exit_ts = _parse_iso8601(str(recent_exit.get("timestamp") or ""))
        if exit_ts is None or now - exit_ts >= timedelta(minutes=cooldown_minutes):
            return False
        min_move = float(self.config.shadow_paper_reentry_min_price_move_pct or 0.0)
        if min_move <= 0:
            return True
        if str(recent_exit.get("reason") or "") != "shadow_take_profit_hit":
            return True
        exit_price = float(recent_exit.get("exit_price") or 0.0)
        if exit_price <= 0:
            return True
        side = str(recent_exit.get("side") or "long").lower()
        if side == "short":
            return snapshot.mark_price < exit_price * (1 + min_move)
        return snapshot.mark_price > exit_price * (1 - min_move)

    def _shadow_paper_trigger(self, shadow_scores: dict) -> dict | None:
        scores = shadow_scores.get("scores") if isinstance(shadow_scores, dict) else {}
        if not isinstance(scores, dict):
            return None
        threshold = float(self.config.shadow_paper_prob_moon_threshold)
        best: dict | None = None
        for model_name, score in scores.items():
            if not isinstance(score, dict):
                continue
            try:
                prob_moon = float(score.get("prob_moon") or 0.0)
                model_label = int(score.get("model_label") or 0)
            except (TypeError, ValueError):
                continue
            if prob_moon < threshold:
                continue
            candidate = {
                "model": str(model_name),
                "prob_moon": round(prob_moon, 4),
                "model_label": model_label,
                "threshold": threshold,
            }
            if best is None or candidate["prob_moon"] > best["prob_moon"]:
                best = candidate
        return best

    def _fetch_fallback_snapshot(self, symbol: str, position: dict) -> MarketSnapshot | None:
        if not self._mark_price_fetcher or not symbol:
            return None
        try:
            mark_price = self._mark_price_fetcher(symbol)
            if mark_price is None or mark_price <= 0:
                return None
            return MarketSnapshot(
                symbol=symbol,
                mark_price=mark_price,
                score=float(position.get("score") or 0.0),
            )
        except Exception as exc:
            logger.warning("shadow position %s: mark_price_fetcher error: %s", symbol, exc)
            return None

    def _shadow_exit_reason(
        self,
        position: dict,
        current: MarketSnapshot,
        market_guard: dict,
        now: datetime,
    ) -> str | None:
        if market_guard.get("triggered"):
            return "shadow_market_flash_crash_protection"
        side = str(position.get("side") or "long").lower()
        price = float(current.mark_price)
        stop_loss = float(position.get("stop_loss_price") or 0.0)
        entry_price = float(position.get("entry_price") or 0.0)
        stage = int(position.get("take_profit_stage") or 0)
        initial_risk_pct = float(position.get("initial_risk_pct") or 0.0)
        if initial_risk_pct <= 0:
            initial_risk_pct = self.config.stop_loss_pct

        if side == "short":
            if stop_loss > 0 and price >= stop_loss:
                return "shadow_stop_loss_hit"
            if stage == 0:
                target = entry_price * (1 - initial_risk_pct * self.config.shadow_paper_tp_stage1_r)
                if price <= target:
                    return "shadow_partial_take_profit"
            elif stage == 1:
                target = entry_price * (1 - initial_risk_pct * self.config.shadow_paper_tp_stage2_r)
                if price <= target:
                    return "shadow_partial_take_profit"
            elif stage >= 2:
                max_target = entry_price * (1 - initial_risk_pct * self.config.shadow_paper_tp_stage3_r)
                if price <= max_target:
                    return "shadow_runner_max_tp"
                low_watermark = min(float(position.get("low_watermark") or price), price)
                if low_watermark > 0 and price >= low_watermark * (1 + self.config.shadow_paper_trailing_pct):
                    return "shadow_trailing_stop"
        else:
            if stop_loss > 0 and price <= stop_loss:
                return "shadow_stop_loss_hit"
            if stage == 0:
                target = entry_price * (1 + initial_risk_pct * self.config.shadow_paper_tp_stage1_r)
                if price >= target:
                    return "shadow_partial_take_profit"
            elif stage == 1:
                target = entry_price * (1 + initial_risk_pct * self.config.shadow_paper_tp_stage2_r)
                if price >= target:
                    return "shadow_partial_take_profit"
            elif stage >= 2:
                max_target = entry_price * (1 + initial_risk_pct * self.config.shadow_paper_tp_stage3_r)
                if price >= max_target:
                    return "shadow_runner_max_tp"
                high_watermark = max(float(position.get("high_watermark") or price), price)
                if high_watermark > 0 and price <= high_watermark * (1 - self.config.shadow_paper_trailing_pct):
                    return "shadow_trailing_stop"
        opened_at = _parse_iso8601(str(position.get("opened_at") or ""))
        if opened_at and now - opened_at >= timedelta(hours=self.config.shadow_paper_max_hold_hours):
            return "shadow_max_hold_reached"
        return None

    @staticmethod
    def _shadow_position_pnl_pct(position: dict, current_price: float) -> float:
        entry_price = float(position.get("entry_price") or 0.0)
        if entry_price <= 0:
            return 0.0
        if str(position.get("side") or "long").lower() == "short":
            return (entry_price - current_price) / entry_price
        return (current_price - entry_price) / entry_price

    @classmethod
    def _shadow_position_unrealized_pnl(cls, position: dict) -> float:
        current_price = float(position.get("current_price") or position.get("entry_price") or 0.0)
        return float(position.get("notional_usdt") or 0.0) * cls._shadow_position_pnl_pct(position, current_price)

    @staticmethod
    def _shadow_initial_risk_pct(side: str, entry_price: float, stop_loss_price: float) -> float:
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0.0
        if side == "short":
            return round(max(0.0, (stop_loss_price - entry_price) / entry_price), 8)
        return round(max(0.0, (entry_price - stop_loss_price) / entry_price), 8)

    def _build_learning_index(
        self, journal: list[dict]
    ) -> tuple[
        dict[str, dict[str, float]],
        dict[tuple[str, str], dict[str, float]],
        dict[str, dict[str, float]],
    ]:
        closed = [item for item in journal if str(item.get("action")) == "exit"]
        # Symbol-level expectancy should use the full closed-trade history.
        # A short recent lookback let repeat losers back in after one quiet window,
        # which kept feeding low-information stop-loss samples into training.
        recent_closed = closed
        buckets: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "wins": 0.0, "total_pnl_pct": 0.0})
        setup_buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"count": 0.0, "wins": 0.0, "total_pnl_pct": 0.0})
        setup_type_buckets: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "wins": 0.0, "total_pnl_pct": 0.0})
        for item in recent_closed:
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            bucket = buckets[symbol]
            bucket["count"] += 1.0
            pnl_pct = float(item.get("pnl_pct") or 0.0)
            bucket["total_pnl_pct"] += pnl_pct
            if pnl_pct > 0:
                bucket["wins"] += 1.0
            setup_type = self._infer_setup_type(item)
            setup_bucket = setup_buckets[(symbol, setup_type)]
            setup_bucket["count"] += 1.0
            setup_bucket["total_pnl_pct"] += pnl_pct
            if pnl_pct > 0:
                setup_bucket["wins"] += 1.0
        for item in closed:
            setup_type = self._infer_setup_type(item)
            if not setup_type:
                continue
            pnl_pct = float(item.get("pnl_pct") or 0.0)
            setup_type_bucket = setup_type_buckets[setup_type]
            setup_type_bucket["count"] += 1.0
            setup_type_bucket["total_pnl_pct"] += pnl_pct
            if pnl_pct > 0:
                setup_type_bucket["wins"] += 1.0
        return buckets, setup_buckets, setup_type_buckets

    def _build_recent_exit_index(self, journal: list[dict]) -> dict[str, dict[str, float | str]]:
        exits = [item for item in journal if str(item.get("action")) == "exit"]
        index: dict[str, dict[str, float | str]] = {}
        for item in exits:
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            index[symbol] = {
                "timestamp": str(item.get("timestamp") or ""),
                "entry_price": float(item.get("entry_price") or 0.0),
                "exit_price": float(item.get("exit_price") or 0.0),
                "score": float(item.get("score") or 0.0),
                "pnl_pct": float(item.get("pnl_pct") or 0.0),
                "reason": str(item.get("reason") or ""),
            }
        return index

    def _build_recent_loss_streak_index(self, journal: list[dict]) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
        closed = [item for item in journal if str(item.get("action")) == "exit"]
        symbol_streaks: dict[str, int] = defaultdict(int)
        setup_streaks: dict[tuple[str, str], int] = defaultdict(int)
        for item in closed:
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            setup_type = self._infer_setup_type(item)
            pnl_pct = float(item.get("pnl_pct") or 0.0)
            if pnl_pct > 0:
                symbol_streaks[symbol] = 0
                setup_streaks[(symbol, setup_type)] = 0
                continue
            symbol_streaks[symbol] += 1
            setup_streaks[(symbol, setup_type)] += 1
        return dict(symbol_streaks), dict(setup_streaks)

    def _build_flat_score_fade_index(self, journal: list[dict]) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
        symbol_counts: dict[str, int] = defaultdict(int)
        setup_counts: dict[tuple[str, str], int] = defaultdict(int)
        max_abs_pnl = max(0.0, self.config.flat_score_fade_max_abs_pnl_pct)
        for item in journal:
            if str(item.get("action")) != "exit":
                continue
            if str(item.get("reason") or "") != "score_faded":
                continue
            pnl_pct = float(item.get("pnl_pct") or 0.0)
            if pnl_pct > 0 or abs(pnl_pct) > max_abs_pnl:
                continue
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            setup_type = self._infer_setup_type(item)
            symbol_counts[symbol] += 1
            setup_counts[(symbol, setup_type)] += 1
        return dict(symbol_counts), dict(setup_counts)

    def _build_stop_loss_index(self, journal: list[dict]) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
        symbol_counts: dict[str, int] = defaultdict(int)
        setup_counts: dict[tuple[str, str], int] = defaultdict(int)
        for item in journal:
            if str(item.get("action")) != "exit":
                continue
            if str(item.get("reason") or "") not in STOP_LOSS_EXIT_REASONS:
                continue
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            setup_type = self._infer_setup_type(item)
            symbol_counts[symbol] += 1
            setup_counts[(symbol, setup_type)] += 1
        return dict(symbol_counts), dict(setup_counts)

    def _build_consecutive_stop_loss_index(self, journal: list[dict]) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
        """检测连续止损：从后往前数，直到遇到非止损的退出为止"""
        symbol_consecutive: dict[str, int] = {}
        setup_consecutive: dict[tuple[str, str], int] = {}
        symbol_closed: set[str] = set()
        setup_closed: set[tuple[str, str]] = set()
        for item in reversed(journal):
            if str(item.get("action")) != "exit":
                continue
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            setup_key = (symbol, self._infer_setup_type(item))
            is_stop_loss = str(item.get("reason") or "") in STOP_LOSS_EXIT_REASONS
            if symbol not in symbol_closed:
                if is_stop_loss:
                    symbol_consecutive[symbol] = symbol_consecutive.get(symbol, 0) + 1
                else:
                    symbol_consecutive.setdefault(symbol, 0)
                    symbol_closed.add(symbol)
            if setup_key not in setup_closed:
                if is_stop_loss:
                    setup_consecutive[setup_key] = setup_consecutive.get(setup_key, 0) + 1
                else:
                    setup_consecutive.setdefault(setup_key, 0)
                    setup_closed.add(setup_key)
        return symbol_consecutive, setup_consecutive

    def _build_non_progress_index(self, journal: list[dict]) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
        symbol_counts: dict[str, int] = defaultdict(int)
        setup_counts: dict[tuple[str, str], int] = defaultdict(int)
        for item in journal:
            if str(item.get("action")) != "exit":
                continue
            if str(item.get("reason") or "") not in NON_PROGRESS_EXIT_REASONS:
                continue
            if float(item.get("pnl_pct") or 0.0) > 0.0:
                continue
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            setup_type = self._infer_setup_type(item)
            symbol_counts[symbol] += 1
            setup_counts[(symbol, setup_type)] += 1
        return dict(symbol_counts), dict(setup_counts)

    def _daily_loss_usdt(self, journal: list[dict]) -> float:
        """计算今日已实现 PnL（USDT），负值表示亏损。"""
        today_start = _utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_pnl = 0.0
        for item in journal:
            if str(item.get("action")) != "exit":
                continue
            ts = _parse_iso8601(str(item.get("timestamp") or ""))
            if ts and ts >= today_start:
                pnl_pct = float(item.get("pnl_pct") or 0.0)
                notional = float(item.get("notional_usdt") or 0.0)
                daily_pnl += pnl_pct * notional
        return daily_pnl

    def _pdf_strategy_hard_block(
        self,
        snapshot: MarketSnapshot,
        recent_exit_index: dict[str, dict[str, float | str]],
    ) -> str | None:
        """PDF 策略硬拦截：止损冷却、逆势拦截、追高拦截（side-aware）。"""
        recent = recent_exit_index.get(snapshot.symbol.upper())
        if recent and str(recent.get("reason")) in STOP_LOSS_EXIT_REASONS:
            ts = _parse_iso8601(str(recent.get("timestamp") or ""))
            if ts:
                elapsed_min = (_utc_now() - ts).total_seconds() / 60
                if elapsed_min < self.config.pdf_cooldown_after_sl_minutes:
                    return "pdf_cooldown_after_sl"

        is_short = snapshot.setup_type in ("squeeze_short_candidate", "base_breakdown_confirmed", "base_breakdown")

        if is_short:
            if snapshot.kline_direction > 0 and snapshot.kline_ma20_distance_pct > abs(self.config.pdf_trend_min_ma20_distance):
                if snapshot.kline_consecutive_direction >= abs(self.config.pdf_trend_min_consecutive):
                    return "pdf_against_trend"
            if snapshot.price_position_7d is not None and snapshot.price_position_7d < 0.15:
                if snapshot.kline_lower_shadow_pct > 0.4:
                    return "pdf_exhaustion_at_bottom"
            if self.config.pdf_close_confirmation_enabled:
                close_pos = float(snapshot.kline_close_position or 1.0)
                if close_pos > (1.0 - self.config.pdf_close_confirmation_min):
                    return "pdf_close_not_confirmed"
        else:
            if snapshot.kline_direction < 0 and snapshot.kline_ma20_distance_pct < self.config.pdf_trend_min_ma20_distance:
                if snapshot.kline_consecutive_direction <= self.config.pdf_trend_min_consecutive:
                    return "pdf_against_trend"
            if snapshot.price_position_7d is not None and snapshot.price_position_7d > 0.85:
                if snapshot.kline_upper_shadow_pct > 0.4:
                    return "pdf_exhaustion_at_top"
            if self.config.pdf_close_confirmation_enabled:
                close_pos = float(snapshot.kline_close_position or 0)
                if 0 < close_pos < self.config.pdf_close_confirmation_min:
                    return "pdf_close_not_confirmed"

        return None

    def _detect_behavioral_traps(
        self,
        snapshot: MarketSnapshot,
        recent_exit_index: dict[str, dict[str, float | str]],
        daily_pnl: float,
    ) -> list[str]:
        """检测行为陷阱：沉没成本偏差（止损后宽窗口内重入同币）、tilt（亏损扩大时加仓）。"""
        traps: list[str] = []
        symbol = snapshot.symbol.upper()
        recent = recent_exit_index.get(symbol)
        if recent and str(recent.get("reason")) in STOP_LOSS_EXIT_REASONS:
            ts = _parse_iso8601(str(recent.get("timestamp") or ""))
            if ts:
                elapsed_min = (_utc_now() - ts).total_seconds() / 60
                if elapsed_min < self.config.behavioral_sunk_cost_window_minutes:
                    traps.append("behavioral_sunk_cost_bias")
        tilt_threshold = -self.config.pdf_daily_max_loss_usdt * self.config.behavioral_tilt_daily_loss_pct
        if daily_pnl <= tilt_threshold:
            traps.append("behavioral_tilt_block")
        return traps

    def _exceeds_correlation_limit(self, snapshot: MarketSnapshot, current_positions: list) -> bool:
        """PDF 相关性风控：同 setup_type 持仓不超限，高相关币对不超限。"""
        setup = snapshot.setup_type or "trend"
        same_setup_count = sum(1 for p in current_positions if p.setup_type == setup)
        if same_setup_count >= self.config.pdf_max_same_setup_positions:
            return True
        direction = float(snapshot.kline_direction or 0)
        if direction != 0:
            correlated = sum(
                1 for p in current_positions
                if p.entry_coin_features
                and float(p.entry_coin_features.get("kline_direction") or 0) * direction > 0
                and abs(float(p.entry_coin_features.get("kline_ma20_distance_pct") or 0) - float(snapshot.kline_ma20_distance_pct or 0)) < 2.0
            )
            if correlated >= self.config.pdf_max_correlated_positions:
                return True
        return False

    def _manual_hard_block_symbols(self) -> set[str]:
        raw = str(self.config.manual_hard_block_symbols or "")
        return {item.strip().upper() for item in raw.replace(";", ",").split(",") if item.strip()}

    def _should_skip_entry(
        self,
        snapshot: MarketSnapshot,
        symbol_learning: dict[str, dict[str, float]],
        setup_learning: dict[tuple[str, str], dict[str, float]],
        setup_type_learning: dict[str, dict[str, float]],
        recent_symbol_loss_streaks: dict[str, int],
        recent_setup_loss_streaks: dict[tuple[str, str], int],
        stop_loss_symbols: dict[str, int],
        stop_loss_setups: dict[tuple[str, str], int],
        cons_sl_symbols: dict[str, int],
        cons_sl_setups: dict[tuple[str, str], int],
        flat_score_fade_symbols: dict[str, int],
        flat_score_fade_setups: dict[tuple[str, str], int],
        non_progress_symbols: dict[str, int],
        non_progress_setups: dict[tuple[str, str], int],
        recent_exit_index: dict[str, dict[str, float | str]],
        bb_touch_counts: dict | None = None,
        consecutive_entries: dict | None = None,
    ) -> bool:
        if snapshot.symbol.upper() in self._manual_hard_block_symbols():
            logger.info("manual_hard_blocked %s", snapshot.symbol)
            return True

        # --- BB Touch Count Hard Block ---
        if self.config.bb_touch_skip_enabled and bb_touch_counts:
            touch_key = f"{snapshot.symbol}:{snapshot.bb_touch_zone}"
            touch_info = bb_touch_counts.get(touch_key)
            if touch_info and touch_info.get("count", 0) >= self.config.bb_touch_max_count:
                logger.info("bb_touch_blocked %s: zone=%s count=%d", snapshot.symbol, snapshot.bb_touch_zone, touch_info["count"])
                return True

        # --- Consecutive Entry Hard Block ---
        if self.config.consecutive_entry_skip_enabled and consecutive_entries:
            ce_info = consecutive_entries.get(snapshot.symbol)
            if ce_info:
                now = _utc_now()
                last_entry_at = datetime.fromisoformat(ce_info["last_entry_at"]) if ce_info.get("last_entry_at") else None
                if last_entry_at and (now - last_entry_at).total_seconds() < self.config.consecutive_entry_reset_hours * 3600:
                    if ce_info.get("count", 0) >= self.config.consecutive_entry_max_count:
                        logger.info("consecutive_entry_blocked %s: count=%d", snapshot.symbol, ce_info["count"])
                        return True

        # --- PDF Strategy Hard Blocks (unconditional, no bypass) ---
        pdf_block = self._pdf_strategy_hard_block(snapshot, recent_exit_index)
        if pdf_block:
            return True

        # --- 4H Structure Hard Block (long side only) ---
        if (
            self.config.h4_structure_filter_enabled
            and self._infer_entry_side(snapshot) == "long"
            and snapshot.h4_structure_state == "broken"
        ):
            logger.info("h4_structure_blocked %s: state=broken", snapshot.symbol)
            return True

        has_external_confirmation = self._has_trusted_sentiment_confirmation(snapshot)
        has_trend_followthrough_probe = (
            snapshot.setup_type == "trend"
            and snapshot.score >= self.config.trend_followthrough_min_score
            and snapshot.hotness_priority_score >= self.config.trend_followthrough_min_hotness_priority_score
            and snapshot.sentiment_mentions >= self.config.trend_followthrough_min_sentiment_mentions
            and snapshot.price_change_pct_24h >= self.config.trend_followthrough_min_price_change_pct_24h
            and snapshot.taker_buy_sell_ratio >= self.config.mania_followthrough_min_taker_ratio
            and snapshot.quote_volume_usdt_24h >= self.config.mania_followthrough_min_quote_volume_usdt
        )

        has_repeated_signal_4h_support = self._has_repeated_signal_4h_support(snapshot)
        has_first_signal_probe = self._has_first_signal_probe(snapshot)
        has_first_signal_source_support = self._has_first_signal_source_support(snapshot)
        has_retest_add_support = self._has_retest_add_support(snapshot)
        has_structure_exception = has_repeated_signal_4h_support or has_first_signal_source_support or has_retest_add_support
        if self._is_predictive_hard_blocked(snapshot):
            return True
        if self._is_hf_quality_blocked(snapshot.symbol, side=self._infer_entry_side(snapshot)):
            return True

        # 致命过滤：setup_type 未知或普通趋势型且无外部确认 → 不进场；强趋势高热度/多次信号4H结构未破允许小仓 probe。
        if (
            snapshot.setup_type in {"trend", "?", ""}
            and not has_external_confirmation
            and not has_trend_followthrough_probe
            and not has_structure_exception
            and not has_first_signal_source_support
        ):
            return True
        # oi_divergence 无外部确认的也要拦截
        if snapshot.setup_type == "oi_divergence_long_candidate" and not has_external_confirmation:
            if self.config.oi_divergence_require_trusted_confirmation:
                return True
        # 连续止损拦截（硬拦截，无例外）
        if (
            self.config.stop_loss_consecutive_symbol_block > 0
            and cons_sl_symbols.get(snapshot.symbol.upper(), 0) >= self.config.stop_loss_consecutive_symbol_block
        ):
            return True
        if (
            self.config.stop_loss_consecutive_setup_block > 0
            and cons_sl_setups.get((snapshot.symbol.upper(), snapshot.setup_type), 0) >= self.config.stop_loss_consecutive_setup_block
        ):
            return True
        if snapshot.sentiment_mentions < self.config.absolute_min_sentiment_mentions and not has_repeated_signal_4h_support:
            return True
        if (
            not has_external_confirmation
            and snapshot.hotness_priority_score < self.config.low_info_min_hotness_priority_score
            and snapshot.sentiment_mentions < self.config.untrusted_entry_high_mentions_override
        ):
            return True
        min_mentions_required = 0 if has_external_confirmation else max(
            self.config.min_entry_sentiment_mentions,
            self.config.untrusted_entry_min_sentiment_mentions,
        )
        if snapshot.sentiment_mentions < min_mentions_required and not has_repeated_signal_4h_support and not has_first_signal_source_support:
            return True
        if (
            not has_external_confirmation
            and not has_structure_exception
            and not has_first_signal_source_support
            and snapshot.score < self.config.untrusted_entry_min_score
        ):
            return True
        if (
            not has_external_confirmation
            and not has_structure_exception
            and snapshot.sentiment_mentions < self.config.untrusted_entry_high_mentions_override
            and self._is_discovery_only_untrusted(snapshot)
        ):
            return True
        if (
            not has_external_confirmation
            and not has_structure_exception
            and self._sentiment_source_count(snapshot) < self.config.untrusted_entry_min_source_count
        ):
            return True
        if (
            not has_external_confirmation
            and not has_structure_exception
            and self._is_weak_single_source_untrusted(snapshot)
        ):
            return True
        if (
            has_external_confirmation
            and snapshot.setup_type not in {"base_breakout_confirmed", "oi_divergence_long_candidate", "mania_followthrough"}
            and snapshot.sentiment_mentions < self.config.untrusted_entry_high_mentions_override
            and snapshot.setup_quality <= 0
            and not self._has_primary_conviction_source(snapshot)
            and self._sentiment_source_count(snapshot) < self.config.untrusted_entry_min_source_count
            and not has_structure_exception
        ):
            return True
        if (
            snapshot.setup_type == "oi_divergence_long_candidate"
            and self.config.oi_divergence_require_trusted_confirmation
            and not has_external_confirmation
        ):
            return True
        if snapshot.setup_type == "oi_divergence_long_candidate":
            if snapshot.sentiment_mentions < self.config.oi_divergence_min_sentiment_mentions:
                return True
            if snapshot.oi_price_divergence_score < self.config.oi_divergence_min_divergence_score:
                return True
            if self._sentiment_source_count(snapshot) < self.config.oi_divergence_min_source_count:
                return True
        if snapshot.setup_type == "base_breakout_confirmed":
            if (
                self.config.base_breakout_require_trusted_confirmation
                and not has_external_confirmation
                and self._is_weak_single_source_untrusted(snapshot)
            ):
                return True
            if snapshot.sentiment_mentions < self.config.base_breakout_min_sentiment_mentions and not has_external_confirmation:
                return True
            if (
                not has_external_confirmation
                and self._is_weak_single_source_untrusted(snapshot)
                and self._sentiment_source_count(snapshot) < self.config.base_breakout_min_source_count
            ):
                return True
        if snapshot.setup_type in {"oi_divergence_long_candidate", "base_breakout_confirmed", "mania_followthrough"}:
            min_mentions = self.config.min_entry_sentiment_mentions
            if snapshot.setup_type in {"oi_divergence_long_candidate", "base_breakout_confirmed"}:
                min_mentions = max(min_mentions, self.config.low_info_min_sentiment_mentions)
            if snapshot.sentiment_mentions < min_mentions and not has_external_confirmation:
                return True
        if (
            self.config.recent_setup_loss_streak_block > 0
            and
            recent_setup_loss_streaks.get((snapshot.symbol.upper(), snapshot.setup_type), 0)
            >= self.config.recent_setup_loss_streak_block
            and not has_structure_exception
        ):
            return True
        if (
            self.config.stop_loss_setup_block_min_closed_trades > 0
            and stop_loss_setups.get((snapshot.symbol.upper(), snapshot.setup_type), 0) >= self.config.stop_loss_setup_block_min_closed_trades
        ):
            setup_history = setup_learning.get((snapshot.symbol.upper(), snapshot.setup_type))
            if (
                self.config.stop_loss_block_apply_to_confirmed
                and (not setup_history or setup_history["wins"] <= 0)
                and not (has_repeated_signal_4h_support and snapshot.setup_type == "trend")
            ):
                return True
            if self._is_weak_single_source_untrusted(snapshot) and not has_external_confirmation:
                return True
        if (
            flat_score_fade_setups.get((snapshot.symbol.upper(), snapshot.setup_type), 0)
            >= self.config.flat_score_fade_block_min_closed_trades
        ):
            return True
        if (
            non_progress_setups.get((snapshot.symbol.upper(), snapshot.setup_type), 0)
            >= self.config.non_progress_setup_block_min_closed_trades
        ):
            return True
        if (
            self.config.recent_symbol_loss_streak_block > 0
            and recent_symbol_loss_streaks.get(snapshot.symbol.upper(), 0) >= self.config.recent_symbol_loss_streak_block
            and not has_repeated_signal_4h_support
        ):
            return True
        if (
            self.config.stop_loss_symbol_block_min_closed_trades > 0
            and stop_loss_symbols.get(snapshot.symbol.upper(), 0) >= self.config.stop_loss_symbol_block_min_closed_trades
        ):
            if self.config.stop_loss_block_apply_to_confirmed:
                symbol_history = symbol_learning.get(snapshot.symbol.upper())
                symbol_win_rate = (symbol_history["wins"] / max(symbol_history["count"], 1.0)) if symbol_history else 0.0
                if symbol_history and symbol_win_rate <= self.config.symbol_block_low_win_rate_max_win_rate and not has_repeated_signal_4h_support:
                    return True
            if self._is_weak_single_source_untrusted(snapshot) and not has_external_confirmation:
                return True
        if flat_score_fade_symbols.get(snapshot.symbol.upper(), 0) >= self.config.flat_score_fade_block_min_closed_trades:
            return True
        if non_progress_symbols.get(snapshot.symbol.upper(), 0) >= self.config.non_progress_symbol_block_min_closed_trades:
            return True
        setup_learned = setup_learning.get((snapshot.symbol.upper(), snapshot.setup_type))
        if (
            self.config.setup_block_min_closed_trades > 0
            and
            setup_learned
            and setup_learned["count"] >= self.config.setup_block_min_closed_trades
            and setup_learned["wins"] <= 0
            and setup_learned["total_pnl_pct"] <= 0
        ):
            return True
        setup_type_learned = setup_type_learning.get(snapshot.setup_type)
        if setup_type_learned:
            win_rate = setup_type_learned["wins"] / max(setup_type_learned["count"], 1.0)
            if (
                snapshot.setup_type in {"oi_divergence_long_candidate", "base_breakout_confirmed"}
                and setup_type_learned["count"] >= self.config.low_info_setup_type_block_min_closed_trades
                and win_rate <= self.config.low_info_setup_type_block_max_win_rate
                and setup_type_learned["total_pnl_pct"] <= 0
            ):
                return True
            if (
                snapshot.setup_type == "oi_divergence_long_candidate"
                and setup_type_learned["count"] >= self.config.oi_divergence_setup_block_min_closed_trades
                and win_rate <= self.config.oi_divergence_setup_block_max_win_rate
                and setup_type_learned["total_pnl_pct"] <= 0
            ):
                return True
            if (
                snapshot.setup_type == "base_breakout_confirmed"
                and setup_type_learned["count"] >= self.config.base_breakout_setup_block_min_closed_trades
                and win_rate <= self.config.base_breakout_setup_block_max_win_rate
                and setup_type_learned["total_pnl_pct"] <= 0
            ):
                return True
            if setup_type_learned["count"] >= self.config.setup_type_block_min_closed_trades:
                if win_rate <= self.config.setup_type_block_max_win_rate and setup_type_learned["total_pnl_pct"] <= 0:
                    return True
        learned = symbol_learning.get(snapshot.symbol.upper())
        if learned and learned["total_pnl_pct"] <= 0:
            win_rate = learned["wins"] / max(learned["count"], 1.0)
            avg_pnl_pct = learned["total_pnl_pct"] / max(learned["count"], 1.0)
            if (
                learned["count"] >= self.config.symbol_hard_block_underperform_min_closed_trades
                and win_rate <= self.config.symbol_hard_block_underperform_max_win_rate
                and avg_pnl_pct <= self.config.symbol_hard_block_underperform_max_avg_pnl_pct
            ):
                return True
            if (
                self.config.symbol_block_zero_win_apply_to_confirmed
                and learned["wins"] <= 0
                and learned["count"] >= self.config.symbol_block_zero_win_min_closed_trades
            ):
                return True
            if (
                snapshot.setup_type in {"oi_divergence_long_candidate", "base_breakout_confirmed"}
                and learned["count"] >= self.config.low_info_symbol_block_min_closed_trades
                and win_rate <= self.config.low_info_symbol_block_max_win_rate
            ):
                return True
            if not has_external_confirmation and not has_repeated_signal_4h_support:
                if learned["wins"] <= 0 and learned["count"] >= self.config.symbol_block_zero_win_min_closed_trades:
                    return True
                if learned["count"] >= self.config.symbol_block_min_closed_trades and learned["wins"] <= 0:
                    return True
                if (
                    learned["count"] >= self.config.symbol_block_low_win_rate_min_closed_trades
                    and win_rate <= self.config.symbol_block_low_win_rate_max_win_rate
                ):
                    return True
        recent = recent_exit_index.get(snapshot.symbol.upper())
        if not recent:
            return False
        ts = _parse_iso8601(str(recent.get("timestamp") or ""))
        if ts is None:
            return False
        elapsed = _utc_now() - ts
        ref_price = float(recent.get("exit_price") or recent.get("entry_price") or 0.0)
        price_move_pct = abs(snapshot.mark_price - ref_price) / max(ref_price, 1e-9)
        score_delta = abs(snapshot.score - float(recent.get("score") or 0.0))
        recent_reason = str(recent.get("reason") or "")
        recent_pnl = float(recent.get("pnl_pct") or 0.0)
        if (
            recent_reason in NON_PROGRESS_EXIT_REASONS
            and recent_pnl <= 0.0
            and elapsed <= timedelta(minutes=self.config.non_progress_reentry_cooldown_minutes)
        ):
            if score_delta < self.config.non_progress_reentry_min_score_delta:
                return True
            if snapshot.sentiment_mentions < self.config.non_progress_reentry_min_sentiment_mentions:
                return True
            if not has_external_confirmation and self._sentiment_source_count(snapshot) < self.config.untrusted_entry_min_source_count:
                return True
            return False
        # Hard block: after a stop_loss_hit WITHOUT external confirmation, enforce full cooldown
        if (
            recent_reason in STOP_LOSS_EXIT_REASONS
            and recent_pnl < 0
            and not has_external_confirmation
            and not has_structure_exception
            and elapsed <= timedelta(minutes=self.config.recent_reentry_cooldown_minutes)
        ):
            return True
        if elapsed > timedelta(minutes=self.config.recent_reentry_cooldown_minutes):
            return False
        return price_move_pct < self.config.recent_reentry_min_price_move_pct and score_delta < self.config.recent_reentry_min_score_delta

    def _build_entry(self, snapshot: MarketSnapshot, cons_sl_count: int = 0, bb_touch_count: int = 0) -> TradePlan:
        smart_money_context = self._smart_money_context(snapshot)
        notional_usdt = self._model_notional_usdt(snapshot, smart_money_context, cons_sl_count, bb_touch_count=bb_touch_count)
        rationale_suffix = self._smart_money_rationale_suffix(smart_money_context)
        if snapshot.setup_type == "squeeze_short_candidate":
            return TradePlan(
                action="open_short",
                symbol=snapshot.symbol,
                score=snapshot.score,
                entry_price=snapshot.mark_price,
                stop_loss_price=round(snapshot.mark_price * 1.06, 8),
                take_profit_price=round(snapshot.mark_price * 0.9, 8),
                notional_usdt=notional_usdt,
                rationale="short_squeeze_fade," + ", ".join(snapshot.reasons[:4]) + rationale_suffix,
                setup_type=snapshot.setup_type,
                sentiment_mentions=snapshot.sentiment_mentions,
                sentiment_score=snapshot.sentiment_score,
                smart_money_context=smart_money_context,
            )

        if snapshot.setup_type == "base_breakdown_confirmed":
            risk_pct = self._breakout_risk_pct(snapshot)
            return TradePlan(
                action="open_short",
                symbol=snapshot.symbol,
                score=snapshot.score,
                entry_price=snapshot.mark_price,
                stop_loss_price=round(snapshot.mark_price * (1 + risk_pct), 8),
                take_profit_price=round(snapshot.mark_price * (1 - (risk_pct * self.config.breakout_first_take_profit_r)), 8),
                notional_usdt=notional_usdt,
                rationale="base_breakdown_short," + ", ".join(snapshot.reasons[:4]) + rationale_suffix,
                setup_type=snapshot.setup_type,
                sentiment_mentions=snapshot.sentiment_mentions,
                sentiment_score=snapshot.sentiment_score,
                smart_money_context=smart_money_context,
            )

        if snapshot.setup_type == "mania_followthrough":
            notional = round(notional_usdt * self.config.mania_followthrough_position_scale, 8)
            return TradePlan(
                action="open_long",
                symbol=snapshot.symbol,
                score=snapshot.score,
                entry_price=snapshot.mark_price,
                stop_loss_price=round(snapshot.mark_price * (1 - self.config.mania_followthrough_stop_loss_pct), 8),
                take_profit_price=round(snapshot.mark_price * (1 + self.config.mania_followthrough_take_profit_pct), 8),
                notional_usdt=notional,
                rationale="mania_followthrough," + ", ".join(snapshot.reasons[:5]) + rationale_suffix,
                setup_type=snapshot.setup_type,
                sentiment_mentions=snapshot.sentiment_mentions,
                sentiment_score=snapshot.sentiment_score,
                smart_money_context=smart_money_context,
            )

        if (
            snapshot.setup_type == "trend"
            and snapshot.score >= self.config.trend_followthrough_min_score
            and snapshot.hotness_priority_score >= self.config.trend_followthrough_min_hotness_priority_score
            and snapshot.sentiment_mentions >= self.config.trend_followthrough_min_sentiment_mentions
            and snapshot.price_change_pct_24h >= self.config.trend_followthrough_min_price_change_pct_24h
            and snapshot.taker_buy_sell_ratio >= self.config.mania_followthrough_min_taker_ratio
            and snapshot.quote_volume_usdt_24h >= self.config.mania_followthrough_min_quote_volume_usdt
        ):
            notional = round(notional_usdt * self.config.trend_followthrough_position_scale, 8)
            return TradePlan(
                action="open_long",
                symbol=snapshot.symbol,
                score=snapshot.score,
                entry_price=snapshot.mark_price,
                stop_loss_price=round(snapshot.mark_price * (1 - self.config.mania_followthrough_stop_loss_pct), 8),
                take_profit_price=round(snapshot.mark_price * (1 + self.config.mania_followthrough_take_profit_pct), 8),
                notional_usdt=notional,
                rationale="trend_followthrough," + ", ".join(snapshot.reasons[:5]) + rationale_suffix,
                setup_type="mania_followthrough",
                sentiment_mentions=snapshot.sentiment_mentions,
                sentiment_score=snapshot.sentiment_score,
                smart_money_context=smart_money_context,
            )

        if snapshot.setup_type == "oi_divergence_long_candidate":
            notional = round(notional_usdt * self.config.oi_divergence_position_scale, 8)
            return TradePlan(
                action="open_long",
                symbol=snapshot.symbol,
                score=snapshot.score,
                entry_price=snapshot.mark_price,
                stop_loss_price=round(snapshot.mark_price * (1 - self.config.oi_divergence_stop_loss_pct), 8),
                take_profit_price=round(snapshot.mark_price * (1 + self.config.oi_divergence_take_profit_pct), 8),
                notional_usdt=notional,
                rationale="oi_divergence_probe," + ", ".join(snapshot.reasons[:5]) + rationale_suffix,
                setup_type=snapshot.setup_type,
                sentiment_mentions=snapshot.sentiment_mentions,
                sentiment_score=snapshot.sentiment_score,
                smart_money_context=smart_money_context,
            )

        if snapshot.setup_type == "base_breakout_confirmed":
            risk_pct = self._breakout_risk_pct(snapshot)
            stop_loss_price = round(snapshot.mark_price * (1 - risk_pct), 8)
            take_profit_price = round(snapshot.mark_price * (1 + (risk_pct * self.config.breakout_first_take_profit_r)), 8)
            rationale = "structure_stop,r_based_scaleout," + ", ".join(snapshot.reasons[:4]) + rationale_suffix
        else:
            atr_pct = float(snapshot.kline_atr_pct or 0)
            if atr_pct > 0:
                # FIX(2026-05-30): kline_atr_pct 已是小数比例(0.18=18%), 原代码误除100导致ATR项永久失效,
                # 止损恒等于固定下限。去掉/100后动态止损真正生效。
                dynamic_sl = max(self.config.stop_loss_pct, min(self.config.max_dynamic_stop_loss_pct, atr_pct * self.config.pdf_atr_stop_multiplier))
            else:
                price = snapshot.mark_price
                if price >= 1000:
                    dynamic_sl = self.config.stop_loss_pct
                elif price >= 10:
                    dynamic_sl = max(self.config.stop_loss_pct, 0.04)
                else:
                    dynamic_sl = max(self.config.stop_loss_pct, 0.06)
                vol_proxy = abs(snapshot.price_change_pct_24h) / 100.0
                vol_scale = max(0.5, min(2.0, 1.0 + (vol_proxy - 0.15) * 5.0))
                dynamic_sl = min(self.config.max_dynamic_stop_loss_pct, dynamic_sl * vol_scale)
            stop_loss_price = round(snapshot.mark_price * (1 - dynamic_sl), 8)
            if self._should_use_staged_tp():
                take_profit_price = round(snapshot.mark_price * (1 + dynamic_sl * self.config.staged_tp_stage1_r), 8)
            else:
                take_profit_price = round(snapshot.mark_price * (1 + self.config.take_profit_pct), 8)
            rationale = ", ".join(snapshot.reasons[:5]) + rationale_suffix

        # 市场 regime 调整止损/止盈
        regime = getattr(self, "_last_regime", None)
        if regime and regime.get("regime") != "flash_crash":
            sl_mult = float(regime.get("stop_loss_multiplier", 1.0))
            tp_mult = float(regime.get("take_profit_multiplier", 1.0))
            if abs(sl_mult - 1.0) > 0.001 or abs(tp_mult - 1.0) > 0.001:
                if snapshot.setup_type == "base_breakout_confirmed":
                    risk_pct_adj = risk_pct * sl_mult
                    stop_loss_price = round(snapshot.mark_price * (1 - min(0.06, risk_pct_adj)), 8)
                else:
                    base_sl_pct = abs(stop_loss_price - snapshot.mark_price) / max(snapshot.mark_price, 1e-9)
                    adj_sl_pct = min(0.10, base_sl_pct * sl_mult)
                    stop_loss_price = round(snapshot.mark_price * (1 - adj_sl_pct), 8)
                base_tp_pct = abs(take_profit_price - snapshot.mark_price) / max(snapshot.mark_price, 1e-9)
                adj_tp_pct = min(0.25, base_tp_pct * tp_mult)
                take_profit_price = round(snapshot.mark_price * (1 + adj_tp_pct), 8)

        return TradePlan(
            action="open_long",
            symbol=snapshot.symbol,
            score=snapshot.score,
            entry_price=snapshot.mark_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            notional_usdt=notional_usdt,
            rationale=rationale,
            setup_type=snapshot.setup_type,
            sentiment_mentions=snapshot.sentiment_mentions,
            sentiment_score=snapshot.sentiment_score,
            oi_change_pct_4h=snapshot.oi_change_pct_4h,
            taker_buy_sell_ratio=snapshot.taker_buy_sell_ratio,
            smart_money_context=smart_money_context,
        )

    def _apply_panel_blend(
        self,
        snapshots: list[MarketSnapshot],
        snapshot_map: dict[str, MarketSnapshot],
        positions: list[Position],
    ) -> None:
        """第2段打分: 只对值得调面板的小集合(持仓+技术分top候选)融入面板信号。

        其余几百个币不调 API, score 保持纯技术分。覆写原地生效于 snapshot_map/ranked
        共享的对象, 故同时驱动开仓选币(score排序+门槛)与持仓退出(score_faded)。
        """
        if not getattr(self.config, "panel_blend_enabled", False):
            return

        # 目标币集合: 全部持仓 + 技术分最高的若干候选 (上限 max_positions×mult)
        target_symbols: dict[str, str] = {}  # symbol -> position_side
        for position in positions:
            current = snapshot_map.get(position.symbol)
            if current is not None:
                target_symbols[position.symbol] = "long" if self._is_long_position(position) else "short"

        held = set(target_symbols)
        cand_cap = max(0, int(self.config.max_positions) * max(1, int(self.config.panel_blend_candidate_multiplier)))
        ranked_candidates = sorted(
            (s for s in snapshots if s.symbol not in held),
            key=lambda s: s.score,
            reverse=True,
        )[:cand_cap]
        for snap in ranked_candidates:
            side = "short" if snap.setup_type in self._SHORT_SETUP_TYPES else "long"
            target_symbols.setdefault(snap.symbol, side)

        panel_cache: dict[str, dict] = {}
        for symbol, side in target_symbols.items():
            snap = snapshot_map.get(symbol)
            if snap is None:
                continue
            ctx = panel_cache.get(symbol)
            if ctx is None:
                ctx = self._smart_money_context(snap)
                panel_cache[symbol] = ctx
            final_score, debug = blend_panel_score(snap.score, ctx, side, self.config)
            if debug.get("panel_applied"):
                snap.score = final_score
                if isinstance(getattr(snap, "reasons", None), list):
                    snap.reasons.append(
                        f"panel_blend(ss={debug['single_side_pct']},{debug['dominant_side']},w={debug['weight']},d={debug['delta']:+})"
                    )

    def _smart_money_context(self, snapshot: MarketSnapshot) -> dict:
        try:
            payload = self.smart_money_panel_provider(snapshot.symbol)
        except Exception as exc:
            return {"available": False, "error": str(exc)}
        views = (payload.get("views") or {}) if isinstance(payload, dict) else {}
        view = views.get("smart_money") or {}
        whale_view = views.get("whales") or {}
        long_side = view.get("long_side") or {}
        short_side = view.get("short_side") or {}
        long_notional = float(long_side.get("current_position_usdt") or 0.0)
        short_notional = float(short_side.get("current_position_usdt") or 0.0)
        total = long_notional + short_notional
        if total <= 0:
            return {"available": False, "error": "empty_smart_money_position"}
        long_pct = round(long_notional / total * 100.0, 2)
        short_pct = round(100.0 - long_pct, 2)
        single_side_pct = max(long_pct, short_pct)
        dominant_side = "long" if long_pct >= short_pct else "short"
        if self._is_major_market_cap_symbol(snapshot.symbol):
            if single_side_pct >= 80.0:
                composition = "large_cap_directional_pressure"
            elif snapshot.oi_change_pct_4h > 0:
                composition = "large_cap_broad_participation"
            else:
                composition = "large_cap_neutral"
        elif single_side_pct >= 90.0:
            composition = "single_side_庄家盘"
        elif single_side_pct < 70.0 and snapshot.oi_change_pct_4h > 0:
            composition = "mixed_consensus_slow_grind"
        else:
            composition = "directional_pressure"
        return {
            "available": True,
            "symbol_class": "major_top30" if self._is_major_market_cap_symbol(snapshot.symbol) else "small_mid_or_meme",
            "dominant_side": dominant_side,
            "long_pct": long_pct,
            "short_pct": short_pct,
            "single_side_pct": round(single_side_pct, 2),
            "oi_composition": composition,
            "long_position_usdt": long_notional,
            "short_position_usdt": short_notional,
            "smart_money_long_count": int(float(long_side.get("count") or 0)),
            "smart_money_short_count": int(float(short_side.get("count") or 0)),
            "smart_money_long_profit_ratio": float(long_side.get("profit_ratio_pct") or 0.0),
            "smart_money_short_profit_ratio": float(short_side.get("profit_ratio_pct") or 0.0),
            "whale_long_position_usdt": float((whale_view.get("long_side") or {}).get("current_position_usdt") or 0.0),
            "whale_short_position_usdt": float((whale_view.get("short_side") or {}).get("current_position_usdt") or 0.0),
            "whale_long_count": int(float((whale_view.get("long_side") or {}).get("count") or 0)),
            "whale_short_count": int(float((whale_view.get("short_side") or {}).get("count") or 0)),
            "whale_long_profit_ratio": float((whale_view.get("long_side") or {}).get("profit_ratio_pct") or 0.0),
            "whale_short_profit_ratio": float((whale_view.get("short_side") or {}).get("profit_ratio_pct") or 0.0),
            "monitor_action": self._position_monitor_action(snapshot, composition, dominant_side),
            "oi_change_pct_4h": snapshot.oi_change_pct_4h,
        }

    def _position_monitor_action(self, snapshot: MarketSnapshot, composition: str, dominant_side: str) -> str:
        if composition == "single_side_庄家盘" and dominant_side == "long" and snapshot.taker_buy_sell_ratio >= 1.05:
            return "watch_add_on_shallow_pullback"
        if composition == "single_side_庄家盘" and dominant_side == "short":
            return "watch_reduce_or_avoid_long"
        if composition in {"mixed_consensus_slow_grind", "large_cap_broad_participation"}:
            return "hold_or_small_add_on_trend_repair"
        if composition in {"large_cap_directional_pressure", "directional_pressure"}:
            return "monitor_for_reduce_if_flow_reverses"
        return "monitor_for_exit_on_panel_deterioration"

    def _smart_money_rationale_suffix(self, context: dict) -> str:
        if not context.get("available"):
            return ",smart_money_unavailable"
        composition = str(context.get("oi_composition") or "")
        if composition == "single_side_庄家盘":
            return ",smart_money_single_side_90pct"
        if composition == "mixed_consensus_slow_grind":
            return ",smart_money_mixed_consensus"
        if composition.startswith("large_cap_"):
            return ",smart_money_large_cap_structure"
        return ",smart_money_directional_pressure"

    def _is_major_market_cap_symbol(self, symbol: str) -> bool:
        base = symbol.upper().removesuffix("USDT")
        majors = {
            "BTC", "ETH", "BNB", "XRP", "SOL", "DOGE", "ADA", "TRX", "AVAX", "SHIB",
            "LINK", "TON", "DOT", "BCH", "NEAR", "LTC", "UNI", "APT", "ICP", "ETC",
            "FIL", "ARB", "ATOM", "HBAR", "VET", "OP", "INJ", "IMX", "XLM", "MATIC",
            "POL", "SUI", "TAO", "WIF", "PEPE",
        }
        return base in majors

    def _model_notional_usdt(self, snapshot: MarketSnapshot, context: dict, cons_sl_count: int = 0, bb_touch_count: int = 0) -> float:
        base = float(self.config.position_notional_usdt)
        score_scale = max(0.5, min(2.0, snapshot.score / 80.0))
        setup_scale = 1.0
        if snapshot.setup_type in {"mania_followthrough", "oi_divergence_long_candidate"}:
            setup_scale = 0.8
        if context.get("oi_composition") == "single_side_庄家盘":
            composition_scale = 1.35
            score_scale = max(1.0, score_scale)
        elif str(context.get("oi_composition") or "").startswith("large_cap_"):
            composition_scale = 1.0
            score_scale = 1.0
        elif context.get("oi_composition") == "mixed_consensus_slow_grind":
            composition_scale = 1.0
            score_scale = 1.0
        elif context.get("available"):
            composition_scale = 1.1
        else:
            composition_scale = 1.0
            score_scale = 1.0
        if self._has_retest_add_support(snapshot):
            composition_scale *= self.config.retest_add_position_scale
            score_scale = max(1.0, score_scale)
        elif self._has_repeated_signal_4h_support(snapshot):
            composition_scale *= self.config.repeated_signal_4h_position_scale
            score_scale = max(1.0, score_scale)
        elif self._has_first_signal_probe(snapshot):
            composition_scale *= self.config.first_signal_probe_position_scale
            score_scale = max(1.0, score_scale)
        loss_scale = 1.0
        half_size_after = int(self.config.consecutive_stop_loss_half_size_after or 0)
        if half_size_after > 0 and cons_sl_count >= half_size_after:
            loss_scale = float(self.config.consecutive_stop_loss_half_size_scale)
        atr_pct = float(snapshot.kline_atr_pct or 0)
        if atr_pct > 0:
            # 风险平价: 波动率越高仓位越小。base 与 atr_pct 同为小数比例
            atr_ratio = self.config.pdf_atr_base_pct / atr_pct
            atr_scale = max(self.config.pdf_atr_min_scale, min(self.config.pdf_atr_max_scale, atr_ratio))
        else:
            atr_scale = 1.0
        bb_touch_scale = self.config.bb_touch_2nd_position_scale if (self.config.bb_touch_skip_enabled and bb_touch_count == 2) else 1.0
        notional = max(0.0, base * score_scale * setup_scale * composition_scale * loss_scale * atr_scale * bb_touch_scale)
        min_notional = float(self.config.min_entry_notional_usdt or 0.0)
        if min_notional > 0 and notional > 0:
            notional = max(min_notional, notional)
        return round(notional, 8)

    def _get_bb_touch_count(self, bb_touch_counts: dict, snapshot: MarketSnapshot) -> int:
        if not bb_touch_counts or snapshot.bb_touch_zone == "none":
            return 0
        touch_key = f"{snapshot.symbol}:{snapshot.bb_touch_zone}"
        info = bb_touch_counts.get(touch_key)
        return info.get("count", 0) if info else 0

    def _update_bb_touch_counts(self, bb_touch_counts: dict, snapshots: list[MarketSnapshot]) -> None:
        now = _utc_now()
        for snapshot in snapshots:
            if snapshot.bb_touch_zone == "none":
                continue
            touch_key = f"{snapshot.symbol}:{snapshot.bb_touch_zone}"
            info = bb_touch_counts.get(touch_key)
            if info is None:
                bb_touch_counts[touch_key] = {
                    "count": 1,
                    "first_touch_at": now.isoformat(),
                    "last_touch_at": now.isoformat(),
                }
                continue
            last_touch = datetime.fromisoformat(info["last_touch_at"]) if info.get("last_touch_at") else now
            first_touch = datetime.fromisoformat(info["first_touch_at"]) if info.get("first_touch_at") else now
            if (now - first_touch).total_seconds() > self.config.bb_touch_reset_hours * 3600:
                bb_touch_counts[touch_key] = {"count": 1, "first_touch_at": now.isoformat(), "last_touch_at": now.isoformat()}
                continue
            if (now - last_touch).total_seconds() < self.config.bb_touch_min_gap_minutes * 60:
                continue
            info["count"] = info.get("count", 0) + 1
            info["last_touch_at"] = now.isoformat()

        keys_to_remove = []
        for key, info in bb_touch_counts.items():
            parts = key.split(":", 1)
            if len(parts) != 2:
                keys_to_remove.append(key)
                continue
            symbol, zone = parts
            snap = next((s for s in snapshots if s.symbol == symbol), None)
            if snap is None:
                continue
            bb_ref = self._bb_zone_reference_price(snap, zone)
            if bb_ref > 0 and abs(snap.mark_price - bb_ref) / bb_ref > self.config.bb_touch_reset_distance_pct / 100.0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            bb_touch_counts.pop(key, None)

    @staticmethod
    def _bb_zone_reference_price(snapshot: MarketSnapshot, zone: str) -> float:
        zone_map = {
            "bb96_upper": snapshot.bb96_upper_15m,
            "bb96_lower": snapshot.bb96_lower_15m,
            "bb210_upper": snapshot.bb210_upper_15m,
            "bb210_lower": snapshot.bb210_lower_15m,
            "bb672_15m_upper": snapshot.bb672_upper_15m,
            "bb672_15m_lower": snapshot.bb672_lower_15m,
            "bb1440_upper_15m": snapshot.bb1440_upper_15m,
            "bb1440_lower_15m": snapshot.bb1440_lower_15m,
            "bb1920_upper_15m": snapshot.bb1920_upper_15m,
            "bb1920_lower_15m": snapshot.bb1920_lower_15m,
            "bb2880_upper_15m": snapshot.bb2880_upper_15m,
            "bb2880_lower_15m": snapshot.bb2880_lower_15m,
            "bb4404_upper_15m": snapshot.bb4404_upper_15m,
            "bb4404_lower_15m": snapshot.bb4404_lower_15m,
            "bb4460_upper_15m": snapshot.bb4460_upper_15m,
            "bb4460_lower_15m": snapshot.bb4460_lower_15m,
            "bb672_upper": snapshot.bb672_upper_4h,
            "bb672_lower": snapshot.bb672_lower_4h,
        }
        return zone_map.get(zone, 0.0)

    def _update_consecutive_entries(self, consecutive_entries: dict, symbol: str) -> None:
        now = _utc_now()
        info = consecutive_entries.get(symbol)
        if info is None:
            consecutive_entries[symbol] = {"count": 1, "last_entry_at": now.isoformat()}
            return
        last_entry = datetime.fromisoformat(info["last_entry_at"]) if info.get("last_entry_at") else now
        if (now - last_entry).total_seconds() > self.config.consecutive_entry_reset_hours * 3600:
            consecutive_entries[symbol] = {"count": 1, "last_entry_at": now.isoformat()}
        else:
            info["count"] = info.get("count", 0) + 1
            info["last_entry_at"] = now.isoformat()

    def _evaluate_position(self, position: Position, current: MarketSnapshot) -> tuple[TradePlan | None, Position | None]:
        adjusted_position = self._maybe_tighten_oi_divergence_stop(position, current)
        adjusted_position = self._maybe_trailing_stop(adjusted_position, current)
        exit_reason = self._should_exit(adjusted_position, current)
        if exit_reason in STOP_LOSS_EXIT_REASONS:
            return (
                TradePlan(
                    action="exit",
                    symbol=adjusted_position.symbol,
                    score=current.score,
                    entry_price=adjusted_position.entry_price,
                    stop_loss_price=adjusted_position.stop_loss_price,
                    take_profit_price=adjusted_position.take_profit_price,
                    notional_usdt=adjusted_position.notional_usdt,
                    rationale=exit_reason,
                ),
                None,
            )

        partial_plan, partial_position = self._maybe_scale_out_breakout(adjusted_position, current)
        if partial_plan is not None:
            return partial_plan, partial_position

        if exit_reason == "staged_take_profit":
            staged_plan, staged_position = self._maybe_scale_out_staged(adjusted_position, current)
            if staged_plan is not None:
                return staged_plan, staged_position
            return (
                TradePlan(
                    action="exit",
                    symbol=adjusted_position.symbol,
                    score=current.score,
                    entry_price=adjusted_position.entry_price,
                    stop_loss_price=adjusted_position.stop_loss_price,
                    take_profit_price=adjusted_position.take_profit_price,
                    notional_usdt=adjusted_position.notional_usdt,
                    rationale="take_profit_hit",
                ),
                None,
            )

        if exit_reason is None:
            pyramided = self._maybe_pyramid_add(adjusted_position, current)
            return None, pyramided
        return (
            TradePlan(
                action="exit",
                symbol=adjusted_position.symbol,
                score=current.score,
                entry_price=adjusted_position.entry_price,
                stop_loss_price=adjusted_position.stop_loss_price,
                take_profit_price=adjusted_position.take_profit_price,
                notional_usdt=adjusted_position.notional_usdt,
                rationale=exit_reason,
            ),
            None,
        )

    def _maybe_scale_out_breakout(self, position: Position, current: MarketSnapshot) -> tuple[TradePlan | None, Position | None]:
        if "structure_levels_scaleout" in position.rationale:
            return self._maybe_scale_out_structure_levels(position, current)
        if not self._allows_internal_position_management():
            return None, position
        if "r_based_scaleout" not in position.rationale:
            return None, position
        if not self._is_breakout_sentiment_strong(current):
            return None, position

        risk_pct = position.initial_risk_pct if position.initial_risk_pct > 0 else self._position_risk_pct(position)
        if risk_pct <= 0:
            return None, position
        tp_mult = self.config.trend_up_tp_extension_multiplier if (self._is_trend_up_regime() and self._is_long_position(position)) else 1.0
        if position.take_profit_stage == 0:
            target_price = round(position.entry_price * (1 + (risk_pct * self.config.breakout_first_take_profit_r * tp_mult)), 8)
            exit_fraction = self.config.breakout_first_exit_fraction
            next_stage = 1
            next_take_profit = round(position.entry_price * (1 + (risk_pct * self.config.breakout_second_take_profit_r * tp_mult)), 8)
            next_stop = position.entry_price
        elif position.take_profit_stage == 1:
            target_price = round(position.entry_price * (1 + (risk_pct * self.config.breakout_second_take_profit_r * tp_mult)), 8)
            exit_fraction = self.config.breakout_second_exit_fraction / max(1e-9, (1 - self.config.breakout_first_exit_fraction))
            next_stage = BREAKOUT_RUNNER_STAGE
            next_take_profit = round(position.entry_price * (1 + (risk_pct * self.config.breakout_runner_take_profit_r * tp_mult)), 8)
            next_stop = round(position.entry_price * (1 + (risk_pct * 0.5)), 8)
        else:
            return None, position

        if current.mark_price < target_price:
            return None, position

        partial_notional = round(position.notional_usdt * exit_fraction, 8)
        remaining_notional = round(position.notional_usdt - partial_notional, 8)
        if partial_notional <= 0 or remaining_notional <= 0:
            return None, position

        updated_position = Position(
            symbol=position.symbol,
            opened_at=position.opened_at,
            entry_price=position.entry_price,
            current_price=current.mark_price,
            stop_loss_price=next_stop,
            take_profit_price=next_take_profit,
            notional_usdt=remaining_notional,
            score=current.score,
            rationale=position.rationale,
            setup_type=position.setup_type,
            take_profit_stage=next_stage,
            initial_risk_pct=risk_pct,
            structure_timeframe=position.structure_timeframe,
            last_sentiment_score=current.sentiment_score,
            last_sentiment_mentions=current.sentiment_mentions,
            last_oi_change_pct_4h=current.oi_change_pct_4h,
            last_taker_buy_sell_ratio=current.taker_buy_sell_ratio,
            entry_coin_features=position.entry_coin_features,
            entry_kline_pattern_bullish_count=position.entry_kline_pattern_bullish_count,
            entry_kline_pattern_bearish_count=position.entry_kline_pattern_bearish_count,
            entry_kline_pattern_flags=position.entry_kline_pattern_flags,
            entry_h4_structure=position.entry_h4_structure,
            pyramid_count=position.pyramid_count,
            predictive_snapshot=position.predictive_snapshot,
            high_watermark_pnl_pct=position.high_watermark_pnl_pct,
        )
        return (
            TradePlan(
                action="take_profit_partial",
                symbol=position.symbol,
                score=current.score,
                entry_price=position.entry_price,
                stop_loss_price=next_stop,
                take_profit_price=target_price,
                notional_usdt=partial_notional,
                rationale=f"r_multiple_scaleout_stage_{next_stage}",
                size_fraction=round(partial_notional / max(position.notional_usdt, 1e-9), 8),
            ),
            updated_position,
        )


    def _maybe_scale_out_staged(self, position: Position, current: MarketSnapshot) -> tuple[TradePlan | None, Position | None]:
        """Staged take-profit for main (non-breakout) positions: 1.5R/3R/5R partial closes."""
        if not self._allows_staged_take_profit_management():
            return None, position

        risk_pct = position.initial_risk_pct if position.initial_risk_pct > 0 else self._position_risk_pct(position)
        if risk_pct <= 0:
            return None, position

        cfg = self.config
        tp_mult = cfg.trend_up_tp_extension_multiplier if (self._is_trend_up_regime() and self._is_long_position(position)) else 1.0

        if position.take_profit_stage == 0:
            target_price = round(position.entry_price * (1 + risk_pct * cfg.staged_tp_stage1_r * tp_mult), 8)
            exit_fraction = cfg.staged_tp_stage1_fraction
            next_stage = 1
            next_tp = round(position.entry_price * (1 + risk_pct * cfg.staged_tp_stage2_r * tp_mult), 8)
            next_stop = position.entry_price if cfg.staged_tp_breakeven_after_stage1 else position.stop_loss_price
        elif position.take_profit_stage == 1:
            target_price = round(position.entry_price * (1 + risk_pct * cfg.staged_tp_stage2_r * tp_mult), 8)
            exit_fraction = cfg.staged_tp_stage2_fraction / max(1e-9, (1 - cfg.staged_tp_stage1_fraction))
            next_stage = 2
            next_tp = round(position.entry_price * (1 + risk_pct * cfg.staged_tp_stage3_r * tp_mult), 8)
            next_stop = round(position.entry_price * (1 + risk_pct * 0.5), 8)
        elif position.take_profit_stage == 2:
            # Stage 2+: trailing stop logic
            drawdown = (position.high_watermark_pnl_pct - ((current.mark_price - position.entry_price) / position.entry_price)) if position.high_watermark_pnl_pct > 0 else 0.0
            if drawdown >= cfg.staged_tp_trailing_pct:
                return (
                    TradePlan(
                        action="exit",
                        symbol=position.symbol,
                        score=current.score,
                        entry_price=position.entry_price,
                        stop_loss_price=position.stop_loss_price,
                        take_profit_price=current.mark_price,
                        notional_usdt=position.notional_usdt,
                        rationale="staged_trailing_stop",
                    ),
                    None,
                )
            return None, position
        else:
            return None, position

        if current.mark_price < target_price:
            return None, position

        partial_notional = round(position.notional_usdt * exit_fraction, 8)
        remaining_notional = round(position.notional_usdt - partial_notional, 8)
        if partial_notional <= 0 or remaining_notional <= 0:
            return None, position

        updated_position = Position(
            symbol=position.symbol,
            opened_at=position.opened_at,
            entry_price=position.entry_price,
            current_price=current.mark_price,
            stop_loss_price=next_stop,
            take_profit_price=next_tp,
            notional_usdt=remaining_notional,
            score=current.score,
            rationale=position.rationale,
            setup_type=position.setup_type,
            take_profit_stage=next_stage,
            initial_risk_pct=risk_pct,
            structure_timeframe=position.structure_timeframe,
            last_sentiment_score=current.sentiment_score,
            last_sentiment_mentions=current.sentiment_mentions,
            last_oi_change_pct_4h=current.oi_change_pct_4h,
            last_taker_buy_sell_ratio=current.taker_buy_sell_ratio,
            entry_coin_features=position.entry_coin_features,
            entry_kline_pattern_bullish_count=position.entry_kline_pattern_bullish_count,
            entry_kline_pattern_bearish_count=position.entry_kline_pattern_bearish_count,
            entry_kline_pattern_flags=position.entry_kline_pattern_flags,
            entry_h4_structure=position.entry_h4_structure,
            pyramid_count=position.pyramid_count,
            predictive_snapshot=position.predictive_snapshot,
            high_watermark_pnl_pct=position.high_watermark_pnl_pct,
        )
        return (
            TradePlan(
                action="take_profit_partial",
                symbol=position.symbol,
                score=current.score,
                entry_price=position.entry_price,
                stop_loss_price=next_stop,
                take_profit_price=target_price,
                notional_usdt=partial_notional,
                rationale=f"staged_tp_stage_{next_stage}",
                size_fraction=round(partial_notional / max(position.notional_usdt, 1e-9), 8),
            ),
            updated_position,
        )
    def _maybe_scale_out_structure_levels(self, position: Position, current: MarketSnapshot) -> tuple[TradePlan | None, Position | None]:
        levels = position.entry_h4_structure or {}
        try:
            tp1 = float(levels.get("structural_tp1_price") or 0.0)
            tp2 = float(levels.get("structural_tp2_price") or 0.0)
            runner = float(levels.get("structural_runner_price") or 0.0)
        except (TypeError, ValueError):
            return None, position
        if tp1 <= 0 or tp2 <= 0:
            return None, position

        if position.take_profit_stage == 0:
            target_price = tp1
            exit_fraction = self.config.breakout_first_exit_fraction
            next_stage = 1
            next_take_profit = tp2
            next_stop = max(position.stop_loss_price, position.entry_price)
            rationale = "structure_scaleout_stage_1"
        elif position.take_profit_stage == 1:
            target_price = tp2
            exit_fraction = self.config.breakout_second_exit_fraction / max(1e-9, (1 - self.config.breakout_first_exit_fraction))
            next_stage = BREAKOUT_RUNNER_STAGE
            next_take_profit = runner if runner > tp2 else round(tp2 * 1.05, 8)
            next_stop = max(position.stop_loss_price, tp1)
            rationale = "structure_scaleout_stage_2"
        else:
            return None, position

        if current.mark_price < target_price:
            return None, position

        partial_notional = round(position.notional_usdt * exit_fraction, 8)
        remaining_notional = round(position.notional_usdt - partial_notional, 8)
        if partial_notional <= 0 or remaining_notional <= 0:
            return None, position

        updated_position = Position(
            symbol=position.symbol,
            opened_at=position.opened_at,
            entry_price=position.entry_price,
            current_price=current.mark_price,
            stop_loss_price=round(next_stop, 8),
            take_profit_price=round(next_take_profit, 8),
            notional_usdt=remaining_notional,
            score=current.score,
            rationale=position.rationale,
            setup_type=position.setup_type,
            take_profit_stage=next_stage,
            initial_risk_pct=position.initial_risk_pct,
            structure_timeframe=position.structure_timeframe,
            last_sentiment_score=current.sentiment_score,
            last_sentiment_mentions=current.sentiment_mentions,
            last_oi_change_pct_4h=current.oi_change_pct_4h,
            last_taker_buy_sell_ratio=current.taker_buy_sell_ratio,
            entry_coin_features=position.entry_coin_features,
            entry_kline_pattern_bullish_count=position.entry_kline_pattern_bullish_count,
            entry_kline_pattern_bearish_count=position.entry_kline_pattern_bearish_count,
            entry_kline_pattern_flags=position.entry_kline_pattern_flags,
            entry_h4_structure=position.entry_h4_structure,
            pyramid_count=position.pyramid_count,
            predictive_snapshot=position.predictive_snapshot,
            high_watermark_pnl_pct=position.high_watermark_pnl_pct,
        )
        return (
            TradePlan(
                action="take_profit_partial",
                symbol=position.symbol,
                score=current.score,
                entry_price=position.entry_price,
                stop_loss_price=round(next_stop, 8),
                take_profit_price=target_price,
                notional_usdt=partial_notional,
                rationale=rationale,
                size_fraction=round(partial_notional / max(position.notional_usdt, 1e-9), 8),
            ),
            updated_position,
        )

    def _maybe_pyramid_add(self, position: Position, current: MarketSnapshot) -> Position:
        """PDF 金字塔加仓：盈利 > 2% 且信号仍强时加仓 50%，止损移至成本价。"""
        if not self._allows_internal_position_management():
            return position
        if not self.config.pdf_pyramid_enabled:
            return position
        if position.pyramid_count >= self.config.pdf_pyramid_max_count:
            return position
        pnl_pct = (current.mark_price - position.entry_price) / max(position.entry_price, 1e-9)
        if pnl_pct < self.config.pdf_pyramid_min_pnl_pct:
            return position
        if current.score < self.config.pdf_pyramid_min_score:
            return position
        if float(current.kline_direction or 0) <= 0:
            return position
        add_notional = round(position.notional_usdt * self.config.pdf_pyramid_size_fraction, 8)
        new_notional = round(position.notional_usdt + add_notional, 8)
        new_entry = round(
            (position.entry_price * position.notional_usdt + current.mark_price * add_notional) / new_notional, 8
        )
        return Position(
            symbol=position.symbol,
            opened_at=position.opened_at,
            entry_price=new_entry,
            current_price=current.mark_price,
            stop_loss_price=position.entry_price,
            take_profit_price=position.take_profit_price,
            notional_usdt=new_notional,
            score=current.score,
            rationale=position.rationale,
            setup_type=position.setup_type,
            take_profit_stage=position.take_profit_stage,
            initial_risk_pct=position.initial_risk_pct,
            structure_timeframe=position.structure_timeframe,
            last_sentiment_score=current.sentiment_score,
            last_sentiment_mentions=current.sentiment_mentions,
            last_oi_change_pct_4h=current.oi_change_pct_4h,
            last_taker_buy_sell_ratio=current.taker_buy_sell_ratio,
            entry_coin_features=position.entry_coin_features,
            entry_kline_pattern_bullish_count=position.entry_kline_pattern_bullish_count,
            entry_kline_pattern_bearish_count=position.entry_kline_pattern_bearish_count,
            entry_kline_pattern_flags=position.entry_kline_pattern_flags,
            entry_h4_structure=position.entry_h4_structure,
            pyramid_count=position.pyramid_count + 1,
            predictive_snapshot=position.predictive_snapshot,
            high_watermark_pnl_pct=position.high_watermark_pnl_pct,
        )

    def _maybe_trailing_stop(self, position: Position, current: MarketSnapshot) -> Position:
        """PDF 阶梯式移动止损：+1R 移至成本，+2R 移至 +1R。"""
        if not self._allows_internal_position_management():
            return position
        if not self.config.pdf_trailing_stop_enabled:
            return position
        risk_pct = position.initial_risk_pct
        if risk_pct <= 0:
            risk_pct = abs(position.entry_price - position.stop_loss_price) / max(position.entry_price, 1e-9)
        if risk_pct <= 0:
            return position
        pnl_pct = (current.mark_price - position.entry_price) / max(position.entry_price, 1e-9)
        r_multiple = pnl_pct / risk_pct
        if r_multiple >= 2.0:
            new_stop = round(position.entry_price * (1 + risk_pct), 8)
        elif r_multiple >= 1.0:
            new_stop = position.entry_price
        else:
            return position
        if new_stop <= position.stop_loss_price:
            return position
        return Position(
            symbol=position.symbol,
            opened_at=position.opened_at,
            entry_price=position.entry_price,
            current_price=current.mark_price,
            stop_loss_price=new_stop,
            take_profit_price=position.take_profit_price,
            notional_usdt=position.notional_usdt,
            score=position.score,
            rationale=position.rationale,
            setup_type=position.setup_type,
            take_profit_stage=position.take_profit_stage,
            initial_risk_pct=position.initial_risk_pct,
            structure_timeframe=position.structure_timeframe,
            last_sentiment_score=position.last_sentiment_score,
            last_sentiment_mentions=position.last_sentiment_mentions,
            last_oi_change_pct_4h=position.last_oi_change_pct_4h,
            last_taker_buy_sell_ratio=position.last_taker_buy_sell_ratio,
            entry_coin_features=position.entry_coin_features,
            entry_kline_pattern_bullish_count=position.entry_kline_pattern_bullish_count,
            entry_kline_pattern_bearish_count=position.entry_kline_pattern_bearish_count,
            entry_kline_pattern_flags=position.entry_kline_pattern_flags,
            entry_h4_structure=position.entry_h4_structure,
            pyramid_count=position.pyramid_count,
            predictive_snapshot=position.predictive_snapshot,
            high_watermark_pnl_pct=position.high_watermark_pnl_pct,
        )

    def _maybe_tighten_oi_divergence_stop(self, position: Position, current: MarketSnapshot) -> Position:
        if not self._allows_internal_position_management():
            return position
        if not position.rationale.startswith("oi_divergence_probe"):
            return position
        if position.stop_loss_price >= position.entry_price:
            return position
        trigger_price = round(position.entry_price * (1 + self.config.oi_divergence_break_even_trigger_pct), 8)
        if current.mark_price < trigger_price:
            return position
        if not self._is_oi_divergence_supported(current):
            return position
        return Position(
            symbol=position.symbol,
            opened_at=position.opened_at,
            entry_price=position.entry_price,
            current_price=current.mark_price,
            stop_loss_price=position.entry_price,
            take_profit_price=position.take_profit_price,
            notional_usdt=position.notional_usdt,
            score=current.score,
            rationale=position.rationale,
            setup_type=position.setup_type,
            take_profit_stage=position.take_profit_stage,
            initial_risk_pct=position.initial_risk_pct,
            structure_timeframe=position.structure_timeframe,
            last_sentiment_score=current.sentiment_score,
            last_sentiment_mentions=current.sentiment_mentions,
            last_oi_change_pct_4h=current.oi_change_pct_4h,
            last_taker_buy_sell_ratio=current.taker_buy_sell_ratio,
            predictive_snapshot=position.predictive_snapshot,
            high_watermark_pnl_pct=position.high_watermark_pnl_pct,
        )

    def _build_predictive_snapshot(self, symbol: str, reasons: list[str] | None = None) -> dict | None:
        model = self._predictive_model
        if not model or model.get("status") != "ok":
            return None
        symbol_upper = symbol.upper()
        symbol_stats = model.get("symbol_stats", {}).get(symbol_upper)
        if not symbol_stats:
            return None
        result: dict = {"prediction_source": "symbol_stats_and_scoring_reasons"}
        trained_until = model.get("trained_until")
        if trained_until:
            result["model_trained_until"] = trained_until
        result["feature_schema_version"] = model.get("feature_count")
        result["hist_win_rate"] = symbol_stats.get("win_rate")
        result["hist_avg_pnl"] = symbol_stats.get("avg_pnl")
        result["hist_stop_loss_rate"] = symbol_stats.get("stop_loss_rate")
        result["hist_trade_count"] = symbol_stats.get("count")
        if reasons:
            for r in reasons:
                if r.startswith("pred_model_adj="):
                    try:
                        result["pred_model_adj"] = float(r.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                elif r.startswith("sl_risk_warn="):
                    try:
                        result["predicted_stop_loss_risk"] = float(r.split("=")[1])
                    except (ValueError, IndexError):
                        pass
                elif r.startswith("low_win_prob="):
                    try:
                        result["predicted_win_probability"] = float(r.split("=")[1])
                    except (ValueError, IndexError):
                        pass
        return result

    def _load_state(self) -> dict:
        if not self.state_file.exists():
            return {"schema_version": self.config.state_schema_version, "positions": [], "journal": [], "learning_journal": []}
        return load_state(self.state_file)

    def _sanitize_position(self, position: Position, snapshot_map: dict[str, MarketSnapshot], journal: list[dict]) -> Position | None:
        current = snapshot_map.get(position.symbol)
        if current is None or position.entry_price <= 0:
            return position
        deviation_pct = abs(current.mark_price - position.entry_price) / position.entry_price * 100.0
        lower_bound = position.stop_loss_price * (1 - (self.config.price_sanity_max_deviation_pct / 100.0))
        upper_bound = position.take_profit_price * (1 + (self.config.price_sanity_max_deviation_pct / 100.0))
        is_impossible = current.mark_price < lower_bound or current.mark_price > upper_bound
        if deviation_pct <= self.config.price_sanity_max_deviation_pct or not is_impossible:
            return position
        warning = (
            f"PRICE_SANITY_MISMATCH:{position.symbol}:entry={position.entry_price:.8f}:current={current.mark_price:.8f}:deviation={deviation_pct:.2f}%"
        )
        journal.append(
            {
                "timestamp": _utc_now().isoformat(),
                "action": "state_reset",
                "symbol": position.symbol,
                "reason": warning,
                "entry_price": position.entry_price,
                "exit_price": current.mark_price,
                "notional_usdt": position.notional_usdt,
                "size_fraction": 0.0,
                "pnl_pct": 0.0,
            }
        )
        return None

    def _evaluate_missing_snapshot_position(self, position: Position) -> TradePlan | None:
        opened_at = _parse_iso8601(position.opened_at)
        if opened_at is None:
            return TradePlan(
                action="exit",
                symbol=position.symbol,
                score=position.score,
                entry_price=position.entry_price,
                stop_loss_price=position.stop_loss_price,
                take_profit_price=position.take_profit_price,
                notional_usdt=position.notional_usdt,
                rationale="market_snapshot_missing",
            )
        if _utc_now() - opened_at < timedelta(hours=self.config.max_hold_hours):
            return None
        return TradePlan(
            action="exit",
            symbol=position.symbol,
            score=position.score,
            entry_price=position.entry_price,
            stop_loss_price=position.stop_loss_price,
            take_profit_price=position.take_profit_price,
            notional_usdt=position.notional_usdt,
            rationale="market_snapshot_missing_timeout",
        )

    def _assess_market_guard(self, snapshots: list[MarketSnapshot]) -> dict[str, float | bool | int]:
        crash_names = [
            item.symbol for item in snapshots
            if item.price_change_pct_5m <= -abs(self.config.market_flash_crash_drop_pct_5m)
            and item.taker_buy_sell_ratio <= self.config.liquidity_fuse_min_taker_ratio
        ]
        triggered = len(crash_names) >= max(1, self.config.market_flash_crash_symbol_count)
        return {
            "triggered": triggered,
            "crash_symbol_count": len(crash_names),
            "crash_symbols": crash_names,
        }

    def _is_trend_up_regime(self) -> bool:
        return bool(self._last_regime and self._last_regime.get("regime") == "trend_up")

    def _should_use_staged_tp(self) -> bool:
        """Use staged TP when BTC macro is bullish; BTC is the sole directional indicator."""
        if not self.config.staged_tp_enabled:
            return False
        if self._is_trend_up_regime():
            return True
        try:
            import json
            cache_path = self.config.macro_trend_cache_file
            with open(cache_path) as f:
                macro = json.load(f)
            btc_trend = macro.get("btc_trend", "neutral")
            btc_ma210_slope = macro.get("btc_ma210_slope_pct", 0)
            btc_dist_672 = macro.get("btc_distance_ma672_pct", 0)
            if btc_trend == "bullish":
                return True
            if btc_ma210_slope > 0 and btc_dist_672 > 0:
                return True
            return False
        except Exception:
            return False

    def _is_long_position(self, position: Position) -> bool:
        return not (position.rationale.startswith("short_squeeze_fade") or position.rationale.startswith("base_breakdown_short"))

    def _disaster_hard_stop_price(self, entry_price: float, snapshot: MarketSnapshot) -> float:
        """灾难硬止损绝对价: entry * (1 - clamp(ATR比例 * mult, floor, ceil))。"""
        if entry_price <= 0:
            return 0.0
        atr_ratio = float(getattr(snapshot, "kline_atr_pct", 0.0) or 0.0)
        raw = atr_ratio * self.config.disaster_hard_stop_atr_mult
        stop_pct = max(self.config.disaster_hard_stop_floor_pct, min(self.config.disaster_hard_stop_ceil_pct, raw))
        return round(entry_price * (1 - stop_pct), 8)

    def _structural_stop_price(self, snapshot: MarketSnapshot) -> float:
        """结构止损绝对价: H4 range_low 下方留 buffer。无结构数据则返回 0(回退到固定止损)。"""
        range_low = float(getattr(snapshot, "h4_range_low", 0.0) or 0.0)
        if range_low <= 0 or snapshot.mark_price <= 0:
            return 0.0
        candidate = range_low * (1 - self.config.structural_stop_buffer_pct)
        # 结构位必须低于入场价才有意义(否则等于立即止损)
        if candidate >= snapshot.mark_price:
            return 0.0
        return round(candidate, 8)

    def _position_hard_stop_price(self, position: Position) -> float:
        """读取入场时锚定的灾难硬止损价(兼容旧持仓: 缺字段则按入场价+ceil回退)。"""
        levels = position.entry_h4_structure or {}
        try:
            hs = float(levels.get("hard_stop_price") or 0.0)
        except (TypeError, ValueError):
            hs = 0.0
        if hs > 0:
            return hs
        if position.entry_price > 0:
            return round(position.entry_price * (1 - self.config.disaster_hard_stop_ceil_pct), 8)
        return 0.0

    def _position_structural_stop_price(self, position: Position) -> float:
        """读取入场时锚定的结构止损价(缺字段则返回0, 不启用前置结构止损)。"""
        levels = position.entry_h4_structure or {}
        try:
            ss = float(levels.get("structural_stop_price") or 0.0)
        except (TypeError, ValueError):
            ss = 0.0
        return ss if ss > 0 else 0.0

    def _staged_take_profit_triggered(self, position: Position, current: MarketSnapshot) -> bool:
        if not self._should_use_staged_tp():
            return False
        if not self._allows_staged_take_profit_management():
            return False
        if not self._is_long_position(position):
            return False
        risk_pct = position.initial_risk_pct if position.initial_risk_pct > 0 else self._position_risk_pct(position)
        if risk_pct <= 0:
            return False
        cfg = self.config
        tp_mult = cfg.trend_up_tp_extension_multiplier if self._is_trend_up_regime() else 1.0
        if position.take_profit_stage == 0:
            target_price = round(position.entry_price * (1 + risk_pct * cfg.staged_tp_stage1_r * tp_mult), 8)
            return current.mark_price >= target_price
        if position.take_profit_stage == 1:
            target_price = round(position.entry_price * (1 + risk_pct * cfg.staged_tp_stage2_r * tp_mult), 8)
            return current.mark_price >= target_price
        if position.take_profit_stage == 2:
            pnl_pct = (current.mark_price - position.entry_price) / max(position.entry_price, 1e-9)
            drawdown = (position.high_watermark_pnl_pct - pnl_pct) if position.high_watermark_pnl_pct > 0 else 0.0
            return drawdown >= cfg.staged_tp_trailing_pct
        return False

    def _should_exit(self, position: Position, current: MarketSnapshot) -> str | None:
        opened_at = _parse_iso8601(position.opened_at) or _utc_now()
        # === 灾难硬止损 (2026-05-30): 不可逾越的底线, 任何 defer/hold 都不能拦, 防 -94% 灾难 ===
        if self.config.disaster_hard_stop_enabled and self._is_long_position(position):
            hard_stop = self._position_hard_stop_price(position)
            if hard_stop > 0 and current.mark_price <= hard_stop:
                return "disaster_hard_stop"
        if position.rationale.startswith("short_squeeze_fade") or position.rationale.startswith("base_breakdown_short"):
            if current.mark_price >= position.stop_loss_price:
                return "stop_loss_hit"
            if current.mark_price <= position.take_profit_price:
                return "take_profit_hit"
        else:
            # === 结构止损前置 (2026-05-30): 入场即锚定 range_low 下方, 不等破位确认 ===
            structural_stop = self._position_structural_stop_price(position)
            if (
                structural_stop > 0
                and current.mark_price <= structural_stop
                and current.mark_price > position.stop_loss_price
            ):
                return "structural_stop_hit"
            if current.mark_price <= position.stop_loss_price:
                if (
                    not self._is_structure_managed_position(position)
                    and self._allows_internal_position_management()
                    and self._should_defer_stop_for_intact_4h_structure(position, current)
                ):
                    return None
                return "stop_loss_hit"
            if position.rationale.startswith("oi_divergence_probe"):
                if current.mark_price >= position.take_profit_price:
                    return "take_profit_hit"
                if not self._is_oi_divergence_supported(current) and current.mark_price > position.entry_price:
                    return "oi_divergence_signal_fade"
            uses_structured_scaleout = (
                "r_based_scaleout" in position.rationale
                or "structure_levels_scaleout" in position.rationale
            )
            staged_tp_controls_exit = self._should_use_staged_tp() and not uses_structured_scaleout
            if staged_tp_controls_exit and self._staged_take_profit_triggered(position, current):
                return "staged_take_profit"
            if (
                position.take_profit_stage >= 1
                and not self._is_breakout_sentiment_strong(current)
                and "structure_levels_scaleout" not in position.rationale
            ):
                return "sentiment_take_profit"
            if position.take_profit_stage < BREAKOUT_RUNNER_STAGE:
                tp_price = position.take_profit_price
                if self._is_trend_up_regime() and self._is_long_position(position):
                    tp_distance = position.take_profit_price - position.entry_price
                    tp_price = round(position.entry_price + tp_distance * self.config.trend_up_tp_extension_multiplier, 8)
                if current.mark_price >= tp_price:
                    if uses_structured_scaleout:
                        return None
                    if staged_tp_controls_exit:
                        return None
                    return "staged_take_profit" if self._should_use_staged_tp() else "take_profit_hit"
            if position.take_profit_stage >= BREAKOUT_RUNNER_STAGE:
                if self._is_breakout_structure_broken(current):
                    return "structure_take_profit"
                if current.mark_price >= position.take_profit_price:
                    return "take_profit_hit"
        exit_thresh = self.config.exit_threshold
        if self._is_trend_up_regime() and self._is_long_position(position):
            exit_thresh = self.config.trend_up_exit_threshold
        if current.score < exit_thresh:
            if self._is_long_position(position):
                pnl_pct = (current.mark_price - position.entry_price) / max(position.entry_price, 1e-9)
            else:
                pnl_pct = (position.entry_price - current.mark_price) / max(position.entry_price, 1e-9)
            if pnl_pct > self.config.pdf_protect_profit_pct:
                if position.high_watermark_pnl_pct > self.config.score_faded_profit_protect_threshold:
                    drawdown_from_peak = position.high_watermark_pnl_pct - pnl_pct
                    if drawdown_from_peak > position.high_watermark_pnl_pct * self.config.score_faded_profit_drawdown_exit:
                        return "score_faded_profit_drawdown"
                return None
            if self._should_hold_score_faded_trend_runner(position, current, pnl_pct):
                return None
            if self._should_hold_structure_managed_position(position, current):
                return None
            if self._should_hold_score_faded_structure_intact(position, current, pnl_pct):
                return None
            return "score_faded"
        if self._has_hold_extension(position, current):
            return None
        max_hold_hours = self.config.max_hold_hours
        if self._is_trend_up_regime() and self._is_long_position(position):
            max_hold_hours = max(max_hold_hours, self.config.trend_up_max_hold_hours)
        if position.rationale.startswith("trend_followthrough"):
            max_hold_hours = max(max_hold_hours, self.config.trend_followthrough_max_hold_hours)
        elif position.rationale.startswith("mania_followthrough"):
            max_hold_hours = max(max_hold_hours, self.config.mania_followthrough_max_hold_hours)
        elif position.rationale.startswith("base_breakout_short") or position.rationale.startswith("short_squeeze_fade"):
            max_hold_hours = min(max_hold_hours, self.config.long_hold_hours)
        if _utc_now() - opened_at >= timedelta(hours=max_hold_hours):
            return "max_hold_reached"
        return None

    def _has_hold_extension(self, position: Position, current: MarketSnapshot) -> bool:
        pnl_pct = (current.mark_price - position.entry_price) / max(position.entry_price, 1e-9)
        sentiment_ok = (
            current.sentiment_score >= self.config.hold_extension_min_sentiment_score
            and current.sentiment_mentions >= self.config.hold_extension_min_mentions
        )
        derivatives_ok = (
            current.oi_change_pct_4h >= self.config.hold_extension_min_oi_change_pct
            and current.taker_buy_sell_ratio >= self.config.hold_extension_min_taker_ratio
        )
        liquidity_ok = current.quote_volume_usdt_24h >= self.config.hold_extension_min_quote_volume_usdt
        has_momentum_or_confirmation = (
            self._has_trusted_sentiment_confirmation(current)
            or current.sentiment_score >= position.last_sentiment_score
            or current.sentiment_mentions >= position.last_sentiment_mentions
            or current.oi_change_pct_4h >= position.last_oi_change_pct_4h
        )
        return bool(
            pnl_pct >= self.config.hold_extension_max_adverse_pnl_pct
            and sentiment_ok
            and derivatives_ok
            and liquidity_ok
            and has_momentum_or_confirmation
        )

    def _should_hold_score_faded_structure_intact(self, position: Position, current: MarketSnapshot, pnl_pct: float) -> bool:
        """Treat score fade as an entry-signal warning, not an exit, while structure is intact."""
        if not self.config.score_faded_structure_guard_enabled:
            return False
        if not self._is_long_position(position):
            return False
        if pnl_pct < self.config.score_faded_structure_guard_max_adverse_pnl_pct:
            return False
        if current.h4_structure_state == "broken" or current.h4_low_break_recent_20bars:
            return False
        if current.price_change_pct_4h <= -abs(self.config.max_dynamic_stop_loss_pct * 100.0):
            return False
        if (
            current.breakout_distance_7d is not None
            and current.breakout_distance_7d > self.config.score_faded_structure_guard_max_breakout_distance
        ):
            return False
        if self._is_momo_15m_anchor_intact(current):
            return True
        return True

    def _should_hold_score_faded_trend_runner(self, position: Position, current: MarketSnapshot, pnl_pct: float) -> bool:
        if not self.config.trend_runner_score_fade_hold_enabled:
            return False
        if not self._is_long_position(position):
            return False
        if pnl_pct < self.config.trend_runner_max_adverse_pnl_pct:
            return False
        if current.score < self.config.trend_runner_min_score_floor:
            return False
        if current.sentiment_score < self.config.trend_runner_min_sentiment_score:
            return False
        if current.oi_change_pct_4h < self.config.trend_runner_min_oi_change_pct_4h:
            return False
        if current.taker_buy_sell_ratio < self.config.trend_runner_min_taker_ratio:
            return False
        if current.quote_volume_usdt_24h < self.config.trend_runner_min_quote_volume_usdt:
            return False
        has_momentum = (
            current.price_change_pct_24h >= self.config.trend_runner_min_price_change_pct_24h
            or current.price_change_pct_4h >= self.config.trend_runner_min_price_change_pct_4h
        )
        if not has_momentum:
            return False
        if current.h4_structure_state in {"broken", "weakening"}:
            return False
        if current.breakout_distance_7d is not None and current.breakout_distance_7d > self.config.trend_runner_max_breakout_distance:
            return False
        if not self._is_trend_runner_structure_intact(current):
            return False
        wr14_ok = current.williams_r_15m_zone in {"overbought", "upper_mid"}
        wr120_ok = current.williams_r_15m_120_zone in {"overbought", "upper_mid", "lower_mid"}
        momo_ok = (
            current.momo_15m_bars >= self.config.trend_runner_min_momo_15m_bars
            and (wr14_ok or wr120_ok)
        )
        momo_anchor_ok = self._is_momo_15m_anchor_intact(current)
        structure_ok = (
            (momo_ok and momo_anchor_ok)
            or current.h4_price_location in {"upper_half", "above_resistance"}
            or (current.price_position_7d is not None and current.price_position_7d >= 0.9)
        )
        return bool(structure_ok)

    def _is_structure_managed_position(self, position: Position) -> bool:
        return "structure_stop" in position.rationale or "structure_levels_scaleout" in position.rationale

    def _should_hold_structure_managed_position(self, position: Position, current: MarketSnapshot) -> bool:
        if not self._is_structure_managed_position(position):
            return False
        if not self._is_long_position(position):
            return False
        if current.mark_price <= position.stop_loss_price:
            return False
        if current.h4_structure_state == "broken" or current.h4_low_break_recent_20bars:
            return False
        return True

    def _allows_internal_position_management(self) -> bool:
        order_mode = str(self.config.execution_order_mode or "").strip().lower()
        execution_mode = str(self.config.execution_mode or "").strip().lower()
        if order_mode and order_mode not in {"auto", "paper", "paper_only", "none"}:
            return False
        return execution_mode in {"paper", "replay"} or order_mode in {"paper", "paper_only"}

    def _allows_staged_take_profit_management(self) -> bool:
        """Allow staged TP where the downstream runner can safely submit partial reduce-only closes."""
        if self._allows_internal_position_management():
            return True
        order_mode = str(self.config.execution_order_mode or "").strip().lower()
        return order_mode in {"testnet_order", "testnet", "demo"}

    def _has_repeated_signal_4h_support(self, snapshot: MarketSnapshot) -> bool:
        if not self.config.repeated_signal_4h_enabled:
            return False
        if snapshot.setup_type not in {"trend", "mania_followthrough", "base_breakout_confirmed", "oi_divergence_long_candidate"}:
            return False
        if snapshot.score < self.config.repeated_signal_4h_min_score:
            return False
        if snapshot.sentiment_mentions < self.config.repeated_signal_4h_min_mentions:
            return False
        if self._sentiment_source_count(snapshot) < self.config.repeated_signal_4h_min_source_count:
            return False
        if snapshot.oi_change_pct_4h < self.config.repeated_signal_4h_min_oi_change_pct:
            return False
        if snapshot.taker_buy_sell_ratio < self.config.repeated_signal_4h_min_taker_ratio:
            return False
        return self._is_4h_structure_intact(snapshot)

    def _has_first_signal_probe(self, snapshot: MarketSnapshot) -> bool:
        if not self.config.first_signal_probe_enabled:
            return False
        is_early_probe = snapshot.score < self.config.entry_threshold or self._has_first_signal_marker(snapshot)
        return bool(
            snapshot.setup_type in {"trend", "mania_followthrough", "oi_divergence_long_candidate"}
            and is_early_probe
            and snapshot.score >= self.config.first_signal_probe_min_score
            and snapshot.sentiment_score >= self.config.first_signal_probe_min_sentiment_score
            and snapshot.sentiment_mentions >= self.config.first_signal_probe_min_mentions
            and snapshot.taker_buy_sell_ratio >= self.config.first_signal_probe_min_taker_ratio
            and snapshot.quote_volume_usdt_24h >= self.config.first_signal_probe_min_quote_volume_usdt
            and (snapshot.breakout_distance_7d is None or snapshot.breakout_distance_7d <= self.config.first_signal_probe_max_breakout_distance)
            and self._is_4h_structure_intact(snapshot)
        )

    def _has_retest_add_support(self, snapshot: MarketSnapshot) -> bool:
        if not self.config.retest_add_enabled:
            return False
        return bool(
            self._has_repeated_signal_4h_support(snapshot)
            and self._has_retest_signal_marker(snapshot)
            and (snapshot.price_position_7d is None or snapshot.price_position_7d >= self.config.retest_add_min_price_position)
            and (snapshot.breakout_distance_7d is None or snapshot.breakout_distance_7d <= self.config.repeated_signal_4h_max_breakout_distance)
        )

    def _has_first_signal_source_support(self, snapshot: MarketSnapshot) -> bool:
        if not self._has_first_signal_probe(snapshot):
            return False
        return not self._is_weak_single_source_untrusted(snapshot)

    @staticmethod
    def _has_first_signal_marker(snapshot: MarketSnapshot) -> bool:
        text = " ".join(str(reason).lower() for reason in snapshot.reasons)
        return "first_signal_probe" in text

    @staticmethod
    def _has_retest_signal_marker(snapshot: MarketSnapshot) -> bool:
        text = " ".join(str(reason).lower() for reason in snapshot.reasons)
        return any(marker in text for marker in ("retest", "pullback", "回踩", "回测"))

    def _should_defer_stop_for_intact_4h_structure(self, position: Position, current: MarketSnapshot) -> bool:
        if not self.config.repeated_signal_4h_enabled:
            return False
        if not (
            position.rationale.startswith("trend_followthrough")
            or position.rationale.startswith("mania_followthrough")
            or position.rationale.startswith("structure_stop")
            or position.rationale.startswith("oi_divergence_probe")
        ):
            return False
        adverse = (current.mark_price - position.entry_price) / max(position.entry_price, 1e-9)
        structure_support = self._has_repeated_signal_4h_support(current) or self._is_momo_15m_anchor_intact(current)
        return bool(
            adverse >= self.config.repeated_signal_4h_max_adverse_pnl_pct
            and structure_support
            and (current.oi_change_pct_4h >= self.config.repeated_signal_4h_min_oi_change_pct or current.sentiment_score >= position.last_sentiment_score)
        )

    def _is_4h_structure_intact(self, snapshot: MarketSnapshot) -> bool:
        price_position = snapshot.price_position_20d if snapshot.price_position_20d is not None else snapshot.price_position_7d
        if price_position is not None and price_position < self.config.repeated_signal_4h_min_price_position:
            return False
        if snapshot.breakout_distance_7d is not None and snapshot.breakout_distance_7d > self.config.repeated_signal_4h_max_breakout_distance:
            return False
        if snapshot.price_change_pct_4h <= -abs(self.config.max_dynamic_stop_loss_pct * 100.0):
            return False
        return True

    def _is_trend_runner_structure_intact(self, snapshot: MarketSnapshot) -> bool:
        price_position = snapshot.price_position_20d if snapshot.price_position_20d is not None else snapshot.price_position_7d
        if price_position is not None and price_position < self.config.repeated_signal_4h_min_price_position:
            return False
        if snapshot.price_change_pct_4h <= -abs(self.config.max_dynamic_stop_loss_pct * 100.0):
            return False
        return True

    def _is_momo_15m_anchor_intact(self, snapshot: MarketSnapshot) -> bool:
        if snapshot.momo_15m_bars < self.config.trend_runner_min_bb672_15m_bars:
            return False
        price = snapshot.mark_price
        if price <= 0:
            return False
        if snapshot.bb672_lower_15m > 0 and price < snapshot.bb672_lower_15m:
            return False
        if (
            snapshot.bb210_lower_15m > 0
            and price < snapshot.bb210_lower_15m
            and (snapshot.bb672_mid_15m <= 0 or price < snapshot.bb672_mid_15m)
        ):
            return False
        if snapshot.bb672_mid_15m > 0 and price >= snapshot.bb672_mid_15m:
            return True
        if snapshot.bb210_mid_15m > 0 and price >= snapshot.bb210_mid_15m:
            return True
        if snapshot.bb_price_position_672_15m >= self.config.trend_runner_bb672_15m_min_position:
            return True
        return snapshot.bb_price_position_210 >= self.config.trend_runner_bb210_min_position

    def _is_oi_divergence_supported(self, current: MarketSnapshot) -> bool:
        return bool(
            current.oi_change_pct_4h >= self.config.oi_divergence_hold_min_oi_change_pct
            and current.taker_buy_sell_ratio >= self.config.oi_divergence_hold_min_taker_ratio
            and current.sentiment_score >= self.config.oi_divergence_hold_min_sentiment_score
        )

    def _is_predictive_hard_blocked(self, snapshot: MarketSnapshot) -> bool:
        if not self.config.predictive_hard_block_enabled:
            return False
        win_prob = self._extract_reason_float(snapshot, "low_win_prob")
        sl_risk = self._extract_reason_float(snapshot, "sl_risk_warn")
        if win_prob is None and sl_risk is None:
            return False
        if win_prob is not None and win_prob <= self.config.predictive_soft_block_max_win_prob:
            return True
        if sl_risk is not None and sl_risk >= self.config.predictive_soft_block_min_stop_loss_risk:
            return True
        return bool(
            win_prob is not None
            and sl_risk is not None
            and win_prob <= self.config.predictive_hard_block_max_win_prob
            and sl_risk >= self.config.predictive_hard_block_min_stop_loss_risk
        )

    def _is_hf_quality_blocked(self, symbol: str, side: str = "long") -> bool:
        quality = self._hf_quality_lookup.get(symbol.upper(), {})
        by_side = quality.get("by_side")
        if by_side is not None:
            directional = by_side.get(side)
            if directional is None:
                return False
        else:
            directional = quality
        sample_count = int(directional.get("hf_signal_count_total", 0))
        if sample_count < self.config.hf_entry_block_min_samples:
            return False
        win_rate = float(directional.get("hf_win_rate_3bar", 0.5))
        return win_rate <= self.config.hf_entry_block_max_win_rate

    _SHORT_SETUP_TYPES = {"squeeze_short_candidate", "base_breakdown_confirmed"}

    @staticmethod
    def _infer_entry_side(snapshot: MarketSnapshot) -> str:
        if snapshot.setup_type in PaperExecutionEngine._SHORT_SETUP_TYPES:
            return "short"
        return "long"

    def _directional_hf_quality(self, snapshot: MarketSnapshot) -> dict:
        """Return directional HF quality metrics for journal storage."""
        quality = self._hf_quality_lookup.get(snapshot.symbol.upper(), {})
        side = self._infer_entry_side(snapshot)
        by_side = quality.get("by_side")
        if by_side is not None:
            directional = by_side.get(side)
            if directional is None:
                return {"hf_entry_side": side}
            result = dict(directional)
        else:
            result = {k: v for k, v in quality.items()}
        result["hf_entry_side"] = side
        return result

    @staticmethod
    def _extract_reason_float(snapshot: MarketSnapshot, key: str) -> float | None:
        pattern = re.compile(rf"(?:^|[,\s]){re.escape(key)}=([-+]?\d+(?:\.\d+)?)")
        for reason in snapshot.reasons:
            match = pattern.search(str(reason))
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
        return None

    @staticmethod
    def _has_trusted_sentiment_confirmation(snapshot: MarketSnapshot) -> bool:
        source = (snapshot.sentiment_source or "").lower()
        return any(item in source for item in BREAKOUT_SENTIMENT_SOURCES)

    @staticmethod
    def _has_primary_conviction_source(snapshot: MarketSnapshot) -> bool:
        source = (snapshot.sentiment_source or "").lower()
        return any(item in source for item in ("manual_distill", "manual_sentiment"))

    @staticmethod
    def _sentiment_source_count(snapshot: MarketSnapshot) -> int:
        return len({item.strip().lower() for item in (snapshot.sentiment_source or "").split(",") if item.strip()})

    def _is_weak_single_source_untrusted(self, snapshot: MarketSnapshot) -> bool:
        tokens = {item.strip().lower() for item in (snapshot.sentiment_source or "").split(",") if item.strip()}
        if not tokens:
            return True
        meaningful_tokens = tokens.difference(WEAK_DISCOVERY_SOURCES)
        if meaningful_tokens:
            return False
        return len(tokens) == 1

    def _is_discovery_only_untrusted(self, snapshot: MarketSnapshot) -> bool:
        tokens = {item.strip().lower() for item in (snapshot.sentiment_source or "").split(",") if item.strip()}
        return bool(tokens) and tokens.issubset(WEAK_DISCOVERY_SOURCES)

    def _is_breakout_sentiment_strong(self, current: MarketSnapshot) -> bool:
        has_external_confirmation = self._has_trusted_sentiment_confirmation(current)
        return bool(
            has_external_confirmation
            and current.sentiment_score >= self.config.breakout_retain_min_sentiment_score
            and current.sentiment_mentions >= self.config.breakout_retain_min_mentions
            and current.oi_change_pct_4h >= self.config.breakout_retain_min_oi_change_pct
            and current.taker_buy_sell_ratio >= self.config.breakout_retain_min_taker_ratio
        )

    @staticmethod
    def _is_breakout_structure_broken(current: MarketSnapshot) -> bool:
        return bool(
            (current.price_position_7d is not None and current.price_position_7d < 0.82)
            or (current.compression_score is not None and current.compression_score < 0.45)
            or (current.breakout_distance_7d is not None and current.breakout_distance_7d > 0.06)
        )

    def _breakout_risk_pct(self, snapshot: MarketSnapshot) -> float:
        raw = (snapshot.breakout_distance_7d or 0.0) * self.config.breakout_risk_multiple_from_distance
        return round(min(self.config.breakout_stop_loss_cap_pct, max(self.config.breakout_stop_loss_floor_pct, raw)), 8)

    @staticmethod
    def _position_risk_pct(position: Position) -> float:
        if position.entry_price <= 0:
            return 0.0
        return round(max(0.0, (position.entry_price - position.stop_loss_price) / position.entry_price), 8)

    @staticmethod
    def _position_from_dict(item: dict) -> Position | None:
        try:
            return Position(
                symbol=item["symbol"],
                opened_at=item["opened_at"],
                entry_price=float(item["entry_price"]),
                current_price=float(item["current_price"]),
                stop_loss_price=float(item["stop_loss_price"]),
                take_profit_price=float(item["take_profit_price"]),
                notional_usdt=float(item["notional_usdt"]),
                score=float(item["score"]),
                rationale=str(item.get("rationale", "")),
                setup_type=str(item.get("setup_type", PaperExecutionEngine._infer_setup_type(item))),
                take_profit_stage=int(item.get("take_profit_stage", 0)),
                initial_risk_pct=float(item.get("initial_risk_pct", 0.0) or 0.0),
                structure_timeframe=str(item.get("structure_timeframe") or "daily"),
                last_sentiment_score=float(item.get("last_sentiment_score", 0.0) or 0.0),
                last_sentiment_mentions=int(float(item.get("last_sentiment_mentions", 0.0) or 0.0)),
                last_oi_change_pct_4h=float(item.get("last_oi_change_pct_4h", 0.0) or 0.0),
                last_taker_buy_sell_ratio=float(item.get("last_taker_buy_sell_ratio", 0.0) or 0.0),
                entry_coin_features=dict(item.get("entry_coin_features") or item.get("coin_features") or {}) or None,
                entry_kline_pattern_bullish_count=float(item.get("entry_kline_pattern_bullish_count") or 0.0),
                entry_kline_pattern_bearish_count=float(item.get("entry_kline_pattern_bearish_count") or 0.0),
                entry_kline_pattern_flags=str(item.get("entry_kline_pattern_flags") or ""),
                entry_h4_structure=item.get("entry_h4_structure"),
                state_warning=str(item.get("state_warning", "")),
                predictive_snapshot=item.get("predictive_snapshot"),
                high_watermark_pnl_pct=float(item.get("high_watermark_pnl_pct", 0.0) or 0.0),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _infer_setup_type(item: dict) -> str:
        setup_type = str(item.get("setup_type") or "").strip()
        if setup_type:
            return setup_type
        risk_band = str(item.get("risk_band") or "").strip().lower()
        if risk_band == "oi_divergence":
            return "oi_divergence_long_candidate"
        if risk_band == "base_breakout":
            return "base_breakout_confirmed"
        reason = str(item.get("reason") or item.get("rationale") or "")
        if reason.startswith("oi_divergence_probe"):
            return "oi_divergence_long_candidate"
        if reason.startswith("structure_stop"):
            return "base_breakout_confirmed"
        if reason.startswith("mania_followthrough"):
            return "mania_followthrough"
        if reason.startswith("short_squeeze_fade"):
            return "squeeze_short_candidate"
        return "trend"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso8601(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None
