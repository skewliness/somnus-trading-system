from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from trading_system.entry_gate_policy import EntryGateFinding, evaluate_entry_gate_findings
from trading_system.v6_execution import SYMBOL_BB_ENTRY_PROFILES, V6ExecutionConfig, V6ExecutionEngine, V6Signal


class EntryGatePolicyUnitTest(unittest.TestCase):
    def test_later_hard_gate_wins_after_soft_gate(self) -> None:
        decision = evaluate_entry_gate_findings(
            [
                EntryGateFinding("profile_unlisted", "soft", notional_multiplier=0.85),
                EntryGateFinding("stale_market_data", "hard"),
            ],
            minimum_notional_multiplier=0.60,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.state, "blocked")
        self.assertEqual(decision.blocking_finding.reason, "stale_market_data")
        self.assertEqual(decision.notional_multiplier, 0.85)

    def test_watch_gate_never_becomes_probe_order(self) -> None:
        decision = evaluate_entry_gate_findings(
            [
                EntryGateFinding("macro_mismatch", "soft", notional_multiplier=0.75),
                EntryGateFinding("wait_for_15m_selector", "watch"),
            ]
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.state, "watch")
        self.assertEqual(decision.blocking_finding.reason, "wait_for_15m_selector")


class V6EntryGatePolicyIntegrationTest(unittest.TestCase):
    def make_engine(self, **overrides) -> V6ExecutionEngine:
        self.temp_dir = tempfile.TemporaryDirectory()
        state_dir = Path(self.temp_dir.name)
        config_values = dict(
            symbols=["BTCUSDT"],
            order_mode="dry_run",
            state_dir=state_dir,
            state_file=state_dir / "state.json",
            signals_file=state_dir / "signals.jsonl",
            orders_file=state_dir / "orders.jsonl",
            wal_file=state_dir / "execution_wal.jsonl",
            latest_summary_file=state_dir / "latest_run_summary.json",
            closed_trade_review_queue_file=state_dir / "closed_trade_review_queue.jsonl",
            post_trade_training_samples_file=state_dir / "post_trade_training_samples.jsonl",
            kline_cache_dir=state_dir / "klines",
            entry_gate_events_file=state_dir / "entry_gate_events.jsonl",
            technical_entry_only_symbols=("BTCUSDT",),
            entry_gate_policy_v2_enabled=True,
            mixed_structure_macro_arbitration_enabled=True,
            base_notional_usdt=510.0,
            min_notional_usdt=510.0,
        )
        config_values.update(overrides)
        config = V6ExecutionConfig(**config_values)
        return V6ExecutionEngine(config)

    @staticmethod
    def make_signal(*, stale: bool = False) -> V6Signal:
        return V6Signal(
            symbol="BTCUSDT",
            computed_at="2026-07-15T00:00:00+00:00",
            price=100.0,
            side_bias="short",
            long_probability=0.30,
            short_probability=0.45,
            wait_probability=0.25,
            model_quality_score=0.5,
            reasons=[],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            levels={
                "price": 100.0,
                "structure_entry_scan": {
                    "long_candidates": [
                        {
                            "name": "bb999_lower_15m",
                            "price": 99.5,
                            "distance_pct": 0.502513,
                            "abs_distance_pct": 0.502513,
                            "period": 999,
                            "band": "lower",
                            "timeframe": "15m",
                            "structure_signal": "long_lower_support_rebound",
                        }
                    ]
                },
            },
            indicators={"recent_extremes": {"1m": {"low": 99.0}}},
            market_data_freshness={"stale": stale, "stale_sources": ["15m"]} if stale else {},
        )

    def tearDown(self) -> None:
        temp_dir = getattr(self, "temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()

    def test_unlisted_profile_candidate_is_retained_and_soft_sized(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_long", context)
            intent = engine.open_intent(signal, "open_long", entry_context=context)
        finally:
            engine.close()

        self.assertEqual(context["selected_entry_level"]["name"], "bb999_lower_15m")
        self.assertEqual(context["symbol_bb_entry_profile"]["mode"], "soft_prior")
        self.assertEqual(context["symbol_bb_entry_profile"]["candidate_count_after_profile"], 1)
        self.assertEqual(context["symbol_bb_entry_profile"]["blocked_candidate_count"], 0)
        self.assertEqual(context["symbol_bb_entry_profile"]["unlisted_candidate_count"], 1)
        self.assertTrue(decision["allowed"])
        self.assertIn("symbol_bb_entry_profile_selected_level_not_allowed", decision["soft_reasons"])
        self.assertLess(intent.quantity_hint_usdt, 510.0)
        self.assertGreaterEqual(intent.quantity_hint_usdt, 510.0 * 0.30)

    def test_soft_profile_and_macro_do_not_skip_later_stale_data_hard_gate(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal(stale=True)
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_long", context)
            block = engine.entry_gate_policy_v2_block(decision, context)
        finally:
            engine.close()

        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["state"], "blocked")
        self.assertEqual(decision["blocking_reason"], "entry_blocked_stale_market_data")
        self.assertIn("macro_direction_mismatch", decision["soft_reasons"])
        self.assertIsNotNone(block)
        self.assertEqual(block["reason"], "entry_blocked_stale_market_data")

    @staticmethod
    def add_mixed_structure(signal: V6Signal) -> None:
        scan = signal.levels["structure_entry_scan"]
        scan["short_candidates"] = [
            {
                "name": "bb120_upper_15m",
                "price": 100.5,
                "distance_pct": -0.497512,
                "abs_distance_pct": 0.497512,
                "period": 120,
                "band": "upper",
                "timeframe": "15m",
                "structure_signal": "short_upper_rejection",
            }
        ]

    @staticmethod
    def set_daily_4h_conflict(
        signal: V6Signal,
        *,
        daily_direction: str,
        h4_direction: str,
        weakened: bool,
    ) -> None:
        signal.levels["daily_ma7_trend"] = {
            "ready": True,
            "bullish_stack": daily_direction == "long",
            "bearish_stack": daily_direction == "short",
            "long_confirmed": False,
            "short_confirmed": False,
        }
        weakening = {
            "enabled": True,
            "required_closed_candles": 2,
            "checked_closed_candles": 2,
            "long_logic_weakened": weakened and h4_direction == "long",
            "short_logic_weakened": weakened and h4_direction == "short",
            "long_weakened_count": 2 if weakened and h4_direction == "long" else 0,
            "short_weakened_count": 2 if weakened and h4_direction == "short" else 0,
        }
        right_side = {
            "ready": True,
            "long_confirmed": h4_direction == "long",
            "short_confirmed": h4_direction == "short",
            "state": f"{h4_direction}_right_side",
            "logic_weakening_exit": weakening,
        }
        signal.levels["h4_right_side"] = right_side
        signal.h4 = {
            "above_ma20": h4_direction == "long",
            "ma20_slope_pct": 0.5 if h4_direction == "long" else -0.5,
            "right_side": right_side,
        }

    @staticmethod
    def add_manager_reversal_observation(
        state: dict,
        *,
        closed_side: str = "short",
        created_at: str = "2026-07-01T00:00:00+00:00",
        minimum_until: str = "2026-07-01T04:00:00+00:00",
        expires_at: str = "2099-07-22T00:00:00+00:00",
    ) -> None:
        target_side = "long" if closed_side == "short" else "short"
        state["manager_reversal_observations"]["BTCUSDT"] = {
            "symbol": "BTCUSDT",
            "closed_side": closed_side,
            "target_side": target_side,
            "reason_type": "logic_weakening_exit",
            "timeframe": "4h",
            "created_at": created_at,
            "minimum_until": minimum_until,
            "expires_at": expires_at,
        }

    @staticmethod
    def confirm_h4_reversal(signal: V6Signal, *, target_side: str, last_closed_open_time: int = 1782882000000) -> None:
        right_side = {
            "ready": True,
            "long_confirmed": target_side == "long",
            "short_confirmed": target_side == "short",
            "state": f"{target_side}_right_side",
            "last_closed_open_time": last_closed_open_time,
            "last_closed_close_time": last_closed_open_time + 14_400_000,
            "logic_weakening_exit": {
                "enabled": True,
                "long_logic_weakened": False,
                "short_logic_weakened": False,
                "long_weakened_count": 0,
                "short_weakened_count": 0,
            },
        }
        signal.levels["h4_right_side"] = right_side
        signal.h4 = {"right_side": right_side}

    def test_mixed_structure_hard_blocks_macro_opposite_entry(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            block = engine.macro_direction_entry_block(signal, "open_long", context)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_long", context)
        finally:
            engine.close()

        self.assertIsNotNone(block)
        self.assertEqual(block["reason"], "mixed_structure_macro_direction_mismatch")
        self.assertEqual(block["effective_bias"], "short")
        self.assertTrue(block["structure_conflict"]["active"])
        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["blocking_reason"], "mixed_structure_macro_direction_mismatch")
        self.assertIn("mixed_structure_macro_direction_mismatch", decision["hard_reasons"])

    def test_mixed_structure_allows_only_macro_aligned_entry(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        try:
            context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            block = engine.macro_direction_entry_block(signal, "open_short", context)
        finally:
            engine.close()

        self.assertIsNone(block)
        self.assertEqual(context["mixed_structure_macro_arbitration"]["macro_direction"], "short")
        self.assertEqual(context["mixed_structure_macro_arbitration"]["allowed_action"], "open_short")
        self.assertTrue(context["mixed_structure_macro_arbitration"]["aligned"])

    def test_mixed_structure_blocks_both_sides_when_macro_is_neutral(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        signal.side_bias = "neutral"
        signal.long_probability = 0.40
        signal.short_probability = 0.40
        self.add_mixed_structure(signal)
        try:
            long_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            short_context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            long_block = engine.macro_direction_entry_block(signal, "open_long", long_context)
            short_block = engine.macro_direction_entry_block(signal, "open_short", short_context)
        finally:
            engine.close()

        self.assertEqual(long_block["reason"], "mixed_structure_macro_direction_neutral")
        self.assertEqual(short_block["reason"], "mixed_structure_macro_direction_neutral")

    def test_mixed_structure_uses_daily_macro_before_model_probability(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        signal.levels["daily_ma7_trend"] = {
            "ready": True,
            "bullish_stack": True,
            "bearish_stack": False,
            "long_confirmed": False,
            "short_confirmed": False,
        }
        try:
            long_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            short_context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            long_block = engine.macro_direction_entry_block(signal, "open_long", long_context)
            short_block = engine.macro_direction_entry_block(signal, "open_short", short_context)
        finally:
            engine.close()

        self.assertIsNone(long_block)
        self.assertEqual(long_context["mixed_structure_macro_arbitration"]["macro_source"], "daily_ma7_above_ma20")
        self.assertTrue(long_context["mixed_structure_macro_arbitration"]["aligned"])
        self.assertEqual(short_block["reason"], "mixed_structure_macro_direction_mismatch")
        self.assertEqual(short_block["effective_bias"], "long")
        self.assertEqual(short_block["macro_source"], "daily_ma7_above_ma20")

    def test_mixed_structure_conflicting_daily_macro_uses_higher_probability_side(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        signal.levels["daily_ma7_trend"] = {
            "ready": True,
            "bullish_stack": True,
            "bearish_stack": True,
            "long_confirmed": False,
            "short_confirmed": False,
        }
        try:
            long_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            short_context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            long_block = engine.macro_direction_entry_block(signal, "open_long", long_context, state={})
            short_block = engine.macro_direction_entry_block(signal, "open_short", short_context, state={})
        finally:
            engine.close()

        self.assertEqual(long_block["reason"], "mixed_structure_macro_direction_mismatch")
        self.assertIsNone(short_block)
        arbitration = short_context["mixed_structure_macro_arbitration"]
        self.assertTrue(arbitration["probability_fallback"])
        self.assertEqual(arbitration["macro_state_source"], "daily_ma_stack_conflict")
        self.assertEqual(arbitration["macro_direction"], "short")
        self.assertEqual(arbitration["selected_probability"], 0.45)

    def test_cross_timeframe_conflict_keeps_healthy_4h_direction(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=False,
        )
        try:
            long_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            short_context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            long_block = engine.macro_direction_entry_block(signal, "open_long", long_context, state={})
            short_block = engine.macro_direction_entry_block(signal, "open_short", short_context, state={})
        finally:
            engine.close()

        self.assertIsNone(long_block)
        self.assertEqual(short_block["reason"], "mixed_structure_cross_timeframe_4h_direction_active")
        arbitration = long_context["mixed_structure_macro_arbitration"]
        self.assertFalse(arbitration["probability_fallback"])
        self.assertEqual(arbitration["arbitration_mode"], "daily_4h_weakening")
        self.assertEqual(arbitration["allowed_action"], "open_long")

    def test_active_4h_long_downgrades_conflicting_daily_long_invalidation_gate(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=False,
        )
        signal.levels["daily_ma7_trend"].update({
            "ma7_loss_held": True,
            "ma20_loss_held": True,
            "long_reversal_on_ma20_loss": True,
        })
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_long", context)
        finally:
            engine.close()

        self.assertTrue(decision["allowed"])
        self.assertEqual(context["mixed_structure_macro_arbitration"]["status"], "active_4h")
        self.assertIn("long_blocked_daily_ma20_loss", decision["soft_reasons"])
        self.assertNotIn("long_blocked_daily_ma20_loss", decision["hard_reasons"])
        daily_finding = next(
            item for item in decision["findings"] if item["reason"] == "long_blocked_daily_ma20_loss"
        )
        self.assertEqual(daily_finding["payload"]["precedence"], "active_4h_over_conflicting_daily_gate")

    def test_active_4h_long_entry_is_not_immediately_closed_by_the_same_daily_gate(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=False,
        )
        signal.levels["daily_ma7_trend"].update({
            "ma7_lost": True,
            "ma7_loss_held": True,
            "ma20_lost": True,
            "ma20_loss_held": True,
            "long_reversal_on_ma20_loss": True,
        })
        state: dict = {"positions": []}
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            macro_block = engine.macro_direction_entry_block(signal, "open_long", context, state=state)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_long", context)
            position = {
                "symbol": signal.symbol,
                "side": "long",
                "entry_price": 100.0,
                "stop_loss_price": 90.0,
                "take_profit_price": 120.0,
                "notional_usdt": 100.0,
                "leverage": 8,
                "risk_plan": {
                    "horizon": "15m_scalp",
                    "entry_timeframe": "15m",
                    "entry_context": {
                        "mixed_structure_macro_arbitration": dict(context["mixed_structure_macro_arbitration"]),
                    },
                },
            }

            exit_intent = engine.exit_intent_if_needed(signal, position, state=state)
        finally:
            engine.close()

        self.assertIsNone(macro_block)
        self.assertTrue(decision["allowed"])
        self.assertIsNone(exit_intent)
        self.assertEqual(position["daily_ma20_long_reversal_watch"]["status"], "deferred_active_4h")

    def test_active_4h_short_entry_is_not_immediately_closed_by_the_same_daily_gate(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="long",
            h4_direction="short",
            weakened=False,
        )
        signal.levels["market_daily_ma7_reversal"] = {
            "enabled": True,
            "ready": True,
            "reversal_confirmed": True,
            "ma7_reclaim_confirmed": True,
            "ma20_reclaim_confirmed": True,
            "ma7_held_count": 2,
            "ma20_held_count": 2,
            "held_count": 2,
            "reclaimed_count": 2,
            "market_symbols": ["BTCUSDT", "ETHUSDT"],
            "require_all": True,
        }
        state: dict = {"positions": []}
        try:
            context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            macro_block = engine.macro_direction_entry_block(signal, "open_short", context, state=state)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_short", context)
            position = {
                "symbol": signal.symbol,
                "side": "short",
                "entry_price": 100.0,
                "stop_loss_price": 110.0,
                "take_profit_price": 80.0,
                "notional_usdt": 100.0,
                "leverage": 8,
                "risk_plan": {
                    "horizon": "15m_scalp",
                    "entry_timeframe": "15m",
                    "entry_context": {
                        "mixed_structure_macro_arbitration": dict(context["mixed_structure_macro_arbitration"]),
                    },
                },
            }

            exit_intent = engine.exit_intent_if_needed(signal, position, state=state)
        finally:
            engine.close()

        self.assertIsNone(macro_block)
        self.assertTrue(decision["allowed"])
        self.assertIsNone(exit_intent)
        self.assertEqual(position["daily_ma7_short_reversal_watch"]["status"], "deferred_active_4h")

    def test_daily_exit_reactivates_after_two_closed_4h_weakening_candles(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=False,
        )
        signal.levels["daily_ma7_trend"].update({
            "ma7_lost": True,
            "ma7_loss_held": True,
            "ma20_lost": True,
            "ma20_loss_held": True,
            "long_reversal_on_ma20_loss": True,
        })
        state: dict = {"positions": []}
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            self.assertIsNone(engine.macro_direction_entry_block(signal, "open_long", context, state=state))
            position = {
                "symbol": signal.symbol,
                "side": "long",
                "entry_price": 100.0,
                "stop_loss_price": 90.0,
                "take_profit_price": 120.0,
                "notional_usdt": 100.0,
                "leverage": 8,
                "risk_plan": {
                    "horizon": "15m_scalp",
                    "entry_timeframe": "15m",
                    "entry_context": {
                        "mixed_structure_macro_arbitration": dict(context["mixed_structure_macro_arbitration"]),
                    },
                },
            }

            self.set_daily_4h_conflict(
                signal,
                daily_direction="short",
                h4_direction="long",
                weakened=True,
            )
            signal.levels["daily_ma7_trend"].update({
                "ma7_lost": True,
                "ma7_loss_held": True,
                "ma20_lost": True,
                "ma20_loss_held": True,
                "long_reversal_on_ma20_loss": True,
            })
            engine.refresh_cross_timeframe_macro_observation(signal, state)
            exit_intent = engine.exit_intent_if_needed(signal, position, state=state)
        finally:
            engine.close()

        self.assertIsNotNone(exit_intent)
        self.assertEqual(exit_intent.action, "close_long")
        self.assertIn("daily_ma20_long_reversal_exit", exit_intent.risk_plan)
        self.assertEqual(position["daily_ma20_long_reversal_watch"]["status"], "exit_ready")

    def test_short_daily_exit_reactivates_after_two_closed_4h_weakening_candles(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="long",
            h4_direction="short",
            weakened=False,
        )
        signal.levels["market_daily_ma7_reversal"] = {
            "enabled": True,
            "ready": True,
            "reversal_confirmed": True,
            "ma7_reclaim_confirmed": True,
            "ma20_reclaim_confirmed": True,
            "ma7_held_count": 2,
            "ma20_held_count": 2,
            "held_count": 2,
            "reclaimed_count": 2,
            "market_symbols": ["BTCUSDT", "ETHUSDT"],
            "require_all": True,
        }
        state: dict = {"positions": []}
        try:
            context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            self.assertIsNone(engine.macro_direction_entry_block(signal, "open_short", context, state=state))
            position = {
                "symbol": signal.symbol,
                "side": "short",
                "entry_price": 100.0,
                "stop_loss_price": 110.0,
                "take_profit_price": 80.0,
                "notional_usdt": 100.0,
                "leverage": 8,
                "risk_plan": {
                    "horizon": "15m_scalp",
                    "entry_timeframe": "15m",
                    "entry_context": {
                        "mixed_structure_macro_arbitration": dict(context["mixed_structure_macro_arbitration"]),
                    },
                },
            }

            self.set_daily_4h_conflict(
                signal,
                daily_direction="long",
                h4_direction="short",
                weakened=True,
            )
            signal.levels["market_daily_ma7_reversal"] = {
                "enabled": True,
                "ready": True,
                "reversal_confirmed": True,
                "ma7_reclaim_confirmed": True,
                "ma20_reclaim_confirmed": True,
                "ma7_held_count": 2,
                "ma20_held_count": 2,
                "held_count": 2,
                "reclaimed_count": 2,
                "market_symbols": ["BTCUSDT", "ETHUSDT"],
                "require_all": True,
            }
            engine.refresh_cross_timeframe_macro_observation(signal, state)
            exit_intent = engine.exit_intent_if_needed(signal, position, state=state)
        finally:
            engine.close()

        self.assertIsNotNone(exit_intent)
        self.assertEqual(exit_intent.action, "close_short")
        self.assertIn("daily_ma7_short_reversal_exit", exit_intent.risk_plan)
        self.assertEqual(position["daily_ma7_short_reversal_watch"]["status"], "exit_ready")

    def test_active_4h_short_downgrades_conflicting_market_daily_short_gate(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="long",
            h4_direction="short",
            weakened=False,
        )
        signal.levels["market_daily_ma7_reversal"] = {
            "ready": True,
            "reversal_confirmed": True,
            "market_symbols": ["BTCUSDT", "ETHUSDT"],
            "require_all": True,
        }
        try:
            context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_short", context)
        finally:
            engine.close()

        self.assertTrue(decision["allowed"])
        self.assertEqual(context["mixed_structure_macro_arbitration"]["status"], "active_4h")
        self.assertIn("short_blocked_market_daily_ma20_reclaim", decision["soft_reasons"])
        self.assertNotIn("short_blocked_market_daily_ma20_reclaim", decision["hard_reasons"])
        daily_finding = next(
            item for item in decision["findings"] if item["reason"] == "short_blocked_market_daily_ma20_reclaim"
        )
        self.assertEqual(daily_finding["payload"]["precedence"], "active_4h_over_conflicting_daily_gate")

    def test_daily_gate_remains_hard_after_active_4h_logic_weakens(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=True,
        )
        signal.levels["daily_ma7_trend"].update({
            "ma7_loss_held": True,
            "ma20_loss_held": True,
            "long_reversal_on_ma20_loss": True,
        })
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_long", context)
        finally:
            engine.close()

        self.assertFalse(decision["allowed"])
        self.assertEqual(context["mixed_structure_macro_arbitration"]["status"], "waiting_key_structure_break")
        self.assertIn("long_blocked_daily_ma20_loss", decision["hard_reasons"])
        self.assertNotIn("long_blocked_daily_ma20_loss", decision["soft_reasons"])

    def test_cross_timeframe_conflict_observes_after_two_4h_weakening_candles(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=True,
        )
        try:
            long_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            short_context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            long_block = engine.macro_direction_entry_block(signal, "open_long", long_context, state={})
            short_block = engine.macro_direction_entry_block(signal, "open_short", short_context, state={})
            short_decision = engine.entry_gate_policy_v2_decision(signal, "open_short", short_context)
        finally:
            engine.close()

        self.assertEqual(long_block["reason"], "mixed_structure_cross_timeframe_4h_logic_weakened")
        self.assertEqual(short_block["reason"], "mixed_structure_cross_timeframe_wait_key_structure_break")
        self.assertIsNone(short_context["mixed_structure_macro_arbitration"]["allowed_action"])
        self.assertEqual(short_decision["state"], "watch")
        self.assertEqual(
            short_decision["blocking_reason"],
            "mixed_structure_cross_timeframe_wait_key_structure_break",
        )

    def test_cross_timeframe_observation_survives_current_4h_direction_flip(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        state = engine.load_state()
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=False,
        )
        try:
            first_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            first_block = engine.macro_direction_entry_block(signal, "open_long", first_context, state=state)
            self.assertIsNone(first_block)
            self.assertEqual(
                state["cross_timeframe_macro_observations"]["BTCUSDT"]["status"],
                "active_4h",
            )

            weakening = signal.levels["h4_right_side"]["logic_weakening_exit"]
            weakening.update({
                "long_logic_weakened": True,
                "long_weakened_count": 2,
            })
            signal.levels["h4_right_side"].update({
                "long_confirmed": False,
                "short_confirmed": True,
                "state": "short_right_side",
            })
            signal.h4.update({
                "above_ma20": False,
                "ma20_slope_pct": -0.5,
                "right_side": signal.levels["h4_right_side"],
            })

            long_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            short_context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            long_block = engine.macro_direction_entry_block(signal, "open_long", long_context, state=state)
            short_block = engine.macro_direction_entry_block(signal, "open_short", short_context, state=state)
        finally:
            engine.close()

        self.assertEqual(long_block["reason"], "mixed_structure_cross_timeframe_4h_logic_weakened")
        self.assertEqual(short_block["reason"], "mixed_structure_cross_timeframe_wait_key_structure_break")
        observation = state["cross_timeframe_macro_observations"]["BTCUSDT"]
        self.assertEqual(observation["original_h4_direction"], "long")
        self.assertEqual(observation["current_h4_direction"], "short")
        self.assertEqual(observation["status"], "waiting_key_structure_break")
        self.assertFalse(short_context["mixed_structure_macro_arbitration"]["probability_fallback"])

    def test_persisted_cross_timeframe_observation_allows_reversal_after_structure_break(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        state = engine.load_state()
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=False,
        )
        try:
            first_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            self.assertIsNone(engine.macro_direction_entry_block(signal, "open_long", first_context, state=state))

            weakening = signal.levels["h4_right_side"]["logic_weakening_exit"]
            weakening.update({"long_logic_weakened": True, "long_weakened_count": 2})
            signal.levels["h4_right_side"].update({"long_confirmed": False, "short_confirmed": True})
            signal.h4.update({"above_ma20": False, "ma20_slope_pct": -0.5})
            signal.levels["structure_entry_scan"]["short_candidates"].append({
                "name": "ma20_15m",
                "price": 100.5,
                "distance_pct": -0.497512,
                "abs_distance_pct": 0.497512,
                "period": 20,
                "band": "ma",
                "timeframe": "15m",
                "indicator_family": "ma",
                "source": "ma_cross",
                "previous_close": 101.0,
                "current_close": 100.0,
                "structure_signal": "short_ma_break_cross",
            })
            context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            block = engine.macro_direction_entry_block(signal, "open_short", context, state=state)
            policy_decision = engine.entry_gate_policy_v2_decision(signal, "open_short", context)
        finally:
            engine.close()

        self.assertIsNone(block)
        self.assertEqual(context["selected_entry_level"]["name"], "ma20_15m")
        self.assertEqual(state["cross_timeframe_macro_observations"]["BTCUSDT"]["status"], "reversal_ready")
        self.assertNotEqual(
            policy_decision.get("blocking_reason"),
            "mixed_structure_cross_timeframe_wait_key_structure_break",
        )

    def test_cross_timeframe_observation_refreshes_during_entry_cooldown(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=False,
        )
        state = engine.load_state()
        engine.set_entry_cooldown(
            state,
            signal.symbol,
            hours=1.0,
            reason="test_cooldown",
        )
        try:
            intent = engine.intent_for_signal(signal, state)
        finally:
            engine.close()

        self.assertIsNone(intent)
        observation = state["cross_timeframe_macro_observations"]["BTCUSDT"]
        self.assertEqual(observation["status"], "active_4h")
        self.assertEqual(observation["daily_direction"], "short")
        self.assertEqual(observation["original_h4_direction"], "long")

    def test_entry_cooldown_is_side_specific_and_does_not_block_reversal(self) -> None:
        engine = self.make_engine()
        state = engine.load_state()
        try:
            engine.set_entry_cooldown(
                state,
                "BTCUSDT",
                hours=1,
                reason="short_side_test",
                side="short",
            )

            self.assertTrue(engine.entry_cooldown_active(state, "BTCUSDT", "short"))
            self.assertFalse(engine.entry_cooldown_active(state, "BTCUSDT", "long"))
            self.assertIn("BTCUSDT:short", state["entry_cooldowns"])
        finally:
            engine.close()

    def test_generic_entry_cooldown_still_blocks_both_sides(self) -> None:
        engine = self.make_engine()
        state = engine.load_state()
        try:
            engine.set_entry_cooldown(state, "BTCUSDT", hours=1, reason="generic_test")

            self.assertTrue(engine.entry_cooldown_active(state, "BTCUSDT", "short"))
            self.assertTrue(engine.entry_cooldown_active(state, "BTCUSDT", "long"))
        finally:
            engine.close()

    def test_manager_reentry_lock_blocks_only_invalidated_side(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        state = engine.load_state()
        state["manager_reentry_locks"]["BTCUSDT:short"] = {
            "symbol": "BTCUSDT",
            "side": "short",
            "reason_type": "logic_weakening_exit",
            "timeframe": "4h",
            "created_at": "2026-07-19T00:00:00+00:00",
            "minimum_until": "2099-07-19T04:00:00+00:00",
            "expires_at": "2099-07-22T00:00:00+00:00",
        }
        try:
            short_block = engine.manager_reentry_lock_block(signal, state, "open_short")
            long_block = engine.manager_reentry_lock_block(signal, state, "open_long")
        finally:
            engine.close()

        self.assertIsNotNone(short_block)
        self.assertEqual(short_block["reason"], "manager_reentry_lock_active")
        self.assertIsNone(long_block)

    def test_manager_reentry_lock_is_not_enforced_for_v5_overlay_by_default(self) -> None:
        engine = self.make_engine(v5_overlay_rules_enabled=True)
        signal = self.make_signal()
        state = engine.load_state()
        state["manager_reentry_locks"]["BTCUSDT:short"] = {
            "symbol": "BTCUSDT",
            "side": "short",
            "reason_type": "logic_weakening_exit",
            "timeframe": "4h",
            "created_at": "2026-07-19T00:00:00+00:00",
            "minimum_until": "2099-07-19T04:00:00+00:00",
            "expires_at": "2099-07-22T00:00:00+00:00",
        }
        try:
            block = engine.manager_reentry_lock_block(signal, state, "open_short")
        finally:
            engine.close()

        self.assertIsNone(block)
        self.assertIn("BTCUSDT:short", state["manager_reentry_locks"])

    def test_manager_reentry_lock_requires_fresh_same_side_structure(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        signal.levels["h4_right_side"] = {
            "ready": True,
            "long_confirmed": True,
            "short_confirmed": False,
            "state": "long_right_side",
            "logic_weakening_exit": {
                "enabled": True,
                "long_logic_weakened": False,
                "long_weakened_count": 0,
            },
        }
        state = engine.load_state()
        state["manager_reentry_locks"]["BTCUSDT:long"] = {
            "symbol": "BTCUSDT",
            "side": "long",
            "reason_type": "logic_weakening_exit",
            "timeframe": "4h",
            "created_at": "2026-07-01T00:00:00+00:00",
            "minimum_until": "2026-07-01T04:00:00+00:00",
            "expires_at": "2099-07-22T00:00:00+00:00",
        }
        try:
            block = engine.manager_reentry_lock_block(signal, state, "open_long")
        finally:
            engine.close()

        self.assertIsNone(block)
        self.assertNotIn("BTCUSDT:long", state["manager_reentry_locks"])
        self.assertEqual(state["manager_reentry_lock_history"][-1]["event"], "released")

    def test_stale_market_data_cannot_release_manager_reentry_lock(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal(stale=True)
        signal.levels["h4_right_side"] = {
            "ready": True,
            "long_confirmed": True,
            "short_confirmed": False,
            "logic_weakening_exit": {
                "long_logic_weakened": False,
                "long_weakened_count": 0,
            },
        }
        state = engine.load_state()
        state["manager_reentry_locks"]["BTCUSDT:long"] = {
            "symbol": "BTCUSDT",
            "side": "long",
            "reason_type": "logic_weakening_exit",
            "timeframe": "4h",
            "created_at": "2026-07-01T00:00:00+00:00",
            "minimum_until": "2026-07-01T04:00:00+00:00",
            "expires_at": "2099-07-22T00:00:00+00:00",
        }
        try:
            block = engine.manager_reentry_lock_block(signal, state, "open_long")
        finally:
            engine.close()

        self.assertIsNotNone(block)
        self.assertFalse(block["same_side_revalidated"])
        self.assertIn("BTCUSDT:long", state["manager_reentry_locks"])

    def test_probability_entry_waits_for_manager_reversal_confirmation(self) -> None:
        engine = self.make_engine(
            technical_entry_only_symbols=(),
            entry_structure_filter_enabled=False,
            long_entry_threshold=0.50,
            short_entry_threshold=0.50,
            max_wait_probability_for_entry=0.50,
        )
        signal = self.make_signal()
        signal.long_probability = 0.70
        signal.short_probability = 0.80
        signal.wait_probability = 0.10
        state = engine.load_state()
        state["manager_reentry_locks"]["BTCUSDT:short"] = {
            "symbol": "BTCUSDT",
            "side": "short",
            "reason_type": "logic_weakening_exit",
            "timeframe": "4h",
            "created_at": "2026-07-19T00:00:00+00:00",
            "minimum_until": "2099-07-19T04:00:00+00:00",
            "expires_at": "2099-07-22T00:00:00+00:00",
        }
        self.add_manager_reversal_observation(
            state,
            created_at="2099-07-19T00:00:00+00:00",
            minimum_until="2099-07-19T04:00:00+00:00",
        )
        try:
            intent = engine.intent_for_signal(signal, state)
        finally:
            engine.close()

        self.assertIsNone(intent)
        self.assertEqual(signal.levels["entry_structure_filter"]["reason"], "manager_reversal_observation_active")

    def test_probability_scale_in_cannot_bypass_manager_reversal_confirmation(self) -> None:
        engine = self.make_engine(
            technical_entry_only_symbols=(),
            entry_structure_filter_enabled=False,
            long_entry_threshold=0.50,
            short_entry_threshold=0.50,
            max_wait_probability_for_entry=0.50,
        )
        signal = self.make_signal()
        signal.long_probability = 0.80
        signal.short_probability = 0.20
        signal.wait_probability = 0.10
        state = engine.load_state()
        state["positions"] = [{"symbol": "BTCUSDT", "side": "long"}]
        self.add_manager_reversal_observation(
            state,
            created_at="2099-07-19T00:00:00+00:00",
            minimum_until="2099-07-19T04:00:00+00:00",
        )
        engine.scale_in_intent_if_needed = Mock(return_value=object())
        try:
            intent = engine.intent_for_signal(signal, state)
        finally:
            engine.close()

        self.assertIsNone(intent)
        engine.scale_in_intent_if_needed.assert_not_called()
        self.assertEqual(signal.levels["entry_structure_filter"]["reason"], "manager_reversal_observation_active")

    def test_technical_entry_waits_for_manager_reversal_confirmation(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        state = engine.load_state()
        state["manager_reentry_locks"]["BTCUSDT:short"] = {
            "symbol": "BTCUSDT",
            "side": "short",
            "reason_type": "logic_weakening_exit",
            "timeframe": "4h",
            "created_at": "2026-07-19T00:00:00+00:00",
            "minimum_until": "2099-07-19T04:00:00+00:00",
            "expires_at": "2099-07-22T00:00:00+00:00",
        }
        self.add_manager_reversal_observation(
            state,
            created_at="2099-07-19T00:00:00+00:00",
            minimum_until="2099-07-19T04:00:00+00:00",
        )
        engine.macro_direction_entry_block = Mock(return_value=None)
        engine.technical_entry_ready = Mock(
            side_effect=lambda current_signal, action, context: (action == "open_long", None, context)
        )
        engine.technical_entry_rank = Mock(return_value=(1.0, 0.0, 0.0))
        try:
            intent = engine.intent_for_technical_entry_signal(signal, state, [])
        finally:
            engine.close()

        self.assertIsNone(intent)
        self.assertEqual(engine.technical_entry_ready.call_count, 0)
        self.assertEqual(signal.levels["entry_structure_filter"]["reason"], "manager_reversal_observation_active")

    def test_manager_reversal_requires_new_closed_4h_and_15m_selector(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        state = engine.load_state()
        self.add_manager_reversal_observation(state)
        self.confirm_h4_reversal(signal, target_side="long", last_closed_open_time=1782860400000)
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            block = engine.manager_reversal_entry_block(signal, state, "open_long", context)
        finally:
            engine.close()

        self.assertIsNotNone(block)
        self.assertFalse(block["full_4h_candle_observed"])
        self.assertTrue(block["opposite_h4_confirmed"])
        self.assertTrue(block["selector_15m_confirmed"])

    def test_manager_reversal_allows_fresh_4h_and_15m_confirmation(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        state = engine.load_state()
        self.add_manager_reversal_observation(state)
        self.confirm_h4_reversal(signal, target_side="long")
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            block = engine.manager_reversal_entry_block(signal, state, "open_long", context)
        finally:
            engine.close()

        self.assertIsNone(block)
        confirmation = context["manager_reversal_confirmation"]
        self.assertTrue(confirmation["approved"])
        self.assertEqual(context["horizon"], "4h_trend")
        self.assertEqual(context["entry_timeframe"], "4h")

    def test_probability_reversal_uses_4h_horizon_and_reduces_daily_conflict_size(self) -> None:
        engine = self.make_engine(
            technical_entry_only_symbols=(),
            entry_structure_filter_enabled=False,
            long_entry_threshold=0.50,
            short_entry_threshold=0.50,
            max_wait_probability_for_entry=0.50,
        )
        signal = self.make_signal()
        signal.long_probability = 0.70
        signal.short_probability = 0.20
        signal.wait_probability = 0.10
        signal.levels["daily_ma7_trend"] = {
            "ready": True,
            "bullish_stack": False,
            "bearish_stack": True,
            "long_confirmed": False,
            "short_confirmed": True,
        }
        state = engine.load_state()
        self.add_manager_reversal_observation(state)
        self.confirm_h4_reversal(signal, target_side="long")
        try:
            intent = engine.intent_for_signal(signal, state)
        finally:
            engine.close()

        self.assertIsNotNone(intent)
        self.assertEqual(intent.action, "open_long")
        self.assertEqual(intent.quantity_hint_usdt, 255.0)
        self.assertEqual(intent.risk_plan["horizon"], "4h_trend")
        self.assertEqual(intent.risk_plan["entry_timeframe"], "4h")
        confirmation = intent.risk_plan["manager_reversal_confirmation"]
        self.assertTrue(confirmation["daily_conflict"])
        self.assertEqual(confirmation["notional_multiplier"], 0.5)

    def test_stale_data_cannot_approve_manager_reversal(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal(stale=True)
        state = engine.load_state()
        self.add_manager_reversal_observation(state)
        self.confirm_h4_reversal(signal, target_side="long")
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            block = engine.manager_reversal_entry_block(signal, state, "open_long", context)
        finally:
            engine.close()

        self.assertIsNotNone(block)
        self.assertTrue(block["market_data_stale"])
        self.assertIn("BTCUSDT", state["manager_reversal_observations"])

    def test_manager_reversal_without_15m_selector_remains_blocked(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        signal.levels["structure_entry_scan"]["long_candidates"] = []
        state = engine.load_state()
        self.add_manager_reversal_observation(state)
        self.confirm_h4_reversal(signal, target_side="long")
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            block = engine.manager_reversal_entry_block(signal, state, "open_long", context)
        finally:
            engine.close()

        self.assertIsNotNone(block)
        self.assertFalse(block["selector_15m_confirmed"])

    def test_manager_reversal_observation_is_not_enforced_for_v5_overlay(self) -> None:
        engine = self.make_engine(v5_overlay_rules_enabled=True)
        signal = self.make_signal()
        state = engine.load_state()
        self.add_manager_reversal_observation(state)
        self.confirm_h4_reversal(signal, target_side="long")
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            block = engine.manager_reversal_entry_block(signal, state, "open_long", context)
        finally:
            engine.close()

        self.assertIsNone(block)
        self.assertIn("BTCUSDT", state["manager_reversal_observations"])

    def test_expired_manager_reversal_observation_is_removed(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        state = engine.load_state()
        self.add_manager_reversal_observation(
            state,
            expires_at="2026-07-02T00:00:00+00:00",
        )
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            block = engine.manager_reversal_entry_block(signal, state, "open_long", context)
        finally:
            engine.close()

        self.assertIsNone(block)
        self.assertNotIn("BTCUSDT", state["manager_reversal_observations"])
        self.assertEqual(state["manager_reversal_observation_history"][-1]["event"], "expired")

    def test_daily_manager_lock_releases_only_after_daily_side_revalidation(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        signal.levels["daily_ma7_trend"] = {
            "ready": True,
            "bearish_stack": True,
            "short_confirmed": True,
        }
        signal.levels["market_daily_ma7_reversal"] = {
            "ready": True,
            "reversal_confirmed": True,
        }
        state = engine.load_state()
        state["manager_reentry_locks"]["BTCUSDT:short"] = {
            "symbol": "BTCUSDT",
            "side": "short",
            "reason_type": "daily_ma7_short_reversal_exit",
            "timeframe": "1d",
            "created_at": "2026-07-01T00:00:00+00:00",
            "minimum_until": "2026-07-02T00:00:00+00:00",
            "expires_at": "2099-07-22T00:00:00+00:00",
        }
        try:
            self.assertIsNotNone(engine.manager_reentry_lock_block(signal, state, "open_short"))
            signal.levels["market_daily_ma7_reversal"]["reversal_confirmed"] = False
            self.assertIsNone(engine.manager_reentry_lock_block(signal, state, "open_short"))
        finally:
            engine.close()

        self.assertNotIn("BTCUSDT:short", state["manager_reentry_locks"])

    def test_manager_lock_with_invalid_minimum_time_fails_closed(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        signal.levels["h4_right_side"] = {
            "ready": True,
            "long_confirmed": True,
            "short_confirmed": False,
            "logic_weakening_exit": {
                "long_logic_weakened": False,
                "long_weakened_count": 0,
            },
        }
        state = engine.load_state()
        state["manager_reentry_locks"]["BTCUSDT:long"] = {
            "symbol": "BTCUSDT",
            "side": "long",
            "reason_type": "logic_weakening_exit",
            "timeframe": "4h",
            "created_at": "2026-07-19T00:00:00+00:00",
            "minimum_until": "invalid",
            "expires_at": "2099-07-22T00:00:00+00:00",
        }
        try:
            block = engine.manager_reentry_lock_block(signal, state, "open_long")
        finally:
            engine.close()

        self.assertIsNotNone(block)
        self.assertFalse(block["minimum_elapsed"])
        self.assertTrue(block["same_side_revalidated"])
        self.assertIn("BTCUSDT:long", state["manager_reentry_locks"])

    def test_cross_timeframe_observation_resolves_when_daily_direction_changes(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        state = engine.load_state()
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=False,
        )
        try:
            initial_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            self.assertIsNone(engine.macro_direction_entry_block(signal, "open_long", initial_context, state=state))

            self.set_daily_4h_conflict(
                signal,
                daily_direction="long",
                h4_direction="long",
                weakened=False,
            )
            changed_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            changed_block = engine.macro_direction_entry_block(signal, "open_long", changed_context, state=state)
        finally:
            engine.close()

        self.assertIsNone(changed_block)
        observation = state["cross_timeframe_macro_observations"]["BTCUSDT"]
        self.assertEqual(observation["status"], "resolved")
        self.assertEqual(observation["resolution_reason"], "daily_direction_changed_or_unresolved")
        self.assertEqual(observation["resolved_daily_direction"], "long")

    def test_cross_timeframe_conflict_reverses_only_after_short_structure_break(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="short",
            h4_direction="long",
            weakened=True,
        )
        signal.levels["structure_entry_scan"]["short_candidates"].append(
            {
                "name": "bb210_lower_15m",
                "price": 100.5,
                "distance_pct": -0.497512,
                "abs_distance_pct": 0.497512,
                "period": 210,
                "band": "lower",
                "timeframe": "15m",
                "structure_signal": "short_lower_breakdown_retest",
            }
        )
        signal.indicators["recent_extremes"]["1m"]["high"] = 101.0
        try:
            context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            block = engine.macro_direction_entry_block(signal, "open_short", context, state={})
        finally:
            engine.close()

        self.assertIsNone(block)
        self.assertEqual(context["selected_entry_level"]["name"], "bb210_lower_15m")
        arbitration = context["mixed_structure_macro_arbitration"]
        self.assertEqual(arbitration["allowed_action"], "open_short")
        self.assertTrue(arbitration["reversal_confirmation"]["confirmed"])

    def test_cross_timeframe_conflict_mirrors_for_daily_long_and_4h_short(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        self.set_daily_4h_conflict(
            signal,
            daily_direction="long",
            h4_direction="short",
            weakened=True,
        )
        signal.levels["structure_entry_scan"]["long_candidates"].append(
            {
                "name": "ma20_15m",
                "price": 99.5,
                "distance_pct": 0.502513,
                "abs_distance_pct": 0.502513,
                "period": 20,
                "band": "ma",
                "timeframe": "15m",
                "indicator_family": "ma",
                "source": "ma_cross",
                "previous_close": 99.0,
                "current_close": 100.0,
                "structure_signal": "long_ma_reclaim_cross",
            }
        )
        try:
            short_context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            long_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            short_block = engine.macro_direction_entry_block(signal, "open_short", short_context, state={})
            long_block = engine.macro_direction_entry_block(signal, "open_long", long_context, state={})
        finally:
            engine.close()

        self.assertEqual(short_block["reason"], "mixed_structure_cross_timeframe_4h_logic_weakened")
        self.assertIsNone(long_block)
        self.assertEqual(long_context["selected_entry_level"]["name"], "ma20_15m")
        self.assertEqual(long_context["mixed_structure_macro_arbitration"]["allowed_action"], "open_long")

    def test_mixed_structure_neutral_macro_selects_higher_long_probability(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        signal.side_bias = "neutral"
        signal.long_probability = 0.51
        signal.short_probability = 0.34
        self.add_mixed_structure(signal)
        state = engine.load_state()
        try:
            long_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            short_context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            long_block = engine.macro_direction_entry_block(signal, "open_long", long_context, state=state)
            short_block = engine.macro_direction_entry_block(signal, "open_short", short_context, state=state)
        finally:
            engine.close()

        self.assertIsNone(long_block)
        self.assertEqual(short_block["reason"], "mixed_structure_macro_direction_mismatch")
        self.assertEqual(long_context["mixed_structure_macro_arbitration"]["macro_source"], "model_probability_fallback")
        self.assertEqual(long_context["mixed_structure_macro_arbitration"]["allowed_action"], "open_long")

    def test_three_probability_fallback_losses_pause_until_explicit_macro_returns(self) -> None:
        engine = self.make_engine()
        state = engine.load_state()
        try:
            for index in range(3):
                position = {
                    "symbol": "BTCUSDT",
                    "side": "short",
                    "opened_at": f"2026-07-{10 + index:02d}T00:00:00+00:00",
                    "entry_price": 100.0,
                    "leverage": 8,
                    "mixed_structure_probability_fallback": True,
                    "lots": [{"entry_price": 100.0, "quantity": 1.0, "remaining_quantity": 1.0}],
                }
                event = engine.record_mixed_probability_fallback_outcome(
                    state,
                    position,
                    final_close_price=101.0,
                    closed_at=f"2026-07-{10 + index:02d}T04:00:00+00:00",
                    order_id=f"loss-{index}",
                    source="test",
                )
                self.assertEqual(event["outcome"], "loss")

            signal = self.make_signal()
            self.add_mixed_structure(signal)
            paused_context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            paused = engine.macro_direction_entry_block(signal, "open_short", paused_context, state=state)
            self.assertEqual(paused["reason"], "mixed_structure_probability_fallback_loss_streak")
            self.assertEqual(paused["consecutive_losses"], 3)
            self.assertEqual(
                state["mixed_structure_probability_fallback_guards"]["BTCUSDT"]["status"],
                "observing",
            )

            signal.levels["daily_ma7_trend"] = {
                "ready": True,
                "bullish_stack": True,
                "bearish_stack": False,
                "long_confirmed": False,
                "short_confirmed": False,
            }
            resumed_context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            resumed = engine.macro_direction_entry_block(signal, "open_long", resumed_context, state=state)
        finally:
            engine.close()

        self.assertIsNone(resumed)
        guard = state["mixed_structure_probability_fallback_guards"]["BTCUSDT"]
        self.assertEqual(guard["consecutive_losses"], 0)
        self.assertEqual(guard["status"], "active")
        self.assertEqual(guard["last_reset_reason"], "explicit_macro_direction")

    def test_probability_fallback_pause_blocks_the_full_technical_entry_path(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        self.add_mixed_structure(signal)
        state = engine.load_state()
        guard = engine.mixed_structure_probability_fallback_guard(state, "BTCUSDT")
        guard.update({"consecutive_losses": 3, "status": "observing"})
        try:
            intent = engine.intent_for_technical_entry_signal(signal, state, [])
        finally:
            engine.close()

        self.assertIsNone(intent)
        block = signal.levels["entry_structure_filter"]
        self.assertEqual(block["reason"], "mixed_structure_probability_fallback_loss_streak")

    def test_non_loss_resets_probability_fallback_loss_streak(self) -> None:
        engine = self.make_engine()
        state = engine.load_state()
        position = {
            "symbol": "BTCUSDT",
            "side": "long",
            "opened_at": "2026-07-12T00:00:00+00:00",
            "entry_price": 100.0,
            "leverage": 8,
            "mixed_structure_probability_fallback": True,
            "lots": [{"entry_price": 100.0, "quantity": 1.0, "remaining_quantity": 1.0}],
        }
        try:
            guard = engine.mixed_structure_probability_fallback_guard(state, "BTCUSDT")
            guard["consecutive_losses"] = 2
            guard["status"] = "active"
            event = engine.record_mixed_probability_fallback_outcome(
                state,
                position,
                final_close_price=101.0,
                closed_at="2026-07-12T04:00:00+00:00",
                order_id="profit-1",
                source="test",
            )
        finally:
            engine.close()

        self.assertEqual(event["outcome"], "non_loss")
        self.assertEqual(guard["consecutive_losses"], 0)

    def test_partial_exits_are_aggregated_for_probability_fallback_outcome(self) -> None:
        engine = self.make_engine()
        position = {
            "symbol": "BTCUSDT",
            "side": "long",
            "entry_price": 100.0,
            "leverage": 8,
            "lot_close_history": [
                {
                    "close_price": 110.0,
                    "allocations": [{"entry_price": 100.0, "closed_quantity": 0.5}],
                }
            ],
            "lots": [{"entry_price": 100.0, "remaining_quantity": 0.5}],
        }
        try:
            result = engine.estimate_closed_position_return(position, final_close_price=95.0)
        finally:
            engine.close()

        self.assertGreater(result["realized_pnl_quote_estimate"], 0.0)
        self.assertAlmostEqual(result["raw_return_pct"], 2.5, places=6)

    def test_gate_diagnostics_remain_json_serializable(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal(stale=True)
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            engine.entry_gate_policy_v2_decision(signal, "open_long", context)
            encoded = json.dumps(signal.to_dict(), ensure_ascii=False)
        finally:
            engine.close()

        self.assertIn("entry_gate_policy_v2", encoded)

    def test_profile_can_preserve_legacy_15m_gate_exemption(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        ma20_gate = Mock(side_effect=AssertionError("15M MA20 gate must be skipped"))
        second_test_gate = Mock(side_effect=AssertionError("15M second-test gate must be skipped"))
        engine.ma20_trend_entry_block = ma20_gate
        engine.bb96_210_second_test_opposite_block = second_test_gate
        profile = {
            "name": "test_ignore_15m",
            "ignore_15m_entry_blocks": True,
            "allow": (),
        }
        try:
            with patch.dict(SYMBOL_BB_ENTRY_PROFILES, {"BTCUSDT": profile}):
                context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
                engine.entry_gate_policy_v2_decision(signal, "open_long", context)
        finally:
            engine.close()

        ma20_gate.assert_not_called()
        second_test_gate.assert_not_called()

    def test_nontechnical_symbol_keeps_legacy_gate_path(self) -> None:
        engine = self.make_engine()
        try:
            self.assertTrue(engine.config.entry_gate_policy_v2_for("BTCUSDT"))
            self.assertFalse(engine.config.entry_gate_policy_v2_for("TESTUSDT"))
        finally:
            engine.close()

    def test_three_line_confirmation_converts_4h_countertrend_block_to_small_probe(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        signal.levels["structure_entry_scan"] = {
            "short_candidates": [
                {
                    "name": "bb300_upper_15m",
                    "price": 100.2,
                    "distance_pct": -0.2,
                    "abs_distance_pct": 0.2,
                    "period": 300,
                    "band": "upper",
                    "timeframe": "15m",
                    "structure_signal": "short_upper_pressure_rejection",
                }
            ]
        }
        engine.directional_entry_level_touched = Mock(return_value=False)
        engine.scale_in_trigger_candidate = Mock(return_value={"name": "three_line_rejection", "line_count": 3})
        engine.macro_direction_entry_block = Mock(return_value=None)
        engine.symbol_bb_entry_profile_block = Mock(return_value=None)
        engine.market_data_freshness_entry_block = Mock(return_value=None)
        engine.smart_signal_entry_block = Mock(return_value=None)
        engine.ma20_trend_entry_block = Mock(return_value=None)
        engine.daily_ma7_short_reversal_entry_block = Mock(return_value=None)
        engine.daily_ma20_long_reversal_entry_block = Mock(return_value=None)
        engine.trend_big_entry_block = Mock(return_value=None)
        engine.h4_lower_chase_short_block = Mock(return_value=None)
        engine.bb96_210_second_test_opposite_block = Mock(return_value=None)
        engine.generic_indicator_entry_block = Mock(return_value=None)
        engine.four_h_entry_15m_selector_block = Mock(return_value=None)
        engine.short_15m_entry_quality_block = Mock(
            return_value={
                "reason": "short_15m_blocked_by_4h_right_side_long",
                "message": "4H right-side long remains active",
            }
        )
        engine.fib_structure_confirmation_block = Mock(return_value=None)
        engine.indicator_confluence_hold = Mock(return_value=None)
        engine.stop_take_profit = Mock(
            return_value=(110.0, 80.0, {"stop_loss_price": 110.0, "take_profit_price": 80.0})
        )
        try:
            context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_short", context)
            intent = engine.open_intent(signal, "open_short", entry_context=context)
        finally:
            engine.close()

        self.assertTrue(decision["allowed"])
        self.assertTrue(decision["transition_probe"])
        self.assertEqual(decision["notional_multiplier"], 0.30)
        self.assertIn("short_15m_blocked_by_4h_right_side_long", decision["probe_reasons"])
        self.assertEqual(intent.quantity_hint_usdt, 153.0)
        self.assertAlmostEqual(intent.stop_loss_price, 101.5, places=8)
        self.assertTrue(intent.risk_plan["transition_probe_risk"]["enabled"])

    def test_4h_countertrend_without_three_line_confirmation_remains_watch(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        engine.directional_entry_level_touched = Mock(return_value=False)
        engine.scale_in_trigger_candidate = Mock(return_value=None)
        engine.macro_direction_entry_block = Mock(return_value=None)
        engine.symbol_bb_entry_profile_block = Mock(return_value=None)
        engine.market_data_freshness_entry_block = Mock(return_value=None)
        engine.smart_signal_entry_block = Mock(return_value=None)
        engine.ma20_trend_entry_block = Mock(return_value=None)
        engine.daily_ma7_short_reversal_entry_block = Mock(return_value=None)
        engine.daily_ma20_long_reversal_entry_block = Mock(return_value=None)
        engine.trend_big_entry_block = Mock(return_value=None)
        engine.h4_lower_chase_short_block = Mock(return_value=None)
        engine.bb96_210_second_test_opposite_block = Mock(return_value=None)
        engine.generic_indicator_entry_block = Mock(return_value=None)
        engine.four_h_entry_15m_selector_block = Mock(return_value=None)
        engine.short_15m_entry_quality_block = Mock(
            return_value={
                "reason": "short_15m_blocked_by_4h_right_side_long",
                "message": "4H right-side long remains active",
            }
        )
        engine.fib_structure_confirmation_block = Mock(return_value=None)
        engine.indicator_confluence_hold = Mock(return_value=None)
        engine.entry_boundary_chase_candidate = Mock(return_value=None)
        try:
            context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_short", context)
        finally:
            engine.close()

        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["state"], "watch")
        self.assertFalse(decision["transition_probe"])

    def test_4h_selector_wait_becomes_reduced_probe_after_directional_level_touch(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        engine.directional_entry_level_touched = Mock(return_value=True)
        engine.scale_in_trigger_candidate = Mock(return_value=None)
        engine.macro_direction_entry_block = Mock(return_value=None)
        engine.symbol_bb_entry_profile_block = Mock(return_value=None)
        engine.market_data_freshness_entry_block = Mock(return_value=None)
        engine.smart_signal_entry_block = Mock(return_value=None)
        engine.ma20_trend_entry_block = Mock(return_value=None)
        engine.daily_ma7_short_reversal_entry_block = Mock(return_value=None)
        engine.daily_ma20_long_reversal_entry_block = Mock(return_value=None)
        engine.trend_big_entry_block = Mock(return_value=None)
        engine.h4_lower_chase_short_block = Mock(return_value=None)
        engine.bb96_210_second_test_opposite_block = Mock(return_value=None)
        engine.generic_indicator_entry_block = Mock(return_value=None)
        engine.four_h_entry_15m_selector_block = Mock(
            return_value={
                "reason": "four_h_trend_waiting_for_15m_selector",
                "message": "4H zone exists but selector is waiting",
            }
        )
        engine.short_15m_entry_quality_block = Mock(return_value=None)
        engine.fib_structure_confirmation_block = Mock(return_value=None)
        engine.indicator_confluence_hold = Mock(return_value=None)
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_long", context)
        finally:
            engine.close()

        self.assertTrue(decision["allowed"])
        self.assertEqual(decision["notional_multiplier"], 0.50)
        self.assertTrue(decision["transition_probe"])
        self.assertIn("four_h_trend_waiting_for_15m_selector", decision["probe_reasons"])

    def test_4h_selector_wait_without_structure_confirmation_remains_watch(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        engine.directional_entry_level_touched = Mock(return_value=False)
        engine.scale_in_trigger_candidate = Mock(return_value=None)
        for name in (
            "macro_direction_entry_block",
            "symbol_bb_entry_profile_block",
            "market_data_freshness_entry_block",
            "smart_signal_entry_block",
            "ma20_trend_entry_block",
            "daily_ma7_short_reversal_entry_block",
            "daily_ma20_long_reversal_entry_block",
            "trend_big_entry_block",
            "h4_lower_chase_short_block",
            "bb96_210_second_test_opposite_block",
            "generic_indicator_entry_block",
            "short_15m_entry_quality_block",
            "fib_structure_confirmation_block",
            "indicator_confluence_hold",
            "entry_boundary_chase_candidate",
        ):
            setattr(engine, name, Mock(return_value=None))
        engine.four_h_entry_15m_selector_block = Mock(
            return_value={
                "reason": "four_h_trend_waiting_for_15m_selector",
                "message": "4H zone exists but selector is waiting",
            }
        )
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_long", context)
        finally:
            engine.close()

        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["state"], "watch")
        self.assertEqual(decision["blocking_reason"], "four_h_trend_waiting_for_15m_selector")
        self.assertFalse(decision["transition_probe"])

    def test_gann_position_sizing_is_btc_only_and_never_a_gate(self) -> None:
        engine = self.make_engine(
            gann_position_sizing_enabled=True,
            gann_position_sizing_symbols=("BTCUSDT",),
        )
        signal = self.make_signal()
        signal.side_bias = "long"
        signal.levels["gann_feature"] = {
            "zone": "above_1x1",
            "bias": "support_long",
            "confidence": 1.0,
            "stale": False,
        }
        context = {
            "entry_gate_policy_v2": {
                "allowed": True,
                "notional_multiplier": 1.0,
                "transition_probe": False,
            }
        }
        try:
            btc_intent = engine.open_intent(signal, "open_long", entry_context=context)
            signal.symbol = "ETHUSDT"
            eth_intent = engine.open_intent(signal, "open_long", entry_context=context)
        finally:
            engine.close()

        self.assertEqual(btc_intent.quantity_hint_usdt, 637.5)
        self.assertEqual(btc_intent.risk_plan["position_sizing_overlays"]["gann_multiplier"], 1.25)
        self.assertEqual(eth_intent.quantity_hint_usdt, 510.0)
        self.assertEqual(eth_intent.risk_plan["position_sizing_overlays"]["gann_multiplier"], 1.0)

    def test_soft_sizing_never_falls_below_configured_min_notional(self) -> None:
        engine = self.make_engine(base_notional_usdt=100.0, min_notional_usdt=40.0)
        signal = self.make_signal()
        context = {
            "entry_gate_policy_v2": {
                "allowed": True,
                "notional_multiplier": 0.30,
                "transition_probe": False,
            }
        }
        try:
            intent = engine.open_intent(signal, "open_long", entry_context=context)
        finally:
            engine.close()

        self.assertEqual(intent.quantity_hint_usdt, 40.0)
        self.assertEqual(intent.risk_plan["entry_gate_adjusted_notional_usdt"], 40.0)

    def test_15m_position_does_not_use_4h_logic_weakening_exit(self) -> None:
        engine = self.make_engine(logic_weakening_exit_enabled=True)
        signal = self.make_signal()
        signal.h4 = {
            "right_side": {
                "state": "short_right_side",
                "logic_weakening_exit": {
                    "enabled": True,
                    "required_closed_candles": 2,
                    "checked_closed_candles": 2,
                    "long_logic_weakened": True,
                    "long_weakened_count": 2,
                },
            }
        }
        position = {
            "symbol": "BTCUSDT",
            "side": "long",
            "horizon": "15m_scalp",
            "entry_timeframe": "15m",
            "risk_plan": {"horizon": "15m_scalp", "entry_timeframe": "15m"},
        }
        try:
            candidate = engine.logic_weakening_exit_candidate(signal, position)
        finally:
            engine.close()

        self.assertIsNone(candidate)
        self.assertNotIn("logic_weakening_watch", position)

    def test_4h_position_keeps_4h_logic_weakening_exit(self) -> None:
        engine = self.make_engine(logic_weakening_exit_enabled=True)
        signal = self.make_signal()
        signal.h4 = {
            "right_side": {
                "state": "short_right_side",
                "logic_weakening_exit": {
                    "enabled": True,
                    "required_closed_candles": 2,
                    "checked_closed_candles": 2,
                    "long_logic_weakened": True,
                    "long_weakened_count": 2,
                },
            }
        }
        position = {
            "symbol": "BTCUSDT",
            "side": "long",
            "horizon": "4h_trend",
            "entry_timeframe": "4h",
            "risk_plan": {"horizon": "4h_trend", "entry_timeframe": "4h"},
        }
        try:
            candidate = engine.logic_weakening_exit_candidate(signal, position)
        finally:
            engine.close()

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["status"], "exit_ready")
        self.assertEqual(candidate["weakened_count"], 2)

    def test_final_unconfirmed_15m_state_overwrites_early_allowed_gate_event(self) -> None:
        engine = self.make_engine(entry_gate_event_recording_enabled=True)
        signal = self.make_signal()
        engine.directional_entry_level_touched = Mock(return_value=False)
        engine.scale_in_trigger_candidate = Mock(return_value=None)
        engine.macro_direction_entry_block = Mock(return_value=None)
        engine.symbol_bb_entry_profile_block = Mock(return_value=None)
        engine.market_data_freshness_entry_block = Mock(return_value=None)
        engine.smart_signal_entry_block = Mock(return_value=None)
        engine.ma20_trend_entry_block = Mock(return_value=None)
        engine.daily_ma7_short_reversal_entry_block = Mock(return_value=None)
        engine.daily_ma20_long_reversal_entry_block = Mock(return_value=None)
        engine.trend_big_entry_block = Mock(return_value=None)
        engine.h4_lower_chase_short_block = Mock(return_value=None)
        engine.bb96_210_second_test_opposite_block = Mock(return_value=None)
        engine.generic_indicator_entry_block = Mock(return_value=None)
        engine.four_h_entry_15m_selector_block = Mock(return_value=None)
        engine.fib_structure_confirmation_block = Mock(return_value=None)
        engine.indicator_confluence_hold = Mock(return_value=None)
        engine.entry_boundary_chase_candidate = Mock(return_value=None)
        try:
            ready, block, _ = engine.technical_entry_ready(signal, "open_long")
            event_rows = [
                json.loads(line)
                for line in engine.config.entry_gate_events_file.read_text(encoding="utf-8").splitlines()
            ]
        finally:
            engine.close()

        self.assertFalse(ready)
        self.assertEqual(block["reason"], "technical_entry_only_unconfirmed_15m_trigger")
        self.assertGreaterEqual(len(event_rows), 2)
        self.assertFalse(event_rows[-1]["decision"]["allowed"])
        self.assertEqual(
            event_rows[-1]["decision"]["blocking_reason"],
            "technical_entry_only_unconfirmed_15m_trigger",
        )

    def test_technical_entry_reuses_directional_touch_from_evaluated_context(self) -> None:
        engine = self.make_engine(entry_gate_event_recording_enabled=True)
        signal = self.make_signal()
        for name in (
            "macro_direction_entry_block",
            "symbol_bb_entry_profile_block",
            "market_data_freshness_entry_block",
            "smart_signal_entry_block",
            "ma20_trend_entry_block",
            "daily_ma7_short_reversal_entry_block",
            "daily_ma20_long_reversal_entry_block",
            "trend_big_entry_block",
            "h4_lower_chase_short_block",
            "bb96_210_second_test_opposite_block",
            "generic_indicator_entry_block",
            "four_h_entry_15m_selector_block",
            "short_15m_entry_quality_block",
            "fib_structure_confirmation_block",
            "indicator_confluence_hold",
            "entry_boundary_chase_candidate",
        ):
            setattr(engine, name, Mock(return_value=None))
        engine.scale_in_trigger_candidate = Mock(return_value=None)
        try:
            context = engine.entry_context_for_action("open_long", signal.levels, symbol=signal.symbol)
            engine.directional_entry_level_touched = Mock(
                side_effect=lambda _signal, _action, evaluated_context=None: evaluated_context is context
            )
            ready, block, returned_context = engine.technical_entry_ready(
                signal,
                "open_long",
                context=context,
            )
        finally:
            engine.close()

        self.assertTrue(ready)
        self.assertIsNone(block)
        self.assertIs(returned_context, context)
        self.assertTrue(
            returned_context["entry_gate_policy_v2"]["structural_confirmation"]["directional_level_touched"]
        )
        engine.directional_entry_level_touched.assert_called_once_with(signal, "open_long", context)

    def test_over_gates_become_sizing_factors_when_structure_is_touched(self) -> None:
        engine = self.make_engine()
        signal = self.make_signal()
        signal.levels["structure_entry_scan"] = {
            "short_candidates": [
                {
                    "name": "bb300_upper_15m",
                    "price": 100.2,
                    "distance_pct": -0.2,
                    "abs_distance_pct": 0.2,
                    "period": 300,
                    "band": "upper",
                    "timeframe": "15m",
                    "structure_signal": "short_upper_pressure_rejection",
                }
            ]
        }
        engine.directional_entry_level_touched = Mock(return_value=True)
        engine.scale_in_trigger_candidate = Mock(return_value=None)
        engine.macro_direction_entry_block = Mock(return_value=None)
        engine.symbol_bb_entry_profile_block = Mock(return_value=None)
        engine.market_data_freshness_entry_block = Mock(return_value=None)
        engine.smart_signal_entry_block = Mock(return_value=None)
        engine.ma20_trend_entry_block = Mock(
            return_value={"reason": "short_blocked_15m_ma20_trend_alive", "message": "MA20 conflict"}
        )
        engine.daily_ma7_short_reversal_entry_block = Mock(return_value=None)
        engine.daily_ma20_long_reversal_entry_block = Mock(return_value=None)
        engine.trend_big_entry_block = Mock(
            return_value={"reason": "trend_big_blocked_without_4h_right_side", "message": "4H confirmation missing"}
        )
        engine.h4_lower_chase_short_block = Mock(
            return_value={"reason": "short_blocked_4h_lower_band_chase", "message": "lower chase"}
        )
        engine.bb96_210_second_test_opposite_block = Mock(return_value=None)
        engine.generic_indicator_entry_block = Mock(return_value=None)
        engine.four_h_entry_15m_selector_block = Mock(return_value=None)
        engine.short_15m_entry_quality_block = Mock(return_value=None)
        engine.fib_structure_confirmation_block = Mock(return_value=None)
        engine.indicator_confluence_hold = Mock(
            return_value={"reason": "short_support_confluence_hold", "message": "support confluence"}
        )
        try:
            context = engine.entry_context_for_action("open_short", signal.levels, symbol=signal.symbol)
            decision = engine.entry_gate_policy_v2_decision(signal, "open_short", context)
        finally:
            engine.close()

        self.assertTrue(decision["allowed"])
        self.assertEqual(decision["state"], "allowed")
        self.assertEqual(decision["notional_multiplier"], 0.30)
        self.assertTrue(decision["transition_probe"])
        self.assertIn("short_blocked_15m_ma20_trend_alive", decision["soft_reasons"])
        self.assertIn("trend_big_blocked_without_4h_right_side", decision["soft_reasons"])
        self.assertIn("short_blocked_4h_lower_band_chase", decision["soft_reasons"])
        self.assertIn("short_support_confluence_hold", decision["soft_reasons"])
