from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


from trading_system.v6_execution import (
    LIVE_TRADING_CONFIRM_VALUE,
    V6BinanceClient,
    V6ExecutionConfig,
    V6ExecutionEngine,
    V6ExecutionError,
    V6OrderIntent,
    V6Signal,
    canonical_order_mode,
    parse_symbols,
)


class V6ExecutionConfigTest(unittest.TestCase):
    def test_v6_config_does_not_read_v5_binance_envs(self) -> None:
        env = {
            "BINANCE_API_KEY": "v5-key",
            "BINANCE_API_SECRET": "v5-secret",
            "BINANCE_TESTNET_API_KEY": "v5-testnet-key",
            "BINANCE_TESTNET_API_SECRET": "v5-testnet-secret",
            "TS_V6_SYMBOLS": "btcusdt, ethusdt",
            "TS_V6_EXECUTION_DIR": "data/trading_system/v6_execution_test",
        }
        with patch.dict(os.environ, env, clear=True):
            config = V6ExecutionConfig.from_env()

        self.assertEqual(config.symbols, ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(config.demo_api_key, "")
        self.assertEqual(config.demo_api_secret, "")
        self.assertEqual(config.live_api_key, "")
        self.assertEqual(config.live_api_secret, "")
        self.assertEqual(config.state_dir, Path("data/trading_system/v6_execution_test"))
        self.assertIn("v6_execution_test", str(config.state_file))
        self.assertEqual(config.kline_cache_dir, Path("data/trading_system/cache/execution_klines"))

    def test_v6_config_reads_only_v6_credentials(self) -> None:
        env = {
            "BINANCE_V6_DEMO_API_KEY": "v6-demo-key",
            "BINANCE_V6_DEMO_API_SECRET": "v6-demo-secret",
            "BINANCE_V6_API_KEY": "v6-live-key",
            "BINANCE_V6_API_SECRET": "v6-live-secret",
            "TS_V6_LIVE_TRADING_CONFIRMED": LIVE_TRADING_CONFIRM_VALUE,
        }
        with patch.dict(os.environ, env, clear=True):
            config = V6ExecutionConfig.from_env()

        self.assertEqual(config.demo_api_key, "v6-demo-key")
        self.assertEqual(config.demo_api_secret, "v6-demo-secret")
        self.assertEqual(config.live_api_key, "v6-live-key")
        self.assertEqual(config.live_api_secret, "v6-live-secret")
        self.assertTrue(config.live_trading_confirmed())

    def test_v6_live_trading_confirmation_requires_exact_value(self) -> None:
        with patch.dict(os.environ, {"TS_V6_LIVE_TRADING_CONFIRMED": "yes"}, clear=True):
            config = V6ExecutionConfig.from_env()
        self.assertFalse(config.live_trading_confirmed())

        with patch.dict(os.environ, {"TS_V6_LIVE_TRADING_CONFIRMED": LIVE_TRADING_CONFIRM_VALUE}, clear=True):
            config = V6ExecutionConfig.from_env()
        self.assertTrue(config.live_trading_confirmed())

    def test_v6_protective_stop_enabled_by_default_with_env_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = V6ExecutionConfig.from_env()
        self.assertTrue(config.submit_protective_stop)

        with patch.dict(os.environ, {"TS_V6_SUBMIT_PROTECTIVE_STOP": "false"}, clear=True):
            config = V6ExecutionConfig.from_env()
        self.assertFalse(config.submit_protective_stop)

    def test_v6_kline_limits_default_to_execution_cache_history(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = V6ExecutionConfig.from_env()

        self.assertEqual(config.kline_limit_for_interval("1m"), 4707)
        self.assertEqual(config.kline_limit_for_interval("15m"), 4707)
        self.assertEqual(config.kline_limit_for_interval("4h"), 4707)
        self.assertEqual(config.kline_limit_for_interval("1d"), 4707)

        self.assertTrue(config.kline_cache_enabled)
        self.assertEqual(config.kline_cache_dir, Path("data/trading_system/cache/execution_klines"))
        self.assertEqual(config.closed_trade_review_queue_file, Path("data/trading_system/v6_execution/closed_trade_review_queue.jsonl"))
        self.assertEqual(config.post_trade_training_samples_file, Path("data/trading_system/v6_execution/post_trade_training_samples.jsonl"))
        self.assertEqual(config.public_mark_price_ttl_seconds, 120)
        self.assertEqual(config.public_ticker_24h_ttl_seconds, 300)
        self.assertEqual(config.exchange_info_cache_ttl_seconds, 180)
        self.assertTrue(config.close_resolver_enabled)
        self.assertTrue(config.close_intent_coalescing_enabled)
        self.assertEqual(config.kline_refresh_interval_seconds_for_interval("1m"), 20)
        self.assertEqual(config.kline_refresh_interval_seconds_for_interval("15m"), 120)
        self.assertEqual(config.kline_refresh_interval_seconds_for_interval("4h"), 600)
        self.assertEqual(config.kline_refresh_interval_seconds_for_interval("1d"), 1800)

    def test_symbol_notional_multiplier_reduces_configured_base_for_zec(self) -> None:
        env = {
            "TS_V6_BASE_NOTIONAL_USDT": "1000",
            "TS_V6_MIN_NOTIONAL_USDT": "1000",
            "TS_V6_DEFAULT_LEVERAGE": "8",
            "TS_V6_SYMBOL_NOTIONAL_MULTIPLIERS": "ZECUSDT:0.60",
        }
        with patch.dict(os.environ, env, clear=True):
            config = V6ExecutionConfig.from_env()

        self.assertEqual(config.default_leverage, 8)
        self.assertEqual(config.base_notional_for("BTCUSDT"), 1000.0)
        self.assertEqual(config.base_notional_for("ZECUSDT"), 600.0)
        self.assertEqual(config.base_notional_for("zec"), 600.0)

    def test_zec_open_intent_keeps_600_notional_when_soft_gate_reduces_size(self) -> None:
        config = V6ExecutionConfig(
            symbols=["ZECUSDT"],
            base_notional_usdt=1000.0,
            min_notional_usdt=1000.0,
            default_leverage=8,
            symbol_notional_multipliers={"ZECUSDT": 0.60},
            technical_entry_only_symbols=("ZECUSDT",),
            entry_gate_policy_v2_enabled=True,
            entry_gate_min_notional_multiplier=1.0,
            smart_signal_risk_guard_enabled=False,
        )
        engine = V6ExecutionEngine(config)
        signal = SimpleNamespace(
            symbol="ZECUSDT",
            price=500.0,
            levels={
                "nearest_supports": [("bb120_mid_4h", 480.0)],
                "nearest_resistances": [("bb300_upper_4h", 540.0)],
                "all_supports": [("bb120_mid_4h", 480.0)],
                "all_resistances": [("bb300_upper_4h", 540.0)],
                "structure_entry_scan": {"long_candidate_count": 1},
            },
            long_probability=0.7,
            short_probability=0.1,
            wait_probability=0.2,
            indicators={},
            reasons=["test fixed ZEC size"],
        )
        try:
            intent = engine.open_intent(
                signal,
                "open_long",
                entry_context={
                    "entry_gate_policy_v2": {
                        "allowed": True,
                        "notional_multiplier": 0.30,
                    }
                },
            )
            normal_intent = engine.open_intent(signal, "open_long")
        finally:
            engine.close()

        self.assertEqual(intent.leverage, 8)
        self.assertEqual(intent.quantity_hint_usdt, 600.0)
        self.assertEqual(intent.risk_plan["configured_base_notional_usdt"], 1000.0)
        self.assertEqual(intent.risk_plan["symbol_notional_multiplier"], 0.60)
        self.assertEqual(intent.risk_plan["symbol_base_notional_usdt"], 600.0)
        self.assertEqual(normal_intent.quantity_hint_usdt, 600.0)

    def test_v6_symbol_entry_thresholds_override_default_thresholds(self) -> None:
        env = {
            "TS_V6_LONG_ENTRY_THRESHOLD": "0.60",
            "TS_V6_SHORT_ENTRY_THRESHOLD": "0.60",
            "TS_V6_SYMBOL_ENTRY_THRESHOLDS": "BTC:0.45,ETHUSDT:0.45,BNB:0.45",
        }
        with patch.dict(os.environ, env, clear=True):
            config = V6ExecutionConfig.from_env()

        self.assertEqual(config.long_entry_threshold_for("BTCUSDT"), 0.45)
        self.assertEqual(config.short_entry_threshold_for("ETHUSDT"), 0.45)
        self.assertEqual(config.short_entry_threshold_for("BNBUSDT"), 0.45)
        self.assertEqual(config.short_entry_threshold_for("HYPEUSDT"), 0.60)

    def test_parse_symbols_and_order_mode_aliases(self) -> None:
        self.assertEqual(parse_symbols("btcusdt; ethusdt, BTCUSDT"), ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(canonical_order_mode("demo"), "testnet_order")
        self.assertEqual(canonical_order_mode("paper"), "dry_run")
        self.assertEqual(canonical_order_mode("live"), "live_order")
        self.assertEqual(canonical_order_mode("unknown"), "dry_run")

    def test_strategy_dust_threshold_defaults_to_five_usdt(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = V6ExecutionConfig.from_env()

        self.assertEqual(config.dust_position_notional_usdt, 1.0)
        self.assertEqual(config.strategy_dust_position_notional_usdt, 5.0)

    def test_status_unknown_detector_treats_signed_5xx_as_unknown(self) -> None:
        self.assertTrue(
            V6ExecutionEngine.is_status_unknown_error(
                "V6ExecutionError: Binance signed request failed 502: <html><h1>502 Bad Gateway</h1></html>"
            )
        )
        self.assertTrue(V6ExecutionEngine.is_status_unknown_error("Binance signed request failed 504: Gateway Timeout"))
        self.assertFalse(V6ExecutionEngine.is_status_unknown_error('Binance signed request failed 400: {"code":-4131}'))


class V6PositionCapScaleInTest(unittest.TestCase):
    def test_existing_same_side_position_can_scale_in_when_position_cap_is_reached(self) -> None:
        config = V6ExecutionConfig(
            symbols=["BTCUSDT", "ETHUSDT"],
            order_mode="dry_run",
            max_positions=2,
            long_entry_threshold=0.45,
            short_entry_threshold=0.45,
            technical_entry_only_symbols=("BTCUSDT",),
        )
        engine = V6ExecutionEngine(config)
        signal = V6Signal(
            symbol="BTCUSDT",
            computed_at="2026-07-20T05:13:24+00:00",
            price=64161.47,
            side_bias="long",
            long_probability=0.46,
            short_probability=0.46,
            wait_probability=0.08,
            model_quality_score=1.0,
            reasons=[],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            levels={},
        )
        btc_position = {
            "symbol": "BTCUSDT",
            "side": "long",
            "position_qty": 0.002,
            "entry_price": 64454.2,
            "current_price": 64161.47,
            "notional_usdt": 128.32,
        }
        state = {
            "positions": [
                btc_position,
                {
                    "symbol": "ETHUSDT",
                    "side": "long",
                    "position_qty": 0.058,
                    "entry_price": 1859.67,
                    "current_price": 1850.0,
                    "notional_usdt": 107.3,
                },
            ]
        }
        expected_intent = V6OrderIntent(
            action="scale_in",
            symbol="BTCUSDT",
            side="BUY",
            quantity_hint_usdt=100.0,
            leverage=8,
            reduce_only=False,
            entry_price=64161.47,
            stop_loss_price=63486.16,
            take_profit_price=65000.0,
            long_probability=0.46,
            short_probability=0.46,
            wait_probability=0.08,
            rationale="test scale-in at position cap",
            generated_at="2026-07-20T05:13:24+00:00",
        )
        try:
            with (
                patch.object(engine, "manager_reentry_lock_block", return_value=None),
                patch.object(engine, "manager_reversal_entry_block", return_value=None),
                patch.object(engine, "macro_direction_entry_block", return_value=None),
                patch.object(engine, "scale_in_intent_if_needed", return_value=expected_intent) as technical_scale_in,
            ):
                technical_intent = engine._entry_intent_for_signal(signal, state, [btc_position])
            self.assertIs(technical_intent, expected_intent)
            technical_scale_in.assert_called_once_with(
                signal,
                btc_position,
                require_probability=False,
                state=state,
            )

            config.technical_entry_only_symbols = ()
            with (
                patch.object(engine, "manager_reentry_lock_block", return_value=None),
                patch.object(engine, "manager_reversal_entry_block", return_value=None),
                patch.object(engine, "scale_in_intent_if_needed", return_value=expected_intent) as probability_scale_in,
            ):
                probability_intent = engine._entry_intent_for_signal(signal, state, [btc_position])
            self.assertIs(probability_intent, expected_intent)
            probability_scale_in.assert_called_once_with(signal, btc_position, state=state)

            btc_position["scale_in_count"] = config.max_scale_ins_per_position
            config.technical_entry_only_symbols = ("BTCUSDT",)
            with (
                patch.object(engine, "manager_reentry_lock_block", return_value=None),
                patch.object(engine, "manager_reversal_entry_block", return_value=None),
                patch.object(engine, "macro_direction_entry_block", return_value=None),
            ):
                self.assertIsNone(engine._entry_intent_for_signal(signal, state, [btc_position]))

            config.technical_entry_only_symbols = ()
            with (
                patch.object(engine, "manager_reentry_lock_block", return_value=None),
                patch.object(engine, "manager_reversal_entry_block", return_value=None),
            ):
                self.assertIsNone(engine._entry_intent_for_signal(signal, state, [btc_position]))
        finally:
            engine.close()

    def test_position_cap_still_blocks_a_new_symbol(self) -> None:
        config = V6ExecutionConfig(
            symbols=["BTCUSDT", "ETHUSDT", "ZECUSDT"],
            order_mode="dry_run",
            max_positions=2,
            long_entry_threshold=0.45,
            short_entry_threshold=0.45,
        )
        engine = V6ExecutionEngine(config)
        signal = V6Signal(
            symbol="ZECUSDT",
            computed_at="2026-07-20T05:13:24+00:00",
            price=550.0,
            side_bias="long",
            long_probability=0.70,
            short_probability=0.10,
            wait_probability=0.20,
            model_quality_score=1.0,
            reasons=[],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            levels={},
        )
        state = {
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "side": "long",
                    "position_qty": 0.002,
                    "entry_price": 64454.2,
                    "current_price": 64161.47,
                    "notional_usdt": 128.32,
                },
                {
                    "symbol": "ETHUSDT",
                    "side": "long",
                    "position_qty": 0.058,
                    "entry_price": 1859.67,
                    "current_price": 1850.0,
                    "notional_usdt": 107.3,
                },
            ]
        }
        try:
            with (
                patch.object(engine, "manager_reentry_lock_block", return_value=None),
                patch.object(engine, "manager_reversal_entry_block", return_value=None),
                patch.object(engine, "open_intent") as open_intent,
            ):
                self.assertIsNone(engine._entry_intent_for_signal(signal, state, []))
            open_intent.assert_not_called()

            config.technical_entry_only_symbols = ("ZECUSDT",)
            with (
                patch.object(engine, "manager_reentry_lock_block", return_value=None),
                patch.object(engine, "manager_reversal_entry_block", return_value=None),
                patch.object(engine, "macro_direction_entry_block", return_value=None),
                patch.object(engine, "technical_entry_ready", return_value=(True, None, {})) as technical_ready,
                patch.object(engine, "open_intent") as technical_open_intent,
            ):
                self.assertIsNone(engine._entry_intent_for_signal(signal, state, []))
            technical_ready.assert_not_called()
            technical_open_intent.assert_not_called()
        finally:
            engine.close()


class V6LiveTradingGateTest(unittest.TestCase):
    def test_live_order_mutating_signed_request_requires_confirmation(self) -> None:
        config = V6ExecutionConfig(
            order_mode="live_order",
            live_api_key="live-key",
            live_api_secret="live-secret",
            live_trading_confirmation="",
        )
        client = V6BinanceClient(config)
        try:
            with self.assertRaisesRegex(V6ExecutionError, "TS_V6_LIVE_TRADING_CONFIRMED"):
                client.signed_request("POST", "/fapi/v1/order", {"symbol": "BTCUSDT", "timestamp": 1})
        finally:
            client.close()

    def test_live_order_mutating_signed_request_allows_explicit_confirmation(self) -> None:
        config = V6ExecutionConfig(
            order_mode="live_order",
            live_api_key="live-key",
            live_api_secret="live-secret",
            live_trading_confirmation=LIVE_TRADING_CONFIRM_VALUE,
        )
        client = V6BinanceClient(config)

        class DummyResponse:
            status_code = 200
            text = '{"status":"ok"}'

            def json(self) -> dict[str, Any]:
                return {"status": "ok"}

        captured: dict[str, Any] = {}

        def fake_request(method: str, url: str, **kwargs: Any) -> DummyResponse:
            captured["method"] = method
            captured["url"] = url
            captured["data"] = kwargs.get("data")
            return DummyResponse()

        client._request = fake_request  # type: ignore[method-assign]
        try:
            result = client.signed_request("POST", "/fapi/v1/order", {"symbol": "BTCUSDT", "timestamp": 1})
        finally:
            client.close()

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(captured["method"], "POST")
        self.assertIn("signature=", str(captured["data"]))

    def test_exchange_info_cache_is_shared_across_short_lived_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = V6ExecutionConfig(
                order_mode="dry_run",
                kline_cache_dir=Path(tmp),
                exchange_info_cache_ttl_seconds=180,
            )
            payload = {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING"}]}
            first = V6BinanceClient(config)
            second = V6BinanceClient(config)
            try:
                with patch.object(first, "public_get", return_value=payload) as first_get:
                    self.assertEqual(first.exchange_info(), payload)
                with patch.object(second, "public_get", side_effect=AssertionError("disk cache should be used")) as second_get:
                    self.assertEqual(second.exchange_info(), payload)
                first_get.assert_called_once()
                second_get.assert_not_called()
            finally:
                first.close()
                second.close()


class V6DailyTrendTwoStepExitTest(unittest.TestCase):
    def make_engine(self) -> V6ExecutionEngine:
        return V6ExecutionEngine(V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run"))

    def make_signal(self, levels: dict[str, Any]) -> V6Signal:
        return V6Signal(
            symbol="BTCUSDT",
            computed_at="2026-07-04T00:00:00+00:00",
            price=100.0,
            side_bias="wait",
            long_probability=0.33,
            short_probability=0.33,
            wait_probability=0.34,
            model_quality_score=1.0,
            reasons=[],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            levels=levels,
        )

    def test_short_daily_ma7_reclaim_is_only_watch_until_ma20_second_checkpoint(self) -> None:
        engine = self.make_engine()
        try:
            signal = self.make_signal(
                {
                    "market_daily_ma7_reversal": {
                        "enabled": True,
                        "ready": True,
                        "reversal_confirmed": False,
                        "ma7_reclaim_confirmed": True,
                        "ma20_reclaim_confirmed": False,
                        "ma7_held_count": 2,
                        "ma20_held_count": 0,
                        "held_count": 0,
                        "reclaimed_count": 2,
                        "market_symbols": ["BTCUSDT", "ETHUSDT"],
                        "require_all": True,
                    }
                }
            )
            position = {"symbol": "BTCUSDT", "side": "short"}

            self.assertIsNone(engine.daily_ma7_short_reversal_exit_candidate(signal, position))
            self.assertEqual(position["daily_ma7_short_reversal_watch"]["status"], "watching")

            signal = self.make_signal(
                {
                    "market_daily_ma7_reversal": {
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
                }
            )

            result = engine.daily_ma7_short_reversal_exit_candidate(signal, position)

            self.assertIsNotNone(result)
            self.assertEqual(result["status"], "exit_ready")
            self.assertIn("MA20", result["reason"])
        finally:
            engine.close()

    def test_long_daily_ma7_loss_is_only_watch_until_ma20_second_checkpoint(self) -> None:
        engine = self.make_engine()
        try:
            signal = self.make_signal(
                {
                    "daily_ma7_trend": {
                        "ready": True,
                        "ma7_lost": True,
                        "ma7_loss_held": True,
                        "ma20_lost": False,
                        "ma20_loss_held": False,
                        "long_reversal_on_ma20_loss": False,
                    }
                }
            )
            position = {"symbol": "BTCUSDT", "side": "long"}

            self.assertIsNone(engine.daily_ma20_long_reversal_exit_candidate(signal, position))
            self.assertEqual(position["daily_ma20_long_reversal_watch"]["status"], "watching")

            signal = self.make_signal(
                {
                    "daily_ma7_trend": {
                        "ready": True,
                        "ma7_lost": True,
                        "ma7_loss_held": True,
                        "ma20_lost": True,
                        "ma20_loss_held": True,
                        "long_reversal_on_ma20_loss": True,
                    }
                }
            )

            result = engine.daily_ma20_long_reversal_exit_candidate(signal, position)

            self.assertIsNotNone(result)
            self.assertEqual(result["status"], "exit_ready")
            self.assertIn("MA20", result["reason"])
        finally:
            engine.close()


class V6ExecutionStateTest(unittest.TestCase):
    def make_engine(self) -> V6ExecutionEngine:
        config = V6ExecutionConfig(
            symbols=["BTCUSDT"],
            state_dir=Path("data/trading_system/v6_execution_test"),
            state_file=Path("data/trading_system/v6_execution_test/state.json"),
            signals_file=Path("data/trading_system/v6_execution_test/signals.jsonl"),
            orders_file=Path("data/trading_system/v6_execution_test/orders.jsonl"),
            wal_file=Path("data/trading_system/v6_execution_test/execution_wal.jsonl"),
            latest_summary_file=Path("data/trading_system/v6_execution_test/latest_run_summary.json"),
            closed_trade_review_queue_file=Path("data/trading_system/v6_execution_test/closed_trade_review_queue.jsonl"),
            post_trade_training_samples_file=Path("data/trading_system/v6_execution_test/post_trade_training_samples.jsonl"),
        )
        return V6ExecutionEngine(config)

    def test_confirmed_manager_full_close_creates_reentry_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = V6ExecutionConfig(
                symbols=["BTCUSDT"],
                order_mode="dry_run",
                state_dir=root,
                state_file=root / "state.json",
                signals_file=root / "signals.jsonl",
                orders_file=root / "orders.jsonl",
                wal_file=root / "execution_wal.jsonl",
                latest_summary_file=root / "latest_run_summary.json",
                closed_trade_review_queue_file=root / "closed_trade_review_queue.jsonl",
                post_trade_training_samples_file=root / "post_trade_training_samples.jsonl",
                close_resolver_enabled=False,
            )
            engine = V6ExecutionEngine(config)
            state = {
                "positions": [{
                    "symbol": "BTCUSDT",
                    "side": "short",
                    "entry_price": 100.0,
                    "position_qty": 1.0,
                    "notional_usdt": 100.0,
                    "leverage": 8,
                    "risk_plan": {"horizon": "4h_trend", "entry_timeframe": "4h"},
                }],
                "journal": [],
            }
            intent = V6OrderIntent(
                action="close_short",
                symbol="BTCUSDT",
                side="BUY",
                quantity_hint_usdt=100.0,
                leverage=8,
                reduce_only=True,
                entry_price=99.0,
                stop_loss_price=103.0,
                take_profit_price=90.0,
                long_probability=0.6,
                short_probability=0.2,
                wait_probability=0.2,
                rationale="short original 4H logic weakened for 2 closed 4H candles",
                generated_at="2026-07-19T00:00:00+00:00",
                risk_plan={"logic_weakening_exit": {"status": "exit_ready", "timeframe": "4h"}},
            )
            try:
                engine.apply_intent_to_state(
                    state,
                    intent,
                    {
                        "status": "submitted",
                        "submitted_at": "2026-07-19T00:00:01+00:00",
                        "response": {"orderId": 123, "avgPrice": "99.0", "executedQty": "1.0"},
                        "position_reconcile": {
                            "confirmation_state": "confirmed",
                            "action": "close_short",
                            "after_state": "flat",
                            "delta_qty_abs": 1.0,
                        },
                    },
                )
            finally:
                engine.close()

        self.assertEqual(state["positions"], [])
        lock = state["manager_reentry_locks"]["BTCUSDT:short"]
        self.assertEqual(lock["reason_type"], "logic_weakening_exit")
        self.assertEqual(lock["timeframe"], "4h")
        self.assertEqual(lock["close_order_id"], 123)
        observation = state["manager_reversal_observations"]["BTCUSDT"]
        self.assertEqual(observation["closed_side"], "short")
        self.assertEqual(observation["target_side"], "long")
        self.assertEqual(observation["close_order_id"], 123)

    def test_unconfirmed_manager_close_does_not_create_reentry_lock(self) -> None:
        engine = self.make_engine()
        state = {
            "positions": [{"symbol": "BTCUSDT", "side": "long", "position_qty": 1.0, "entry_price": 100.0}],
            "journal": [],
        }
        intent = V6OrderIntent(
            action="close_long",
            symbol="BTCUSDT",
            side="SELL",
            quantity_hint_usdt=100.0,
            leverage=8,
            reduce_only=True,
            entry_price=99.0,
            stop_loss_price=97.0,
            take_profit_price=110.0,
            long_probability=0.2,
            short_probability=0.6,
            wait_probability=0.2,
            rationale="long original 4H logic weakened for 2 closed 4H candles",
            generated_at="2026-07-19T00:00:00+00:00",
            risk_plan={"logic_weakening_exit": {"status": "exit_ready", "timeframe": "4h"}},
        )
        try:
            engine.apply_intent_to_state(
                state,
                intent,
                {
                    "status": "submitted_no_position_delta",
                    "submitted_at": "2026-07-19T00:00:01+00:00",
                    "position_reconcile": {
                        "confirmation_state": "unconfirmed",
                        "action": "close_long",
                        "after_state": "open",
                        "delta_qty_abs": 0.0,
                    },
                },
            )
        finally:
            engine.close()

        self.assertEqual(len(state["positions"]), 1)
        self.assertFalse(state.get("manager_reentry_locks"))
        self.assertFalse(state.get("manager_reversal_observations"))

    def test_v5_overlay_manager_close_does_not_create_v6_reentry_lock_by_default(self) -> None:
        engine = self.make_engine()
        engine.config.v5_overlay_rules_enabled = True
        engine.config.technical_entry_only_symbols = ("BTCUSDT",)
        position = {"symbol": "BTCUSDT", "side": "short", "position_qty": 1.0, "entry_price": 100.0}
        intent = V6OrderIntent(
            action="close_short",
            symbol="BTCUSDT",
            side="BUY",
            quantity_hint_usdt=100.0,
            leverage=8,
            reduce_only=True,
            entry_price=99.0,
            stop_loss_price=103.0,
            take_profit_price=90.0,
            long_probability=0.6,
            short_probability=0.2,
            wait_probability=0.2,
            rationale="short original 4H logic weakened for 2 closed 4H candles",
            generated_at="2026-07-19T00:00:00+00:00",
            risk_plan={"logic_weakening_exit": {"status": "exit_ready", "timeframe": "4h"}},
        )
        state = {"manager_reentry_locks": {}, "manager_reentry_lock_history": []}
        result = {
            "status": "submitted",
            "submitted_at": "2026-07-19T00:00:01+00:00",
            "response": {"orderId": 456, "avgPrice": "99.0", "executedQty": "1.0"},
        }
        try:
            lock = engine.register_manager_reentry_lock(state, position, intent, result)
        finally:
            engine.close()

        self.assertIsNone(lock)
        self.assertFalse(state["manager_reentry_locks"])
        self.assertFalse(state.get("manager_reversal_observations"))

    def test_confirmed_opposite_open_completes_manager_reversal_observation(self) -> None:
        engine = self.make_engine()
        state = {
            "positions": [],
            "journal": [],
            "manager_reversal_observations": {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "closed_side": "short",
                    "target_side": "long",
                    "reason_type": "logic_weakening_exit",
                    "created_at": "2026-07-19T00:00:00+00:00",
                    "minimum_until": "2026-07-19T04:00:00+00:00",
                    "expires_at": "2099-07-22T00:00:00+00:00",
                }
            },
            "manager_reversal_observation_history": [],
        }
        intent = V6OrderIntent(
            action="open_long",
            symbol="BTCUSDT",
            side="BUY",
            quantity_hint_usdt=100.0,
            leverage=8,
            reduce_only=False,
            entry_price=101.0,
            stop_loss_price=98.0,
            take_profit_price=110.0,
            long_probability=0.6,
            short_probability=0.2,
            wait_probability=0.2,
            rationale="confirmed manager reversal",
            generated_at="2026-07-19T08:00:00+00:00",
            risk_plan={
                "horizon": "4h_trend",
                "entry_timeframe": "4h",
                "manager_reversal_confirmation": {"approved": True},
            },
        )
        result = {
            "status": "submitted",
            "submitted_at": "2026-07-19T08:00:01+00:00",
            "response": {"orderId": 789, "avgPrice": "101.0", "executedQty": "1.0"},
            "position_reconcile": {
                "confirmation_state": "confirmed",
                "action": "open_long",
                "delta_qty_abs": 1.0,
                "after": {
                    "position_amt": "1.0",
                    "entry_price": "101.0",
                    "mark_price": "101.0",
                    "notional": "101.0",
                },
            },
        }
        try:
            engine.apply_intent_to_state(state, intent, result)
        finally:
            engine.close()

        self.assertEqual(len(state["positions"]), 1)
        self.assertFalse(state["manager_reversal_observations"])
        event = state["manager_reversal_observation_history"][-1]
        self.assertEqual(event["event"], "completed")
        self.assertEqual(event["open_order_id"], 789)

    def test_unconfirmed_open_keeps_manager_reversal_observation(self) -> None:
        engine = self.make_engine()
        observation = {
            "symbol": "BTCUSDT",
            "closed_side": "short",
            "target_side": "long",
            "reason_type": "logic_weakening_exit",
            "created_at": "2026-07-19T00:00:00+00:00",
            "minimum_until": "2026-07-19T04:00:00+00:00",
            "expires_at": "2099-07-22T00:00:00+00:00",
        }
        state = {
            "positions": [],
            "journal": [],
            "manager_reversal_observations": {"BTCUSDT": observation},
            "manager_reversal_observation_history": [],
        }
        intent = V6OrderIntent(
            action="open_long",
            symbol="BTCUSDT",
            side="BUY",
            quantity_hint_usdt=100.0,
            leverage=8,
            reduce_only=False,
            entry_price=101.0,
            stop_loss_price=98.0,
            take_profit_price=110.0,
            long_probability=0.6,
            short_probability=0.2,
            wait_probability=0.2,
            rationale="unconfirmed manager reversal",
            generated_at="2026-07-19T08:00:00+00:00",
        )
        try:
            engine.apply_intent_to_state(
                state,
                intent,
                {
                    "status": "submitted_no_position_delta",
                    "submitted_at": "2026-07-19T08:00:01+00:00",
                    "position_reconcile": {
                        "confirmation_state": "unconfirmed",
                        "action": "open_long",
                        "delta_qty_abs": 0.0,
                    },
                },
            )
        finally:
            engine.close()

        self.assertEqual(state["manager_reversal_observations"]["BTCUSDT"], observation)
        self.assertFalse(state["manager_reversal_observation_history"])

    def test_confirmed_old_side_reopen_cancels_manager_reversal_observation(self) -> None:
        engine = self.make_engine()
        state = {
            "manager_reversal_observations": {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "closed_side": "short",
                    "target_side": "long",
                    "reason_type": "logic_weakening_exit",
                    "created_at": "2026-07-19T00:00:00+00:00",
                    "minimum_until": "2026-07-19T04:00:00+00:00",
                    "expires_at": "2099-07-22T00:00:00+00:00",
                }
            },
            "manager_reversal_observation_history": [],
        }
        intent = V6OrderIntent(
            action="open_short",
            symbol="BTCUSDT",
            side="SELL",
            quantity_hint_usdt=100.0,
            leverage=8,
            reduce_only=False,
            entry_price=99.0,
            stop_loss_price=102.0,
            take_profit_price=90.0,
            long_probability=0.2,
            short_probability=0.6,
            wait_probability=0.2,
            rationale="old side revalidated",
            generated_at="2026-07-19T08:00:00+00:00",
        )
        try:
            engine.complete_manager_reversal_observation_after_open(
                state,
                intent,
                {"submitted_at": "2026-07-19T08:00:01+00:00", "response": {"orderId": 790}},
            )
        finally:
            engine.close()

        self.assertFalse(state["manager_reversal_observations"])
        event = state["manager_reversal_observation_history"][-1]
        self.assertEqual(event["event"], "cancelled")
        self.assertEqual(event["resolve_reason"], "closed_side_revalidated_and_reopened")

    def test_strategy_dust_position_is_ignored_for_entry_and_scale_in(self) -> None:
        config = V6ExecutionConfig(
            dust_position_notional_usdt=1.0,
            strategy_dust_position_notional_usdt=5.0,
        )
        engine = V6ExecutionEngine(config)
        state = {
            "positions": [
                {
                    "symbol": "SOLUSDT",
                    "side": "long",
                    "position_qty": 0.03,
                    "entry_price": 70.0,
                    "current_price": 72.0,
                    "notional_usdt": 2.16,
                }
            ]
        }

        self.assertEqual(engine.find_positions(state, "SOLUSDT"), [])
        self.assertIsNone(engine.find_position(state, "SOLUSDT", side="long"))
        self.assertEqual(len(engine.find_positions(state, "SOLUSDT", include_strategy_dust=True)), 1)
        self.assertEqual(state["positions"][0]["strategy_status"], "dust_ignored")
        self.assertTrue(state["positions"][0]["strategy_dust_ignored"])
        engine.close()

    def test_v5_staged_entry_uses_dedicated_order_floor_instead_of_v6_base_floor(self) -> None:
        config = V6ExecutionConfig(
            symbols=["ALTUSDT"],
            base_notional_usdt=510.0,
            min_notional_usdt=510.0,
            v5_overlay_rules_enabled=True,
            v5_staged_entry_enabled=True,
            v5_initial_entry_notional_fraction=0.30,
            v5_staged_entry_min_notional_usdt=5.0,
            smart_signal_risk_guard_enabled=False,
        )
        engine = V6ExecutionEngine(config)
        signal = SimpleNamespace(
            symbol="ALTUSDT",
            price=100.0,
            levels={
                "nearest_supports": [("bb210_lower_15m", 95.0)],
                "nearest_resistances": [("bb96_mid_15m", 105.0)],
                "all_supports": [("bb210_lower_15m", 95.0)],
                "all_resistances": [("bb96_mid_15m", 105.0)],
                "structure_entry_scan": {"long_candidate_count": 1},
            },
            long_probability=0.7,
            short_probability=0.1,
            wait_probability=0.2,
            indicators={},
            reasons=["test staged entry"],
        )
        try:
            intent = engine.open_intent(signal, "open_long")
        finally:
            engine.close()

        self.assertAlmostEqual(intent.quantity_hint_usdt, 153.0)
        self.assertEqual(intent.risk_plan["v5_staged_entry"]["order_min_notional_usdt"], 5.0)
        self.assertEqual(intent.risk_plan["v5_staged_entry"]["initial_notional_usdt"], 153.0)

    def test_v5_overlay_uses_structured_risk_flags_not_reason_text(self) -> None:
        config = V6ExecutionConfig(
            v5_overlay_rules_enabled=True,
            v5_block_unclean_15m_ma20_initial_entry=True,
        )
        engine = V6ExecutionEngine(config)
        signal = SimpleNamespace(
            symbol="ALTUSDT",
            price=100.0,
            reasons=["这段文案可以任意修改"],
            fifteen_m={},
            mid_retest={},
            levels={},
            model_quality_score=0.5,
            indicators={
                "entry_risk_flags": {
                    "schema_version": 1,
                    "ma20_trend_broken_or_mixed": True,
                    "mixed_bb_structure": False,
                    "avoid_chasing_4h_mids": False,
                    "low_model_quality": False,
                }
            },
        )
        try:
            blocked = engine.v5_overlay_entry_block(signal, "open_long", {"selected_entry_level": {}})
        finally:
            engine.close()

        self.assertIsNotNone(blocked)
        self.assertEqual(blocked["reason"], "v5_wait_clean_15m_ma20_pullback")

    def test_intents_for_signal_returns_risk_reductions_for_both_hedge_sides(self) -> None:
        engine = self.make_engine()
        signal = SimpleNamespace(symbol="BTCUSDT")
        state = {
            "positions": [
                {"symbol": "BTCUSDT", "side": "long", "position_qty": 1.0, "entry_price": 100.0, "notional_usdt": 100.0},
                {"symbol": "BTCUSDT", "side": "short", "position_qty": 1.0, "entry_price": 100.0, "notional_usdt": 100.0},
            ]
        }

        def close_for_position(
            _signal: Any,
            position: dict[str, Any],
            *,
            state: dict[str, Any] | None = None,
        ) -> V6OrderIntent:
            self.assertIsNotNone(state)
            side = str(position["side"])
            action = f"close_{side}"
            return V6OrderIntent(
                action=action,
                symbol="BTCUSDT",
                side="SELL" if side == "long" else "BUY",
                quantity_hint_usdt=100.0,
                leverage=8,
                reduce_only=True,
                entry_price=100.0,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                long_probability=0.5,
                short_probability=0.5,
                wait_probability=0.0,
                rationale="risk reduction",
                generated_at="2026-07-18T00:00:00+00:00",
            )

        try:
            with (
                patch.object(engine, "refresh_cross_timeframe_macro_observation", return_value=None),
                patch.object(engine, "exit_intent_if_needed", side_effect=close_for_position),
            ):
                intents = engine.intents_for_signal(signal, state)
        finally:
            engine.close()

        self.assertEqual([item.action for item in intents], ["close_long", "close_short"])

    def test_exchange_snapshot_below_strategy_dust_threshold_is_dust(self) -> None:
        config = V6ExecutionConfig(
            dust_position_notional_usdt=1.0,
            strategy_dust_position_notional_usdt=5.0,
        )
        client = V6BinanceClient(config)
        try:
            snapshot = {
                "symbol": "SOLUSDT",
                "side": "long",
                "position_amt": 0.03,
                "abs_position_qty": 0.03,
                "mark_price": 72.0,
                "entry_price": 70.0,
                "notional": 2.16,
            }

            self.assertTrue(client.is_dust_exchange_snapshot(snapshot))
        finally:
            client.close()

    @staticmethod
    def kline(open_time: int, close: float, *, high: float | None = None, low: float | None = None) -> list[float]:
        high_value = high if high is not None else close
        low_value = low if low is not None else close
        return [open_time, close, high_value, low_value, close]

    def test_intent_for_signal_uses_symbol_specific_entry_threshold(self) -> None:
        config = V6ExecutionConfig(
            short_entry_threshold=0.60,
            long_entry_threshold=0.60,
            symbol_short_entry_thresholds={"BTCUSDT": 0.45},
            entry_structure_filter_enabled=False,
        )
        engine = V6ExecutionEngine(config)
        signal = SimpleNamespace(
            symbol="BTCUSDT",
            price=100.0,
            long_probability=0.1,
            short_probability=0.46,
            wait_probability=0.2,
            levels={
                "nearest_supports": [("bb96_lower_15m", 95.0)],
                "nearest_resistances": [("bb96_mid_15m", 105.0)],
                "all_supports": [("bb96_lower_15m", 95.0)],
                "all_resistances": [("bb96_mid_15m", 105.0)],
                "structure_entry_scan": {"short_candidate_count": 1},
            },
            reasons=["test"],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            indicators={},
        )

        intent = engine.intent_for_signal(signal, {"positions": []})
        signal.symbol = "HYPEUSDT"
        non_override_intent = engine.intent_for_signal(signal, {"positions": []})
        engine.close()

        self.assertIsNotNone(intent)
        self.assertEqual(intent.action, "open_short")
        self.assertIsNone(non_override_intent)

    def test_fifteen_m_state_detects_failed_long_cycle(self) -> None:
        engine = self.make_engine()
        bb = SimpleNamespace(
            bb1440_mid_15m=100.0,
            bb1440_lower_15m=80.0,
            bb_price_position_1440_15m=0.2,
            bb1920_mid_15m=99.0,
            bb1920_lower_15m=79.0,
            bb_price_position_1920_15m=0.25,
            bb2880_mid_15m=98.0,
            bb2880_lower_15m=78.0,
            bb_price_position_2880_15m=0.3,
            bb4707_mid_15m=96.0,
            bb4707_lower_15m=76.0,
            bb_price_position_4707_15m=0.35,
        )

        state = engine.fifteen_m_state(95.0, bb)
        engine.close()

        self.assertTrue(state["long_cycle_failed"])
        self.assertEqual(state["below_mid_count"], 4)
        self.assertEqual(state["below_lower_count"], 0)
        self.assertIn("bb4707", state["below_mid"])

    def test_fetch_klines_with_cache_backfills_and_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = V6ExecutionConfig(
                state_dir=Path(tmp) / "state",
                kline_cache_dir=Path(tmp) / "kline_cache",
                kline_refresh_interval_1m_seconds=0,
            )
            client = V6BinanceClient(config)
            client.save_kline_cache(
                "BTCUSDT",
                "1m",
                [
                    [1_000, "1", "1", "1", "1"],
                    [2_000, "2", "2", "2", "2"],
                ],
                earliest_exhausted=False,
            )
            calls = []

            def fake_forward(symbol, interval, *, start_time, limit):
                calls.append(("forward", symbol, interval, start_time, limit))
                return [[3_000, "3", "3", "3", "3"]]

            def fake_backward(symbol, interval, limit, *, end_time=None):
                calls.append(("backward", symbol, interval, limit, end_time))
                if end_time is None:
                    return [[3_000, "3", "3", "3", "3"]], False
                return [[0, "0", "0", "0", "0"]], True

            client.fetch_klines_from_api_forward = fake_forward  # type: ignore[method-assign]
            client.fetch_klines_from_api_backward = fake_backward  # type: ignore[method-assign]

            rows = client.fetch_klines("BTCUSDT", "1m", 4)
            payload = client.load_kline_cache("BTCUSDT", "1m")
            client.close()

        self.assertEqual([row[0] for row in rows], [0, 1000, 2000, 3000])
        self.assertEqual(calls[0], ("backward", "BTCUSDT", "1m", 4, None))
        self.assertEqual(calls[1], ("backward", "BTCUSDT", "1m", 1, 999))
        self.assertTrue(all(call[0] != "forward" for call in calls))
        self.assertTrue(payload["earliest_exhausted"])
        self.assertEqual([row[0] for row in payload["klines"]], [0, 1000, 2000, 3000])

    def test_fetch_klines_with_cache_skips_tail_refresh_until_interval_expires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = V6ExecutionConfig(
                state_dir=Path(tmp) / "state",
                kline_cache_dir=Path(tmp) / "kline_cache",
                kline_refresh_interval_15m_seconds=120,
            )
            client = V6BinanceClient(config)
            try:
                client.save_kline_cache(
                    "BTCUSDT",
                    "15m",
                    [
                        [1_000, "1", "1", "1", "1"],
                        [2_000, "2", "2", "2", "2"],
                        [3_000, "3", "3", "3", "3"],
                    ],
                    earliest_exhausted=True,
                )

                def fake_backward(*args, **kwargs):
                    raise AssertionError("tail refresh should be skipped while cache is fresh")

                client.fetch_klines_from_api_backward = fake_backward  # type: ignore[method-assign]
                rows = client.fetch_klines("BTCUSDT", "15m", 3)
            finally:
                client.close()

        self.assertEqual([row[0] for row in rows], [1_000, 2_000, 3_000])

    def test_fetch_klines_with_cache_refreshes_tail_after_interval_expires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = V6ExecutionConfig(
                state_dir=Path(tmp) / "state",
                kline_cache_dir=Path(tmp) / "kline_cache",
                kline_refresh_interval_15m_seconds=120,
            )
            client = V6BinanceClient(config)
            try:
                client.save_kline_cache(
                    "BTCUSDT",
                    "15m",
                    [
                        [1_000, "1", "1", "1", "1"],
                        [2_000, "2", "2", "2", "2"],
                    ],
                    earliest_exhausted=True,
                )
                cache_path = client.kline_cache_path("BTCUSDT", "15m")
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                payload["updated_at_epoch"] = 1.0
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
                calls = []

                def fake_backward(symbol, interval, limit, *, end_time=None):
                    calls.append((symbol, interval, limit, end_time))
                    return [[3_000, "3", "3", "3", "3"]], False

                client.fetch_klines_from_api_backward = fake_backward  # type: ignore[method-assign]
                rows = client.fetch_klines("BTCUSDT", "15m", 3)
            finally:
                client.close()

        self.assertEqual(calls, [("BTCUSDT", "15m", 3, None)])
        self.assertEqual([row[0] for row in rows], [1_000, 2_000, 3_000])

    def test_public_market_metadata_uses_endpoint_specific_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = V6ExecutionConfig(
                state_dir=Path(tmp) / "state",
                kline_cache_dir=Path(tmp) / "kline_cache",
                public_mark_price_ttl_seconds=120,
                public_ticker_24h_ttl_seconds=300,
            )
            client = V6BinanceClient(config)
            try:
                client.save_public_cache("premiumIndex", "BTCUSDT", {"markPrice": "100"})
                client.save_public_cache("ticker_24hr", "BTCUSDT", {"lastPrice": "101"})

                def fake_public_get(*args, **kwargs):
                    raise AssertionError("fresh public cache should be used")

                client.public_get = fake_public_get  # type: ignore[method-assign]

                mark_price = client.mark_price("BTCUSDT")
                ticker = client.ticker_24h("BTCUSDT")
            finally:
                client.close()

        self.assertEqual(mark_price, 100.0)
        self.assertEqual(ticker["lastPrice"], "101")

    def test_4h_supplement_fills_bb2880_window_without_overwriting_binance_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = V6ExecutionConfig(
                state_dir=Path(tmp) / "state",
                kline_cache_dir=Path(tmp) / "kline_cache",
            )
            client = V6BinanceClient(config)
            binance_rows = [[i, "b", "b", "b", "binance"] for i in range(2000, 4229)]
            supplemental_rows = [[i, "x", "x", "x", "external"] for i in range(4707)]
            calls = []

            def fake_supplemental(symbol, limit):
                calls.append((symbol, limit))
                return supplemental_rows

            client.fetch_supplemental_4h_klines = fake_supplemental  # type: ignore[method-assign]
            rows = client.supplement_4h_klines_for_bb2880("HYPEUSDT", binance_rows, 4707)
            client.close()

        self.assertEqual(calls, [("HYPEUSDT", 4707)])
        self.assertGreaterEqual(len(rows), 4707)
        by_open = {row[0]: row for row in rows}
        self.assertEqual(by_open[1999][4], "external")
        self.assertEqual(by_open[2000][4], "binance")

    def test_4h_supplement_skips_when_binance_already_has_bb4707_window(self) -> None:
        config = V6ExecutionConfig()
        client = V6BinanceClient(config)
        binance_rows = [[i, "b", "b", "b", "binance"] for i in range(4707)]
        client.fetch_supplemental_4h_klines = lambda symbol, limit: (_ for _ in ()).throw(AssertionError("should not fetch"))  # type: ignore[method-assign]

        rows = client.supplement_4h_klines_for_bb2880("BTCUSDT", binance_rows, 4707)
        client.close()

        self.assertEqual(rows, binance_rows)

    def test_fifteen_m_state_exposes_bb672_lower_context(self) -> None:
        engine = self.make_engine()
        bb = SimpleNamespace(
            bb96_mid_15m=65.0,
            bb96_lower_15m=61.0,
            bb_price_position_96=0.05,
            bb120_mid_15m=66.0,
            bb120_lower_15m=60.0,
            bb_price_position_120=0.12,
            bb210_mid_15m=69.0,
            bb210_lower_15m=61.3,
            bb_price_position_210=0.01,
            bb300_mid_15m=70.0,
            bb300_lower_15m=62.8,
            bb_price_position_300=-0.08,
            bb672_mid_15m=69.4,
            bb672_lower_15m=62.25,
            bb_price_position_672_15m=-0.05,
            bb1440_mid_15m=64.0,
            bb1440_lower_15m=50.0,
            bb_price_position_1440_15m=0.4,
            bb1920_mid_15m=60.0,
            bb1920_lower_15m=40.0,
            bb_price_position_1920_15m=0.55,
            bb2880_mid_15m=54.0,
            bb2880_lower_15m=30.0,
            bb_price_position_2880_15m=0.67,
            bb4707_mid_15m=49.0,
            bb4707_lower_15m=27.0,
            bb_price_position_4707_15m=0.78,
        )

        state = engine.fifteen_m_state(61.5271381, bb)
        engine.close()

        self.assertTrue(state["bb672_lower_active"])
        self.assertIn("bb672", state["below_lower"])
        self.assertIn("bb672", state["near_lower"])
        self.assertIn("bb672", state["positions"])
        self.assertAlmostEqual(state["bb672_lower_distance_pct"], -1.161224, places=6)
        self.assertFalse(state["long_cycle_failed"])

    def test_ma20_trend_state_uses_two_closed_15m_candles_for_hold_and_rejection(self) -> None:
        engine = self.make_engine()
        bb = SimpleNamespace(ma20_15m=100.0)
        bullish_rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(20)]
        bullish_rows.extend([
            self.kline(20, 101.0, high=102.0, low=100.2),
            self.kline(21, 101.2, high=102.0, low=100.3),
            self.kline(22, 99.0, high=103.0, low=98.0),
        ])
        bearish_rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(20)]
        bearish_rows.extend([
            self.kline(20, 99.0, high=99.5, low=98.5),
            self.kline(21, 98.8, high=99.4, low=98.0),
            self.kline(22, 101.0, high=102.0, low=97.0),
        ])

        bullish = engine.ma20_trend_state(101.0, bb, bullish_rows)
        bearish = engine.ma20_trend_state(99.0, bb, bearish_rows)
        engine.close()

        self.assertTrue(bullish["ready"])
        self.assertTrue(bullish["uses_closed_candles"])
        self.assertEqual(bullish["checked_candles"], 2)
        self.assertEqual(bullish["state"], "bullish_trend_alive")
        self.assertTrue(bullish["bullish_trend_alive"])
        self.assertFalse(bullish["bearish_trend_alive"])
        self.assertEqual(bullish["candles"][-1]["open_time"], 21)

        self.assertTrue(bearish["ready"])
        self.assertEqual(bearish["checked_candles"], 2)
        self.assertEqual(bearish["state"], "bearish_trend_alive")
        self.assertTrue(bearish["bearish_trend_alive"])
        self.assertFalse(bearish["bullish_trend_alive"])
        self.assertEqual(bearish["candles"][-1]["open_time"], 21)

    def test_ma20_trend_entry_block_blocks_only_opposite_15m_direction(self) -> None:
        engine = self.make_engine()
        bullish_signal = SimpleNamespace(
            fifteen_m={"ma20_trend": {"ready": True, "bullish_trend_alive": True, "bearish_trend_alive": False}}
        )
        bearish_signal = SimpleNamespace(
            fifteen_m={"ma20_trend": {"ready": True, "bullish_trend_alive": False, "bearish_trend_alive": True}}
        )

        short_block = engine.ma20_trend_entry_block(bullish_signal, "open_short")
        long_block = engine.ma20_trend_entry_block(bearish_signal, "open_long")
        allowed_long = engine.ma20_trend_entry_block(bullish_signal, "open_long")
        allowed_short = engine.ma20_trend_entry_block(bearish_signal, "open_short")
        engine.close()

        self.assertEqual(short_block["reason"], "short_blocked_15m_ma20_trend_alive")
        self.assertEqual(long_block["reason"], "long_blocked_15m_ma20_downtrend_alive")
        self.assertIsNone(allowed_long)
        self.assertIsNone(allowed_short)

    def test_h4_right_side_state_requires_cross_and_retest_before_trend_confirmation(self) -> None:
        engine = self.make_engine()
        bb = SimpleNamespace()
        long_rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(20)]
        long_rows.extend([
            self.kline(20, 101.0, high=102.0, low=100.2),
            self.kline(21, 101.2, high=102.0, low=99.8),
            self.kline(22, 101.5, high=102.5, low=100.4),
            self.kline(23, 101.8, high=102.6, low=100.6),
            self.kline(24, 102.0, high=102.8, low=100.7),
            self.kline(25, 99.0, high=103.0, low=98.0),
        ])
        short_rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(20)]
        short_rows.extend([
            self.kline(20, 99.0, high=99.8, low=98.0),
            self.kline(21, 98.8, high=100.2, low=98.0),
            self.kline(22, 98.5, high=99.5, low=97.5),
            self.kline(23, 98.2, high=99.2, low=97.0),
            self.kline(24, 98.0, high=99.0, low=96.8),
            self.kline(25, 101.0, high=102.0, low=97.0),
        ])

        long_state = engine.h4_right_side_state(102.0, bb, long_rows)
        short_state = engine.h4_right_side_state(98.0, bb, short_rows)
        engine.close()

        self.assertEqual(long_state["state"], "long_right_side")
        self.assertTrue(long_state["cross_up_seen"])
        self.assertIsNotNone(long_state["retest_hold"])
        self.assertTrue(long_state["long_confirmed"])
        self.assertFalse(long_state["short_confirmed"])

        self.assertEqual(short_state["state"], "short_right_side")
        self.assertTrue(short_state["cross_down_seen"])
        self.assertIsNotNone(short_state["retest_reject"])
        self.assertTrue(short_state["short_confirmed"])
        self.assertFalse(short_state["long_confirmed"])

    def test_daily_ma7_trend_state_locks_long_short_rebreak_and_bearish_ma7_reclaim(self) -> None:
        engine = self.make_engine()
        bb = SimpleNamespace()
        long_rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(20)]
        long_rows.extend([
            self.kline(20, 101.0, high=102.0, low=100.2),
            self.kline(21, 100.8, high=101.4, low=100.2),
            self.kline(22, 102.0, high=103.0, low=101.0),
            self.kline(23, 102.5, high=103.0, low=101.5),
            self.kline(24, 103.0, high=104.0, low=102.0),
            self.kline(25, 99.0, high=105.0, low=98.0),
        ])
        short_rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(20)]
        short_rows.extend([
            self.kline(20, 99.0, high=99.8, low=98.0),
            self.kline(21, 99.2, high=99.85, low=98.8),
            self.kline(22, 98.0, high=99.0, low=97.0),
            self.kline(23, 97.5, high=98.0, low=96.5),
            self.kline(24, 97.0, high=98.0, low=96.0),
            self.kline(25, 101.0, high=102.0, low=96.0),
        ])
        reclaim_rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(20)]
        reclaim_rows.extend([
            self.kline(20, 90.0, high=91.0, low=89.0),
            self.kline(21, 90.0, high=91.0, low=89.0),
            self.kline(22, 90.0, high=91.0, low=89.0),
            self.kline(23, 90.0, high=91.0, low=89.0),
            self.kline(24, 96.0, high=97.0, low=95.0),
            self.kline(25, 90.0, high=100.0, low=89.0),
        ])

        long_state = engine.daily_ma7_trend_state(103.0, bb, long_rows)
        short_state = engine.daily_ma7_trend_state(97.0, bb, short_rows)
        reclaim_state = engine.daily_ma7_trend_state(96.0, bb, reclaim_rows)
        engine.close()

        self.assertEqual(long_state["state"], "long_ma7_rebreak")
        self.assertTrue(long_state["bullish_stack"])
        self.assertTrue(long_state["long_confirmed"])
        self.assertFalse(long_state["short_confirmed"])
        self.assertEqual(long_state["entry_level"]["name"], "ma7_1d")

        self.assertEqual(short_state["state"], "short_ma7_rebreak")
        self.assertTrue(short_state["bearish_stack"])
        self.assertTrue(short_state["short_confirmed"])
        self.assertFalse(short_state["long_confirmed"])
        self.assertEqual(short_state["entry_level"]["name"], "ma7_1d")

        self.assertEqual(reclaim_state["state"], "bearish_stack_ma20_above_ma7")
        self.assertTrue(reclaim_state["bearish_stack"])
        self.assertTrue(reclaim_state["ma7_reclaimed"])
        self.assertTrue(reclaim_state["ma7_reclaim_held"])
        self.assertTrue(reclaim_state["short_reversal_on_ma7_reclaim"])
        self.assertEqual(reclaim_state["ma7_reclaim_hold"]["held_count"], 1)

    def test_trend_big_entry_block_requires_matching_4h_right_side_confirmation(self) -> None:
        engine = self.make_engine()
        signal = SimpleNamespace()

        blocked = engine.trend_big_entry_block(
            signal,
            "open_long",
            {"horizon": "4h_trend", "h4_right_side": {"ready": True, "long_confirmed": False}},
        )
        allowed = engine.trend_big_entry_block(
            signal,
            "open_long",
            {"horizon": "4h_trend", "h4_right_side": {"ready": True, "long_confirmed": True}},
        )
        engine.close()

        self.assertEqual(blocked["reason"], "trend_big_blocked_without_4h_right_side")
        self.assertIsNone(allowed)

    def test_structure_entry_scan_classifies_near_bb_and_ma_retests_without_1d_ma_entries(self) -> None:
        engine = self.make_engine()
        scan = engine.structure_entry_scan(
            100.0,
            [
                ("bb96_lower_15m", 99.6),
                ("bb120_mid_15m", 100.5),
                ("bb210_upper_4h", 100.4),
                ("bb300_lower_1d", 90.0),
            ],
            [
                ("ma20_15m", 99.5),
                ("ma60_4h", 100.6),
                ("ma20_1d", 99.0),
            ],
        )
        engine.close()

        long_by_name = {item["name"]: item for item in scan["long_candidates"]}
        short_by_name = {item["name"]: item for item in scan["short_candidates"]}
        near_names = {item["name"] for item in scan["near_levels"]}

        self.assertEqual(long_by_name["bb96_lower_15m"]["structure_signal"], "long_lower_support_rebound")
        self.assertEqual(long_by_name["ma20_15m"]["structure_signal"], "long_ma_support_retest")
        self.assertEqual(short_by_name["bb120_mid_15m"]["structure_signal"], "short_mid_resistance_retest")
        self.assertEqual(short_by_name["bb210_upper_4h"]["structure_signal"], "short_upper_rejection")
        self.assertEqual(short_by_name["ma60_4h"]["structure_signal"], "short_ma_resistance_retest")
        self.assertNotIn("bb300_lower_1d", near_names)
        self.assertNotIn("ma20_1d", near_names)

    def test_structure_entry_scan_adds_15m_ma_reclaim_cross_from_closed_klines(self) -> None:
        engine = self.make_engine()
        rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(20)]
        rows.extend([
            self.kline(20, 99.0, high=100.0, low=98.0),
            self.kline(21, 101.0, high=102.0, low=99.5),
            self.kline(22, 100.5, high=103.0, low=98.0),
        ])

        scan = engine.structure_entry_scan(
            100.2,
            [],
            [],
            {"15m": rows},
        )
        engine.close()

        ma20_reclaims = [
            item for item in scan["long_candidates"]
            if item["name"] == "ma20_15m" and item.get("source") == "ma_cross"
        ]
        self.assertEqual(len(ma20_reclaims), 1)
        self.assertEqual(ma20_reclaims[0]["structure_signal"], "long_ma_reclaim_cross")
        self.assertEqual(ma20_reclaims[0]["previous_close"], 99.0)
        self.assertEqual(ma20_reclaims[0]["current_close"], 101.0)

    def test_structure_entry_scan_adds_4h_ma300_and_1d_ma_break_crosses(self) -> None:
        engine = self.make_engine()
        h4_rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(300)]
        h4_rows.extend([
            self.kline(300, 101.0, high=102.0, low=100.0),
            self.kline(301, 99.8, high=100.0, low=99.0),
            self.kline(302, 99.7, high=100.0, low=99.0),
        ])
        one_d_rows = [self.kline(i, 100.0, high=101.0, low=99.0) for i in range(20)]
        one_d_rows.extend([
            self.kline(20, 101.0, high=102.0, low=100.0),
            self.kline(21, 99.8, high=100.0, low=99.0),
            self.kline(22, 99.7, high=100.0, low=99.0),
        ])

        scan = engine.structure_entry_scan(
            99.8,
            [],
            [],
            {"4h": h4_rows, "1d": one_d_rows},
        )
        engine.close()

        short_rules = {
            (item["timeframe"], item["period"], item["band"], item["structure_signal"])
            for item in scan["short_candidates"]
            if item.get("source") == "ma_cross"
        }
        self.assertIn(("4h", 300, "ma", "short_ma_break_cross"), short_rules)
        self.assertIn(("1d", 20, "ma", "short_ma_break_cross"), short_rules)
        self.assertIn(("1d", 7, "ma", "short_ma_break_cross"), short_rules)

    def test_entry_context_prefers_daily_ma7_candidate_over_15m_candidate(self) -> None:
        engine = self.make_engine()
        levels = {
            "price": 100.0,
            "structure_entry_scan": {
                "long_candidates": [
                    {
                        "name": "bb96_lower_15m",
                        "price": 99.0,
                        "distance_pct": 1.0,
                        "abs_distance_pct": 1.0,
                        "period": 96,
                        "band": "lower",
                        "timeframe": "15m",
                        "structure_signal": "long_lower_support_rebound",
                    }
                ],
            },
            "daily_ma7_trend": {
                "long_confirmed": True,
                "entry_level": {"name": "ma7_1d", "price": 98.0, "distance_pct": 2.040816},
            },
        }

        context = engine.entry_context_for_action("open_long", levels)
        engine.close()

        self.assertEqual(context["horizon"], "1d_trend")
        self.assertEqual(context["entry_timeframe"], "1d")
        self.assertEqual(context["max_hold_days"], 365.0)
        self.assertEqual(context["selected_entry_level"]["name"], "ma7_1d")
        self.assertEqual(context["daily_ma7_candidate"]["structure_signal"], "long_1d_ma7_rebreak")
        self.assertEqual(context["fifteen_m_candidate_count"], 1)

    def test_entry_context_promotes_15m_candidate_to_4h_trend_when_h4_is_confirmed(self) -> None:
        engine = self.make_engine()
        levels = {
            "price": 100.0,
            "structure_entry_scan": {
                "long_candidates": [
                    {
                        "name": "bb96_lower_15m",
                        "price": 99.0,
                        "distance_pct": 1.0,
                        "abs_distance_pct": 1.0,
                        "period": 96,
                        "band": "lower",
                        "timeframe": "15m",
                        "structure_signal": "long_lower_support_rebound",
                    }
                ],
            },
            "h4_right_side": {"long_confirmed": True, "short_confirmed": False},
        }

        context = engine.entry_context_for_action("open_long", levels)
        engine.close()

        self.assertEqual(context["horizon"], "4h_trend")
        self.assertEqual(context["entry_timeframe"], "15m")
        self.assertEqual(context["max_hold_days"], 180.0)
        self.assertTrue(context["h4_confirmed_15m_selector"])
        self.assertEqual(context["selected_entry_level"]["name"], "bb96_lower_15m")

    def test_entry_context_short_branch_covers_1d_4h_and_15m_selection(self) -> None:
        engine = self.make_engine()
        candidate = {
            "name": "bb96_upper_15m",
            "price": 101.0,
            "distance_pct": -1.0,
            "abs_distance_pct": 1.0,
            "period": 96,
            "band": "upper",
            "timeframe": "15m",
            "structure_signal": "short_upper_rejection",
        }
        daily_context = engine.entry_context_for_action(
            "open_short",
            {
                "price": 100.0,
                "structure_entry_scan": {"short_candidates": [dict(candidate)]},
                "daily_ma7_trend": {
                    "short_confirmed": True,
                    "entry_level": {"name": "ma7_1d", "price": 102.0, "distance_pct": -1.960784},
                },
            },
        )
        h4_context = engine.entry_context_for_action(
            "open_short",
            {
                "price": 100.0,
                "structure_entry_scan": {"short_candidates": [dict(candidate)]},
                "h4_right_side": {"long_confirmed": False, "short_confirmed": True},
            },
        )
        scalp_context = engine.entry_context_for_action(
            "open_short",
            {
                "price": 100.0,
                "structure_entry_scan": {"short_candidates": [dict(candidate)]},
            },
        )
        engine.close()

        self.assertEqual(daily_context["horizon"], "1d_trend")
        self.assertEqual(daily_context["entry_timeframe"], "1d")
        self.assertEqual(daily_context["selected_entry_level"]["structure_signal"], "short_1d_ma7_rebreak")
        self.assertEqual(daily_context["max_hold_days"], 365.0)

        self.assertEqual(h4_context["horizon"], "4h_trend")
        self.assertEqual(h4_context["entry_timeframe"], "15m")
        self.assertTrue(h4_context["h4_confirmed_15m_selector"])
        self.assertEqual(h4_context["max_hold_days"], 180.0)

        self.assertEqual(scalp_context["horizon"], "15m_scalp")
        self.assertEqual(scalp_context["entry_timeframe"], "15m")
        self.assertFalse(scalp_context["h4_confirmed_15m_selector"])
        self.assertEqual(scalp_context["max_hold_days"], 14.0)

    def test_pump_profile_selects_allowed_daily_ma_break_cross(self) -> None:
        engine = self.make_engine()
        levels = {
            "price": 99.0,
            "structure_entry_scan": {
                "short_candidates": [
                    {
                        "name": "ma20_1d",
                        "price": 100.0,
                        "distance_pct": -1.0,
                        "abs_distance_pct": 1.0,
                        "period": 20,
                        "band": "ma",
                        "timeframe": "1d",
                        "structure_signal": "short_ma_break_cross",
                        "source": "ma_cross",
                    },
                    {
                        "name": "bb96_upper_15m",
                        "price": 101.0,
                        "distance_pct": -1.0,
                        "abs_distance_pct": 1.0,
                        "period": 96,
                        "band": "upper",
                        "timeframe": "15m",
                        "structure_signal": "short_upper_rejection",
                    },
                ],
            },
        }

        context = engine.entry_context_for_action("open_short", levels, symbol="PUMPUSDT")
        engine.close()

        self.assertEqual(context["horizon"], "1d_trend")
        self.assertEqual(context["entry_timeframe"], "1d")
        self.assertEqual(context["one_d_candidate_count"], 1)
        self.assertEqual(context["selected_entry_level"]["name"], "ma20_1d")
        self.assertEqual(context["symbol_bb_entry_profile"]["allowed_candidate_count"], 1)
        self.assertEqual(context["symbol_bb_entry_profile"]["blocked_candidate_count"], 1)

    def test_technical_entry_ready_blocks_missing_level_and_allows_trend_horizon(self) -> None:
        engine = self.make_engine()
        engine.config.entry_structure_filter_enabled = False
        signal = V6Signal(
            symbol="BTCUSDT",
            computed_at="2026-06-19T00:00:00+00:00",
            price=100.0,
            side_bias="long",
            long_probability=0.6,
            short_probability=0.2,
            wait_probability=0.2,
            model_quality_score=0.5,
            reasons=[],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            levels={},
            indicators={},
        )

        missing_ready, missing_block, _ = engine.technical_entry_ready(signal, "open_long", {"horizon": "15m_scalp"})
        trend_ready, trend_block, _ = engine.technical_entry_ready(
            signal,
            "open_long",
            {
                "horizon": "4h_trend",
                "selected_entry_level": {
                    "name": "bb96_lower_15m",
                    "price": 99.0,
                    "timeframe": "15m",
                    "structure_signal": "long_lower_support_rebound",
                },
            },
        )
        engine.close()

        self.assertFalse(missing_ready)
        self.assertEqual(missing_block["reason"], "technical_entry_only_no_indicator_level")
        self.assertTrue(trend_ready)
        self.assertIsNone(trend_block)

    def test_technical_entry_ready_requires_15m_touch_or_confirmation_for_scalp(self) -> None:
        engine = self.make_engine()
        engine.config.entry_structure_filter_enabled = False
        levels = {
            "structure_entry_scan": {
                "long_candidates": [
                    {
                        "name": "bb96_lower_15m",
                        "price": 100.0,
                        "distance_pct": 0.2,
                        "abs_distance_pct": 0.2,
                        "period": 96,
                        "band": "lower",
                        "timeframe": "15m",
                        "structure_signal": "long_lower_support_rebound",
                    }
                ],
            },
            "all_supports": [("bb96_lower_15m", 100.0)],
        }
        base_signal = {
            "symbol": "TESTUSDT",
            "computed_at": "2026-06-19T00:00:00+00:00",
            "side_bias": "long",
            "long_probability": 0.6,
            "short_probability": 0.2,
            "wait_probability": 0.2,
            "model_quality_score": 0.5,
            "reasons": [],
            "h4": {},
            "fifteen_m": {},
            "one_m": {},
            "mid_retest": {},
            "levels": levels,
        }
        untouched_signal = V6Signal(
            **base_signal,
            price=101.0,
            indicators={"recent_extremes": {"1m": {"low": 100.8}}},
        )
        touched_signal = V6Signal(
            **base_signal,
            price=100.2,
            indicators={"recent_extremes": {"1m": {"low": 99.8}}},
        )

        untouched_ready, untouched_block, _ = engine.technical_entry_ready(untouched_signal, "open_long")
        touched_ready, touched_block, _ = engine.technical_entry_ready(touched_signal, "open_long")
        engine.close()

        self.assertFalse(untouched_ready)
        self.assertEqual(untouched_block["reason"], "technical_entry_only_unconfirmed_15m_trigger")
        self.assertTrue(touched_ready)
        self.assertIsNone(touched_block)

    def test_technical_entry_ready_short_branch_uses_15m_touch_and_4h_trend_ready(self) -> None:
        engine = self.make_engine()
        engine.config.entry_structure_filter_enabled = False
        levels = {
            "structure_entry_scan": {
                "short_candidates": [
                    {
                        "name": "bb96_upper_15m",
                        "price": 100.0,
                        "distance_pct": -0.2,
                        "abs_distance_pct": 0.2,
                        "period": 96,
                        "band": "upper",
                        "timeframe": "15m",
                        "structure_signal": "short_upper_rejection",
                    }
                ],
            },
            "all_resistances": [("bb96_upper_15m", 100.0)],
        }
        base_signal = {
            "symbol": "TESTUSDT",
            "computed_at": "2026-06-19T00:00:00+00:00",
            "side_bias": "short",
            "long_probability": 0.2,
            "short_probability": 0.6,
            "wait_probability": 0.2,
            "model_quality_score": 0.5,
            "reasons": [],
            "h4": {},
            "fifteen_m": {},
            "one_m": {},
            "mid_retest": {},
            "levels": levels,
        }
        untouched_signal = V6Signal(
            **base_signal,
            price=99.0,
            indicators={"recent_extremes": {"1m": {"high": 99.2}}},
        )
        touched_signal = V6Signal(
            **base_signal,
            price=99.8,
            indicators={"recent_extremes": {"1m": {"high": 100.2}}},
        )
        trend_context = {
            "horizon": "4h_trend",
            "selected_entry_level": {
                "name": "bb96_upper_15m",
                "price": 100.0,
                "timeframe": "15m",
                "structure_signal": "short_upper_rejection",
            },
        }

        untouched_ready, untouched_block, _ = engine.technical_entry_ready(untouched_signal, "open_short")
        touched_ready, touched_block, _ = engine.technical_entry_ready(touched_signal, "open_short")
        trend_ready, trend_block, _ = engine.technical_entry_ready(untouched_signal, "open_short", trend_context)
        engine.close()

        self.assertFalse(untouched_ready)
        self.assertEqual(untouched_block["reason"], "technical_entry_only_unconfirmed_15m_trigger")
        self.assertTrue(touched_ready)
        self.assertIsNone(touched_block)
        self.assertTrue(trend_ready)
        self.assertIsNone(trend_block)

    def test_structure_entry_scan_near_pct_boundary_and_invalid_levels_are_safe(self) -> None:
        engine = self.make_engine()
        scan = engine.structure_entry_scan(
            100.0,
            [
                ("bb96_upper_15m", 100.4),
                ("bb120_upper_15m", 100.6),
                ("bb210_mid_15m", 100.7),
                ("bb300_mid_15m", 100.9),
                ("bb672_upper_15m", 99.2),
                ("bb1440_upper_15m", 98.8),
                ("bb1920_lower_4h", 100.8),
                ("bb2880_lower_4h", 101.2),
                ("bad_name", 100.0),
            ],
            [
                ("ma20_15m", 99.3),
                ("ma60_4h", 99.1),
                ("ma20_1d", 100.0),
                ("not_ma", 100.0),
            ],
        )
        engine.close()

        near_names = {item["name"] for item in scan["near_levels"]}
        self.assertIn("bb96_upper_15m", near_names)
        self.assertIn("bb210_mid_15m", near_names)
        self.assertIn("bb672_upper_15m", near_names)
        self.assertIn("bb1920_lower_4h", near_names)
        self.assertIn("ma20_15m", near_names)
        self.assertNotIn("bb120_upper_15m", near_names)
        self.assertNotIn("bb300_mid_15m", near_names)
        self.assertNotIn("bb1440_upper_15m", near_names)
        self.assertNotIn("bb2880_lower_4h", near_names)
        self.assertNotIn("ma60_4h", near_names)
        self.assertNotIn("ma20_1d", near_names)
        self.assertNotIn("bad_name", near_names)
        self.assertEqual(scan["long_candidate_count"], 2)
        self.assertEqual(scan["short_candidate_count"], 3)
        self.assertEqual(
            scan["near_pct_by_structure"],
            {"rebound_rejection": 0.5, "mid_ma_retest": 0.8, "breakout_breakdown": 1.0},
        )

    def test_mid_retest_state_uses_shared_4h_mids(self) -> None:
        engine = self.make_engine()
        bb = SimpleNamespace(
            bb105_mid_4h=100.0,
            bb96_mid_4h=100.0,
            bb210_mid_4h=110.0,
            bb300_mid_4h=115.0,
            bb672_mid_4h=125.0,
            bb1440_mid_4h=150.0,
            bb4707_mid_4h=180.0,
        )

        state = engine.mid_retest_state(99.0, bb)
        engine.close()

        self.assertEqual(state["nearest_mid"], "bb96_4h")
        self.assertIn("bb96_4h", state["retest_candidates"])
        self.assertIn("bb105_4h", state["periods"])
        self.assertFalse(state["periods"]["bb96_4h"]["far_below_mid"])
        self.assertTrue(state["periods"]["bb672_4h"]["far_below_mid"])
        self.assertIn("bb4707_4h", state["periods"])

    def test_levels_include_mid_cycle_4h_bb_supports(self) -> None:
        engine = self.make_engine()
        bb = SimpleNamespace(
            bb105_lower_4h=420.0,
            bb210_lower_4h=360.0,
            bb300_lower_4h=314.0,
            bb672_lower_4h=72.0,
            bb1440_lower_4h=116.0,
            bb4707_lower_15m=276.0,
            bb120_mid_1d=350.0,
            bb105_mid_4h=580.0,
            bb210_mid_4h=430.0,
            bb300_mid_4h=410.0,
            bb672_mid_4h=357.0,
            bb1440_mid_4h=381.0,
            bb105_upper_4h=690.0,
            bb210_upper_4h=500.0,
            bb300_upper_4h=470.0,
            bb672_upper_4h=642.0,
        )
        h4 = SimpleNamespace(h4_recent_swing_low=386.0, h4_range_low=386.0)

        levels = engine.levels(380.0, bb, h4)
        engine.close()

        self.assertEqual(levels["nearest_supports"][0], ("bb210_lower_4h", 360.0))
        self.assertEqual(levels["nearest_supports"][1], ("bb672_mid_4h", 357.0))
        self.assertIn(("bb120_mid_1d", 350.0), levels["nearest_supports"])
        self.assertNotEqual(levels["nearest_supports"][0][0], "bb1440_lower_4h")
        self.assertEqual(levels["shared_bb_periods"][-1], 4707)
        self.assertEqual(levels["bb_coverage"]["configured_level_count"], 90)
        self.assertGreaterEqual(levels["bb_coverage"]["active_level_count"], 13)
        self.assertIn(("bb4707_lower_15m", 276.0), levels["all_supports"])
        self.assertGreater(levels["structure_entry_scan"]["short_candidate_count"], 0)
        self.assertEqual(levels["structure_entry_scan"]["short_candidates"][0]["name"], "bb1440_mid_4h")

    def test_structure_risk_plan_uses_enum_bb_ladder_for_staged_take_profit(self) -> None:
        engine = self.make_engine()
        levels = {
            "nearest_supports": [("bb210_lower_4h", 360.0)],
            "nearest_resistances": [("bb1440_mid_4h", 381.0)],
            "all_supports": [
                ("bb96_mid_15m", 362.0),
                ("bb120_mid_15m", 352.0),
                ("bb210_lower_4h", 360.0),
                ("bb672_mid_4h", 357.0),
                ("bb120_mid_1d", 350.0),
                ("bb672_lower_15m", 336.0),
                ("bb300_lower_4h", 314.0),
                ("bb4707_lower_15m", 276.0),
                ("bb672_lower_4h", 72.0),
            ],
            "all_resistances": [("bb1440_mid_4h", 381.0)],
            "structure_entry_scan": {"short_candidate_count": 1},
        }

        plan = engine.structure_risk_plan(380.0, "open_short", levels)
        engine.close()

        self.assertEqual(plan["runner"]["name"], "bb672_lower_15m")
        self.assertEqual(plan["take_profit_price"], 336.0)
        self.assertAlmostEqual(plan["stop_loss_price"], 391.4)
        self.assertEqual(plan["target_ladder"][-1]["name"], "bb4707_lower_15m")
        self.assertEqual([stage["name"] for stage in plan["staged_take_profit_targets"]], [
            "bb96_mid_15m",
            "bb120_mid_15m",
            "bb672_lower_15m",
        ])
        self.assertEqual([stage["fraction_of_initial"] for stage in plan["staged_take_profit_targets"]], [0.5, 0.25, 0.25])
        self.assertEqual(plan["staged_take_profit_targets"][0]["selection_strategy"], "bb_enum_mid_band_v1")

    def test_structure_risk_plan_for_long_uses_support_stop_and_resistance_ladder(self) -> None:
        engine = self.make_engine()
        levels = {
            "nearest_supports": [("bb96_lower_15m", 95.0)],
            "nearest_resistances": [("bb96_mid_15m", 102.0)],
            "all_supports": [
                ("bb96_lower_15m", 95.0),
                ("bb120_lower_4h", 90.0),
            ],
            "all_resistances": [
                ("bb96_mid_15m", 102.0),
                ("bb120_upper_15m", 106.0),
                ("bb672_upper_4h", 120.0),
            ],
            "structure_entry_scan": {
                "long_candidate_count": 1,
                "long_candidates": [
                    {
                        "name": "bb96_lower_15m",
                        "price": 95.0,
                        "distance_pct": 5.0,
                        "abs_distance_pct": 5.0,
                        "period": 96,
                        "band": "lower",
                        "timeframe": "15m",
                        "structure_signal": "long_lower_support_rebound",
                    }
                ],
            },
        }

        plan = engine.structure_risk_plan(100.0, "open_long", levels)
        engine.close()

        self.assertEqual(plan["horizon"], "15m_scalp")
        self.assertEqual(plan["entry_timeframe"], "15m")
        self.assertEqual(plan["stop_anchor"]["name"], "bb96_lower_15m")
        self.assertAlmostEqual(plan["default_stop_price"], 97.0)
        self.assertAlmostEqual(plan["stop_loss_price"], 94.525)
        self.assertEqual(plan["target_ladder"][-1]["name"], "bb672_upper_4h")
        self.assertEqual(plan["tp1"]["name"], "bb96_mid_15m")
        self.assertEqual(plan["tp2"]["name"], "bb120_upper_15m")
        self.assertEqual(plan["runner"]["name"], "bb120_upper_15m")
        self.assertEqual(plan["take_profit_price"], 106.0)
        self.assertEqual([stage["name"] for stage in plan["staged_take_profit_targets"]], [
            "bb96_mid_15m",
            "bb120_upper_15m",
        ])

    def test_structure_risk_plan_1d_and_4h_trend_contexts_keep_directional_ladders(self) -> None:
        engine = self.make_engine()
        long_levels = {
            "nearest_supports": [("bb96_lower_15m", 95.0)],
            "nearest_resistances": [("bb96_mid_15m", 102.0)],
            "all_supports": [("bb96_lower_15m", 95.0), ("bb120_lower_4h", 90.0)],
            "all_resistances": [("bb96_mid_15m", 102.0), ("bb120_upper_15m", 106.0), ("bb672_upper_4h", 120.0)],
            "structure_entry_scan": {
                "long_candidates": [
                    {
                        "name": "bb96_lower_15m",
                        "price": 95.0,
                        "distance_pct": 5.0,
                        "abs_distance_pct": 5.0,
                        "period": 96,
                        "band": "lower",
                        "timeframe": "15m",
                        "structure_signal": "long_lower_support_rebound",
                    }
                ],
            },
        }
        short_levels = {
            "nearest_supports": [("bb96_mid_15m", 98.0)],
            "nearest_resistances": [("bb96_upper_15m", 105.0)],
            "all_supports": [("bb96_mid_15m", 98.0), ("bb120_lower_15m", 94.0), ("bb672_lower_4h", 80.0)],
            "all_resistances": [("bb96_upper_15m", 105.0), ("bb120_upper_4h", 110.0)],
            "structure_entry_scan": {
                "short_candidates": [
                    {
                        "name": "bb96_upper_15m",
                        "price": 105.0,
                        "distance_pct": -4.7619,
                        "abs_distance_pct": 4.7619,
                        "period": 96,
                        "band": "upper",
                        "timeframe": "15m",
                        "structure_signal": "short_upper_rejection",
                    }
                ],
            },
        }

        long_1d = dict(long_levels)
        long_1d["daily_ma7_trend"] = {"long_confirmed": True, "entry_level": {"name": "ma7_1d", "price": 98.0, "distance_pct": 2.040816}}
        long_4h = dict(long_levels)
        long_4h["h4_right_side"] = {"long_confirmed": True, "short_confirmed": False}
        short_1d = dict(short_levels)
        short_1d["daily_ma7_trend"] = {"short_confirmed": True, "entry_level": {"name": "ma7_1d", "price": 102.0, "distance_pct": -1.960784}}
        short_4h = dict(short_levels)
        short_4h["h4_right_side"] = {"long_confirmed": False, "short_confirmed": True}

        plans = {
            "long_1d": engine.structure_risk_plan(100.0, "open_long", long_1d),
            "long_4h": engine.structure_risk_plan(100.0, "open_long", long_4h),
            "short_1d": engine.structure_risk_plan(100.0, "open_short", short_1d),
            "short_4h": engine.structure_risk_plan(100.0, "open_short", short_4h),
        }
        engine.close()

        self.assertEqual(plans["long_1d"]["horizon"], "1d_trend")
        self.assertEqual(plans["long_1d"]["entry_context"]["selected_entry_level"]["name"], "ma7_1d")
        self.assertEqual(plans["long_1d"]["max_hold_days"], 365.0)
        self.assertEqual(plans["long_1d"]["runner"]["name"], "bb120_upper_15m")

        self.assertEqual(plans["long_4h"]["horizon"], "4h_trend")
        self.assertEqual(plans["long_4h"]["entry_timeframe"], "15m")
        self.assertEqual(plans["long_4h"]["max_hold_days"], 180.0)
        self.assertEqual(plans["long_4h"]["runner"]["name"], "bb672_upper_4h")
        self.assertEqual(plans["long_4h"]["take_profit_price"], 120.0)

        self.assertEqual(plans["short_1d"]["horizon"], "1d_trend")
        self.assertEqual(plans["short_1d"]["entry_context"]["selected_entry_level"]["name"], "ma7_1d")
        self.assertEqual(plans["short_1d"]["max_hold_days"], 365.0)
        self.assertEqual(plans["short_1d"]["runner"]["name"], "bb120_lower_15m")

        self.assertEqual(plans["short_4h"]["horizon"], "4h_trend")
        self.assertEqual(plans["short_4h"]["entry_timeframe"], "15m")
        self.assertEqual(plans["short_4h"]["max_hold_days"], 180.0)
        self.assertEqual(plans["short_4h"]["runner"]["name"], "bb672_lower_4h")
        self.assertEqual(plans["short_4h"]["take_profit_price"], 80.0)

    def test_open_intent_rationale_deduplicates_reasons_preserving_first_seen_order(self) -> None:
        engine = self.make_engine()
        signal = SimpleNamespace(
            symbol="BTCUSDT",
            price=100.0,
            levels={
                "nearest_supports": [("bb210_lower_15m", 95.0)],
                "nearest_resistances": [("bb96_mid_15m", 101.0)],
                "all_supports": [("bb210_lower_15m", 95.0)],
                "all_resistances": [("bb96_mid_15m", 101.0)],
                "structure_entry_scan": {"short_candidate_count": 1},
            },
            long_probability=0.1,
            short_probability=0.7,
            wait_probability=0.2,
            indicators={},
            reasons=[" first reason ", "second reason", "first reason", "", "second reason"],
        )

        intent = engine.open_intent(signal, "open_short")
        engine.close()

        self.assertEqual(intent.rationale, "first reason; second reason")

    def test_open_intent_carries_structure_risk_plan(self) -> None:
        engine = self.make_engine()
        signal = SimpleNamespace(
            symbol="BTCUSDT",
            price=100.0,
            levels={
                "nearest_supports": [("bb210_lower_15m", 95.0)],
                "nearest_resistances": [("bb96_mid_15m", 101.0)],
                "all_supports": [("bb96_mid_15m", 96.0), ("bb210_lower_15m", 95.0), ("bb672_lower_4h", 82.0)],
                "all_resistances": [("bb96_mid_15m", 101.0)],
                "structure_entry_scan": {"short_candidate_count": 1},
            },
            long_probability=0.1,
            short_probability=0.7,
            wait_probability=0.2,
            reasons=["test"],
        )

        intent = engine.open_intent(signal, "open_short")
        engine.close()

        self.assertEqual(intent.risk_plan["mode"], "bb_structure_runner")
        self.assertEqual(intent.take_profit_price, 95.0)
        self.assertEqual(intent.risk_plan["runner"]["name"], "bb210_lower_15m")
        self.assertEqual([stage["name"] for stage in intent.risk_plan["staged_take_profit_targets"]], ["bb96_mid_15m", "bb210_lower_15m"])

    def test_staged_take_profit_closes_one_stage_and_keeps_position(self) -> None:
        engine = self.make_engine()
        engine.config.roi_take_profit_enabled = False
        risk_plan = engine.structure_risk_plan(
            380.0,
            "open_short",
            {
                "all_supports": [
                    ("bb96_mid_15m", 362.0),
                    ("bb120_mid_15m", 352.0),
                    ("bb210_lower_4h", 360.0),
                    ("bb672_mid_4h", 357.0),
                    ("bb672_lower_15m", 336.0),
                    ("bb300_lower_4h", 314.0),
                    ("bb4707_lower_15m", 276.0),
                ],
                "all_resistances": [("bb1440_mid_4h", 381.0)],
                "structure_entry_scan": {"short_candidate_count": 1},
            },
        )
        position = {
            "symbol": "ZECUSDT",
            "side": "short",
            "entry_price": 380.0,
            "stop_loss_price": 391.4,
            "take_profit_price": risk_plan["take_profit_price"],
            "notional_usdt": 300.0,
            "risk_plan": risk_plan,
            "remaining_fraction": 1.0,
        }
        signal = V6Signal(
            symbol="ZECUSDT",
            computed_at="2026-06-06T00:00:00+00:00",
            price=359.0,
            side_bias="short",
            long_probability=0.1,
            short_probability=0.61,
            wait_probability=0.29,
            model_quality_score=0.4,
            reasons=[],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            levels={},
            indicators={"tick_volume_15m": {"ready": True, "active": True}},
        )

        intent = engine.exit_intent_if_needed(signal, position)
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.action, "close_short")
        self.assertEqual(intent.side, "BUY")
        self.assertEqual(intent.target_name, "bb96_mid_15m")
        self.assertAlmostEqual(intent.close_fraction, 0.5)

        state = {"positions": [position], "journal": []}
        engine.apply_intent_to_state(
            state,
            intent,
            {
                "status": "submitted",
                "submitted_at": "2026-06-06T00:00:01+00:00",
                "response": {"orderId": 99, "executedQty": "0.280", "avgPrice": "359.0"},
            },
        )
        self.assertEqual(len(state["positions"]), 1)
        updated = state["positions"][0]
        self.assertAlmostEqual(updated["remaining_fraction"], 0.5)
        self.assertAlmostEqual(updated["notional_usdt"], 150.0)
        self.assertEqual(updated["take_profit_stages"][0]["status"], "executed")
        self.assertEqual(updated["take_profit_stages"][0]["order_id"], 99)

        repeated = engine.exit_intent_if_needed(signal, updated)
        engine.close()
        self.assertIsNone(repeated)

    def test_staged_take_profit_combines_multiple_touched_targets(self) -> None:
        engine = self.make_engine()
        engine.config.roi_take_profit_enabled = False
        risk_plan = engine.structure_risk_plan(
            380.0,
            "open_short",
            {
                "all_supports": [
                    ("bb96_mid_15m", 362.0),
                    ("bb120_mid_15m", 352.0),
                    ("bb672_lower_15m", 336.0),
                    ("bb4707_lower_15m", 276.0),
                ],
                "all_resistances": [("bb1440_mid_4h", 381.0)],
                "structure_entry_scan": {"short_candidate_count": 1},
            },
        )
        position = {
            "symbol": "ZECUSDT",
            "side": "short",
            "entry_price": 380.0,
            "stop_loss_price": 391.4,
            "take_profit_price": risk_plan["take_profit_price"],
            "notional_usdt": 300.0,
            "risk_plan": risk_plan,
            "remaining_fraction": 1.0,
        }
        signal = V6Signal(
            symbol="ZECUSDT",
            computed_at="2026-06-06T00:00:00+00:00",
            price=351.0,
            side_bias="short",
            long_probability=0.1,
            short_probability=0.61,
            wait_probability=0.29,
            model_quality_score=0.4,
            reasons=[],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            levels={},
            indicators={"tick_volume_15m": {"ready": True, "active": True}},
        )

        intent = engine.exit_intent_if_needed(signal, position)
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.target_name, "bb96_mid_15m,bb120_mid_15m")
        self.assertAlmostEqual(intent.close_fraction, 0.75)

        state = {"positions": [position], "journal": []}
        engine.apply_intent_to_state(
            state,
            intent,
            {
                "status": "submitted",
                "submitted_at": "2026-06-06T00:00:01+00:00",
                "response": {"orderId": 99, "executedQty": "0.560", "avgPrice": "356.0"},
            },
        )
        updated = state["positions"][0]
        engine.close()

        self.assertAlmostEqual(updated["remaining_fraction"], 0.25)
        self.assertEqual(updated["take_profit_stages"][0]["status"], "executed")
        self.assertEqual(updated["take_profit_stages"][1]["status"], "executed")
        self.assertEqual(updated["take_profit_stages"][2]["status"], "pending")

    def test_legacy_position_without_risk_plan_gets_staged_targets_from_signal_levels(self) -> None:
        engine = self.make_engine()
        position = {
            "symbol": "HYPEUSDT",
            "side": "short",
            "entry_price": 62.0,
            "stop_loss_price": 64.0,
            "take_profit_price": 48.0,
            "notional_usdt": 300.0,
        }
        signal = V6Signal(
            symbol="HYPEUSDT",
            computed_at="2026-06-06T00:00:00+00:00",
            price=61.0,
            side_bias="short",
            long_probability=0.1,
            short_probability=0.55,
            wait_probability=0.35,
            model_quality_score=0.4,
            reasons=[],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            levels={
                "all_supports": [
                    ("bb672_mid_15m", 58.0),
                    ("bb210_lower_4h", 55.0),
                    ("bb672_lower_15m", 48.0),
                ],
                "all_resistances": [("bb96_mid_15m", 63.0)],
                "structure_entry_scan": {"short_candidate_count": 1},
            },
        )

        intent = engine.exit_intent_if_needed(signal, position)
        engine.close()

        self.assertIsNone(intent)
        self.assertIn("risk_plan", position)
        self.assertEqual([stage["name"] for stage in position["take_profit_stages"]], [
            "bb672_mid_15m",
            "bb210_lower_4h",
        ])

    def test_manual_quarantine_blocks_close_intent_for_unclosable_position(self) -> None:
        engine = self.make_engine()
        position = {
            "symbol": "HYPEUSDT",
            "side": "long",
            "entry_price": 58.0,
            "stop_loss_price": 55.0,
            "take_profit_price": 60.0,
            "notional_usdt": 300.0,
            "manual_quarantine": {
                "enabled": True,
                "until": "2099-01-01T00:00:00+00:00",
                "reason": "binance_demo_unclosable",
            },
        }
        signal = V6Signal(
            symbol="HYPEUSDT",
            computed_at="2026-06-06T00:00:00+00:00",
            price=70.0,
            side_bias="long",
            long_probability=0.6,
            short_probability=0.1,
            wait_probability=0.3,
            model_quality_score=0.4,
            reasons=[],
            h4={},
            fifteen_m={},
            one_m={},
            mid_retest={},
            levels={},
        )

        intent = engine.exit_intent_if_needed(signal, position)
        engine.close()

        self.assertIsNone(intent)

    def test_old_runner_take_profit_stages_are_migrated_to_enum_strategy(self) -> None:
        engine = self.make_engine()
        risk_plan = {
            "action": "open_short",
            "target_ladder": [
                {"name": "bb96_mid_15m", "price": 360.0, "distance_pct": 3.0},
                {"name": "bb672_mid_4h", "price": 357.0, "distance_pct": 4.0},
                {"name": "bb120_mid_15m", "price": 352.0, "distance_pct": 6.0},
                {"name": "bb672_lower_15m", "price": 336.0, "distance_pct": 11.0},
                {"name": "bb1920_mid_4h", "price": 299.0, "distance_pct": 25.0},
            ],
        }
        position = {
            "symbol": "ZECUSDT",
            "side": "short",
            "entry_price": 374.0,
            "notional_usdt": 90.0,
            "remaining_fraction": 0.3,
            "risk_plan": risk_plan,
            "take_profit_stages": [
                {"index": 1, "name": "bb96_mid_15m", "price": 360.0, "fraction_of_initial": 0.35, "status": "executed"},
                {"index": 2, "name": "bb672_mid_4h", "price": 357.0, "fraction_of_initial": 0.35, "status": "executed"},
                {"index": 3, "name": "bb1920_mid_4h", "price": 299.0, "fraction_of_initial": 0.3, "status": "pending"},
            ],
        }

        stages = engine.ensure_position_take_profit_stages(position)
        engine.close()

        self.assertEqual([stage["name"] for stage in stages], [
            "bb96_mid_15m",
            "bb120_mid_15m",
            "bb672_lower_15m",
        ])
        self.assertEqual(stages[0]["status"], "executed")
        self.assertEqual(stages[1]["status"], "pending")
        self.assertEqual(stages[2]["status"], "pending")
        self.assertAlmostEqual(stages[1]["fraction_of_initial"] + stages[2]["fraction_of_initial"], 0.3)
        self.assertEqual(position["take_profit_price"], 336.0)
        self.assertEqual(position["risk_plan"]["runner"]["name"], "bb672_lower_15m")

    def test_full_staged_take_profit_close_enqueues_post_trade_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            config = V6ExecutionConfig(
                state_dir=state_dir,
                state_file=state_dir / "state.json",
                signals_file=state_dir / "signals.jsonl",
                orders_file=state_dir / "orders.jsonl",
                wal_file=state_dir / "execution_wal.jsonl",
                latest_summary_file=state_dir / "latest_run_summary.json",
                closed_trade_review_queue_file=state_dir / "closed_trade_review_queue.jsonl",
                post_trade_training_samples_file=state_dir / "post_trade_training_samples.jsonl",
            )
            engine = V6ExecutionEngine(config)
            state = {
                "positions": [{
                    "symbol": "ZECUSDT",
                    "side": "short",
                    "opened_at": "2026-06-06T13:00:00+00:00",
                    "entry_price": 374.0,
                    "notional_usdt": 45.0,
                    "original_notional_usdt": 300.0,
                    "leverage": 8,
                    "risk_plan": {"action": "open_short", "target_ladder": []},
                    "take_profit_stages": [
                        {"name": "bb96_mid_15m", "price": 360.0, "fraction_of_initial": 0.5, "status": "executed"},
                        {"name": "bb672_lower_15m", "price": 336.0, "fraction_of_initial": 0.5, "status": "pending"},
                    ],
                    "partial_take_profit_history": [{"target_name": "bb96_mid_15m"}],
                }],
                "journal": [],
            }
            intent = V6OrderIntent(
                action="close_short",
                symbol="ZECUSDT",
                side="BUY",
                quantity_hint_usdt=45.0,
                leverage=8,
                reduce_only=True,
                entry_price=336.0,
                stop_loss_price=385.0,
                take_profit_price=336.0,
                long_probability=0.1,
                short_probability=0.5,
                wait_probability=0.4,
                rationale="staged short take profit touched: bb672_lower_15m close_initial=0.50 close_remaining=1.0000",
                generated_at="2026-06-06T14:00:00+00:00",
                risk_plan={"partial_take_profit": {"fraction_of_initial": 0.5}},
                close_fraction=1.0,
                target_name="bb672_lower_15m",
            )

            engine.apply_intent_to_state(
                state,
                intent,
                {
                    "status": "submitted",
                    "submitted_at": "2026-06-06T14:00:00+00:00",
                    "response": {"orderId": 1, "avgPrice": "336.0", "executedQty": "0.1"},
                },
            )
            engine.close()

            self.assertEqual(state["positions"], [])
            rows = [json.loads(line) for line in config.closed_trade_review_queue_file.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["symbol"], "ZECUSDT")
            self.assertEqual(rows[0]["review_due_at"], "2026-06-07T02:00:00+00:00")
            self.assertEqual(rows[0]["target_name"], "bb672_lower_15m")

    def test_direction_probabilities_use_structure_entry_scan(self) -> None:
        engine = self.make_engine()
        h4 = SimpleNamespace(
            h4_close_above_ma20=False,
            h4_ma20_slope_pct=0.0,
            h4_structure_state="range",
            h4_close_position_20bars=0.5,
        )
        base_args = {
            "mark_price": 100.0,
            "quality": 0.5,
            "h4": h4,
            "fifteen_m": {},
            "mid_retest": {},
            "one_m": {},
        }

        base_long, base_short, base_wait, _ = engine.direction_probabilities(**base_args)
        scan_long, scan_short, scan_wait, reasons = engine.direction_probabilities(
            **base_args,
            entry_scan={
                "long_candidate_count": 2,
                "short_candidate_count": 0,
                "long_candidates": [
                    {"name": "bb672_lower_15m"},
                    {"name": "bb1440_mid_4h"},
                ],
                "short_candidates": [],
            },
        )
        engine.close()

        self.assertGreater(scan_long, base_long)
        self.assertLess(scan_short, base_short)
        self.assertLess(scan_wait, base_wait)
        self.assertTrue(any("BB structure long entry candidates" in reason for reason in reasons))

    def test_features_include_real_clipped_structure_distances(self) -> None:
        engine = self.make_engine()
        bb = SimpleNamespace(
            bb_price_position_96=0.4,
            bb_price_position_105_4h=0.5,
            bb_price_position_672_4h=0.6,
            bb_price_position_1440_4h=0.7,
            bb_price_position_105_1m=0.8,
            bb_price_position_300_1m=0.9,
            bb105_mid_4h=100.0,
            bb672_mid_4h=120.0,
            bb1440_mid_4h=200.0,
            bb105_upper_1m=110.0,
            bb300_upper_1m=125.0,
            bb96_upper_15m=105.0,
        )
        h4 = SimpleNamespace(h4_close_position_20bars=0.3)
        klines_4h = [[0, 0, 0, 0, 90.0], [0, 0, 0, 0, 100.0]]
        klines_15m = [[0, 0, 0, 0, 98.0], [0, 0, 0, 0, 100.0], [0, 0, 0, 0, 101.0], [0, 0, 0, 0, 102.0], [0, 0, 0, 0, 103.0]]

        features = engine.features_from_context("BTCUSDT", 90.0, {}, bb, h4, klines_4h, klines_15m)
        engine.close()

        self.assertAlmostEqual(features.bb105_mid_4h_distance_pct, -10.0)
        self.assertAlmostEqual(features.bb672_mid_4h_distance_pct, -25.0)
        self.assertAlmostEqual(features.bb1440_mid_4h_distance_pct, -30.0)
        self.assertAlmostEqual(features.bb105_upper_1m_distance_pct, -18.1818181818)
        self.assertAlmostEqual(features.bb300_upper_1m_distance_pct, -28.0)
        self.assertAlmostEqual(features.bb96_upper_15m_distance_pct, -14.2857142857)
        self.assertAlmostEqual(features.short_4h_mid_retest_distance_min_pct, 10.0)

    def test_open_order_submits_exchange_stop_but_no_exchange_take_profit(self) -> None:
        config = V6ExecutionConfig(order_mode="testnet_order", demo_api_key="key", demo_api_secret="secret")
        client = V6BinanceClient(config)
        calls = []
        client.resolve_quantity = lambda symbol, notional, price: "0.01"  # type: ignore[method-assign]
        client.resolve_price = lambda symbol, price: "103"  # type: ignore[method-assign]
        client.mark_price = lambda symbol: 100.0  # type: ignore[method-assign]
        client.current_position_mode = lambda: "one_way"  # type: ignore[method-assign]
        position_risk_calls = 0

        def fake_signed_request(method, endpoint, params):
            nonlocal position_risk_calls
            calls.append((endpoint, dict(params)))
            if endpoint == "/fapi/v2/positionRisk":
                position_risk_calls += 1
                if position_risk_calls >= 2:
                    return [{"symbol": "BTCUSDT", "positionAmt": "-0.01", "positionSide": "BOTH"}]
                return [{"symbol": "BTCUSDT", "positionAmt": "0", "positionSide": "BOTH"}]
            if endpoint == "/fapi/v1/marginType":
                return {"symbol": "BTCUSDT", "marginType": "ISOLATED"}
            if endpoint == "/fapi/v1/leverageBracket":
                return [{"symbol": "BTCUSDT", "brackets": [{"initialLeverage": 125, "notionalCap": "1000000"}]}]
            if endpoint == "/fapi/v1/leverage":
                self.assertEqual(params["symbol"], "BTCUSDT")
                self.assertEqual(params["leverage"], "3")
                return {"symbol": "BTCUSDT", "leverage": 3}
            if endpoint == "/fapi/v1/userTrades":
                return []
            if params.get("type") == "MARKET":
                return {"orderId": 100, "status": "FILLED", "clientOrderId": "entry"}
            if params.get("type") == "STOP_MARKET":
                self.assertEqual(endpoint, "/fapi/v1/algoOrder")
                self.assertEqual(params["algoType"], "CONDITIONAL")
                self.assertEqual(params["triggerPrice"], "103")
                self.assertEqual(params["quantity"], "0.01")
                self.assertEqual(params["reduceOnly"], "true")
                self.assertEqual(params["positionSide"], "BOTH")
                self.assertNotIn("closePosition", params)
                return {"algoId": 200, "algoStatus": "NEW", "clientAlgoId": "stop"}
            raise AssertionError(params["type"])

        client.signed_request = fake_signed_request  # type: ignore[method-assign]
        intent = V6OrderIntent(
            action="open_short",
            symbol="BTCUSDT",
            side="SELL",
            quantity_hint_usdt=300.0,
            leverage=3,
            reduce_only=False,
            entry_price=100.0,
            stop_loss_price=103.0,
            take_profit_price=90.0,
            long_probability=0.1,
            short_probability=0.7,
            wait_probability=0.2,
            rationale="test",
            generated_at="2026-06-05T00:00:00+00:00",
        )

        result = client.submit_order(intent)
        client.close()

        self.assertEqual(result["status"], "submitted")
        order_types = [call[1].get("type") for call in calls]
        self.assertIn("MARKET", order_types)
        self.assertIn("STOP_MARKET", order_types)
        self.assertNotIn("TAKE_PROFIT_MARKET", order_types)
        self.assertEqual(result["leverage"]["response"]["leverage"], 3)
        self.assertEqual(result["protective_stop"]["response"]["algoId"], 200)

    def test_manual_profit_mode_submits_close_position_stop_without_fixed_quantity(self) -> None:
        config = V6ExecutionConfig(
            order_mode="testnet_order",
            demo_api_key="key",
            demo_api_secret="secret",
            exit_control_mode="stop_only_manual_profit",
        )
        client = V6BinanceClient(config)
        client.resolve_price = lambda symbol, price: "97"  # type: ignore[method-assign]
        client.current_position_mode = lambda: "hedge"  # type: ignore[method-assign]
        submitted = []

        def fake_signed_request(method, endpoint, params):
            submitted.append((method, endpoint, dict(params)))
            return {
                "algoId": 201,
                "algoStatus": "NEW",
                "clientAlgoId": "manual-profit-stop",
                "closePosition": True,
            }

        client.signed_request = fake_signed_request  # type: ignore[method-assign]
        intent = V6OrderIntent(
            action="open_long",
            symbol="BTCUSDT",
            side="BUY",
            quantity_hint_usdt=1000.0,
            leverage=8,
            reduce_only=False,
            entry_price=100.0,
            stop_loss_price=97.0,
            take_profit_price=110.0,
            long_probability=0.7,
            short_probability=0.1,
            wait_probability=0.2,
            rationale="manual profit account",
            generated_at="2026-07-21T00:00:00+00:00",
        )

        result = client.submit_protective_stop(intent, quantity="10")
        client.close()

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(len(submitted), 1)
        method, endpoint, params = submitted[0]
        self.assertEqual((method, endpoint), ("POST", "/fapi/v1/algoOrder"))
        self.assertEqual(params["side"], "SELL")
        self.assertEqual(params["positionSide"], "LONG")
        self.assertEqual(params["closePosition"], "true")
        self.assertNotIn("quantity", params)
        self.assertNotIn("reduceOnly", params)

    def test_open_order_closes_immediately_when_exchange_stop_fails(self) -> None:
        config = V6ExecutionConfig(order_mode="testnet_order", demo_api_key="key", demo_api_secret="secret")
        client = V6BinanceClient(config)
        calls = []
        client.resolve_quantity = lambda symbol, notional, price: "0.01"  # type: ignore[method-assign]
        client.resolve_price = lambda symbol, price: "103"  # type: ignore[method-assign]
        client.mark_price = lambda symbol: 100.0  # type: ignore[method-assign]
        client.current_position_mode = lambda: "one_way"  # type: ignore[method-assign]
        position_risk_calls = 0

        def fake_signed_request(method, endpoint, params):
            nonlocal position_risk_calls
            calls.append((method, endpoint, dict(params)))
            if endpoint == "/fapi/v2/positionRisk":
                position_risk_calls += 1
                return [{"symbol": "BTCUSDT", "positionAmt": "0", "positionSide": "BOTH"}]
            if endpoint == "/fapi/v1/marginType":
                return {"symbol": "BTCUSDT", "marginType": "ISOLATED"}
            if endpoint == "/fapi/v1/leverageBracket":
                return [{"symbol": "BTCUSDT", "brackets": [{"initialLeverage": 125, "notionalCap": "1000000"}]}]
            if endpoint == "/fapi/v1/leverage":
                self.assertEqual(params["leverage"], "3")
                return {"symbol": "BTCUSDT", "leverage": 3}
            if endpoint == "/fapi/v1/order" and params["side"] == "SELL":
                return {"orderId": 100, "status": "FILLED", "executedQty": "0.01"}
            if endpoint == "/fapi/v1/userTrades":
                return []
            if endpoint == "/fapi/v1/algoOrder":
                raise RuntimeError("stop failed")
            if endpoint == "/fapi/v1/order" and params["side"] == "BUY":
                self.assertEqual(params["reduceOnly"], "true")
                self.assertEqual(params["quantity"], "0.01")
                return {"orderId": 101, "status": "FILLED", "executedQty": "0.01"}
            raise AssertionError(params)

        client.signed_request = fake_signed_request  # type: ignore[method-assign]
        intent = V6OrderIntent(
            action="open_short",
            symbol="BTCUSDT",
            side="SELL",
            quantity_hint_usdt=300.0,
            leverage=3,
            reduce_only=False,
            entry_price=100.0,
            stop_loss_price=103.0,
            take_profit_price=90.0,
            long_probability=0.1,
            short_probability=0.7,
            wait_probability=0.2,
            rationale="test",
            generated_at="2026-06-05T00:00:00+00:00",
        )

        result = client.submit_order(intent)
        client.close()

        self.assertEqual(result["status"], "closed_after_protective_stop_failed")
        self.assertEqual(result["protective_stop"]["status"], "failed")
        self.assertEqual(result["emergency_close"]["status"], "submitted")
        execution_calls = [
            call[1]
            for call in calls
            if call[1] in {"/fapi/v1/marginType", "/fapi/v1/leverage", "/fapi/v1/order", "/fapi/v1/algoOrder"}
        ]
        self.assertEqual(execution_calls[:3], ["/fapi/v1/marginType", "/fapi/v1/leverage", "/fapi/v1/order"])
        self.assertEqual(execution_calls.count("/fapi/v1/algoOrder"), 1)
        self.assertEqual(execution_calls[-1], "/fapi/v1/order")

    def test_close_order_uses_exchange_position_amount(self) -> None:
        config = V6ExecutionConfig(order_mode="testnet_order", demo_api_key="key", demo_api_secret="secret")
        client = V6BinanceClient(config)
        submitted = []
        client.quantity_rules = lambda symbol: (0.001, 0.001)  # type: ignore[method-assign]
        client.current_position_mode = lambda: "one_way"  # type: ignore[method-assign]
        position_risk_calls = 0

        def fake_signed_request(method, endpoint, params):
            nonlocal position_risk_calls
            if endpoint == "/fapi/v2/positionRisk":
                position_risk_calls += 1
                if position_risk_calls >= 3:
                    return [{"symbol": "BTCUSDT", "positionAmt": "0", "positionSide": "BOTH"}]
                return [{"symbol": "BTCUSDT", "positionAmt": "-0.123", "positionSide": "BOTH"}]
            if endpoint == "/fapi/v1/order":
                submitted.append(dict(params))
                return {"orderId": 101, "status": "FILLED", "executedQty": params["quantity"]}
            raise AssertionError(endpoint)

        client.signed_request = fake_signed_request  # type: ignore[method-assign]
        intent = V6OrderIntent(
            action="close_short",
            symbol="BTCUSDT",
            side="BUY",
            quantity_hint_usdt=300.0,
            leverage=3,
            reduce_only=True,
            entry_price=90.0,
            stop_loss_price=103.0,
            take_profit_price=80.0,
            long_probability=0.1,
            short_probability=0.2,
            wait_probability=0.7,
            rationale="take profit",
            generated_at="2026-06-05T00:00:00+00:00",
        )

        result = client.submit_order(intent)
        client.close()

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(submitted[0]["quantity"], "0.123")
        self.assertEqual(submitted[0]["reduceOnly"], "true")

    def test_partial_take_profit_close_uses_fraction_of_exchange_position(self) -> None:
        config = V6ExecutionConfig(order_mode="testnet_order", demo_api_key="key", demo_api_secret="secret")
        client = V6BinanceClient(config)
        submitted = []
        client.quantity_rules = lambda symbol: (0.001, 0.001)  # type: ignore[method-assign]
        client.current_position_mode = lambda: "one_way"  # type: ignore[method-assign]
        position_risk_calls = 0

        def fake_signed_request(method, endpoint, params):
            nonlocal position_risk_calls
            if endpoint == "/fapi/v2/positionRisk":
                position_risk_calls += 1
                if position_risk_calls >= 3:
                    return [{"symbol": "ZECUSDT", "positionAmt": "0", "positionSide": "BOTH"}]
                return [{"symbol": "ZECUSDT", "positionAmt": "-0.801", "positionSide": "BOTH"}]
            if endpoint == "/fapi/v1/order":
                submitted.append(dict(params))
                return {"orderId": 101, "status": "FILLED", "executedQty": params["quantity"]}
            raise AssertionError(endpoint)

        client.signed_request = fake_signed_request  # type: ignore[method-assign]
        intent = V6OrderIntent(
            action="close_short",
            symbol="ZECUSDT",
            side="BUY",
            quantity_hint_usdt=300.0,
            leverage=8,
            reduce_only=True,
            entry_price=359.0,
            stop_loss_price=391.0,
            take_profit_price=292.0,
            long_probability=0.1,
            short_probability=0.6,
            wait_probability=0.3,
            rationale="staged take profit",
            generated_at="2026-06-06T00:00:00+00:00",
            risk_plan={"partial_take_profit": {"fraction_of_initial": 0.35}},
            close_fraction=0.35,
            target_name="bb210_lower_4h",
        )

        result = client.submit_order(intent)
        client.close()

        self.assertEqual(result["status"], "submitted")
        self.assertEqual(submitted[0]["quantity"], "0.28")
        self.assertEqual(submitted[0]["reduceOnly"], "true")

    def test_state_stores_and_cancels_protective_stop_order(self) -> None:
        engine = self.make_engine()
        intent = V6OrderIntent(
            action="open_short",
            symbol="BTCUSDT",
            side="SELL",
            quantity_hint_usdt=300.0,
            leverage=3,
            reduce_only=False,
            entry_price=100.0,
            stop_loss_price=103.0,
            take_profit_price=90.0,
            long_probability=0.1,
            short_probability=0.7,
            wait_probability=0.2,
            rationale="test",
            generated_at="2026-06-05T00:00:00+00:00",
        )
        result = {
            "status": "submitted",
            "order_mode": "testnet_order",
            "submitted_at": "2026-06-05T00:00:01+00:00",
            "protective_stop": {
                "status": "submitted",
                "response": {"algoId": 200, "clientAlgoId": "stop", "algoStatus": "NEW"},
            },
        }
        state = {"positions": [], "journal": []}
        engine.apply_intent_to_state(state, intent, result)
        self.assertEqual(state["positions"][0]["protective_stop_order_id"], 200)

        class FakeClient:
            def __init__(self) -> None:
                self.cancelled = []

            def cancel_order(self, symbol, order_id):
                self.cancelled.append((symbol, order_id))
                return {"status": "submitted", "response": {"orderId": order_id}}

            def close(self):
                pass

        fake_client = FakeClient()
        engine.client = fake_client  # type: ignore[assignment]
        cancellation = engine.cancel_protective_stop_if_present(state["positions"][0])
        engine.close()

        self.assertEqual(fake_client.cancelled, [("BTCUSDT", 200)])
        self.assertEqual(cancellation["status"], "submitted")

    def test_reconcile_removes_state_position_missing_on_exchange(self) -> None:
        engine = self.make_engine()
        engine.config.order_mode = "testnet_order"
        state = {
            "positions": [
                {"symbol": "BTCUSDT", "side": "short", "protective_stop_order_id": 100},
                {"symbol": "HYPEUSDT", "side": "short", "protective_stop_order_id": 200, "stop_loss_price": 103.0, "entry_price": 100.0, "take_profit_price": 90.0},
            ],
            "journal": [],
        }

        class FakeClient:
            def __init__(self) -> None:
                self.cancelled = []

            def signed_params(self):
                return {}

            def actual_position_rows(self, symbol=None):
                return [
                    {"symbol": "HYPEUSDT", "positionAmt": "-4.87", "positionSide": "BOTH"},
                ]

            def is_dust_exchange_snapshot(self, snapshot):
                return False

            def exchange_snapshot_notional(self, snapshot):
                return 0.0

            def signed_request(self, method, endpoint, params):
                if endpoint == "/fapi/v1/openAlgoOrders":
                    return [{"symbol": "HYPEUSDT", "side": "BUY", "orderType": "STOP_MARKET", "quantity": "4.87", "triggerPrice": "103.000", "reduceOnly": True, "algoStatus": "NEW", "algoId": 200}]
                raise AssertionError(endpoint)

            def quantity_rules(self, symbol):
                return 0.01, 0.01

            def resolve_price(self, symbol, price):
                return "103"

            def cancel_order(self, symbol, order_id):
                self.cancelled.append((symbol, order_id))
                return {"status": "submitted", "response": {"algoId": order_id}}

            def close(self):
                pass

        fake_client = FakeClient()
        engine.client = fake_client  # type: ignore[assignment]
        result = engine.reconcile_state_with_exchange(state)
        engine.close()

        self.assertEqual([item["symbol"] for item in state["positions"]], ["HYPEUSDT"])
        self.assertEqual(fake_client.cancelled, [("BTCUSDT", 100)])
        self.assertEqual(result["removed"], ["BTCUSDT"])

    def test_reconcile_replaces_mismatched_protective_stop_quantity(self) -> None:
        engine = self.make_engine()
        engine.config.order_mode = "testnet_order"
        state = {
            "positions": [
                {"symbol": "HYPEUSDT", "side": "short", "protective_stop_order_id": 200, "stop_loss_price": 63.372, "entry_price": 61.5, "take_profit_price": 47.7, "notional_usdt": 300.0, "leverage": 3},
            ],
            "journal": [],
        }

        class FakeClient:
            def __init__(self) -> None:
                self.cancelled = []
                self.submitted = []

            def signed_params(self):
                return {}

            def actual_position_rows(self, symbol=None):
                return [{"symbol": "HYPEUSDT", "positionAmt": "-5.00", "positionSide": "BOTH"}]

            def is_dust_exchange_snapshot(self, snapshot):
                return False

            def exchange_snapshot_notional(self, snapshot):
                return 0.0

            def signed_request(self, method, endpoint, params):
                if endpoint == "/fapi/v1/openAlgoOrders":
                    return [{"symbol": "HYPEUSDT", "side": "BUY", "orderType": "STOP_MARKET", "quantity": "4.87", "triggerPrice": "63.372", "reduceOnly": True, "algoStatus": "NEW", "algoId": 200}]
                raise AssertionError(endpoint)

            def quantity_rules(self, symbol):
                return 0.01, 0.01

            def resolve_price(self, symbol, price):
                return "63.372"

            def cancel_order(self, symbol, order_id):
                self.cancelled.append((symbol, order_id))
                return {"status": "submitted", "response": {"algoId": order_id}}

            def submit_protective_stop(self, intent, quantity=None):
                self.submitted.append((intent.symbol, quantity))
                return {"status": "submitted", "response": {"algoId": 300, "clientAlgoId": "new-stop", "algoStatus": "NEW"}}

            def protective_stop_precheck(self, intent):
                return {"checked": True, "would_immediately_trigger": False}

            def close(self):
                pass

        fake_client = FakeClient()
        engine.client = fake_client  # type: ignore[assignment]
        result = engine.reconcile_state_with_exchange(state)
        engine.close()

        self.assertEqual(fake_client.cancelled, [("HYPEUSDT", 200)])
        self.assertEqual(fake_client.submitted, [("HYPEUSDT", "5")])
        self.assertEqual(state["positions"][0]["protective_stop_order_id"], 300)
        self.assertEqual(result["protective_stop_sync"][0]["action"], "cancel_stale_protective_stop")

    def test_partial_take_profit_syncs_protective_stop_to_remaining_exchange_quantity(self) -> None:
        engine = self.make_engine()
        engine.config.order_mode = "testnet_order"
        position = {
            "symbol": "HYPEUSDT",
            "side": "short",
            "protective_stop_order_id": 200,
            "stop_loss_price": 63.372,
            "entry_price": 61.5,
            "take_profit_price": 47.7,
            "notional_usdt": 195.0,
            "leverage": 8,
        }

        class FakeClient:
            def __init__(self) -> None:
                self.cancelled = []
                self.submitted = []

            def signed_params(self):
                return {}

            def actual_position_rows(self, symbol=None):
                return [{"symbol": "HYPEUSDT", "positionAmt": "-0.650", "positionSide": "BOTH"}]

            def signed_request(self, method, endpoint, params):
                if endpoint == "/fapi/v1/openAlgoOrders":
                    return [{"symbol": "HYPEUSDT", "side": "BUY", "orderType": "STOP_MARKET", "quantity": "1.000", "triggerPrice": "63.372", "reduceOnly": True, "algoStatus": "NEW", "algoId": 200}]
                raise AssertionError(endpoint)

            def quantity_rules(self, symbol):
                return 0.001, 0.001

            def resolve_price(self, symbol, price):
                return "63.372"

            def cancel_order(self, symbol, order_id):
                self.cancelled.append((symbol, order_id))
                return {"status": "submitted", "response": {"algoId": order_id}}

            def submit_protective_stop(self, intent, quantity=None):
                self.submitted.append((intent.symbol, quantity))
                return {"status": "submitted", "response": {"algoId": 301, "clientAlgoId": "new-stop", "algoStatus": "NEW"}}

            def protective_stop_precheck(self, intent):
                return {"checked": True, "would_immediately_trigger": False}

            def close(self):
                pass

        fake_client = FakeClient()
        engine.client = fake_client  # type: ignore[assignment]
        result = engine.refresh_protective_stop_after_partial_take_profit(position)
        engine.close()

        assert result is not None
        self.assertEqual(result["status"], "ok")
        self.assertEqual(fake_client.cancelled, [("HYPEUSDT", 200)])
        self.assertEqual(fake_client.submitted, [("HYPEUSDT", "0.65")])
        self.assertEqual(position["protective_stop_order_id"], 301)

    def test_manual_profit_close_position_stop_survives_partial_manual_reduce(self) -> None:
        engine = self.make_engine()
        engine.config.order_mode = "testnet_order"
        engine.config.exit_control_mode = "stop_only_manual_profit"
        position = {
            "symbol": "HYPEUSDT",
            "side": "long",
            "protective_stop_order_id": 400,
            "stop_loss_price": 61.221,
            "entry_price": 62.144,
            "take_profit_price": 70.0,
            "notional_usdt": 700.0,
            "leverage": 8,
        }

        class FakeClient:
            def __init__(self) -> None:
                self.cancelled = []
                self.submitted = []

            def signed_params(self):
                return {}

            def signed_request(self, method, endpoint, params):
                if endpoint == "/fapi/v1/openAlgoOrders":
                    return [{
                        "symbol": "HYPEUSDT",
                        "side": "SELL",
                        "positionSide": "LONG",
                        "orderType": "STOP_MARKET",
                        "quantity": "0",
                        "triggerPrice": "61.221",
                        "reduceOnly": False,
                        "closePosition": True,
                        "algoStatus": "NEW",
                        "algoId": 400,
                    }]
                raise AssertionError(endpoint)

            def quantity_rules(self, symbol):
                return 0.01, 0.01

            def resolve_price(self, symbol, price):
                return "61.221"

            def cancel_order(self, symbol, order_id):
                self.cancelled.append((symbol, order_id))
                return {"status": "submitted", "response": {"algoId": order_id}}

            def submit_protective_stop(self, intent, quantity=None):
                self.submitted.append((intent.symbol, quantity))
                raise AssertionError("matching closePosition stop must not be replaced")

            def protective_stop_precheck(self, intent):
                return {"checked": True, "would_immediately_trigger": False}

            def close(self):
                pass

        fake_client = FakeClient()
        engine.client = fake_client  # type: ignore[assignment]
        actions = engine.sync_protective_stops_with_exchange(
            [position],
            {("HYPEUSDT", "long"): {"symbol": "HYPEUSDT", "positionAmt": "8.04", "positionSide": "LONG"}},
        )
        engine.close()

        self.assertEqual(actions, [])
        self.assertEqual(fake_client.cancelled, [])
        self.assertEqual(fake_client.submitted, [])
        self.assertEqual(position["protective_stop_order_id"], 400)
        self.assertTrue(position["protective_stop_close_position"])

    def test_manual_profit_sync_replaces_quantity_bound_stop(self) -> None:
        engine = self.make_engine()
        engine.config.order_mode = "testnet_order"
        engine.config.exit_control_mode = "stop_only_manual_profit"
        position = {
            "symbol": "BTCUSDT",
            "side": "long",
            "protective_stop_order_id": 500,
            "stop_loss_price": 63501.7,
            "entry_price": 65498.6,
            "take_profit_price": 70000.0,
            "notional_usdt": 980.0,
            "leverage": 8,
        }

        class FakeClient:
            def __init__(self) -> None:
                self.cancelled = []
                self.submitted = []

            def signed_params(self):
                return {}

            def signed_request(self, method, endpoint, params):
                if endpoint == "/fapi/v1/openAlgoOrders":
                    return [{
                        "symbol": "BTCUSDT",
                        "side": "SELL",
                        "positionSide": "LONG",
                        "orderType": "STOP_MARKET",
                        "quantity": "0.015",
                        "triggerPrice": "63501.7",
                        "reduceOnly": True,
                        "closePosition": False,
                        "algoStatus": "NEW",
                        "algoId": 500,
                    }]
                raise AssertionError(endpoint)

            def quantity_rules(self, symbol):
                return 0.001, 0.001

            def resolve_price(self, symbol, price):
                return "63501.7"

            def cancel_order(self, symbol, order_id):
                self.cancelled.append((symbol, order_id))
                return {"status": "submitted", "response": {"algoId": order_id}}

            def submit_protective_stop(self, intent, quantity=None):
                self.submitted.append((intent.symbol, quantity))
                return {
                    "status": "submitted",
                    "response": {
                        "algoId": 501,
                        "clientAlgoId": "close-position-stop",
                        "algoStatus": "NEW",
                        "closePosition": True,
                    },
                }

            def protective_stop_precheck(self, intent):
                return {"checked": True, "would_immediately_trigger": False}

            def close(self):
                pass

        fake_client = FakeClient()
        engine.client = fake_client  # type: ignore[assignment]
        actions = engine.sync_protective_stops_with_exchange(
            [position],
            {("BTCUSDT", "long"): {"symbol": "BTCUSDT", "positionAmt": "0.015", "positionSide": "LONG"}},
        )
        engine.close()

        self.assertEqual(fake_client.cancelled, [("BTCUSDT", 500)])
        self.assertEqual(fake_client.submitted, [("BTCUSDT", "0.015")])
        self.assertEqual([item["action"] for item in actions], ["cancel_stale_protective_stop", "submit_protective_stop"])
        self.assertEqual(position["protective_stop_order_id"], 501)
        self.assertTrue(position["protective_stop_close_position"])


if __name__ == "__main__":
    unittest.main()
