from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import patch

import scripts.telegram_monitor_v5_v6 as tg
from trading_system.v6_execution import (
    V6BinanceClient,
    V6ExecutionConfig,
    V6ExecutionEngine,
    V6ExecutionError,
    V6OrderIntent,
)


def make_close_intent(*, action: str = "close_long", close_fraction: float = 1.0) -> V6OrderIntent:
    return V6OrderIntent(
        action=action,
        symbol="BTCUSDT",
        side="SELL" if action == "close_long" else "BUY",
        quantity_hint_usdt=100.0,
        leverage=5,
        reduce_only=True,
        entry_price=100.0,
        stop_loss_price=95.0,
        take_profit_price=110.0,
        long_probability=0.5,
        short_probability=0.3,
        wait_probability=0.2,
        rationale="test close",
        generated_at="2026-07-06T00:00:00+00:00",
        close_fraction=close_fraction,
    )


class CloseResolverConfigTest(unittest.TestCase):
    def test_env_flags_parse_and_default_off(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = V6ExecutionConfig.from_env()
        self.assertFalse(config.close_resolver_enabled)
        self.assertFalse(config.close_intent_coalescing_enabled)
        env = {
            "TS_V6_CLOSE_RESOLVER_ENABLED": "1",
            "TS_V6_CLOSE_COALESCE_ENABLED": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = V6ExecutionConfig.from_env()
        self.assertTrue(config.close_resolver_enabled)
        self.assertTrue(config.close_intent_coalescing_enabled)


class ResolveCloseQuantityViaResolverTest(unittest.TestCase):
    def make_client(self, *, position_amt: float | None, mark_price: float = 100.0) -> V6BinanceClient:
        config = V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run", close_resolver_enabled=True)
        client = V6BinanceClient(config)
        row = None
        if position_amt is not None:
            row = {"symbol": "BTCUSDT", "positionAmt": str(position_amt), "markPrice": str(mark_price)}
        patch.object(client, "actual_position_for_intent", return_value=row).start()
        patch.object(client, "quantity_rules", return_value=(0.001, 5.0)).start()
        self.addCleanup(patch.stopall)
        return client

    def test_full_close_uses_exchange_position_amt(self) -> None:
        client = self.make_client(position_amt=4.7)
        intent = make_close_intent(close_fraction=1.0)
        quantity = client.resolve_close_quantity(intent)
        self.assertEqual(float(quantity), 4.7)
        decision = intent.risk_plan.get("close_decision") or {}
        self.assertEqual(decision.get("mode"), "full_flatten")

    def test_fraction_above_090_escalates_to_full_flatten(self) -> None:
        client = self.make_client(position_amt=4.0)
        intent = make_close_intent(close_fraction=0.95)
        quantity = client.resolve_close_quantity(intent)
        self.assertEqual(float(quantity), 4.0)
        self.assertEqual((intent.risk_plan.get("close_decision") or {}).get("mode"), "full_flatten")

    def test_partial_remainder_below_dust_escalates_to_full_flatten(self) -> None:
        # remaining after 50% close = 0.005 * 100 = 0.5 USDT < dust threshold (5.0)
        client = self.make_client(position_amt=0.010, mark_price=100.0)
        intent = make_close_intent(close_fraction=0.5)
        quantity = client.resolve_close_quantity(intent)
        self.assertEqual(float(quantity), 0.010)
        decision = intent.risk_plan.get("close_decision") or {}
        self.assertEqual(decision.get("mode"), "full_flatten")
        self.assertIn("dust", decision.get("reason") or "")

    def test_missing_exchange_position_raises_legacy_phrase(self) -> None:
        client = self.make_client(position_amt=None)
        intent = make_close_intent()
        with self.assertRaises(V6ExecutionError) as ctx:
            client.resolve_close_quantity(intent)
        self.assertIn("no exchange position found", str(ctx.exception))
        decision = intent.risk_plan.get("close_decision") or {}
        self.assertEqual(decision.get("mode"), "skip_missing_exchange_position")

    def test_zero_quantity_raises_legacy_phrase(self) -> None:
        # 5% of 0.010 rounds down to zero at step 0.001
        client = self.make_client(position_amt=0.010, mark_price=1_000_000.0)
        intent = make_close_intent(close_fraction=0.05)
        with self.assertRaises(V6ExecutionError) as ctx:
            client.resolve_close_quantity(intent)
        self.assertIn("resolved zero partial close quantity", str(ctx.exception))


class ResidualPositionAfterFullCloseTest(unittest.TestCase):
    def make_engine(self) -> V6ExecutionEngine:
        config = V6ExecutionConfig(symbols=["BTCUSDT"], order_mode="dry_run", close_resolver_enabled=True)
        return V6ExecutionEngine(config)

    def make_position(self) -> dict[str, Any]:
        return {
            "symbol": "BTCUSDT",
            "side": "long",
            "quantity": 1.0,
            "entry_price": 100.0,
            "current_price": 100.0,
            "lots": [
                {"quantity": 1.0, "remaining_quantity": 1.0, "status": "open", "entry_price": 100.0},
            ],
        }

    def make_result(self, after_amt: float, *, after_error: str = "") -> dict[str, Any]:
        return {
            "status": "submitted",
            "submitted_at": "2026-07-06T00:00:00+00:00",
            "response": {"orderId": 123456, "executedQty": "0.6", "avgPrice": "101.0"},
            "position_reconcile": {
                "after": {"position_amt": after_amt, "notional": abs(after_amt) * 100.0, "mark_price": 100.0},
                "after_error": after_error,
                "delta_qty_abs": 0.6,
            },
        }

    def test_flat_exchange_marks_lots_closed_and_drops_position(self) -> None:
        engine = self.make_engine()
        position = self.make_position()
        residual = engine.residual_position_after_full_close(position, make_close_intent(), self.make_result(0.0))
        self.assertIsNone(residual)
        self.assertTrue(all(lot["status"] == "closed" for lot in position["lots"]))

    def test_residual_exchange_quantity_keeps_scaled_position(self) -> None:
        engine = self.make_engine()
        position = self.make_position()
        residual = engine.residual_position_after_full_close(position, make_close_intent(), self.make_result(0.4))
        self.assertIsNotNone(residual)
        assert residual is not None
        self.assertAlmostEqual(residual["quantity"], 0.4)
        remaining = sum(float(lot.get("remaining_quantity") or 0.0) for lot in residual["lots"])
        self.assertAlmostEqual(remaining, 0.4, places=6)
        events = residual.get("close_events") or []
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["order_id"], 123456)

    def test_after_error_returns_none_without_touching_lots(self) -> None:
        engine = self.make_engine()
        position = self.make_position()
        residual = engine.residual_position_after_full_close(
            position, make_close_intent(), self.make_result(0.4, after_error="timeout")
        )
        self.assertIsNone(residual)
        self.assertEqual(position["lots"][0]["status"], "open")


class TelegramCloseEventTest(unittest.TestCase):
    def test_zh_close_mode_mappings(self) -> None:
        self.assertEqual(tg.zh_close_mode({"mode": "partial"}), "部分平仓")
        self.assertEqual(tg.zh_close_mode({"mode": "full_flatten", "reason": "x"}), "全平")
        self.assertEqual(
            tg.zh_close_mode({"mode": "full_flatten", "reason": "partial_remainder_below_dust_notional"}),
            "粉尘全平",
        )
        self.assertEqual(tg.zh_close_mode({"mode": "skip_zero"}), "跳过(数量为零)")
        self.assertEqual(tg.zh_close_mode({"mode": "skip_missing_exchange_position"}), "跳过(交易所无持仓)")

    def test_close_intent_shows_mode(self) -> None:
        intent = {
            "symbol": "BTCUSDT",
            "action": "close_long",
            "quantity_hint_usdt": 100.0,
            "rationale": "tp stage 1",
            "risk_plan": {"close_decision": {"mode": "full_flatten", "reason": "requested_fraction"}},
        }
        text = tg.format_compact_execution_intent(intent)
        self.assertIn("模式=全平", text)

    def test_close_failure_fingerprint_stable_across_timestamps(self) -> None:
        def failure(submitted_at: str) -> dict[str, Any]:
            return {
                "status": "failed",
                "error": "APIError -2022 ReduceOnly Order is rejected",
                "submitted_at": submitted_at,
                "intent": {
                    "symbol": "BTCUSDT",
                    "action": "close_long",
                    "side": "SELL",
                    "target_name": "tp_stage_1",
                    "risk_plan": {"close_decision": {"mode": "partial", "reason": "requested_fraction"}},
                },
            }

        v6_a = {"intents": [], "results": [failure("2026-07-06T00:00:00+00:00")]}
        v6_b = {"intents": [], "results": [failure("2026-07-06T00:03:00+00:00")]}
        key_a = tg.short_hash(tg.stable_v6_activity_fingerprint(v6_a))
        key_b = tg.short_hash(tg.stable_v6_activity_fingerprint(v6_b))
        self.assertEqual(key_a, key_b)

    def test_different_error_codes_produce_different_fingerprints(self) -> None:
        base = {
            "status": "failed",
            "intent": {"symbol": "BTCUSDT", "action": "close_long", "side": "SELL"},
        }
        v6_a = {"intents": [], "results": [{**base, "error": "code -2022 rejected"}]}
        v6_b = {"intents": [], "results": [{**base, "error": "code -1007 timeout"}]}
        key_a = tg.short_hash(tg.stable_v6_activity_fingerprint(v6_a))
        key_b = tg.short_hash(tg.stable_v6_activity_fingerprint(v6_b))
        self.assertNotEqual(key_a, key_b)

    def test_successful_result_fingerprint_unchanged_semantics(self) -> None:
        success = {
            "status": "submitted",
            "intent": {"symbol": "BTCUSDT", "action": "close_long", "side": "SELL"},
            "response": {"orderId": 1},
        }
        fp = tg.stable_v6_activity_fingerprint({"intents": [], "results": [success]})
        self.assertEqual(fp["results"][0], success)


if __name__ == "__main__":
    unittest.main()
