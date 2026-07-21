from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import scripts.check_v6_live_readiness as live_readiness
from trading_system.v6_execution import (
    V6BinanceClient,
    V6ExecutionConfig,
    V6ExecutionEngine,
    V6ExecutionError,
    V6OrderIntent,
)


class V6DualInstanceTest(unittest.TestCase):
    @staticmethod
    def live_config(root: Path) -> V6ExecutionConfig:
        return V6ExecutionConfig(
            order_mode="live_order",
            state_dir=root,
            state_file=root / "state.json",
            signals_file=root / "signals.jsonl",
            entry_gate_events_file=root / "entry_gate_events.jsonl",
            orders_file=root / "orders.jsonl",
            wal_file=root / "execution_wal.jsonl",
            latest_summary_file=root / "latest_run_summary.json",
            closed_trade_review_queue_file=root / "closed_trade_review_queue.jsonl",
            post_trade_training_samples_file=root / "post_trade_training_samples.jsonl",
            demo_api_key="demo-key",
            demo_api_secret="demo-secret",
            live_api_key="live-key",
            live_api_secret="live-secret",
        )

    def test_live_runtime_accepts_isolated_state_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.live_config(Path(tmp) / "v6_execution_live")
            config.validate_runtime_isolation()

    def test_live_runtime_rejects_demo_state_directory(self) -> None:
        config = V6ExecutionConfig(order_mode="live_order")
        with self.assertRaisesRegex(V6ExecutionError, "Demo state directory"):
            config.validate_runtime_isolation()

    def test_live_runtime_rejects_mutable_path_outside_live_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "v6_execution_live"
            config = self.live_config(root)
            config.orders_file = Path(tmp) / "v6_execution" / "orders.jsonl"
            with self.assertRaisesRegex(V6ExecutionError, "must stay under"):
                config.validate_runtime_isolation()

    def test_live_runtime_rejects_reused_demo_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.live_config(Path(tmp) / "v6_execution_live")
            config.live_api_key = config.demo_api_key
            config.live_api_secret = config.demo_api_secret
            with self.assertRaisesRegex(V6ExecutionError, "credentials must be different"):
                config.validate_runtime_isolation()

    def test_from_env_routes_every_mutable_path_to_live_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "v6_execution_live"
            env = {
                "TS_V6_ORDER_MODE": "live_order",
                "TS_V6_EXECUTION_DIR": str(root),
                "BINANCE_V6_DEMO_API_KEY": "demo-key",
                "BINANCE_V6_DEMO_API_SECRET": "demo-secret",
                "BINANCE_V6_API_KEY": "live-key",
                "BINANCE_V6_API_SECRET": "live-secret",
            }
            with patch.dict(os.environ, env, clear=True):
                config = V6ExecutionConfig.from_env()
            config.validate_runtime_isolation()
            self.assertEqual(config.state_dir, root)
            for path in config.mutable_runtime_paths():
                self.assertIn(root.resolve(), (path.resolve(), *path.resolve().parents))

    def test_demo_and_live_clients_select_different_credentials_and_hosts(self) -> None:
        demo = V6ExecutionConfig(
            order_mode="testnet_order",
            demo_api_key="demo-key",
            demo_api_secret="demo-secret",
            live_api_key="live-key",
            live_api_secret="live-secret",
        )
        live = V6ExecutionConfig(
            order_mode="live_order",
            demo_api_key="demo-key",
            demo_api_secret="demo-secret",
            live_api_key="live-key",
            live_api_secret="live-secret",
        )
        demo_client = V6BinanceClient(demo)
        live_client = V6BinanceClient(live)
        try:
            self.assertEqual(demo_client.api_key(), "demo-key")
            self.assertEqual(live_client.api_key(), "live-key")
            self.assertNotEqual(demo_client.order_base_url(), live_client.order_base_url())
        finally:
            demo_client.close()
            live_client.close()

    def test_live_readiness_blocks_isolated_margin_under_multi_assets_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.live_config(Path(tmp) / "v6_execution_live")
            config.margin_type = "ISOLATED"

            class FakeClient:
                def __init__(self, _config: V6ExecutionConfig) -> None:
                    pass

                @staticmethod
                def signed_params() -> dict[str, int]:
                    return {}

                @staticmethod
                def signed_request(_method: str, path: str, _params: dict[str, int]) -> object:
                    responses = {
                        "/fapi/v2/account": {
                            "canTrade": True,
                            "positions": [],
                            "assets": [{"asset": "USDT", "availableBalance": "1000"}],
                        },
                        "/fapi/v1/positionSide/dual": {"dualSidePosition": False},
                        "/fapi/v1/openOrders": [],
                        "/fapi/v1/multiAssetsMargin": {"multiAssetsMargin": True},
                    }
                    return responses[path]

                @staticmethod
                def close() -> None:
                    pass

            output = io.StringIO()
            with (
                patch.object(live_readiness.V6ExecutionConfig, "from_env", return_value=config),
                patch.object(live_readiness, "V6BinanceClient", FakeClient),
                redirect_stdout(output),
            ):
                result = live_readiness.main()

            report = json.loads(output.getvalue())
            self.assertEqual(result, 2)
            self.assertEqual(report["status"], "blocked")
            self.assertTrue(report["multi_assets_mode"])
            self.assertEqual(report["configured_margin_type"], "ISOLATED")
            self.assertFalse(report["margin_mode_compatible"])
            self.assertIn("ISOLATED margin is unavailable", " ".join(report["blocked_reasons"]))

    def test_live_readiness_accepts_cross_margin_under_multi_assets_mode(self) -> None:
        self.assertTrue(
            live_readiness.margin_mode_is_compatible(
                multi_assets_mode=True,
                configured_margin_type="CROSSED",
            )
        )

    def test_live_multi_asset_virtual_risk_is_written_to_intent_and_lot_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.live_config(Path(tmp) / "v6_execution_live")
            config.margin_type = "CROSSED"
            config.account_asset_mode = "MULTI_ASSET"
            config.virtual_risk_isolation_enabled = True
            engine = object.__new__(V6ExecutionEngine)
            engine.config = config
            intent = V6OrderIntent(
                action="open_short",
                symbol="ZECUSDT",
                side="SELL",
                quantity_hint_usdt=153.0,
                leverage=8,
                reduce_only=False,
                entry_price=544.8,
                stop_loss_price=561.144,
                take_profit_price=419.199,
                long_probability=0.2,
                short_probability=0.6,
                wait_probability=0.2,
                rationale="test",
                generated_at="2026-07-18T00:00:00Z",
                risk_plan={"entry_timeframe": "4h"},
            )

            engine.annotate_intent_execution_risk(intent)
            compact = engine.compact_risk_plan_for_lot(intent.risk_plan)

            self.assertEqual(compact["execution_account"]["account_asset_mode"], "MULTI_ASSET")
            self.assertEqual(compact["execution_account"]["exchange_margin_type"], "CROSSED")
            self.assertEqual(compact["execution_account"]["local_isolation_mode"], "virtual_subledger")
            self.assertEqual(compact["local_risk_budget"]["initial_margin_budget_usdt"], 19.125)
            self.assertAlmostEqual(compact["local_risk_budget"]["planned_loss_at_stop_usdt"], 4.59, places=6)

    def test_live_readiness_account_asset_mode_must_match_exchange(self) -> None:
        self.assertTrue(
            live_readiness.account_asset_mode_matches(
                multi_assets_mode=True,
                configured_account_asset_mode="MULTI_ASSET",
            )
        )
        self.assertFalse(
            live_readiness.account_asset_mode_matches(
                multi_assets_mode=True,
                configured_account_asset_mode="SINGLE_ASSET",
            )
        )

    def test_exchange_recovered_lot_keeps_virtual_risk_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.live_config(Path(tmp) / "v6_execution_live")
            config.margin_type = "CROSSED"
            config.account_asset_mode = "MULTI_ASSET"
            config.virtual_risk_isolation_enabled = True
            engine = object.__new__(V6ExecutionEngine)
            engine.config = config

            lot = engine.recovered_lot_from_position(
                {
                    "symbol": "ZECUSDT",
                    "side": "short",
                    "entry_price": 544.0,
                    "current_price": 540.0,
                    "stop_loss_price": 560.0,
                    "leverage": 8,
                },
                actual_qty=0.25,
                reason="test_recovery",
            )

            accounting = lot["local_risk_accounting"]
            self.assertEqual(accounting["source"], "exchange_recovered")
            self.assertEqual(accounting["execution_account"]["local_isolation_mode"], "virtual_subledger")
            self.assertGreater(accounting["risk_budget"]["planned_loss_at_stop_usdt"], 0.0)


if __name__ == "__main__":
    unittest.main()
