from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch


_CREDENTIAL_ENV_KEYS = (
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "BINANCE_TESTNET_API_KEY",
    "BINANCE_TESTNET_API_SECRET",
    "BINANCE_TESTNET_SECRET_KEY",
    "BINANCE_V6_DEMO_API_KEY",
    "BINANCE_V6_DEMO_API_SECRET",
)

for _key in _CREDENTIAL_ENV_KEYS:
    os.environ[_key] = ""
os.environ["TS_EXECUTION_ORDER_MODE"] = "auto"
os.environ["SOMNUS_SKIP_LOCAL_ENV"] = "1"

from scripts.run_v5_v6_execution import filter_exchange_supported_symbols, run_locked
from trading_system.v6_execution import V6ExecutionConfig


for _key in _CREDENTIAL_ENV_KEYS:
    os.environ.pop(_key, None)
os.environ.pop("TS_EXECUTION_ORDER_MODE", None)
os.environ.pop("SOMNUS_SKIP_LOCAL_ENV", None)


class FakeClient:
    def __init__(self, config: V6ExecutionConfig) -> None:
        self.config = config

    def exchange_info(self) -> dict:
        return {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING"},
                {"symbol": "PUMPUSDT", "status": "SETTLING"},
            ]
        }

    def close(self) -> None:
        return None


def test_filter_exchange_supported_symbols_requires_trading_status() -> None:
    config = V6ExecutionConfig(order_mode="dry_run")
    with patch("scripts.run_v5_v6_execution.V6BinanceClient", FakeClient):
        kept, skipped, error = filter_exchange_supported_symbols(config, ["BTCUSDT", "PUMPUSDT"])

    assert kept == ["BTCUSDT"]
    assert skipped == ["PUMPUSDT"]
    assert error == ""


def test_positions_only_does_not_read_entry_watchlist(tmp_path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "entry_watchlist.json").write_text(
        '{"symbols":{"WATCHUSDT":{"symbol":"WATCHUSDT","updated_at":"2099-01-01T00:00:00+00:00","expires_at":"2099-01-01T01:00:00+00:00"}}}',
        encoding="utf-8",
    )
    output_report = tmp_path / "positions_only_report.json"
    args = SimpleNamespace(
        positions_only=True,
        report=str(tmp_path / "missing_report.json"),
        limit=30,
        output_report=str(output_report),
        submit=False,
        archive_v5_requests=False,
    )

    with patch("scripts.run_v5_v6_execution.load_v5_entry_watchlist", side_effect=AssertionError("positions_only must not read entry_watchlist")):
        payload = run_locked(args, state_dir)

    assert payload["run_scope"] == "positions_only"
    assert payload["source_symbols"] == []
    assert payload["symbols"] == []
    assert payload["entry_watchlist"]["used_for_scan"] is False
    assert payload["entry_watchlist"]["updated"] is False
    assert payload["entry_watchlist"]["count"] == 0
    assert payload["entry_watchlist"]["symbols"] == []
