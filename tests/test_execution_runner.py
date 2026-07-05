import json
from pathlib import Path

import pytest

from trading_system.config import StrategyConfig
from trading_system.execution_bridge import summarize_execution_results
from trading_system.execution_runner import BinanceExecutionError, BinanceExecutionRunner


class StubRunner(BinanceExecutionRunner):
    def __post_init__(self) -> None:
        self.config.execution_result_dir.mkdir(parents=True, exist_ok=True)
        self.config.execution_archive_dir.mkdir(parents=True, exist_ok=True)
        self._require_credentials(self._config_request_mode())
        self.calls = []

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        self.calls.append(("leverage", symbol, leverage))

    def _signed_request(self, method: str, endpoint: str, params: dict[str, str]) -> dict:
        self.calls.append((method, endpoint, dict(params), self._active_request_mode, self._base_url(), self._api_key()))
        return {"ok": True, "symbol": params.get("symbol"), "endpoint": endpoint, "params": dict(params)}

    def _get_mark_price(self, symbol: str) -> float:
        return 2.0

    def _get_symbol_quantity_rules(self, symbol: str) -> tuple[float, float]:
        return (0.1, 0.1)

    def _get_symbol_tick_size(self, symbol: str) -> float:
        return 0.01

    def _get_exchange_symbols(self) -> set[str]:
        return {"AIXBTUSDT", "BTCUSDT", "ETHUSDT", "PLAYUSDT"}

    def _get_position_quantity_for_close(self, symbol: str, action: str) -> float:
        return 1000.0


class FailingRunner(StubRunner):
    def _signed_request(self, method: str, endpoint: str, params: dict[str, str]) -> dict:
        raise BinanceExecutionError(f"Binance execution failed 400: bad request for {params.get('symbol')}")


class NoPositionRunner(StubRunner):
    def _get_position_quantity_for_close(self, symbol: str, action: str) -> float:
        return 0.0


class ProtectiveStopFailingRunner(StubRunner):
    def _signed_request(self, method: str, endpoint: str, params: dict[str, str]) -> dict:
        if endpoint == "/fapi/v1/algoOrder":
            raise BinanceExecutionError("Binance execution failed 400: protective stop rejected")
        return super()._signed_request(method, endpoint, params)


def test_execution_runner_requires_api_credentials(tmp_path: Path) -> None:
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        binance_api_key="",
        binance_api_secret="",
    )
    with pytest.raises(BinanceExecutionError, match="BINANCE_API_KEY"):
        BinanceExecutionRunner(config)


def test_execution_runner_executes_request_and_writes_result(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_1.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-13T00:00:00+00:00",
                "request_mode": "live_order",
                "skill_name": "derivatives-trading-usds-futures",
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "AIXBTUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "leverage": 3, "quantity_hint_usdt": 100, "stop_loss_price": 1.87},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        binance_api_key="key",
        binance_api_secret="secret",
        live_execution_confirmation="CONFIRM_LIVE_BINANCE_FUTURES",
        binance_position_mode="hedge",
    )
    runner = StubRunner(config)
    summary = runner.execute_request_file(request_file)

    assert summary["status"] == "submitted"
    assert summary["result_count"] == 1
    assert Path(summary["result_file"]).exists()
    assert Path(summary["archived_request_file"]).exists()
    assert not request_file.exists()
    result_payload = json.loads(Path(summary["result_file"]).read_text(encoding="utf-8"))
    assert result_payload["results"][0]["symbol"] == "AIXBTUSDT"
    assert runner.calls[0] == ("leverage", "AIXBTUSDT", 3)
    order_call = runner.calls[1]
    assert order_call[0:2] == ("POST", "/fapi/v1/order")
    assert order_call[2]["positionSide"] == "LONG"
    assert order_call[2]["quantity"] == "50"
    assert "reduceOnly" not in order_call[2]
    assert not any(call[0:2] == ("DELETE", "/fapi/v1/allOpenOrders") for call in runner.calls)
    algo_query_call = runner.calls[2]
    assert algo_query_call[0:2] == ("GET", "/fapi/v1/algoOrders/openOrders")
    stop_call = runner.calls[3]
    assert stop_call[0:2] == ("POST", "/fapi/v1/algoOrder")
    assert stop_call[2]["algoType"] == "CONDITIONAL"
    assert stop_call[2]["type"] == "STOP_MARKET"
    assert stop_call[2]["closePosition"] == "true"
    assert stop_call[2]["triggerPrice"] == "1.87"
    assert stop_call[2]["clientAlgoId"].startswith("somnussl")


def test_execution_runner_executes_all_pending_requests(tmp_path: Path) -> None:
    for idx in range(2):
        (tmp_path / f"execution_request_{idx}.json").write_text(
            json.dumps(
                {
                    "generated_at": "2026-04-13T00:00:00+00:00",
                    "request_mode": "test_order",
                    "skill_name": "derivatives-trading-usds-futures",
                    "binance_requests": [
                        {
                            "endpoint": "/fapi/v1/order/test",
                            "method": "POST",
                            "params": {"symbol": "AIXBTUSDT", "side": "BUY", "type": "MARKET"},
                            "meta": {"action": "open_long", "quantity_hint_usdt": 10},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    config = StrategyConfig(
        execution_request_dir=tmp_path,
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        binance_api_key="key",
        binance_api_secret="secret",
    )
    runner = StubRunner(config)
    summary = runner.execute_pending_requests()

    assert summary["request_count"] == 2
    assert summary["executed_count"] == 2
    assert len(summary["summaries"]) == 2


def test_execution_runner_records_trade_context_when_request_fails(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_fail.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-13T00:00:00+00:00",
                "request_mode": "live_order",
                "skill_name": "derivatives-trading-usds-futures",
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "leverage": 5, "quantity_hint_usdt": 250, "rationale": "oi_divergence_probe"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        binance_api_key="key",
        binance_api_secret="secret",
        live_execution_confirmation="CONFIRM_LIVE_BINANCE_FUTURES",
        binance_position_mode="hedge",
    )
    runner = FailingRunner(config)

    summary = runner.execute_request_file(request_file)

    assert summary["status"] == "partial"
    result = summary["results"][0]
    assert result["status"] == "failed"
    assert result["symbol"] == "BTCUSDT"
    assert result["request"]["params"]["side"] == "BUY"
    assert result["meta"]["quantity_hint_usdt"] == 250
    assert "bad request for BTCUSDT" in result["error"]

    digest = summarize_execution_results([{"payload": summary}])
    assert digest["latest_requests"][0]["symbol"] == "BTCUSDT"
    assert digest["latest_errors"][0]["params"]["side"] == "BUY"


def test_execution_runner_blocks_live_without_confirmation(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_live.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-13T00:00:00+00:00",
                "request_mode": "live_order",
                "risk_controls": {"status": "ok", "worst_case_stop_loss_usdt": 0},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "quantity_hint_usdt": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        binance_api_key="key",
        binance_api_secret="secret",
    )
    runner = StubRunner(config)

    with pytest.raises(BinanceExecutionError, match="TS_LIVE_EXECUTION_CONFIRMATION"):
        runner.execute_request_file(request_file)


def test_execution_runner_executes_testnet_without_live_confirmation(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_testnet.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-09T00:00:00+00:00",
                "request_mode": "testnet_order",
                "risk_controls": {"status": "ok", "worst_case_stop_loss_usdt": 1.0},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "leverage": 3, "quantity_hint_usdt": 100, "stop_loss_price": 95000},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_order_mode="testnet_order",
        binance_testnet_base_url="https://demo-fapi.binance.com",
        binance_testnet_api_key="testnet-key",
        binance_testnet_api_secret="testnet-secret",
        binance_position_mode="hedge",
    )
    runner = StubRunner(config)

    summary = runner.execute_request_file(request_file)

    assert summary["request_mode"] == "testnet_order"
    assert summary["status"] == "submitted"
    order_call = next(call for call in runner.calls if call[0:2] == ("POST", "/fapi/v1/order"))
    assert order_call[3] == "testnet_order"
    assert order_call[4] == "https://demo-fapi.binance.com"
    assert order_call[5] == "testnet-key"


def test_execution_runner_omits_position_side_for_one_way_mode(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_one_way.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-09T00:00:00+00:00",
                "request_mode": "testnet_order",
                "risk_controls": {"status": "ok", "worst_case_stop_loss_usdt": 1.0},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "positionSide": "LONG"},
                        "meta": {"action": "open_long", "quantity_hint_usdt": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_order_mode="testnet_order",
        binance_testnet_api_key="testnet-key",
        binance_testnet_api_secret="testnet-secret",
        binance_position_mode="one_way",
    )
    runner = StubRunner(config)

    summary = runner.execute_request_file(request_file)

    assert summary["status"] == "submitted"
    order_call = next(call for call in runner.calls if call[0:2] == ("POST", "/fapi/v1/order"))
    assert "positionSide" not in order_call[2]


def test_execution_runner_keeps_reduce_only_for_one_way_close(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_one_way_close.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-09T00:00:00+00:00",
                "request_mode": "testnet_order",
                "risk_controls": {"status": "ok", "worst_case_stop_loss_usdt": 1.0},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {
                            "symbol": "BTCUSDT",
                            "side": "SELL",
                            "type": "MARKET",
                            "reduceOnly": "true",
                            "positionSide": "LONG",
                        },
                        "meta": {"action": "close_long", "quantity_hint_usdt": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_order_mode="testnet_order",
        binance_testnet_api_key="testnet-key",
        binance_testnet_api_secret="testnet-secret",
        binance_position_mode="one_way",
    )
    runner = StubRunner(config)

    summary = runner.execute_request_file(request_file)

    assert summary["status"] == "submitted"
    order_call = next(call for call in runner.calls if call[0:2] == ("POST", "/fapi/v1/order"))
    assert "positionSide" not in order_call[2]
    assert order_call[2]["reduceOnly"] == "true"


def test_execution_runner_skips_symbol_missing_from_current_endpoint(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_missing_symbol.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-09T00:00:00+00:00",
                "request_mode": "testnet_order",
                "risk_controls": {"status": "ok", "worst_case_stop_loss_usdt": 1.0},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "BILLUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "quantity_hint_usdt": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_order_mode="testnet_order",
        binance_testnet_api_key="testnet-key",
        binance_testnet_api_secret="testnet-secret",
    )
    runner = StubRunner(config)

    summary = runner.execute_request_file(request_file)

    assert summary["status"] == "partial"
    assert summary["results"][0]["status"] == "skipped_unsupported_symbol"
    assert "symbol not available" in summary["results"][0]["reason"]
    assert not any(call[0:2] == ("POST", "/fapi/v1/order") for call in runner.calls)


def test_execution_runner_skips_close_when_binance_position_missing(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_close_without_position.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-09T00:00:00+00:00",
                "request_mode": "testnet_order",
                "risk_controls": {"status": "ok", "worst_case_stop_loss_usdt": 1.0},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "PLAYUSDT", "side": "SELL", "type": "MARKET", "reduceOnly": "true"},
                        "meta": {"action": "close_long", "quantity_hint_usdt": 15},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_order_mode="testnet_order",
        binance_testnet_api_key="testnet-key",
        binance_testnet_api_secret="testnet-secret",
        binance_position_mode="one_way",
    )
    runner = NoPositionRunner(config)

    summary = runner.execute_request_file(request_file)

    assert summary["status"] == "submitted"
    assert summary["results"][0]["status"] == "submitted"
    assert summary["results"][0]["response"] == {"info": "no_exchange_position_to_close"}
    assert not any(call[0:2] == ("POST", "/fapi/v1/order") for call in runner.calls)


def test_execution_runner_emergency_closes_when_protective_stop_fails(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_protective_stop_fail.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-09T00:00:00+00:00",
                "request_mode": "testnet_order",
                "risk_controls": {"status": "ok", "worst_case_stop_loss_usdt": 1.0},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {
                            "action": "open_long",
                            "quantity_hint_usdt": 100,
                            "stop_loss_price": 95000,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_order_mode="testnet_order",
        binance_testnet_api_key="testnet-key",
        binance_testnet_api_secret="testnet-secret",
        binance_position_mode="one_way",
    )
    runner = ProtectiveStopFailingRunner(config)

    summary = runner.execute_request_file(request_file)

    assert summary["status"] == "partial"
    result = summary["results"][0]
    assert result["status"] == "closed_after_protective_stop_failed"
    assert result["response"]["endpoint"] == "/fapi/v1/order"
    assert result["follow_up_orders"][0]["status"] == "failed"
    assert result["follow_up_orders"][0]["type"] == "protective_stop"
    assert result["follow_up_orders"][1]["status"] == "submitted"
    assert result["follow_up_orders"][1]["type"] == "emergency_close_after_protective_stop_failed"
    close_call = [call for call in runner.calls if call[0:2] == ("POST", "/fapi/v1/order")][-1]
    assert close_call[2]["side"] == "SELL"
    assert close_call[2]["reduceOnly"] == "true"


def test_execution_runner_blocks_live_risk_control_violation(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_blocked.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-13T00:00:00+00:00",
                "request_mode": "live_order",
                "risk_controls": {"status": "blocked", "violations": ["paper cycle loss fuse triggered"]},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "quantity_hint_usdt": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        binance_api_key="key",
        binance_api_secret="secret",
        live_execution_confirmation="CONFIRM_LIVE_BINANCE_FUTURES",
    )
    runner = StubRunner(config)

    with pytest.raises(BinanceExecutionError, match="risk controls"):
        runner.execute_request_file(request_file)


class InvalidSymbolRunner(StubRunner):
    """First call to /fapi/v1/order for unknown symbols returns -1121."""

    def _get_exchange_symbols(self) -> set[str]:
        return {"TRUTHUSDT", "BTCUSDT", "ETHUSDT"}

    def _signed_request(self, method: str, endpoint: str, params: dict[str, str]) -> dict:
        self.calls.append((method, endpoint, dict(params), self._active_request_mode, self._base_url(), self._api_key()))
        if endpoint == "/fapi/v1/order" and params.get("symbol") == "TRUTHUSDT":
            raise BinanceExecutionError('Binance execution failed 400: {"code":-1121,"msg":"Invalid symbol."}')
        return {"ok": True, "symbol": params.get("symbol"), "endpoint": endpoint}


def test_execution_runner_blacklists_invalid_symbol_after_1121(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_invalid_symbol.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-14T00:00:00+00:00",
                "request_mode": "testnet_order",
                "risk_controls": {"status": "ok", "worst_case_stop_loss_usdt": 1.0},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "TRUTHUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "quantity_hint_usdt": 15, "leverage": 3},
                    },
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "TRUTHUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "quantity_hint_usdt": 15, "leverage": 3},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_order_mode="testnet_order",
        binance_testnet_api_key="testnet-key",
        binance_testnet_api_secret="testnet-secret",
    )
    runner = InvalidSymbolRunner(config)

    summary = runner.execute_request_file(request_file)

    assert summary["result_count"] == 2
    assert summary["results"][0]["status"] == "failed"
    assert "-1121" in summary["results"][0]["error"]
    assert summary["results"][1]["status"] == "skipped_unsupported_symbol"
    assert "previously rejected" in summary["results"][1]["reason"]
    order_calls = [c for c in runner.calls if c[1] == "/fapi/v1/order"]
    assert len(order_calls) == 1


class TimestampCapturingRunner(StubRunner):
    """Captures timestamps at each stage to verify refresh after leverage."""

    def __init__(self, config):
        super().__init__(config)
        self.leverage_called_at: int | None = None
        self.order_timestamp: int | None = None

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        from datetime import datetime, timezone
        self.leverage_called_at = int(datetime.now(timezone.utc).timestamp() * 1000)
        super()._set_leverage(symbol, leverage)

    def _signed_request(self, method: str, endpoint: str, params: dict[str, str]) -> dict:
        if endpoint == "/fapi/v1/order":
            self.order_timestamp = int(params.get("timestamp", "0"))
        return super()._signed_request(method, endpoint, params)


def test_execution_runner_refreshes_timestamp_after_leverage(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_ts.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-14T00:00:00+00:00",
                "request_mode": "testnet_order",
                "risk_controls": {"status": "ok", "worst_case_stop_loss_usdt": 1.0},
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "leverage": 3, "quantity_hint_usdt": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_order_mode="testnet_order",
        binance_testnet_api_key="testnet-key",
        binance_testnet_api_secret="testnet-secret",
        binance_position_mode="one_way",
    )
    runner = TimestampCapturingRunner(config)

    summary = runner.execute_request_file(request_file)

    assert summary["status"] == "submitted"
    assert runner.leverage_called_at is not None
    assert runner.order_timestamp is not None
    assert runner.order_timestamp >= runner.leverage_called_at


def test_execution_runner_recv_window_defaults_to_30000(tmp_path: Path) -> None:
    request_file = tmp_path / "execution_request_rw.json"
    request_file.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-14T00:00:00+00:00",
                "request_mode": "testnet_order",
                "binance_requests": [
                    {
                        "endpoint": "/fapi/v1/order",
                        "method": "POST",
                        "params": {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"},
                        "meta": {"action": "open_long", "quantity_hint_usdt": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = StrategyConfig(
        execution_result_dir=tmp_path / "results",
        execution_archive_dir=tmp_path / "archive",
        execution_order_mode="testnet_order",
        binance_testnet_api_key="testnet-key",
        binance_testnet_api_secret="testnet-secret",
        binance_position_mode="one_way",
    )
    runner = StubRunner(config)

    runner.execute_request_file(request_file)

    order_call = next(call for call in runner.calls if call[0:2] == ("POST", "/fapi/v1/order"))
    assert order_call[2]["recvWindow"] == "30000"
