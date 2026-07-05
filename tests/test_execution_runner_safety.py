from __future__ import annotations

from typing import Any

from trading_system.config import StrategyConfig
from trading_system.execution_runner import BinanceExecutionRunner, BinanceExecutionError


class DummyRunner(BinanceExecutionRunner):
    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _require_credentials(self, request_mode: str) -> None:
        return None

    def _api_key(self) -> str:
        return "dummy"

    def _api_secret(self) -> str:
        return "dummy"

    def _current_position_mode(self) -> str:
        return "one_way"

    def _get_symbol_tick_size(self, symbol: str) -> float:
        return 0.01

    def _signed_request(self, method: str, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, endpoint, dict(params)))
        if endpoint == "/fapi/v1/algoOrders/openOrders":
            return {
                "orders": [
                    {
                        "symbol": "BTCUSDT",
                        "algoId": "111",
                        "clientAlgoId": "somnussl_owned_stop",
                    },
                    {
                        "symbol": "BTCUSDT",
                        "algoId": "222",
                        "clientAlgoId": "manual_or_other_strategy",
                    },
                ]
            }
        return {"ok": True}


def test_cancel_existing_protective_orders_never_cancels_all_symbol_orders() -> None:
    runner = DummyRunner(StrategyConfig())

    runner._cancel_existing_protective_orders("BTCUSDT")

    assert not any(endpoint == "/fapi/v1/allOpenOrders" for _, endpoint, _ in runner.calls)
    cancelled = [params for method, endpoint, params in runner.calls if method == "DELETE" and endpoint == "/fapi/v1/algoOrder"]
    assert [item["algoId"] for item in cancelled] == ["111"]


def test_submit_protective_stop_uses_owned_client_algo_id() -> None:
    runner = DummyRunner(StrategyConfig())

    runner._submit_protective_stop("BTCUSDT", "open_long", 62000.0)

    submissions = [params for method, endpoint, params in runner.calls if method == "POST" and endpoint == "/fapi/v1/algoOrder"]
    assert len(submissions) == 1
    assert submissions[0]["clientAlgoId"].startswith("somnussl")


class FailingStopRunner(DummyRunner):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.wal: list[dict[str, Any]] = []

    def _get_position_quantity_for_close(self, symbol: str, action: str) -> float:
        return 0.25

    def _get_symbol_quantity_rules(self, symbol: str) -> tuple[float, float]:
        return 0.001, 0.001

    def _signed_request(self, method: str, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, endpoint, dict(params)))
        if endpoint == "/fapi/v1/algoOrder":
            raise BinanceExecutionError("protective stop rejected")
        return {"ok": True}

    def _append_wal(self, payload: dict[str, Any]) -> None:
        self.wal.append(payload)


def test_protective_stop_failure_can_emergency_close_open_position() -> None:
    runner = FailingStopRunner(StrategyConfig())

    result = runner._emergency_close_after_protective_stop_failed(
        symbol="BTCUSDT",
        action="open_long",
        request_id="req1",
        generated_at="2026-01-01T00:00:00+00:00",
        error_text="protective stop rejected",
    )

    assert result["status"] == "submitted"
    close_orders = [params for method, endpoint, params in runner.calls if method == "POST" and endpoint == "/fapi/v1/order"]
    assert close_orders[0]["side"] == "SELL"
    assert close_orders[0]["reduceOnly"] == "true"
