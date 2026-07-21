from __future__ import annotations

import pytest

from trading_system.errors import V6ExecutionError
from trading_system.v6_execution import V6BinanceClient, V6ExecutionConfig


def _client(tmp_path) -> V6BinanceClient:
    state_dir = tmp_path / "state"
    return V6BinanceClient(
        V6ExecutionConfig(
            order_mode="testnet_order",
            demo_api_key="test-key",
            demo_api_secret="test-secret",
            state_dir=state_dir,
            order_state_file=state_dir / "order_states.json",
        )
    )


def test_tracked_submit_persists_partial_exchange_fill(tmp_path) -> None:
    client = _client(tmp_path)
    client.signed_request = lambda method, endpoint, params: {  # type: ignore[method-assign]
        "orderId": 123,
        "status": "PARTIALLY_FILLED",
        "executedQty": "0.2",
        "origQty": "1",
    }

    payload, order_key, snapshot = client.submit_tracked_order(
        symbol="BTCUSDT",
        action="open_long",
        endpoint="/fapi/v1/order",
        params={"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"},
    )
    restored = client.order_state_store.get(order_key)
    client.close()

    assert payload["orderId"] == 123
    assert snapshot["current_state"] == "partial_filled"
    assert restored is not None and restored["exchange_order_id"] == "123"


def test_tracked_submit_timeout_is_pending_after_client_restart(tmp_path) -> None:
    first = _client(tmp_path)

    def timeout(*_args, **_kwargs):
        raise V6ExecutionError('Binance 408: {"code":-1007,"msg":"status unknown"}')

    first.signed_request = timeout  # type: ignore[method-assign]
    with pytest.raises(V6ExecutionError) as caught:
        first.submit_tracked_order(
            symbol="ETHUSDT",
            action="close_long",
            endpoint="/fapi/v1/order",
            params={"symbol": "ETHUSDT", "side": "SELL", "type": "MARKET"},
        )
    order_key = str(getattr(caught.value, "order_state_key", ""))
    first.close()

    restarted = _client(tmp_path)
    pending = restarted.pending_order_reconciliation()
    restarted.close()

    assert order_key
    assert [item["order_key"] for item in pending] == [order_key]
    assert pending[0]["current_state"] == "unknown_pending_reconcile"


def test_cancel_failure_is_recorded_but_original_error_still_propagates(tmp_path) -> None:
    client = _client(tmp_path)

    def reject(*_args, **_kwargs):
        raise V6ExecutionError('Binance signed request failed 400: {"code":-2011,"msg":"Unknown order"}')

    client.signed_request = reject  # type: ignore[method-assign]
    with pytest.raises(V6ExecutionError):
        client.cancel_order("SOLUSDT", 77)
    snapshot = client.order_state_store.get("exchange:SOLUSDT:77")
    client.close()

    assert snapshot is not None
    assert snapshot["current_state"] == "rejected"
    assert snapshot["terminal"] is True


def test_sidecar_failure_never_blocks_existing_order_submission(tmp_path) -> None:
    client = _client(tmp_path)
    client._order_state_call = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    client.signed_request = lambda method, endpoint, params: {"orderId": 9, "status": "FILLED"}  # type: ignore[method-assign]

    payload, order_key, snapshot = client.submit_tracked_order(
        symbol="HYPEUSDT",
        action="open_short",
        endpoint="/fapi/v1/order",
        params={"symbol": "HYPEUSDT", "side": "SELL", "type": "MARKET"},
    )
    client.close()

    assert payload["orderId"] == 9
    assert order_key == ""
    assert snapshot == {}
