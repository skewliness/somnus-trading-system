from __future__ import annotations

import json
import logging

from trading_system.errors import (
    ErrorCategory,
    ErrorRiskAction,
    MarketDataError,
    OrderError,
    RiskError,
    V6ExecutionError,
    classify_exception,
    log_classified_exception,
)
from trading_system.v6_execution import V6ExecutionError as LegacyV6ExecutionError


def test_v6_execution_error_keeps_legacy_import_and_runtime_error_contract() -> None:
    assert LegacyV6ExecutionError is V6ExecutionError
    assert issubclass(V6ExecutionError, RuntimeError)
    assert str(V6ExecutionError("same message")) == "same message"


def test_explicit_exception_categories_are_preserved() -> None:
    market = classify_exception(MarketDataError("kline unavailable"), operation="build_signal", symbol="btcusdt")
    risk = classify_exception(RiskError("risk budget breached"), operation="risk_check")
    order = classify_exception(OrderError("submit rejected"), operation="submit_order", action="open_long")

    assert market.category == ErrorCategory.MARKET_DATA.value
    assert market.transient is True
    assert risk.category == ErrorCategory.RISK.value
    assert risk.severity == "critical"
    assert order.category == ErrorCategory.ORDER.value


def test_status_unknown_is_order_error_and_requires_reconciliation() -> None:
    record = classify_exception(
        V6ExecutionError('Binance signed request failed 408: {"code":-1007,"msg":"Timeout waiting for response; status unknown"}'),
        operation="submit_order",
        symbol="ETHUSDT",
        action="close_long",
        account_id="momo",
    )

    assert record.category == "order"
    assert record.code == "order_status_unknown"
    assert record.transient is True
    assert record.ambiguous_order_status is True
    assert record.risk_action == ErrorRiskAction.RECONCILE_ORDER.value


def test_error_categories_map_to_existing_risk_boundaries() -> None:
    market = classify_exception(MarketDataError("kline unavailable"), operation="build_signal")
    risk = classify_exception(RiskError("risk budget breached"), operation="risk_check")
    state = classify_exception(RuntimeError("ledger state mismatch"), operation="state_update")

    assert market.risk_action == "skip_symbol"
    assert risk.risk_action == "block_order"
    assert state.risk_action == "halt_position_mutation"


def test_structured_record_redacts_secrets_in_message_and_context() -> None:
    record = classify_exception(
        RuntimeError("request failed api_key=abc123&signature=deadbeef"),
        operation="public_market_data",
        context={"api_key": "abc123", "url": "https://example.test/path?token=secret-value"},
    ).to_dict()
    encoded = json.dumps(record)

    assert "abc123" not in encoded
    assert "deadbeef" not in encoded
    assert "secret-value" not in encoded
    assert "<redacted>" in encoded


def test_debug_logger_emits_structured_json_without_raising(caplog) -> None:
    logger = logging.getLogger("test_v65_errors")
    with caplog.at_level(logging.DEBUG, logger="test_v65_errors"):
        record = log_classified_exception(logger, ValueError("bad cache timestamp"), operation="cache_fallback")

    assert record.code == "unclassified_exception"
    assert "trade_error=" in caplog.text
