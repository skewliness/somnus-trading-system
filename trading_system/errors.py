from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Any, Mapping

from trading_system.redaction import redact_sensitive_text


SENSITIVE_KEY_PARTS = ("api_key", "secret", "token", "cookie", "authorization", "password", "signature")


class ErrorCategory(str, Enum):
    MARKET_DATA = "market_data"
    EXCHANGE = "exchange"
    ORDER = "order"
    RISK = "risk"
    STATE = "state"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorRiskAction(str, Enum):
    """Operational response already expected at a failed V6 boundary."""

    OBSERVE = "observe"
    SKIP_SYMBOL = "skip_symbol"
    BLOCK_ORDER = "block_order"
    RECONCILE_ORDER = "reconcile_order"
    HALT_POSITION_MUTATION = "halt_position_mutation"
    RETRY_RECOVERY = "retry_recovery"


class TradingSystemError(RuntimeError):
    category = ErrorCategory.UNKNOWN
    code = "trading_system_error"
    severity = ErrorSeverity.ERROR
    transient = False
    ambiguous_order_status = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        transient: bool | None = None,
        ambiguous_order_status: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = str(code or self.code)
        self.is_transient = self.transient if transient is None else bool(transient)
        self.is_ambiguous_order_status = (
            self.ambiguous_order_status if ambiguous_order_status is None else bool(ambiguous_order_status)
        )


class MarketDataError(TradingSystemError):
    category = ErrorCategory.MARKET_DATA
    code = "market_data_error"
    transient = True


class ExchangeError(TradingSystemError):
    category = ErrorCategory.EXCHANGE
    code = "exchange_error"


class OrderError(TradingSystemError):
    category = ErrorCategory.ORDER
    code = "order_error"


class RiskError(TradingSystemError):
    category = ErrorCategory.RISK
    code = "risk_error"
    severity = ErrorSeverity.CRITICAL


class StateError(TradingSystemError):
    category = ErrorCategory.STATE
    code = "state_error"


class RecoveryError(TradingSystemError):
    category = ErrorCategory.RECOVERY
    code = "recovery_error"
    transient = True


class V6ExecutionError(TradingSystemError):
    """Compatibility error retained at the historic import path."""

    category = ErrorCategory.EXCHANGE
    code = "v6_execution_error"


def _safe_context(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(part in lowered for part in SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(item_key): _safe_context(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_context(item) for item in value[:50]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_sensitive_text(value)[:1000]


def _derived_code(message: str) -> str:
    lowered = message.lower()
    if "-1007" in lowered or "status unknown" in lowered or "timeout waiting for response" in lowered:
        return "order_status_unknown"
    if "429" in lowered or "418" in lowered or "too many requests" in lowered:
        return "rate_limited"
    if "name resolution" in lowered or "temporary failure in name resolution" in lowered:
        return "dns_failure"
    if "timeout" in lowered or "timed out" in lowered:
        return "network_timeout"
    if "ssl" in lowered or "unexpected_eof" in lowered:
        return "tls_failure"
    if "-1121" in lowered or "invalid symbol" in lowered or "invalid trading pair" in lowered:
        return "invalid_symbol"
    if "reduceonly order is rejected" in lowered or "-2022" in lowered:
        return "reduce_only_rejected"
    if "no exchange position found" in lowered:
        return "exchange_position_missing"
    return "unclassified_exception"


def _infer_category(message: str, operation: str, action: str) -> ErrorCategory:
    lowered = message.lower()
    operation_lower = operation.lower()
    if "reconcile" in operation_lower:
        return ErrorCategory.RECOVERY
    if action or any(token in operation_lower for token in ("order", "submit", "open", "close", "cancel")):
        return ErrorCategory.ORDER
    if any(token in operation_lower for token in ("signal", "market", "kline", "ticker", "price")):
        return ErrorCategory.MARKET_DATA
    if any(token in lowered for token in ("kline", "ticker", "mark price", "market data")):
        return ErrorCategory.MARKET_DATA
    if any(token in lowered for token in ("binance", "exchange", "http", "ssl", "timeout", "name resolution")):
        return ErrorCategory.EXCHANGE
    if isinstance(message, str) and any(token in lowered for token in ("state", "ledger", "position")):
        return ErrorCategory.STATE
    return ErrorCategory.UNKNOWN


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    occurred_at: str
    category: str
    code: str
    severity: str
    transient: bool
    ambiguous_order_status: bool
    risk_action: str
    exception_type: str
    message: str
    operation: str
    symbol: str
    action: str
    account_id: str
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_exception(
    exc: BaseException,
    *,
    operation: str = "",
    symbol: str = "",
    action: str = "",
    account_id: str = "",
    context: Mapping[str, Any] | None = None,
) -> ErrorRecord:
    raw_message = str(exc)
    message = redact_sensitive_text(raw_message)[:2000]
    lowered = raw_message.lower()
    ambiguous = "-1007" in lowered or "status unknown" in lowered or "timeout waiting for response" in lowered
    transient = any(
        token in lowered
        for token in ("timeout", "timed out", "429", "418", "too many requests", "name resolution", "connection", "ssl")
    )
    category = _infer_category(raw_message, operation, action)
    code = _derived_code(raw_message)
    severity = ErrorSeverity.WARNING if transient else ErrorSeverity.ERROR
    if isinstance(exc, TradingSystemError):
        category = exc.category
        code = exc.error_code
        transient = exc.is_transient or transient
        ambiguous = exc.is_ambiguous_order_status or ambiguous
        severity = exc.severity
        if code == V6ExecutionError.code:
            derived = _derived_code(raw_message)
            code = derived if derived != "unclassified_exception" else code
            category = _infer_category(raw_message, operation, action)
    risk_action = _risk_action_for(
        category=category,
        code=code,
        operation=operation,
        ambiguous_order_status=ambiguous,
    )
    return ErrorRecord(
        occurred_at=datetime.now(timezone.utc).isoformat(),
        category=category.value,
        code=code,
        severity=severity.value,
        transient=bool(transient),
        ambiguous_order_status=bool(ambiguous),
        risk_action=risk_action.value,
        exception_type=type(exc).__name__,
        message=message,
        operation=str(operation or "")[:160],
        symbol=str(symbol or "").upper()[:40],
        action=str(action or "")[:80],
        account_id=str(account_id or "")[:160],
        context=_safe_context(dict(context or {})),
    )


def _risk_action_for(
    *,
    category: ErrorCategory,
    code: str,
    operation: str,
    ambiguous_order_status: bool,
) -> ErrorRiskAction:
    operation_lower = str(operation or "").lower()
    if ambiguous_order_status or code == "order_status_unknown":
        return ErrorRiskAction.RECONCILE_ORDER
    if category == ErrorCategory.MARKET_DATA:
        return ErrorRiskAction.SKIP_SYMBOL
    if category in {ErrorCategory.ORDER, ErrorCategory.RISK}:
        return ErrorRiskAction.BLOCK_ORDER
    if category == ErrorCategory.STATE:
        return ErrorRiskAction.HALT_POSITION_MUTATION
    if category == ErrorCategory.RECOVERY or "reconcile" in operation_lower:
        return ErrorRiskAction.RETRY_RECOVERY
    return ErrorRiskAction.OBSERVE


def record_caught_exception(
    logger: logging.Logger,
    exc: BaseException,
    *,
    operation: str,
    level: int = logging.DEBUG,
    **fields: Any,
) -> ErrorRecord:
    """Classify and log a caught exception without changing control flow."""

    try:
        return log_classified_exception(logger, exc, operation=operation, level=level, **fields)
    except Exception as logging_exc:  # pragma: no cover - logging must never mask the original failure
        logger.error("failed to classify caught exception operation=%s error=%s", operation, logging_exc)
        return classify_exception(exc, operation=operation, **fields)


def log_classified_exception(
    logger: logging.Logger,
    exc: BaseException,
    *,
    operation: str,
    level: int = logging.DEBUG,
    **fields: Any,
) -> ErrorRecord:
    record = classify_exception(exc, operation=operation, **fields)
    if logger.isEnabledFor(level):
        logger.log(level, "trade_error=%s", json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")))
    return record
