# V6.5 Exception Taxonomy

PR2 introduces a behavior-neutral exception taxonomy for V6 diagnostics.

## Categories

- `market_data`: klines, ticker, mark price, freshness, and public data.
- `exchange`: transport, authentication, exchange response, and rate limits.
- `order`: submit, cancel, close, reduce-only, and ambiguous order status.
- `risk`: explicit risk invariant violations.
- `state`: local state, ledger, position, and serialization invariants.
- `recovery`: reconciliation and recovery operations.
- `unknown`: errors that do not yet have enough evidence for a narrower class.

## Compatibility contract

- `V6ExecutionError` remains importable from `trading_system.v6_execution`.
- Existing exception catches, retries, cooldowns, and reconciliation branches are unchanged.
- Existing legacy `error` strings remain present. The new `error_record` is additive.
- Sensitive credentials and signed request values are redacted in structured records.
- Error classification has no authority to open, close, resize, retry, or reconcile an order.
- Order-mutation HTTP 5xx responses are marked as ambiguous and require the
  existing exchange reconciliation path; market-data HTTP 5xx responses remain
  transient read failures and do not become order events.

Every broad defensive catch in `trading_system/v6_execution.py` now emits a
classified record or delegates to a helper that does. The original catch body,
retry behavior, cooldown, return value, and reconciliation behavior are
preserved. `risk_action` is audit metadata describing the existing operational
boundary; it does not execute a new trading action.
