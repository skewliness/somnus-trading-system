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

Broad defensive catches are intentionally not rewritten wholesale in this PR.
Only the primary signal/order failure boundaries receive structured records, and
previously silent fallback catches receive debug-level diagnostics.
