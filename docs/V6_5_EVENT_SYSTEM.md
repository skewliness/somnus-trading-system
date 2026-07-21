# V6.5 Trade Event System

The event stream is an append-only, account-scoped audit sidecar. It records
typed `SignalEvent`, `OrderEvent`, `FillEvent`, `PositionEvent`, and `ExitEvent`
payloads without participating in signal selection, risk decisions, order
submission, or ledger reconciliation.

Every envelope contains:

- `strategy_version`, `config_version`, `model_version`, and `config_hash`;
- `symbol` in the typed payload;
- `occurred_at` plus the migration-plan-compatible `timestamp` alias;
- the relevant `signal_reason`, `order_reason`, or `exit_reason` field;
- account and runtime-mode identity so demo and live streams remain separate.

Fill and exit events retain Binance order identity from either top-level result
fields or the nested exchange response. Explicit zero quantities and prices are
preserved. Sensitive keys and URL query credentials are redacted before the
event is serialized.

The stream is disabled by default to preserve the V6.0 runtime baseline and can
be enabled with `TS_V6_TRADE_EVENT_LOGGING_ENABLED=1`. Write failures are
classified and logged, but the sidecar cannot fail a trading cycle.
