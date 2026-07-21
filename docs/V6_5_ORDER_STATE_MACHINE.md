# V6.5 Shadow Order State Machine

PR3 adds a read-only projection of existing V6 execution results into a
standard order lifecycle:

`created -> submitted -> acked -> partial_filled/filled`

Additional recovery states cover:

- `unknown_pending_reconcile`
- `reconciled_filled`
- `reconciled_not_filled`
- `cancel_pending` / `cancelled`
- `rejected` / `failed` / `expired`

## Authority boundary

The state machine is an audit projection only. It does not submit, retry,
cancel, reconcile, mutate the ledger, or change Telegram output. Existing
Binance responses and V6 reconciliation remain authoritative. The projected
state is added only to the optional PR1 sidecar event stream, which is disabled
by default.

Invalid transitions raise `StateError` in direct callers. Repeated transitions
to the same state are recorded as idempotent no-ops.
