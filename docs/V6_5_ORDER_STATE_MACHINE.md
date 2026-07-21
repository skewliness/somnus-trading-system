# V6.5 Durable Order State Machine

PR3 records every existing V6 order submission in an account-scoped sidecar
before the Binance request is sent, then applies the exchange response to the
same record. The standard lifecycle is:

`created -> submitted -> acked -> partial_filled/filled`

Additional recovery states cover:

- `unknown_pending_reconcile`
- `reconciled_filled`
- `reconciled_not_filled`
- `cancel_pending` / `cancelled`
- `rejected` / `failed` / `expired`

## Restart and authority boundary

The state machine has no strategy authority: it cannot create an intent,
change order fields, retry an order, or mutate a position ledger. Existing
order and cancel calls use `submit_tracked_order()` only to persist lifecycle
state around the unchanged Binance request.

Non-terminal normal orders survive process restart in `order_states.json` and
remain visible through `pending_order_reconciliation()`. An exchange-confirmed
result can then resolve them to `filled`, `reconciled_filled`, or
`reconciled_not_filled`. Exchange state is authoritative. A timeout or 5xx
response from an order mutation remains ambiguous until reconciliation; it is
never treated as an automatic failure or resubmitted by this sidecar.

Acknowledged protective-stop orders are intentionally excluded from generic
pending-order recovery because the existing protective-stop synchronizer owns
their long-lived exchange lifecycle.

Invalid transitions raise `StateError` in direct callers. Repeated transitions
to the same state are recorded as idempotent no-ops.
