# V6.5 Trade Event Chain

Event schema version 2 adds two top-level correlation fields:

- `trade_id` identifies one directional position lifecycle for one account;
- `parent_event_id` links the immediate cause within the append-only stream.

The normal open chain is `signal -> order -> fill -> position`. The close chain
is `signal -> order -> fill -> exit`. Partial closes and scale-ins retain the
same `trade_id`, while their order/fill events remain individually addressable.

Trade IDs are stored in the intent risk plan, root position, and every local
lot. Existing positions receive a deterministic migration ID derived from
account, symbol, side, opening identity, and source. New entries receive their
ID before submission. Close and scale-in intents inherit the current position
ID. These rules keep the ID stable across process restarts and reconciliation.

The event writer also derives a deterministic fallback ID for legacy summaries
without mutating them. Schema v1 JSONL rows remain readable; new rows use schema
v2. This feature is observability-only and does not affect signal selection,
entry/exit priority, sizing, or exchange requests.
