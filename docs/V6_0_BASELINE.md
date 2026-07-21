# V6.0 Frozen Baseline

This document freezes the behavioral baseline used by the V6.5 migration.

- Git commit: `0aaaaa2`
- Git tag: `v6.0-baseline-20260721`
- Full regression on the Mac production Python 3.11 environment: `677 passed`
- Baseline test duration: `113.43s`
- Runtime source: `/Users/ma7/binance-trading-skill-system`

## V6.5 invariants

The V6.5 migration may improve architecture, observability, recovery, and
testing only. It must not change:

- signal or entry predicates;
- indicator periods, thresholds, or symbol profiles;
- notional, leverage, stop-loss, take-profit, or scale-in parameters;
- exit priority, close fractions, or target selection;
- demo/live/momo2 account routing;
- the existing order, WAL, ledger, or reconciliation semantics.

Every migration phase is delivered as an independent pull request. Each pull
request must pass targeted tests and the full regression suite in a clean
archive on the Mac before it is merged.

## PR1 event stream

PR1 adds an optional, account-scoped event stream beside the existing V6 WAL.
It is disabled by default and has no authority over decisions, orders, state,
or reconciliation. Event write failures are logged and cannot fail a trading
cycle.
