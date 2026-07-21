# V6.5 Trading Context

Migration PR6 introduced the frozen `TradingContext` container. Remediation
PR3 separates its directional bias from its market regime. The context carries
the current symbol, price, position, directional bias, market regime,
indicators, features, config, and original signal.

`side_bias` and `market_regime` are deliberately independent. `side_bias`
preserves the signal's `long` / `short` / `neutral` preference. Market regime
is read only from explicit regime metadata and otherwise remains `unknown`; it
never falls back to directional bias. This prevents a long candidate from
being mislabeled as a trending market without adding or changing a gate.

The context preserves object identity for the signal, position, indicators,
features, and config. It does not copy, normalize, recalculate, or deep-freeze
the legacy dictionaries because those operations could alter the existing V6
state-mutation and decision behavior.

Production entry, technical-entry, open, scale-in, and exit boundaries now
accept or construct `TradingContext`. Existing public methods continue to
accept `V6Signal` and normalize it at the boundary, preserving compatibility
for scripts and tests. No thresholds, indicator values, rule ordering, sizing,
or order fields change in this phase.

`V6ExecutionEngine.intents_for_context()` is the canonical decision entrypoint.
The live and dry-run loops normalize their signal into a context and delegate to
that method. `ReplayDecisionAdapter` also delegates to the same method, so a
replay adapter cannot maintain a second copy of the entry/exit decision rules.
