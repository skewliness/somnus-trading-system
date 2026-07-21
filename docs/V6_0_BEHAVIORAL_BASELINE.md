# V6.0 Behavioral Baseline Manifest

This manifest completes Phase 0 of the V6.5 migration plan. The immutable
source of truth is the annotated Git tag `v6.0-baseline-20260721`, which
dereferences to commit `0aaaaa2dc014becf99ec711531b97d94b49f6c30`.

## Restore

The tag contains the complete code, tests, defaults, symbol profiles, entry
rules, exit rules, risk calculations, indicators, and model-loading contract.
Restoration must use a new worktree or branch; production state, credentials,
ledgers, and exchange positions are not part of the source baseline.

```text
git switch --detach v6.0-baseline-20260721
python -m pytest tests -q
```

## Recorded policy surfaces

The machine-readable companion is `docs/v6_0_behavioral_baseline.json`.

- **Entry**: `intent_for_signal`, `intent_for_technical_entry_signal`,
  `entry_structure_block`, `macro_direction_entry_block`, and `open_intent`.
- **Exit**: `exit_intent_if_needed` and `stop_take_profit`.
- **Risk**: position capacity, scale-in capacity, stop distance, leverage,
  close preflight, protective-stop handling, and account isolation defaults.
- **Indicators**: every MA/BB/Fib/Gann period and symbol profile is frozen by
  the tagged `V6ExecutionConfig` source and its 248 default expressions.
- **Model**: `DEFAULT_MODEL_PATH`, feature-vector adaptation, model loading,
  and scoring behavior are frozen by the tagged source.
- **State**: `apply_intent_to_state` and `reconcile_state_with_exchange`.

The manifest records hashes rather than copying hundreds of mutable values
into prose. The tagged source remains the recoverable full snapshot; the
hashes make accidental policy drift mechanically detectable.

## Runtime configuration boundary

Secrets and host-specific paths are deliberately excluded. Environment
variables may select an account or override a value at runtime, but V6.5 must
not alter any pre-existing default expression. New V6.5 fields may only carry
architecture metadata, event paths, or monitoring settings.

