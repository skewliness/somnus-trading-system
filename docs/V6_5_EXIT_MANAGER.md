# V6.5 Exit Manager Extraction

PR4 moves the complete `exit_intent_if_needed()` orchestration body into
`trading_system/exit/manager.py`. The public engine method remains as a thin
compatibility wrapper.

The move preserves, in order:

1. manual-profit stop-only handling;
2. max-hold and hard-stop exits;
3. V5 soft stop and daily reversal exits;
4. logic weakening and smart-signal guards;
5. momentum/profit-giveback guards;
6. ROI and BB staged take profit;
7. fixed take profit and opposite-probability exits;
8. quarantine, retry cooldown, close fractions, and risk-plan construction.

The files under `exit/rules/` convert the already selected legacy exit intent
into immutable `ExitSignal` values with `triggered`, `reason`, and `priority`.
`exit/resolver.py` deterministically resolves those adapters and returns the
same intent object. In this migration phase the resolver is deliberately
downstream of the unchanged legacy decision chain, so it cannot alter whether
an exit is allowed, the existing exit priority, or any order field. All
candidate calculations, thresholds, state mutation, and intent fields remain
the existing V6 implementations.
