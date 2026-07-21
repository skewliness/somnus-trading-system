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

The files under `exit/rules/` classify emitted exit intents for diagnostics.
They do not decide whether an exit is allowed and cannot change exit priority.
All candidate calculations, thresholds, state mutation, and intent fields are
the existing V6 implementations.
