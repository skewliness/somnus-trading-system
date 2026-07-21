# V6.5 Exit Regression Gate

This gate implements remediation item P1-2 without changing exit behavior.

The fixed input deliberately activates competing exit conditions. The frozen
V6 legacy oracle and the current V6.5 implementation must select the same:

- exit reason;
- exit-rule priority;
- exit price.

The collision cases cover max hold, hard stop, daily reversal, 4H logic
weakening, smart-signal protection, ROI take profit, staged BB take profit,
fixed take profit, and opposite-probability fallback. Protective-stop failure
emergency closure remains an execution-safety path outside `ExitManager` and is
covered by `tests/test_execution_runner.py` and
`tests/test_execution_runner_safety.py`.

The legacy oracle must be generated from the annotated
`v6.0-baseline-20260721` tag in a separate checkout. Tests never rewrite the
oracle and never call an exchange.
