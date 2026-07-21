# V6.5 Advisory Risk Engine

The V6.5 risk engine adds four outputs requested by the remediation plan:

- a bounded risk score from 0 to 100;
- a position-size recommendation and multiplier;
- a volatility adjustment;
- a maximum reference risk budget in USDT.

The engine is deliberately advisory-only. It writes its result to
`risk_plan.risk_engine` but does not change entry eligibility, position size,
leverage, stop loss, take profit, close fraction, or any exchange request.

The reference risk budget defaults to the existing base notional multiplied by
the existing stop-loss percentage. `TS_V6_RISK_MAX_BUDGET_USDT` can provide an
explicit advisory budget. `TS_V6_RISK_REFERENCE_VOLATILITY_PCT` can provide the
reference volatility. Both values are telemetry inputs only. Volatility is
read from existing risk metadata when available; otherwise the existing stop
distance is recorded as a transparent proxy. No new market indicator is
introduced.

The score combines budget utilization, observed volatility relative to the
reference, and leverage. It produces `standard`, `reduced`, `defensive`, or
`minimum` advice. These labels are telemetry for future V7 analysis and have no
effect on V6.5 trading behavior.
