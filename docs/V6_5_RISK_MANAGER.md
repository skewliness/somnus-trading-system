# V6.5 Risk Manager Extraction

PR5 moves the existing V6 risk boundary into `trading_system/risk/manager.py`.
The engine keeps compatibility wrappers for existing callers and tests.

The extracted boundary owns the existing implementations for:

1. active-position capacity against `max_positions`;
2. scale-in total-notional capacity, including the existing `1e-9` tolerance;
3. close-quantity preflight before protective-stop cancellation;
4. account isolation metadata and local stop-risk budget calculation;
5. known reduce-only, percent-price, and unknown-status exception guards.

This phase does not introduce a new portfolio-loss limit because V6 did not
previously enforce one at this boundary. Adding such a limit would be a strategy
and trading-frequency change. Existing values, comparisons, exception
propagation, risk-plan fields, leverage, stop distance, and order behavior are
unchanged.
