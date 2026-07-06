# Fable5 Task Request: Gann Structural Feature Test + Execution Refactor

## Objective

Review and extend the current Somnus V6 Gann integration and execution layer. This request is not for a prose-only review. Please return implementation code through a pull request.

The main questions:

1. Does the current Gann structural feature add real value as a macro factor?
2. How should Gann be integrated without becoming an unsafe hard execution gate?
3. Why do close/flatten operations sometimes leave tiny residual balances or split into many small reduce-only closes?
4. What refactor should separate signal, macro factor, sizing, execution, ledger reconciliation, and Telegram reporting more cleanly?
5. Is V6 currently over-gated by serial thresholds across model probability, symbol profile, 15M/4H BB/MA, Fib, Gann, and macro direction?

## Current Code Entry Points

### Gann Feature

- `trading_system/gann_feature.py`
  - `normalize_daily_klines(rows)`
  - `compute_gann_feature(symbol, daily_klines, asof_index=None, lookback_days=365, min_cycle_days=14, min_drawdown_pct=10.0, stability_shift_days=3)`
  - `classify_zone(close, lines)`
  - `anchor_stability_pct(...)`
  - `gann_position_multiplier(feature, side)`

The current Gann implementation is intentionally conservative:

- daily timeframe only
- log-price axis
- historical-only anchors
- lines: `1x2`, `1x1`, `2x1`
- zones: `above_1x1`, `1x2_to_1x1`, `below_1x2`
- default execution effect is disabled unless env/config enables it

### V6 Execution Integration

- `trading_system/v6_execution.py`
  - `V6ExecutionConfig.gann_position_sizing_enabled`
  - env: `TS_V6_GANN_POSITION_SIZING_ENABLED`
  - `V6ExecutionEngine.open_intent(...)`
  - risk-plan field: `position_sizing_overlays`

Gann currently belongs in the macro/sizing layer, not as a direct hard gate.

### Transition Gate Policy

- `trading_system/transition_gate_policy.py`
  - `TransitionGatePolicy`
  - `GatePolicyDecision`
  - `append_gate_decision(...)`
  - `transition_notional_multiplier(...)`

This is the current mechanism for converting selected over-gated blocks into reduced-size probes. It is disabled by default.

## Validation Seed Data Included in This PR

Small, sanitized summary files are included under:

`reports/fable5_validation_data/`

These files are provided so Fable5 can quickly inspect the prior evidence before
rerunning the tests. They are not raw training data, raw smart-signal payloads,
private API responses, or full signal-history exports.

| File | Review dimension |
|---|---|
| `v6_gann_ab_table.csv` | V6 baseline vs V6 + Gann A/B |
| `gann_zone_stratification_multi_summary.csv` | Gann zone outcomes across ETH/SOL/BNB/HYPE/ZEC |
| `v6_model_threshold_block_action_summary.csv` | model probability / threshold blocked candidates by action |
| `v6_model_threshold_block_symbol_top.csv` | model probability / threshold blocked candidates by symbol-action |
| `v6_current_strategy_gate_summary.csv` | current gate coverage and top rejection reasons |
| `hype_profile_blocked_level_summary_top.csv` | symbol profile blocked-level outcomes for HYPE |
| `fib_confluence_top_summary.csv` | prior Fib + MA/BB confluence summaries |

Use these files to review:

- model-probability thresholds
- symbol profile allowed/blocked levels
- 15M and 4H BB/MA entry structures
- Fib confluence
- Gann macro zone
- macro direction mismatch or neutrality

If you disagree with the current transition-gate design, return a code
refactor. Do not respond only with prose. The expected direction is to replace
unnecessary serial hard gates with layered risk handling:

- true hard-risk gate
- soft score penalty
- reduced-size probe
- position-size modifier
- watchlist-only state

## Existing V6 + Gann A/B Table

Source:

`E:\somnus\output\gann_v1_extended_20260704\v6_gann_ab\v6_gann_ab_table.csv`

Summary:

| Version | Return | Max Drawdown | PF | Win Rate | Avg Return | Worst Trade | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| V6 baseline | 0.550312 | -0.600000 | 1.906694 | 50.0% | 1.393139 | -3.000000 | 2 |
| V6 + Gann low-weight scoring | 0.550312 | -0.600000 | 1.906694 | 50.0% | 1.393139 | -3.000000 | 2 |
| V6 + Gann position sizing | 0.837890 | -0.600000 | 2.376571 | 50.0% | 1.393139 | -3.000000 | 2 |
| V6 + Gann hard filter | 1.157256 | 0.000000 | 999.0 | 100.0% | 5.786279 | 5.786279 | 1 |

Initial conclusion:

- low-weight score had no measurable effect in this sample
- position sizing improved PF/return without reducing trade count
- hard filtering looks good only because it left one trade, so it is not reliable
- Gann should remain a macro auxiliary factor until stronger portfolio-level evidence exists

Please independently re-test this conclusion.

## Required Gann Test Plan

Please implement or extend scripts/tests so the result is reproducible.

### 1. Unit-Level Feature Validation

Add or extend tests for:

- no future leak: anchor high/low must be found using only data available at `asof_index`
- log-line math: verify `1x2`, `1x1`, `2x1` against known fixture data
- zone classification: `above_1x1`, `1x2_to_1x1`, `below_1x2`
- anchor stability: shift `asof_index` by +/-3 days and quantify line drift
- stale/invalid behavior: insufficient data, no valid drawdown, zero prices

Relevant test file:

- `tests/test_gann_feature.py`

### 2. Cross-Symbol Outcome Stratification

Run the same Gann zone stratification on:

- BTCUSDT
- ETHUSDT
- SOLUSDT
- BNBUSDT
- HYPEUSDT
- ZECUSDT

For each symbol and zone, compute future returns for:

- 1D
- 3D
- 7D
- 14D
- 30D

Required columns:

- symbol
- date
- close
- zone
- bias
- confidence
- anchor_high_time
- anchor_low_time
- cycle_days
- `line_1x2`
- `line_1x1`
- `line_2x1`
- anchor_stability
- future_return_1d
- future_return_3d
- future_return_7d
- future_return_14d
- future_return_30d
- MFE
- MAE

The output should make it clear whether Gann zones are ranked:

`above_1x1` > `1x2_to_1x1` > `below_1x2`

If this ranking is not stable across BTC/ETH/SOL/BNB/HYPE/ZEC, Gann should not become a global hard gate.

### 3. Portfolio-Level V6 A/B

The previous table was too small. Please run portfolio-level simulation instead of only fixed-horizon outcome stats.

Required variants:

1. V6 baseline
2. V6 + Gann low-weight scoring
3. V6 + Gann position sizing
4. V6 + Gann hard filter

Required metrics:

- total return
- max drawdown
- PF
- win rate
- average trade return
- worst trade
- trade count
- skipped count
- same-symbol skipped count
- average exposure
- max simultaneous positions

Important simulation rules:

- use next-bar execution for daily-close confirmed signals
- include fees
- include slippage
- include funding estimate if available; if not available, explicitly mark funding as omitted
- define same-day multi-symbol allocation rule
- define overlapping signal rule: add, ignore, or resize
- never use future data in Gann anchor selection

### 4. Out-of-Sample Test

Please split:

- train/tune: 2021-2024
- validation: 2025-2026

Do not tune Gann thresholds on 2025-2026 and then report 2025-2026 as proof.

### 5. Deliverable Files

Please add:

- a reproducible script, preferably `scripts/backtest_v6_gann_ab.py` or `scripts/evaluate_gann_structural_features.py`
- CSV outputs under `reports/` or documented output paths
- tests for the script helper functions
- a short markdown report under `docs/` with the result and recommendation

## Required Execution Refactor Review

There is a recurring production issue:

- close/flatten sometimes leaves tiny residual balances
- partial take-profit can generate many small reduce-only closes
- manual exchange-side closes and local lot ledger can drift
- repeated close intents can spam Telegram or execution requests

Relevant current entry points:

- `trading_system/v6_execution.py`
  - `strategy_dust_threshold`
  - `is_strategy_dust_position`
  - `mark_strategy_dust_position`
  - `apply_lot_close_allocation`
  - `apply_partial_take_profit_to_position`
  - `reconcile_lots_to_exchange_quantity`
  - `apply_exchange_snapshot_to_position`
  - `quantity_from_position_amount`
  - `protective_stop_matches`
  - `recent_exchange_close_order_for_missing_position`

Please review and return code for the following.

### 1. Close Quantity Resolver

Create a single shared resolver for all close intents:

- input: symbol, side, requested close fraction, local position, exchange position snapshot, symbol quantity rules
- output: exact reduce-only quantity, close mode, and whether this should be full flatten

Expected behavior:

- if requested close fraction >= 0.90, close full exchange position quantity
- if remaining notional after partial close would be below strategy dust threshold, close full quantity instead
- if quantity after `stepSize` rounding becomes zero, do not submit a useless order
- if exchange position is missing, clear or mark local ledger instead of repeatedly submitting close orders
- use exchange `positionAmt` as the authority for final close quantity

Potential interface:

```python
@dataclass(frozen=True)
class CloseQuantityDecision:
    quantity: str
    quantity_float: float
    mode: str  # partial, full_flatten, skip_zero, skip_missing_exchange_position
    reason: str
    expected_remaining_qty: float
    expected_remaining_notional_usdt: float

def resolve_close_quantity(
    *,
    symbol: str,
    side: str,
    requested_fraction: float,
    local_position: dict[str, Any],
    exchange_position: dict[str, Any] | None,
    mark_price: float,
    step_size: float,
    min_notional_usdt: float,
    dust_notional_usdt: float,
) -> CloseQuantityDecision:
    ...
```

### 2. Close Intent Coalescing

Before submitting reduce-only orders, coalesce multiple close triggers in the same run:

- same symbol
- same side
- same action (`close_long` or `close_short`)
- same run/window

Rules:

- keep the strongest close intent
- combine staged TP fractions when safe
- full close beats partial close
- stop loss / emergency close beats take profit
- never send three tiny reduce-only orders in one loop for the same position

Potential interface:

```python
def coalesce_close_intents(intents: list[V6OrderIntent]) -> list[V6OrderIntent]:
    ...
```

### 3. Ledger Reconciliation After Close

After a close result:

- read exchange position snapshot
- if exchange position is flat or dust, mark all lots closed
- if exchange position remains, scale local lots to exchange quantity
- store one close event with exchange order id, executed quantity, avg price, and reconcile snapshot

Do not rely only on `response.executedQty` when `position_reconcile.after` is available.

### 4. Tests Required

Please add tests for:

- partial close that would leave dust becomes full flatten
- 100% close uses exchange `positionAmt`, not stale local quantity
- repeated TP triggers coalesce into one order
- close order with missing exchange position does not re-submit every loop
- lot allocation remains proportional when exchange quantity remains non-dust
- manual close on exchange clears local lots without new reduce-only order

Suggested tests:

- `tests/test_close_quantity_resolver.py`
- `tests/test_execution_runner.py`
- `tests/test_v6_execution.py`

## Telegram Refactor Requirement

Execution bots should only report execution events and errors. Interactive commands belong to Xuejian/Hermes.

For close/flatten events, Telegram should display one logical close event, not every small internal stage.

Required message fields:

- symbol
- action
- side
- close mode: partial / full_flatten / dust_flatten / skipped_missing_exchange_position
- executed quantity
- avg price if available
- order id if available
- reason
- execution status

Avoid repeated broadcasts for the same unresolved close failure. Add dedup key:

`symbol + side + action + reason + target_name + error_code`

## PR Requirements

Please return a PR with:

1. Gann structural feature tests and backtest script
2. Gann recommendation report
3. close quantity resolver
4. close intent coalescing
5. ledger reconciliation improvements
6. Telegram close-event dedup/format update
7. tests proving the above

Do not change live-trading defaults in the PR unless explicitly required.

Do not include API keys, cookies, WSS tokens, private data, raw smart-signal dumps, model-training data, or large JSONL logs.

## Validation Commands

Run at minimum:

```bash
python -m compileall trading_system scripts
python -m pytest tests/test_gann_feature.py tests/test_transition_gate_policy.py -q
python -m pytest tests/test_execution_runner.py tests/test_v6_execution.py -q
python -m pytest tests/test_telegram_monitor_format.py -q
```

If full pytest fails due existing old PaperExecution test contracts, list them separately and explain why they are unrelated.
