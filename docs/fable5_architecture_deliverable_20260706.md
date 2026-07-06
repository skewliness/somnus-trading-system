# Somnus Fable5 Architecture Deliverable

Status: implementation bridge, not final strategy approval.

## 1. Architecture Review Deliverable

### Runtime boundaries

| Area | V5 | V6 | Xuejian/Hermes |
|---|---|---|---|
| Role | Small/mid-cap scan and execution | Main-symbol strategy and execution | Macro/news/report/chart command layer |
| Trading authority | Yes | Yes | No order submission |
| Telegram role | Execution and error alerts only | Execution and error alerts only | `p`, `fc`, `report`, news and explanations |
| Ledger | V5 state only | `data/trading_system/v6_execution` | Read-only |
| Key material | Must not mix with V6 | Must not mix with V5 | No trading key |

### Code entry points

| Layer | Primary files |
|---|---|
| V6 execution | `trading_system/v6_execution.py` |
| Gate transition policy | `trading_system/transition_gate_policy.py` |
| Gate event recording | `trading_system/gate_registry.py` |
| Gann macro feature framework | `trading_system/gann_feature.py` |
| Telegram split | `scripts/telegram_monitor_v5_v6.py`, `scripts/hermes_somnus_command.py` |
| Optional snapshot fallbacks | `trading_system/optional_integrations.py` |

## 2. Over-Gating Transition Plan

Fable5's main finding was that V6 has too many serial hard gates. The safe implementation path is transition gating:

1. Keep true hard gates hard.
2. Convert selected over-gates into recordable soft score, sizing, or watch/probe gates.
3. Keep the feature disabled by default.
4. Enable with `TS_V6_TRANSITION_GATE_POLICY_ENABLED=1` only for shadow or small-size rollout.

### Gate classes

| Class | Examples | Behavior |
|---|---|---|
| Hard gate | stale market data, no indicator level, smart-signal squeeze/crowding hard risk, daily reversal invalidation | Still blocks |
| Soft score | symbol BB profile not allowed, Fibonacci confirmation missing, confluence hold | Can pass with penalty |
| Sizing | macro mismatch/neutral, 15M MA20 opposite trend, missing 4H/1D right-side confirmation, 4H lower chase | Can pass with reduced notional |
| Watch/probe | four-hour waiting for 15M selector, V5 wait states | Can pass as small probe when enabled |

### Implemented framework

`trading_system/transition_gate_policy.py` now defines:

- `TransitionGatePolicy`
- `GatePolicyDecision`
- `append_gate_decision`
- `transition_notional_multiplier`

`trading_system/v6_execution.py` now:

- Initializes `TransitionGatePolicy`.
- Calls `maybe_downgrade_entry_block()` before returning entry blocks.
- Stores downgraded gates under `signal.levels["transition_gate_policy"]`.
- Applies reduced notional in `open_intent()` when transition gates are active.

Default behavior remains unchanged because the policy is disabled unless explicitly enabled.

## 3. V6 + Gann A/B Result Found Locally

Local verified source:

`E:\somnus\output\gann_v1_extended_20260704\v6_gann_ab\v6_gann_ab_table.csv`

| Version | Return | Max DD | PF | Win rate | Avg return | Worst trade | Trades | Candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V6 baseline | +0.550312% | -0.600000% | 1.906694 | 50.00% | +1.393139% | -3.000000% | 2 | 14 |
| V6 + Gann low-weight scoring | +0.550312% | -0.600000% | 1.906694 | 50.00% | +1.393139% | -3.000000% | 2 | 14 |
| V6 + Gann position sizing | +0.837890% | -0.600000% | 2.376571 | 50.00% | +1.393139% | -3.000000% | 2 | 14 |
| V6 + Gann hard filter | +1.157256% | +0.000000% | 999.000000 | 100.00% | +5.786279% | +5.786279% | 1 | 2 |

Interpretation:

- Low-weight scoring did not change the result.
- Position sizing improved return and PF while keeping the same trade count.
- Hard filter looks best on the table but has only one trade, so it is not reliable.
- Current evidence supports Gann as macro sizing/auxiliary signal, not as a direct execution filter.

## 4. Gann Framework Code

`trading_system/gann_feature.py` provides the concrete framework:

- Daily kline normalization.
- Anchor high/low detection using only historical data.
- Log-scale Gann lines: `1x2`, `1x1`, `2x1`.
- Zone classification:
  - `above_1x1`
  - `1x2_to_1x1`
  - `below_1x2`
- Anchor stability check using +/-3 day shifted anchors.
- Macro-only position multiplier via `gann_position_multiplier()`.

V6 integration is deliberately conservative:

- `TS_V6_GANN_POSITION_SIZING_ENABLED=0` by default.
- If enabled, `signal.levels["gann_feature"]` can adjust notional through `position_sizing_overlays`.
- Gann cannot bypass BB/MA/Fib/risk gates.

## 5. Telegram Remediation

Implemented target:

- V5/V6 bot: execution and error alerts only.
- Manual commands (`p`, `fc`, `report`, `news`) are not executed by V5/V6 by default.
- Mentioned manual commands receive handoff help text instead of running in V5/V6.

Relevant files:

- `scripts/telegram_monitor_v5_v6.py`
- `tests/test_telegram_monitor_format.py`

Key runtime switch:

- Default: `TS_TELEGRAM_MONITOR_EXECUTION_ONLY_COMMANDS=true`
- Restore old manual command behavior only with `--enable-manual-commands`

## 6. Validation Scope

Added tests:

- `tests/test_transition_gate_policy.py`
- `tests/test_gann_feature.py`
- `tests/test_gate_registry.py`
- `tests/test_optional_integrations.py`
- Telegram command split tests in `tests/test_telegram_monitor_format.py`

Minimum validation command:

```bash
python -m compileall trading_system scripts
python -m pytest tests/test_transition_gate_policy.py tests/test_gann_feature.py tests/test_gate_registry.py tests/test_optional_integrations.py tests/test_telegram_monitor_format.py -q
```

Full-suite caveat from this branch: legacy PaperExecution tests still assert old single-exit behavior while the current engine uses staged TP and structured position management.
