# Fable5 Validation Seed Data

This directory contains small, sanitized summary artifacts for Fable5 review.
These are not raw training samples, raw smart-signal payloads, private cookies,
or full signal-history exports.

Use these files as starting evidence only. Fable5 should still rerun the
backtests and return code through a PR.

## Files

| File | Purpose | Original local source |
|---|---|---|
| `v6_gann_ab_table.csv` | Previous V6 baseline vs V6 + Gann A/B summary | `E:\somnus\output\gann_v1_extended_20260704\v6_gann_ab\v6_gann_ab_table.csv` |
| `gann_zone_stratification_multi_summary.csv` | Cross-symbol Gann zone outcome summary | `E:\somnus\output\gann_v1_extended_20260704\gann_zone_stratification_multi_codex_summary.json` |
| `v6_model_threshold_block_action_summary.csv` | Model-threshold blocked-candidate outcome summary by action | `E:\somnus\output\v6_model_threshold_block_backtest_variant_summary.json` |
| `v6_model_threshold_block_symbol_top.csv` | Model-threshold blocked-candidate outcome summary by symbol/action | same as above |
| `v6_current_strategy_gate_summary.csv` | Current strategy replay coverage and top gate rejection reasons | `E:\somnus\output\v6_current_strategy_replay_summary.json` |
| `hype_profile_blocked_level_summary_top.csv` | HYPE profile-blocked level backtest summary | `E:\somnus\output\hype_profile_blocked_pairs_v6_exit_backtest_summary.csv` |
| `fib_confluence_top_summary.csv` | Fib + MA/BB top confluence results from prior BTC/ETH/BNB/ZEC studies | `E:\somnus\output\fib_*_confluence_*.md` |

## Review Guidance

Fable5 should use this data to validate whether the current gate stack is too
serial and too binary:

- model probability threshold
- symbol profile allowed/blocked levels
- 15M/4H BB and MA structure
- Fib confluence
- Gann macro zone
- macro direction mismatch or neutrality

The requested design is not "remove gates". The requested design is to separate:

- true hard-risk gates
- soft scoring penalties
- reduced-size probe entries
- position-size modifiers
- watchlist-only states

If Fable5 disagrees with these summaries or the current implementation, the
expected response is code plus tests, not a prose-only objection.
