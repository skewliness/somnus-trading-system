# V6.5 Acceptance Matrix

This matrix maps every explicit migration-plan failure scenario to executable
evidence. It is an acceptance checklist, not a strategy specification.

| Requirement | Evidence |
| --- | --- |
| Order submission failure | `test_execution_runner_records_trade_context_when_request_fails` |
| API timeout / unknown status | `test_tracked_submit_timeout_is_pending_after_client_restart`, `test_timeout_survives_restart_until_exchange_reconciliation` |
| Partial fill | `test_tracked_submit_persists_partial_exchange_fill`, `test_partial_fill_survives_restart_and_can_finish` |
| Process restart | `test_partial_fill_survives_restart_and_can_finish`, `test_timeout_survives_restart_until_exchange_reconciliation` |
| Cancel failure | `test_cancel_failure_is_recorded_but_original_error_still_propagates`, `test_cancel_rejection_is_persisted_without_losing_exchange_identity` |
| Local/exchange state mismatch | `test_exchange_position_missing_locally_is_recovered_with_a_lot`, `test_ledger_persists_final_mismatch_status_instead_of_premature_ok` |
| Protective stop missing | `test_missing_exchange_protective_stop_is_recreated_and_recorded`, `test_missing_protective_stop_is_critical_only_for_submitted_exchange_mode` |
| Live/dry-run/backtest consistency | `test_live_dry_run_and_replay_use_identical_context_decision_path` |
| Existing V6 defaults unchanged | `test_every_v60_config_default_is_unchanged_in_v65` |

The final acceptance gate is the full `pytest` suite plus `compileall`, Git
diff validation, a sensitive-value scan, and a production-path scan. No phase
is complete when only its isolated unit tests pass.
