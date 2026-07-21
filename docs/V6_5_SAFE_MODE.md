# V6.5 Runtime Safe Mode

Safe Mode is a health guard at the new-exposure boundary. It does not change
signal generation, entry thresholds, indicator calculations, exit ordering,
position sizing, or exchange order parameters.

Safe Mode activates when runtime health reports one of these hazards:

- an exchange position is missing its required protective stop;
- ledger reconciliation failed, is behind the latest position change, or is
  still pending after an uncertain exchange result;
- the current symbol uses stale market data;
- the exchange connection is unavailable, all observed requests failed, or
  request latency is critical.

The engine evaluates exits before Safe Mode. An active decision blocks new
entries and scale-ins but continues to allow stop-loss, take-profit, reduction,
full close, reconciliation, and protective-stop repair paths. Decisions are
reported per symbol in the run summary under `safe_mode`; they are not used as
strategy features and are not fed into the model.

Dry-run/replay compatibility remains intact. The missing-protective-stop check
only applies when an exchange submission is actually requested.
