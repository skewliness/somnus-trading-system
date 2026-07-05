# Somnus Trading System Push Scope

This snapshot contains a filtered V5/V6 code export from the production
Mac workspace:

- `trading_system/`
- `scripts/`
- `tests/`
- `AGENTS.md`
- `pyproject.toml`
- `requirements.txt`

Excluded from this push by request:

- training content, training datasets, and training artifacts
- smart-money, smart-signal, whale, distillation, dataset, and ablation data/content files
- runtime data, ledgers, kline caches, logs, JSONL event streams, backups, venvs
- local environment files, secrets, cookies, tokens, API keys, and private keys

Implementation note:

- Some core V5/V6 execution modules retain optional code paths that can read
  these sources when present in production runtime data directories.
- The source data, snapshots, model artifacts, and generated content for those
  sources are not included in this git snapshot.

V5/V6 runtime note:

- This repository snapshot is code-only.
- Do not treat this directory as the live runtime state.
- The production runtime state remains outside git and must be reconciled from
  the exchange, ledgers, and live service logs before any live operation.
