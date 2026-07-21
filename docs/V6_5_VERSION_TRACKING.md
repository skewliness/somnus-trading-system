# V6.5 Runtime Version Tracking

This phase adds traceability metadata only. It does not participate in signal,
entry, exit, sizing, or exchange decisions.

Every annotated order intent records a `runtime_versions` object containing:

- `strategy_version`
- `config_version`
- `model_version`
- `config_hash`

The same object is retained in execution results through their nested intent,
open lots, compact lot risk plans, positions, run summaries, and sidecar event
envelopes. The configuration fingerprint excludes credentials, confirmations,
tokens, cookies, proxies, URLs, machine-local paths, and account identifiers.

Model artifacts are identified by a short SHA-256 digest unless an explicit
`TS_V6_MODEL_VERSION` label is supplied. No model input or output is changed.
