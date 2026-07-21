# V6/V6.5 Behavioral Compatibility

This test is the executable acceptance gate for V6.5 migration item P1-1.

## Boundary

The compatibility matrix checks the same fixed inputs against two independent
implementations:

- Legacy oracle: Git tag `v6.0-baseline-20260721`, commit
  `0aaaaa2dc014becf99ec711531b97d94b49f6c30`.
- Candidate: the V6.5 working tree under test.

It compares:

1. a signal built from a fixed BTCUSDT historical Binance USD-M futures market
   snapshot;
2. entry intent;
3. exit intent;
4. position size;
5. trade reason.

The expected JSON is a golden master captured by running the harness against
the real legacy tag. It is not produced by the V6.5 implementation and is not
updated automatically during tests.

## Files

- `tests/fixtures/v65_behavioral_compatibility_input.json`: immutable market
  and decision snapshots.
- `tests/fixtures/v6_legacy_behavioral_compatibility_expected.json`: output
  captured from the legacy tag.
- `scripts/capture_v65_behavioral_compatibility.py`: shared capture harness.
- `tests/test_v65_behavioral_compatibility.py`: V6.5 compatibility gate.

## Run

```bash
venv/bin/python -m pytest tests/test_v65_behavioral_compatibility.py -q
```

## Regenerating the legacy oracle

Regeneration is only valid when the input fixture intentionally changes. Run
the capture script with `--repo-root` pointing at a clean checkout of the
legacy tag, then review the full JSON diff before accepting it:

```bash
python scripts/capture_v65_behavioral_compatibility.py \
  --repo-root /path/to/v6.0-baseline-20260721 \
  --fixture tests/fixtures/v65_behavioral_compatibility_input.json \
  --output tests/fixtures/v6_legacy_behavioral_compatibility_expected.json \
  --label v6.0-baseline-20260721 \
  --commit 0aaaaa2dc014becf99ec711531b97d94b49f6c30
```

Do not regenerate the oracle from the V6.5 candidate. That would make the test
self-referential and unable to detect behavioral drift.
