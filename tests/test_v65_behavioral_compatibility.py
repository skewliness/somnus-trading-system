from __future__ import annotations

import json
from pathlib import Path

from scripts.capture_v65_behavioral_compatibility import capture, fixture_sha256
from trading_system import v6_execution


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "v65_behavioral_compatibility_input.json"
LEGACY_RESULT_PATH = ROOT / "tests" / "fixtures" / "v6_legacy_behavioral_compatibility_expected.json"
BASELINE_TAG = "v6.0-baseline-20260721"
BASELINE_COMMIT = "0aaaaa2dc014becf99ec711531b97d94b49f6c30"
BASELINE_SOURCE_SHA256 = "2f36b212d36b60267279a984a0c05d2fc3169f7a72730cf0e1066be373bcf40a"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v65_matches_frozen_v6_behavior_for_identical_snapshots(tmp_path: Path) -> None:
    fixture = load_json(FIXTURE_PATH)
    legacy = load_json(LEGACY_RESULT_PATH)
    original_utc_now_iso = v6_execution.utc_now_iso

    assert legacy["baseline"]["label"] == BASELINE_TAG
    assert legacy["baseline"]["commit"] == BASELINE_COMMIT
    assert legacy["baseline"]["source_sha256"] == BASELINE_SOURCE_SHA256
    assert legacy["fixture_sha256"] == fixture_sha256(fixture)

    candidate = capture(
        v6_execution,
        fixture,
        repo_root=ROOT,
        work_root=tmp_path,
        label="v6.5-candidate",
        commit="working-tree",
    )

    assert candidate["fixture_sha256"] == legacy["fixture_sha256"]
    assert candidate["market_signal"] == legacy["market_signal"]
    assert candidate["cases"] == legacy["cases"]
    assert v6_execution.utc_now_iso is original_utc_now_iso


def test_compatibility_matrix_covers_every_required_behavioral_surface() -> None:
    fixture = load_json(FIXTURE_PATH)
    legacy = load_json(LEGACY_RESULT_PATH)
    cases = legacy["cases"]
    cases_by_id = {case["id"]: case for case in cases}

    assert fixture["provenance"]["exchange"] == "Binance USD-M Futures"
    assert fixture["provenance"]["timezone"] == "UTC"
    assert len(fixture["historical_market_snapshot"]["klines"]["1m"]) == 485
    assert len(fixture["historical_market_snapshot"]["klines"]["15m"]) == 1600
    assert len(fixture["historical_market_snapshot"]["klines"]["4h"]) == 730
    assert len(fixture["historical_market_snapshot"]["klines"]["1d"]) == 400
    assert legacy["market_signal"]["symbol"] == "BTCUSDT"
    assert all(case["signal"] for case in cases)
    assert any(case["entry_intent"] is not None for case in cases)
    assert any(case["exit_intent"] is not None for case in cases)
    assert any(case["position_size_usdt"] is not None for case in cases)
    assert any(case["trade_reason"] for case in cases)
    assert {case["id"] for case in cases} == {
        "btc_probability_short_entry",
        "eth_probability_long_entry",
        "hype_high_wait_no_entry",
        "btc_long_hard_stop_exit",
        "eth_short_hard_stop_exit",
    }
    assert cases_by_id["btc_probability_short_entry"]["entry_intent"]["action"] == "open_short"
    assert cases_by_id["eth_probability_long_entry"]["entry_intent"]["action"] == "open_long"
    assert cases_by_id["hype_high_wait_no_entry"]["entry_intent"] is None
    assert cases_by_id["btc_long_hard_stop_exit"]["exit_intent"]["action"] == "close_long"
    assert cases_by_id["eth_short_hard_stop_exit"]["exit_intent"]["action"] == "close_short"
    assert {
        case["position_size_usdt"]
        for case in cases
        if case["position_size_usdt"] is not None
    } == {300.0}
