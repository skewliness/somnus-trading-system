from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from trading_system import v6_execution


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SCRIPT = ROOT / "scripts" / "capture_v65_exit_regression.py"
INPUT_FIXTURE = ROOT / "tests" / "fixtures" / "v65_exit_regression_input.json"
LEGACY_EXPECTED = ROOT / "tests" / "fixtures" / "v6_legacy_exit_regression_expected.json"


def load_capture_module():
    spec = importlib.util.spec_from_file_location("capture_v65_exit_regression", CAPTURE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exit_regression_matches_frozen_v6_legacy_oracle(tmp_path: Path) -> None:
    capture = load_capture_module()
    fixture = json.loads(INPUT_FIXTURE.read_text(encoding="utf-8"))
    expected = json.loads(LEGACY_EXPECTED.read_text(encoding="utf-8"))
    original_utc_now_iso = v6_execution.utc_now_iso
    actual = capture.capture_current(ROOT, fixture, work_root=tmp_path)

    assert actual["fixture_sha256"] == expected["fixture_sha256"]
    assert actual["cases"] == expected["cases"]
    assert v6_execution.utc_now_iso is original_utc_now_iso


def test_exit_regression_covers_required_priority_collisions() -> None:
    fixture = json.loads(INPUT_FIXTURE.read_text(encoding="utf-8"))
    case_ids = {case["id"] for case in fixture["cases"]}

    assert {
        "max_hold_precedes_stop",
        "stop_precedes_daily_reversal",
        "daily_reversal_precedes_logic_weakening",
        "logic_weakening_precedes_smart_signal",
        "smart_signal_precedes_roi_take_profit",
        "roi_precedes_staged_take_profit",
        "staged_precedes_fixed_take_profit",
        "opposite_probability_fallback",
    } <= case_ids

    expected = json.loads(LEGACY_EXPECTED.read_text(encoding="utf-8"))
    for case in expected["cases"]:
        assert case["exit_reason"]
        assert case["exit_priority"] > 0
        assert case["exit_price"] > 0

