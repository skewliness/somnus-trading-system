from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts import label_v6_entry_gate_events as labeler


def make_rows(count: int, *, start: datetime, entry_open: float = 100.0) -> list[list[object]]:
    rows: list[list[object]] = []
    for index in range(count):
        opened = start + timedelta(minutes=15 * index)
        open_ms = int(opened.timestamp() * 1000)
        close_ms = open_ms + 15 * 60 * 1000 - 1
        close = entry_open + index * 0.1
        rows.append(
            [
                open_ms,
                str(entry_open if index == 1 else close),
                str(close + 2.0),
                str(close - 1.0),
                str(close),
                "1000",
                close_ms,
            ]
        )
    return rows


def make_event(*, computed_at: str, action: str = "open_long") -> dict[str, object]:
    return {
        "event": "entry_gate_decision",
        "event_id": "event-1",
        "computed_at": computed_at,
        "symbol": "BTCUSDT",
        "action": action,
        "price": 100.0,
        "selected_entry_level": {"name": "bb300_mid_15m"},
        "decision": {
            "state": "watch",
            "allowed": False,
            "blocking_reason": "four_h_trend_waiting_for_15m_selector",
            "watch_reasons": ["four_h_trend_waiting_for_15m_selector"],
        },
    }


def test_label_event_uses_next_bar_open_and_directional_mfe_mae() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = make_rows(20, start=start)
    event = make_event(computed_at=(start + timedelta(minutes=14, seconds=59)).isoformat())
    now_ms = int((start + timedelta(hours=8)).timestamp() * 1000)

    outcome = labeler.label_event(event, rows, now_ms)

    assert outcome is not None
    one_hour = outcome["horizons"]["1h"]
    assert one_hour["entry_time"] == (start + timedelta(minutes=15)).isoformat()
    assert one_hour["entry_price"] == 100.0
    assert one_hour["mfe_pct"] > 0
    assert one_hour["mae_pct"] < 0

    short_outcome = labeler.label_event({**event, "action": "open_short"}, rows, now_ms)
    assert short_outcome is not None
    assert short_outcome["horizons"]["1h"]["mfe_pct"] >= 0
    assert short_outcome["horizons"]["1h"]["mae_pct"] <= 0


def test_main_labels_cached_events_without_network_and_is_idempotent(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = make_rows(700, start=start)
    events_file = tmp_path / "entry_gate_events.jsonl"
    cache_dir = tmp_path / "execution_klines"
    symbol_cache = cache_dir / "15m" / "BTCUSDT.json"
    outcomes_file = tmp_path / "entry_gate_outcomes.jsonl"
    state_file = tmp_path / "state.json"
    summary_file = tmp_path / "summary.json"
    recommendations_file = tmp_path / "recommendations.json"
    symbol_cache.parent.mkdir(parents=True)
    event = make_event(computed_at=(start + timedelta(minutes=14, seconds=59)).isoformat())
    events_file.write_text(json.dumps(event) + "\n", encoding="utf-8")
    symbol_cache.write_text(
        json.dumps({"source": "test-cache", "updated_at": "2026-01-08T08:00:00+00:00", "klines": rows}),
        encoding="utf-8",
    )
    now = (start + timedelta(days=8)).isoformat()
    args = [
        "--events-file", str(events_file),
        "--cache-dir", str(cache_dir),
        "--outcomes-file", str(outcomes_file),
        "--state-file", str(state_file),
        "--summary-file", str(summary_file),
        "--recommendations-file", str(recommendations_file),
        "--now", now,
    ]

    assert labeler.main(args) == 0
    assert labeler.main(args) == 0

    outcomes = outcomes_file.read_text(encoding="utf-8").strip().splitlines()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    recommendations = json.loads(recommendations_file.read_text(encoding="utf-8"))
    assert len(outcomes) == 1
    assert summary["events_labeled"] == 1
    assert summary["complete_events"] == 1
    assert summary["outcomes_appended"] == 0
    assert summary["cost_model"]["funding_omitted"] is True
    assert recommendations["auto_apply"] is False


def test_summary_uses_only_complete_24h_rows_for_24h_rates() -> None:
    reason = "four_h_trend_waiting_for_15m_selector"
    complete = {
        "gate_allowed": False,
        "watch_reasons": [reason],
        "horizons": {
            "24h": {
                "net_return_pct": 2.0,
                "mfe_pct": 4.0,
                "mae_pct": -1.0,
            }
        },
        # These row-level flags can be based on a shorter primary horizon and
        # therefore must not drive a 24h recommendation.
        "favorable_counterfactual": False,
        "harmful_counterfactual": True,
    }
    partial = {
        "gate_allowed": False,
        "watch_reasons": [reason],
        "horizons": {
            "4h": {
                "net_return_pct": -4.0,
                "mfe_pct": 0.5,
                "mae_pct": -5.0,
            }
        },
        "favorable_counterfactual": False,
        "harmful_counterfactual": True,
    }

    summary, recommendations = labeler.build_summary([complete, partial])

    row = summary["reasons"][0]
    assert row["samples"] == 1
    assert row["favorable_rate"] == 1.0
    assert row["harmful_rate"] == 0.0
    assert recommendations["recommendations"][reason]["samples"] == 1
