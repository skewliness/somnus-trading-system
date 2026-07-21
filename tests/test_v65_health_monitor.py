from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import requests

from trading_system.health_monitor import HealthThresholds, evaluate_runtime_health
from trading_system.v6_execution import V6BinanceClient, V6ExecutionConfig, V6ExecutionEngine
from scripts.telegram_monitor_v5_v6 import build_alerts


NOW = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


def healthy_snapshot() -> dict:
    return {
        "service_running": True,
        "summary_age_seconds": 30,
        "order_mode": "testnet_order",
        "submit": True,
        "signals": [
            {"symbol": "BTCUSDT", "market_data_freshness": {"stale": False}},
            {"symbol": "ETHUSDT", "market_data_freshness": {"stale": False}},
        ],
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "position_qty": 0.01,
                "opened_at": (NOW - timedelta(minutes=10)).isoformat(),
                "protective_stop_order_id": 123,
            }
        ],
        "exchange_request_health": {
            "request_count": 4,
            "failure_count": 0,
            "average_latency_ms": 120,
            "last_latency_ms": 100,
        },
        "last_ledger_reconcile": {
            "status": "ok",
            "synced_at": (NOW - timedelta(minutes=5)).isoformat(),
        },
        "protective_stop_required": True,
    }


def findings_by_code(report) -> dict:
    return {item.code: item for item in report.findings}


def test_healthy_runtime_is_ok() -> None:
    report = evaluate_runtime_health(healthy_snapshot(), now=NOW)

    assert report.overall_status == "ok"
    assert report.critical_digest() == ""


def test_stopped_service_and_stale_summary_are_critical() -> None:
    snapshot = healthy_snapshot()
    snapshot["service_running"] = False
    snapshot["summary_age_seconds"] = 1200

    report = evaluate_runtime_health(snapshot, now=NOW)

    assert report.overall_status == "critical"
    assert {"service_stopped", "summary_stale"}.issubset(findings_by_code(report))


def test_all_stale_market_data_is_critical_but_partial_is_warning() -> None:
    snapshot = healthy_snapshot()
    for signal in snapshot["signals"]:
        signal["market_data_freshness"]["stale"] = True
    all_stale = evaluate_runtime_health(snapshot, now=NOW)

    snapshot["signals"][0]["market_data_freshness"]["stale"] = False
    partial = evaluate_runtime_health(snapshot, now=NOW)

    assert "market_data_unavailable" in findings_by_code(all_stale)
    assert findings_by_code(all_stale)["market_data_unavailable"].severity == "critical"
    assert findings_by_code(partial)["market_data_partial_stale"].severity == "warning"


def test_exchange_failures_and_latency_are_reported_without_new_requests() -> None:
    snapshot = healthy_snapshot()
    snapshot["exchange_request_health"] = {
        "request_count": 3,
        "failure_count": 3,
        "average_latency_ms": 15000,
        "last_latency_ms": 16000,
    }

    report = evaluate_runtime_health(snapshot, now=NOW)

    assert findings_by_code(report)["exchange_requests_failed"].severity == "critical"


def test_reconcile_must_cover_latest_position_change() -> None:
    snapshot = healthy_snapshot()
    snapshot["last_ledger_reconcile"]["synced_at"] = (NOW - timedelta(minutes=15)).isoformat()

    report = evaluate_runtime_health(snapshot, now=NOW)

    assert findings_by_code(report)["position_reconcile_behind"].severity == "critical"


def test_critical_digest_ignores_volatile_age_values() -> None:
    snapshot = healthy_snapshot()
    snapshot["last_ledger_reconcile"]["synced_at"] = (NOW - timedelta(minutes=15)).isoformat()

    first = evaluate_runtime_health(snapshot, now=NOW)
    later = evaluate_runtime_health(snapshot, now=NOW + timedelta(minutes=1))

    assert first.critical_digest()
    assert first.critical_digest() == later.critical_digest()


def test_missing_protective_stop_is_critical_only_for_submitted_exchange_mode() -> None:
    snapshot = healthy_snapshot()
    snapshot["positions"][0].pop("protective_stop_order_id")
    exchange_report = evaluate_runtime_health(snapshot, now=NOW)

    snapshot["order_mode"] = "dry_run"
    snapshot["submit"] = False
    dry_report = evaluate_runtime_health(snapshot, now=NOW)

    assert findings_by_code(exchange_report)["protective_stop_missing"].severity == "critical"
    assert "protective_stop_missing" not in findings_by_code(dry_report)


def test_request_telemetry_observes_existing_request_without_changing_response(monkeypatch) -> None:
    client = V6BinanceClient(V6ExecutionConfig())
    response = SimpleNamespace(status_code=200)
    monkeypatch.setattr(client.session, "request", lambda *_args, **_kwargs: response)

    returned = client._request("GET", "https://example.invalid")

    assert returned is response
    assert client.request_health()["request_count"] == 1
    assert client.request_health()["failure_count"] == 0


def test_request_telemetry_records_failure_and_preserves_exception_contract(monkeypatch) -> None:
    client = V6BinanceClient(V6ExecutionConfig(binance_request_retries=1))

    def fail(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(client.session, "request", fail)

    try:
        client._request("GET", "https://example.invalid")
    except Exception as exc:
        assert type(exc).__name__ == "V6ExecutionError"
    else:
        raise AssertionError("request failure must preserve the existing V6ExecutionError contract")
    assert client.request_health()["request_count"] == 1
    assert client.request_health()["failure_count"] == 1


def test_engine_request_health_sidecar_is_optional() -> None:
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.client = object()

    assert engine.exchange_request_health() == {}


def test_thresholds_are_configurable() -> None:
    snapshot = healthy_snapshot()
    snapshot["summary_age_seconds"] = 61
    report = evaluate_runtime_health(
        snapshot,
        thresholds=HealthThresholds(summary_warning_seconds=60, summary_critical_seconds=120),
        now=NOW,
    )

    assert findings_by_code(report)["summary_delayed"].severity == "warning"


def test_telegram_alerts_include_new_critical_health_without_duplicate_service_alerts() -> None:
    status = {
        "v5": {"running": True, "report_age": 0, "activity_age": 0, "latest_errors": []},
        "v6": {
            "running": False,
            "summary_age": 0,
            "distill_running": True,
            "distill_activity_age": 0,
            "distill_age": 0,
            "intent_count": 0,
            "result_count": 0,
            "runtime_health": {
                "critical_digest": "abc",
                "findings": [
                    {"code": "service_stopped", "severity": "critical", "summary": "service stopped"},
                    {
                        "code": "protective_stop_missing",
                        "severity": "critical",
                        "summary": "open exchange positions are missing protective stops",
                    },
                ],
            },
        },
        "thresholds": {"v5_stale_seconds": 1800, "v6_stale_seconds": 180, "v6_distill_stale_seconds": 7200},
    }

    alerts = build_alerts(status, "v6")
    keys = [key for key, _message in alerts]

    assert keys.count("v6_down") == 1
    assert "v6_runtime_health_abc" in keys
    assert sum("service stopped" in message for _key, message in alerts) == 0
