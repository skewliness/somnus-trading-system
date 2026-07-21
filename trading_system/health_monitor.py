from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence


SEVERITY_RANK = {"ok": 0, "warning": 1, "critical": 2}
SAFE_MODE_TRIGGER_CODES = frozenset(
    {
        "protective_stop_missing",
        "position_reconcile_failed",
        "position_reconcile_behind",
        "position_reconcile_pending",
        "market_data_unavailable",
        "market_data_partial_stale",
        "exchange_requests_failed",
        "exchange_connection_unavailable",
        "exchange_latency_critical",
    }
)


def parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def age_seconds(value: Any, *, now: datetime) -> float | None:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds())


def _position_identity(position: Mapping[str, Any]) -> str:
    symbol = str(position.get("symbol") or "UNKNOWN").upper()
    side = str(position.get("side") or "unknown").lower()
    return f"{symbol}:{side}"


def _position_quantity(position: Mapping[str, Any]) -> float:
    for key in ("position_qty", "remaining_quantity", "quantity"):
        try:
            value = abs(float(position.get(key) or 0.0))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def latest_position_change_at(positions: Sequence[Mapping[str, Any]]) -> str:
    latest: datetime | None = None
    latest_text = ""
    timestamp_keys = (
        "last_scale_in_at",
        "last_partial_close_at",
        "last_reconciled_at",
        "opened_at",
        "updated_at",
    )
    for position in positions:
        for key in timestamp_keys:
            raw = position.get(key)
            parsed = parse_timestamp(raw)
            if parsed is not None and (latest is None or parsed > latest):
                latest = parsed
                latest_text = str(raw)
        for lot in position.get("lots") or []:
            if not isinstance(lot, Mapping):
                continue
            for key in ("opened_at", "closed_at", "updated_at"):
                raw = lot.get(key)
                parsed = parse_timestamp(raw)
                if parsed is not None and (latest is None or parsed > latest):
                    latest = parsed
                    latest_text = str(raw)
    return latest_text


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    summary_warning_seconds: float = 180.0
    summary_critical_seconds: float = 600.0
    request_latency_warning_ms: float = 3000.0
    request_latency_critical_ms: float = 10000.0


@dataclass(frozen=True, slots=True)
class HealthFinding:
    code: str
    severity: str
    summary: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeHealthReport:
    generated_at: str
    overall_status: str
    findings: tuple[HealthFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        safe_mode = safe_mode_from_report(self)
        return {
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
            "findings": [item.to_dict() for item in self.findings],
            "critical_digest": self.critical_digest(),
            "safe_mode": safe_mode.to_dict(),
        }

    def critical_findings(self) -> tuple[HealthFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "critical")

    def critical_digest(self) -> str:
        stable_detail_keys = {
            "latest_position_change_at",
            "positions",
            "status",
            "symbols",
            "synced_at",
        }
        payload = [
            {
                "code": item.code,
                "summary": item.summary,
                "details": {
                    key: value
                    for key, value in item.details.items()
                    if key in stable_detail_keys
                },
            }
            for item in self.critical_findings()
        ]
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16] if payload else ""


@dataclass(frozen=True, slots=True)
class SafeModeDecision:
    active: bool
    block_new_entries: bool
    allow_risk_actions: bool
    trigger_codes: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_mode_from_report(report: RuntimeHealthReport) -> SafeModeDecision:
    triggers = tuple(item for item in report.findings if item.code in SAFE_MODE_TRIGGER_CODES)
    active = bool(triggers)
    return SafeModeDecision(
        active=active,
        block_new_entries=active,
        allow_risk_actions=True,
        trigger_codes=tuple(item.code for item in triggers),
        reasons=tuple(item.summary for item in triggers),
    )


def _finding(code: str, severity: str, summary: str, **details: Any) -> HealthFinding:
    return HealthFinding(code=code, severity=severity, summary=summary, details=details)


def evaluate_runtime_health(
    snapshot: Mapping[str, Any],
    *,
    thresholds: HealthThresholds | None = None,
    now: datetime | None = None,
) -> RuntimeHealthReport:
    limits = thresholds or HealthThresholds()
    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    findings: list[HealthFinding] = []

    service_running = bool(snapshot.get("service_running"))
    findings.append(
        _finding(
            "service_running" if service_running else "service_stopped",
            "ok" if service_running else "critical",
            "execution service is running" if service_running else "execution service is stopped",
        )
    )

    summary_age = snapshot.get("summary_age_seconds")
    try:
        summary_age_value = float(summary_age) if summary_age is not None else None
    except (TypeError, ValueError):
        summary_age_value = None
    if summary_age_value is None:
        findings.append(_finding("summary_missing", "critical", "execution summary timestamp is unavailable"))
    elif summary_age_value > limits.summary_critical_seconds:
        findings.append(
            _finding("summary_stale", "critical", "execution summary is critically stale", age_seconds=summary_age_value)
        )
    elif summary_age_value > limits.summary_warning_seconds:
        findings.append(
            _finding("summary_delayed", "warning", "execution summary is delayed", age_seconds=summary_age_value)
        )
    else:
        findings.append(_finding("summary_fresh", "ok", "execution summary is fresh", age_seconds=summary_age_value))

    signals = [item for item in snapshot.get("signals") or [] if isinstance(item, Mapping)]
    stale_symbols = sorted(
        str(item.get("symbol") or "UNKNOWN").upper()
        for item in signals
        if isinstance(item.get("market_data_freshness"), Mapping)
        and bool(item["market_data_freshness"].get("stale"))
    )
    if signals and len(stale_symbols) == len(signals):
        findings.append(
            _finding("market_data_unavailable", "critical", "all configured symbols use stale market data", symbols=stale_symbols)
        )
    elif stale_symbols:
        findings.append(
            _finding("market_data_partial_stale", "warning", "some symbols use stale market data", symbols=stale_symbols)
        )
    elif signals:
        findings.append(_finding("market_data_fresh", "ok", "signal market data is fresh", symbol_count=len(signals)))
    else:
        findings.append(_finding("market_data_missing", "warning", "no signal freshness data is available"))

    request_health = snapshot.get("exchange_request_health")
    request_health = request_health if isinstance(request_health, Mapping) else {}
    request_count = max(0, int(request_health.get("request_count") or 0))
    failure_count = max(0, int(request_health.get("failure_count") or 0))
    average_latency = float(request_health.get("average_latency_ms") or 0.0)
    last_latency = float(request_health.get("last_latency_ms") or 0.0)
    observed_latency = max(average_latency, last_latency)
    request_details = {
        "request_count": request_count,
        "failure_count": failure_count,
        "average_latency_ms": average_latency,
        "last_latency_ms": last_latency,
    }
    connection_unavailable = bool(request_health.get("connection_unavailable"))
    if connection_unavailable:
        findings.append(
            _finding("exchange_connection_unavailable", "critical", "exchange connection is unavailable", **request_details)
        )
    elif request_count > 0 and failure_count >= request_count:
        findings.append(_finding("exchange_requests_failed", "critical", "all recorded exchange requests failed", **request_details))
    elif observed_latency >= limits.request_latency_critical_ms:
        findings.append(_finding("exchange_latency_critical", "critical", "exchange request latency is critical", **request_details))
    elif failure_count > 0 or observed_latency >= limits.request_latency_warning_ms:
        findings.append(_finding("exchange_request_degraded", "warning", "exchange requests are degraded", **request_details))
    else:
        findings.append(_finding("exchange_requests_healthy", "ok", "exchange requests are healthy", **request_details))

    positions = [item for item in snapshot.get("positions") or [] if isinstance(item, Mapping) and _position_quantity(item) > 0]
    reconcile = snapshot.get("last_ledger_reconcile")
    reconcile = reconcile if isinstance(reconcile, Mapping) else {}
    reconcile_status = str(reconcile.get("status") or "").lower()
    reconcile_at = reconcile.get("synced_at") or reconcile.get("generated_at")
    reconcile_time = parse_timestamp(reconcile_at)
    latest_change_text = latest_position_change_at(positions)
    latest_change_time = parse_timestamp(latest_change_text)
    reconcile_details = {
        "status": reconcile_status,
        "synced_at": str(reconcile_at or ""),
        "latest_position_change_at": latest_change_text,
        "age_seconds": age_seconds(reconcile_at, now=checked_at),
    }
    pending_reconcile = sorted(_position_identity(position) for position in positions if bool(position.get("reconcile_pending")))
    if pending_reconcile:
        findings.append(
            _finding(
                "position_reconcile_pending",
                "critical",
                "position reconciliation is pending after an uncertain exchange result",
                positions=pending_reconcile,
                **reconcile_details,
            )
        )
    elif reconcile_status in {"failed", "mismatch", "exchange_unavailable"}:
        findings.append(_finding("position_reconcile_failed", "critical", "position reconciliation is not healthy", **reconcile_details))
    elif positions and reconcile_time is None:
        findings.append(
            _finding(
                "position_reconcile_missing",
                "warning",
                "open positions have no reconciliation timestamp",
                **reconcile_details,
            )
        )
    elif positions and latest_change_time is not None and reconcile_time is not None and reconcile_time < latest_change_time:
        findings.append(
            _finding(
                "position_reconcile_behind",
                "critical",
                "position ledger changed after the latest reconciliation",
                **reconcile_details,
            )
        )
    else:
        findings.append(
            _finding(
                "position_reconcile_current",
                "ok",
                "position reconciliation covers the latest change",
                **reconcile_details,
            )
        )

    submitted = bool(snapshot.get("submit"))
    order_mode = str(snapshot.get("order_mode") or "dry_run").lower()
    protective_stop_required = bool(snapshot.get("protective_stop_required", True))
    missing_stops = sorted(
        _position_identity(position)
        for position in positions
        if not (position.get("protective_stop_order_id") or position.get("protective_stop_client_order_id"))
    )
    if submitted and order_mode != "dry_run" and protective_stop_required and missing_stops:
        findings.append(
            _finding("protective_stop_missing", "critical", "open exchange positions are missing protective stops", positions=missing_stops)
        )
    else:
        findings.append(
            _finding(
                "protective_stops_healthy",
                "ok",
                "protective-stop requirement is satisfied",
                checked_position_count=len(positions),
            )
        )

    overall = max((item.severity for item in findings), key=lambda value: SEVERITY_RANK[value], default="ok")
    return RuntimeHealthReport(
        generated_at=checked_at.isoformat(),
        overall_status=overall,
        findings=tuple(findings),
    )
