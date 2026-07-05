from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from trading_system.persistence import atomic_write_json, atomic_write_text


@dataclass(slots=True)
class IssueRecord:
    code: str
    severity: str
    summary: str
    detail: str
    remediation: str
    status: str = "open"

    def to_dict(self) -> dict:
        return asdict(self)


class IssueRegistry:
    def __init__(self) -> None:
        self._issues: dict[str, IssueRecord] = {}

    def add(self, issue: IssueRecord) -> None:
        self._issues[issue.code] = issue

    def ensure(self, code: str, severity: str, summary: str, detail: str, remediation: str) -> None:
        if code not in self._issues:
            self.add(IssueRecord(code=code, severity=severity, summary=summary, detail=detail, remediation=remediation))

    def remove(self, code: str) -> None:
        self._issues.pop(code, None)

    def items(self) -> list[IssueRecord]:
        return sorted(self._issues.values(), key=lambda item: (item.severity, item.code))

    def write(self, json_path: Path, md_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(json_path, {"issues": [item.to_dict() for item in self.items()]})
        lines = ["# Open Issues", ""]
        for issue in self.items():
            lines.extend(
                [
                    f"## {issue.code} [{issue.severity}]",
                    "",
                    f"- Summary: {issue.summary}",
                    f"- Detail: {issue.detail}",
                    f"- Remediation: {issue.remediation}",
                    f"- Status: {issue.status}",
                    "",
                ]
            )
        atomic_write_text(md_path, "\n".join(lines).strip() + "\n")


def sync_live_access_issue(registry: IssueRegistry, error: Exception | None = None) -> None:
    if error is None:
        registry.remove("BINANCE_FUTURES_LIVE_BLOCKED")
        return
    registry.ensure(
        code="BINANCE_FUTURES_LIVE_BLOCKED",
        severity="high",
        summary="Live Binance Futures market data is blocked in the current runtime.",
        detail=str(error),
        remediation="Run live mode from an allowed network or route requests through your own compliant proxy layer.",
    )


def build_default_issue_registry() -> IssueRegistry:
    registry = IssueRegistry()
    registry.ensure(
        code="SQUARE_FEED_SESSION_DEPENDENCY",
        severity="high",
        summary="Binance Square feed reads depend on a fresh logged-in browser session.",
        detail="The feed endpoint works when replaying a browser curl export, but anonymous requests are blocked by Binance WAF.",
        remediation="Refresh the browser export periodically or automate session capture in a separate local bridge.",
    )
    registry.ensure(
        code="BOT_FILTER_RULES_EARLY_STAGE",
        severity="low",
        summary="Square bot filtering is still rule-based.",
        detail="The current framework now checks engagement anomalies, repeated templates, and low text variety, but it does not yet use a trained account-quality model.",
        remediation="Add account-level history features and a learned classifier if the false-positive rate becomes a practical problem.",
    )
    return registry
