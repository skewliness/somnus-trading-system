from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from trading_system.models import SentimentSignal


class SentimentBook:
    def __init__(self, signals: dict[str, SentimentSignal]) -> None:
        self._signals = signals

    @classmethod
    def from_csv(cls, path: Path) -> "SentimentBook":
        signals: dict[str, SentimentSignal] = {}
        if not path.exists():
            return cls(signals)

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                symbol = (row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                updated_at_text = (row.get("updated_at") or "").strip()
                updated_at = _parse_dt(updated_at_text)
                signals[symbol] = SentimentSignal(
                    symbol=symbol,
                    hotness_score=float(row.get("hotness_score") or 0),
                    mention_count=int(float(row.get("mention_count") or 0)),
                    source=(row.get("source") or "manual").strip() or "manual",
                    updated_at=updated_at,
                    notes=(row.get("notes") or "").strip(),
                    earliest_strong_source=(row.get("earliest_strong_source") or "").strip(),
                    earliest_strong_timestamp=_parse_optional_dt(row.get("earliest_strong_timestamp") or ""),
                    okx_confirmed_at=_parse_optional_dt(row.get("okx_confirmed_at") or ""),
                    binance_confirmed_at=_parse_optional_dt(row.get("binance_confirmed_at") or ""),
                    signal_sequence=(row.get("signal_sequence") or "single_source").strip() or "single_source",
                    lead_time_minutes=_parse_optional_float(row.get("lead_time_minutes") or ""),
                )
        return cls(signals)

    def lookup(self, symbol: str) -> SentimentSignal | None:
        return self._signals.get(symbol.upper())

    def symbols(self) -> list[str]:
        return sorted(self._signals)


def _parse_dt(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.now(timezone.utc)


def _parse_optional_dt(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _parse_optional_float(value: str) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
