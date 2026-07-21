from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def _datetime_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def _cache_key(symbol: str, interval: str, start_ms: int, end_ms: int) -> str:
    return f"{symbol.upper()}_{interval}_{start_ms}_{end_ms}.json"


class CoinScout4HLabeler:
    """Small public-kline fallback for exported snapshots.

    The production labeler also trains/labels CoinScout data. That module is
    intentionally excluded from the public snapshot, but Telegram/chart commands
    still need a compatible `fetch_klines` method for read-only public klines.
    """

    def __init__(self, cache_dir: str | Path = "data/trading_system/cache/chart_klines") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        symbol = str(symbol or "").strip().upper()
        interval = str(interval or "").strip().lower()
        if interval not in INTERVAL_MS:
            raise ValueError(f"unsupported interval: {interval}")
        start_ms = _datetime_ms(start)
        end_ms = _datetime_ms(end)
        cache_file = self.cache_dir / _cache_key(symbol, interval, start_ms, end_ms)
        if cache_file.exists() and not force_refresh:
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    return [row for row in cached if isinstance(row, dict)]
            except Exception:
                pass

        rows: list[dict[str, Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            params = urllib.parse.urlencode(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1500,
                }
            )
            request = urllib.request.Request(
                f"https://fapi.binance.com/fapi/v1/klines?{params}",
                headers={"User-Agent": "somnus-exported-snapshot/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Binance API error for {symbol} {interval}: {exc.code} {body}") from exc
            if not isinstance(payload, list) or not payload:
                break
            for item in payload:
                if not isinstance(item, list) or len(item) < 11:
                    continue
                open_time = int(item[0])
                rows.append(
                    {
                        "open_time": open_time,
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                        "close_time": int(item[6]),
                        "quote_volume": float(item[7]),
                        "trades": int(item[8]),
                        "taker_buy_base": float(item[9]),
                        "taker_buy_quote": float(item[10]),
                        "binance_symbol": symbol,
                        "interval": interval,
                    }
                )
            last_open = int(payload[-1][0])
            next_cursor = last_open + INTERVAL_MS[interval]
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(payload) < 1500:
                break
            time.sleep(0.05)

        deduped = sorted({int(row["open_time"]): row for row in rows}.values(), key=lambda row: int(row["open_time"]))
        try:
            cache_file.write_text(json.dumps(deduped, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return deduped


def get_smart_money_panel_payload(symbol: str) -> dict[str, Any]:
    """Fallback payload when the private smart-money panel module is excluded."""

    return {
        "symbol": str(symbol or "").upper(),
        "status": "unavailable",
        "source": "optional_integrations_fallback",
        "views": {},
        "reason": "smart_money_panel module is not included in this code snapshot",
    }


def optional_command_unavailable(command: str, module_name: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "command": command,
        "missing_module": module_name,
        "reason": "optional/private module is not included in this code snapshot",
    }
