from __future__ import annotations

import json
from pathlib import Path

from trading_system.optional_integrations import (
    CoinScout4HLabeler,
    INTERVAL_MS,
    get_smart_money_panel_payload,
    optional_command_unavailable,
)


def test_optional_smart_money_payload_is_safe_fallback() -> None:
    payload = get_smart_money_panel_payload("btcusdt")

    assert payload["symbol"] == "BTCUSDT"
    assert payload["status"] == "unavailable"
    assert payload["views"] == {}


def test_optional_command_unavailable_is_machine_readable() -> None:
    payload = optional_command_unavailable("smart-money-panel", "trading_system.smart_money_panel")

    assert payload["status"] == "unavailable"
    assert payload["command"] == "smart-money-panel"
    assert payload["missing_module"] == "trading_system.smart_money_panel"


def test_optional_labeler_reads_valid_cache_without_network(tmp_path: Path) -> None:
    labeler = CoinScout4HLabeler(cache_dir=tmp_path)
    start_ms = 1_000
    end_ms = 2_000
    cache_file = tmp_path / f"BTCUSDT_15m_{start_ms}_{end_ms}.json"
    rows = [{"open_time": start_ms, "close": 1.0}]
    cache_file.write_text(json.dumps(rows), encoding="utf-8")

    from datetime import datetime, timezone

    start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    assert INTERVAL_MS["15m"] == 900_000
    assert labeler.fetch_klines("BTCUSDT", "15m", start, end, force_refresh=False) == rows
