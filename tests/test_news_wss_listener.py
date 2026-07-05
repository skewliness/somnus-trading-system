from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_system.news_wss_listener import DEFAULT_FILTER_CONFIG, NewsWSSListener


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _listener(tmp_path: Path) -> NewsWSSListener:
    config = json.loads(json.dumps(DEFAULT_FILTER_CONFIG))
    config["cache_retention_hours"] = 24
    return NewsWSSListener(
        token="test-token",
        output_file=tmp_path / "opennews_wss.json",
        priority_hit_file=tmp_path / "opennews_wss_priority_hits.jsonl",
        trigger_file=tmp_path / "news_urgent_trigger",
        health_file=tmp_path / "news_wss_health.json",
        filter_config=config,
    )


def test_wss_cache_loads_only_last_24_hours(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    fresh = {"symbol": "BTCUSDT", "updated_at": _iso(now - timedelta(hours=1))}
    stale = {"symbol": "ETHUSDT", "updated_at": _iso(now - timedelta(hours=25))}
    output_file = tmp_path / "opennews_wss.json"
    output_file.write_text(json.dumps({"signals": [stale, fresh]}), encoding="utf-8")

    listener = _listener(tmp_path)

    assert [item["symbol"] for item in listener.buffer] == ["BTCUSDT"]


def test_wss_disk_caches_are_pruned_to_last_24_hours(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    fresh = {"symbol": "BTCUSDT", "updated_at": _iso(now - timedelta(hours=2))}
    stale = {"symbol": "ETHUSDT", "updated_at": _iso(now - timedelta(hours=26))}

    listener = _listener(tmp_path)
    listener.buffer = [stale, fresh]
    listener.priority_hit_file.write_text(
        json.dumps(stale) + "\n" + json.dumps(fresh) + "\n",
        encoding="utf-8",
    )
    listener.trigger_file.write_text(_iso(now - timedelta(hours=26)), encoding="utf-8")

    listener._maybe_prune_disk_caches(force=True, now=now)
    listener._write_signal_file(_iso(now))

    payload = json.loads(listener.output_file.read_text(encoding="utf-8"))
    assert [item["symbol"] for item in payload["signals"]] == ["BTCUSDT"]
    assert payload["filter_summary"]["cache_retention_hours"] == 24

    priority_records = [json.loads(line) for line in listener.priority_hit_file.read_text(encoding="utf-8").splitlines()]
    assert [item["symbol"] for item in priority_records] == ["BTCUSDT"]
    assert not listener.trigger_file.exists()
