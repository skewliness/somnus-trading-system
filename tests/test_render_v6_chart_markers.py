from __future__ import annotations

import json
from datetime import datetime, timezone

import scripts.render_v6_chart as chart


def test_load_trade_markers_reads_recent_v5_v6_execution_closes(tmp_path, monkeypatch) -> None:
    v5v6_orders = tmp_path / "v5v6_orders.jsonl"
    v5v6_orders.write_text(
        json.dumps(
            {
                "status": "submitted",
                "submitted_at": "2026-06-27T07:56:00+00:00",
                "intent": {
                    "action": "close_long",
                    "symbol": "UBUSDT",
                    "side": "SELL",
                    "entry_price": 0.07879,
                },
                "response": {"avgPrice": "0.07901"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    paper_state = tmp_path / "paper_state.json"
    paper_state.write_text('{"positions": [], "journal": [], "learning_journal": []}', encoding="utf-8")

    monkeypatch.setattr(chart, "V6_ORDERS_FILE", empty)
    monkeypatch.setattr(chart, "V5_V6_ORDERS_FILE", v5v6_orders, raising=False)
    monkeypatch.setattr(chart, "V5_PAPER_STATE_FILE", paper_state)

    candle_time = int(datetime(2026, 6, 27, 7, 45, tzinfo=timezone.utc).timestamp())
    candles = [{"time": candle_time + step * 15 * 60, "open": 1, "high": 1, "low": 1, "close": 1} for step in range(3)]

    markers = chart.load_trade_markers("UBUSDT", "15m", candles)

    assert len(markers) == 1
    assert markers[0]["text"] == "平多"
    assert markers[0]["source"] == "v5/v6"
    assert markers[0]["position"] == "aboveBar"


def test_fetch_klines_for_chart_forces_realtime_refresh(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeLabeler:
        def __init__(self, cache_dir: str):
            self.cache_dir = cache_dir

        def fetch_klines(self, symbol, interval, start, end, **kwargs):
            calls.append({"symbol": symbol, "interval": interval, "force_refresh": kwargs.get("force_refresh")})
            return [
                {
                    "open_time": 2,
                    "open": 2.0,
                    "high": 2.0,
                    "low": 2.0,
                    "close": 2.0,
                    "volume": 1.0,
                    "binance_symbol": symbol,
                    "interval": interval,
                },
                {
                    "open_time": 1,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1.0,
                    "binance_symbol": symbol,
                    "interval": interval,
                },
            ]

    monkeypatch.setattr(chart, "CoinScout4HLabeler", FakeLabeler)

    rows = chart.fetch_klines("BTCUSDT", "15m", 10)

    assert [row["open_time"] for row in rows] == [1, 2]
    assert calls and calls[0]["force_refresh"] is True
