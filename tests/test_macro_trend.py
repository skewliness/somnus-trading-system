from __future__ import annotations

from datetime import datetime, timezone

from trading_system.macro_trend import MacroTrendSignal, compute_macro_trend
from trading_system.persistence import atomic_write_json


class FakeClient:
    def __init__(self, btc_closes=None, eth_closes=None, btc_hourly_closes=None, error: Exception | None = None) -> None:
        self.btc_closes = btc_closes or []
        self.eth_closes = eth_closes or []
        self.btc_hourly_closes = btc_hourly_closes
        self.error = error

    def get_json(self, _path, params=None):
        if self.error:
            raise self.error
        symbol = (params or {}).get("symbol")
        interval = (params or {}).get("interval")
        if symbol == "BTCUSDT" and interval == "1h" and self.btc_hourly_closes is not None:
            closes = self.btc_hourly_closes
        else:
            closes = self.btc_closes if symbol == "BTCUSDT" else self.eth_closes
        return [[index, "0", "0", "0", str(close), "0"] for index, close in enumerate(closes)]


def test_compute_macro_trend_uses_btc_primary_high_confidence(tmp_path):
    cache_file = tmp_path / "macro_cache.json"
    client = FakeClient(
        btc_closes=[float(100 + index) for index in range(30)],
        eth_closes=[float(200 + index) for index in range(30)],
        btc_hourly_closes=[float(100 + index) for index in range(700)],
    )

    signal = compute_macro_trend(client, cache_file=cache_file)

    assert signal.source == "live"
    assert signal.btc_trend == "bullish"
    assert signal.eth_trend == "bullish"
    assert signal.macro_direction == "bullish"
    assert signal.macro_confidence == "high"
    assert signal.btc_ma7_slope_pct_3d > 1.0
    assert cache_file.exists()


def test_compute_macro_trend_forces_neutral_when_ma7_slope_is_flat(tmp_path):
    cache_file = tmp_path / "macro_cache.json"
    client = FakeClient(
        btc_closes=[100.0] * 29 + [101.0],
        eth_closes=[200.0] * 29 + [202.0],
    )

    signal = compute_macro_trend(
        client,
        cache_file=cache_file,
        neutral_band_pct=0.5,
        slope_neutral_threshold_pct=1.0,
    )

    assert signal.btc_distance_pct > 0.5
    assert abs(signal.btc_ma7_slope_pct_3d) < 1.0
    assert signal.btc_trend == "neutral"
    assert signal.macro_direction == "mixed"


def test_compute_macro_trend_falls_back_to_fresh_cache(tmp_path):
    cache_file = tmp_path / "macro_cache.json"
    cached = MacroTrendSignal.neutral(source="live").to_dict()
    cached.update(
        {
            "btc_price": 100.0,
            "btc_ma7": 99.0,
            "btc_trend": "bullish",
            "eth_price": 200.0,
            "eth_ma7": 198.0,
            "eth_trend": "bullish",
            "macro_direction": "bullish",
            "macro_confidence": "high",
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_write_json(cache_file, cached)

    signal = compute_macro_trend(
        FakeClient(error=RuntimeError("network down")),
        cache_file=cache_file,
        cache_ttl_seconds=6 * 60 * 60,
    )

    assert signal.source == "cache"
    assert signal.macro_direction == "bullish"
    assert signal.error == "network down"
    assert signal.cache_age_seconds >= 0
