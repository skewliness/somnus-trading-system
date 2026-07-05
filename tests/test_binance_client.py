from dataclasses import dataclass, fields
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from trading_system.binance import (
    BinanceFuturesClient,
    BinanceMarketDataProvider,
    KlineCache,
    _bb_signal_kwargs,
    _merge_scan_candidates,
    _rank_short_scan_candidates,
)
from trading_system.bb_structure import BBStructureSignal
from trading_system.config import StrategyConfig
from trading_system.models import MarketSnapshot


@dataclass
class DummyBBSignal:
    bb96_mid_15m: float = 123.0
    ma7_15m: float = 456.0
    fake_future_field: float = 789.0


def test_bb_signal_kwargs_accepts_synced_v6_structure_and_filters_unknowns() -> None:
    payload = _bb_signal_kwargs(DummyBBSignal())

    assert payload["bb96_mid_15m"] == 123.0
    assert payload["ma7_15m"] == 456.0
    assert "fake_future_field" not in payload


def test_market_snapshot_accepts_current_v6_bb_structure_fields() -> None:
    bb_fields = {field.name for field in fields(BBStructureSignal)}
    snapshot_fields = {field.name for field in fields(MarketSnapshot)}

    assert bb_fields <= snapshot_fields


def test_v5_kline_cache_reads_v6_dict_payload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "15m" / "BTCUSDT.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "source": "binance_futures_mainnet",
                    "symbol": "BTCUSDT",
                    "interval": "15m",
                    "klines": [
                        [2000, "2", "2", "2", "2"],
                        [1000, "1", "1", "1", "1"],
                    ],
                }
            ),
            encoding="utf-8",
        )

        rows = KlineCache(Path(tmp)).load("BTCUSDT", "15m")

    assert [row[0] for row in rows] == [1000, 2000]


def test_v5_momo_defaults_use_shared_4707_kline_window() -> None:
    config = StrategyConfig()

    assert config.momo_structure_cache_dir == Path("data/trading_system/cache/binance_klines")
    assert config.momo_1m_kline_limit == 4707
    assert config.momo_15m_kline_limit == 4707
    assert config.momo_4h_kline_limit == 4707
    assert config.momo_daily_kline_limit == 4707


def test_v5_provider_uses_shared_cache_for_short_daily_window() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = StrategyConfig(momo_structure_cache_dir=Path(tmp), momo_structure_cache_enabled=True)
        cache_path = Path(tmp) / "1d" / "BTCUSDT.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text(
            json.dumps(
                {
                    "source": "binance_futures_mainnet",
                    "symbol": "BTCUSDT",
                    "interval": "1d",
                    "klines": [
                        [1000, "1", "1", "1", "1"],
                        [2000, "2", "2", "2", "2"],
                        [3000, "3", "3", "3", "3"],
                    ],
                }
            ),
            encoding="utf-8",
        )
        provider = BinanceMarketDataProvider(config)
        provider.client = SimpleNamespace(get_json=Mock(side_effect=AssertionError("unexpected live kline request")))

        rows = provider._get_cached_or_live_klines("BTCUSDT", "1d", 2, 3600)

    assert [row[0] for row in rows] == [2000, 3000]


def test_v5_optional_binance_request_degrades_on_rate_limit() -> None:
    provider = BinanceMarketDataProvider(StrategyConfig())
    provider.client = SimpleNamespace(get_json=Mock(side_effect=requests.HTTPError("429 rate limit")))

    payload = provider._get_optional_json("/fapi/v1/openInterest", {"symbol": "BTCUSDT"}, default={})

    assert payload == {}


def test_v5_5m_price_change_uses_stale_cache_on_rate_limit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = StrategyConfig(
            momo_structure_cache_dir=Path(tmp),
            momo_structure_cache_enabled=True,
            momo_1m_refresh_seconds=0,
        )
        cache_path = Path(tmp) / "5m" / "BTCUSDT.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text(
            json.dumps(
                [
                    [1000, "100", "100", "100", "100"],
                    [2000, "110", "110", "110", "110"],
                ]
            ),
            encoding="utf-8",
        )
        provider = BinanceMarketDataProvider(config)
        provider.client = SimpleNamespace(get_json=Mock(side_effect=requests.HTTPError("429 rate limit")))

        price_change = provider._get_price_change_pct_5m("BTCUSDT", 110.0)

    assert price_change == 10.0


def test_v5_5m_price_change_degrades_to_zero_without_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = StrategyConfig(momo_structure_cache_dir=Path(tmp), momo_structure_cache_enabled=True)
        provider = BinanceMarketDataProvider(config)
        provider.client = SimpleNamespace(get_json=Mock(side_effect=requests.HTTPError("429 rate limit")))

        price_change = provider._get_price_change_pct_5m("BTCUSDT", 110.0)

    assert price_change == 0.0


def test_binance_client_retries_transient_status_codes() -> None:
    client = BinanceFuturesClient("https://example.test")
    first = Mock(status_code=429, text="rate limit")
    second = Mock(status_code=200, text='{"ok": true}')
    second.json.return_value = {"ok": True}
    client.session.get = Mock(side_effect=[first, second])

    with patch("trading_system.binance.time.sleep") as sleep:
        payload = client.get_json("/fapi/v1/ticker/24hr", {"symbol": "AAAUSDT"})

    assert payload == {"ok": True}
    assert client.session.get.call_count == 2
    sleep.assert_called_once()


def test_short_scan_candidates_include_negative_movers_outside_long_filter() -> None:
    config = StrategyConfig(
        short_scan_min_quote_volume_usdt=20_000_000,
        short_scan_min_abs_price_change_pct=3.0,
    )
    symbol_meta = {"BEARUSDT": {}, "BULLUSDT": {}, "THINUSDT": {}}
    tickers = [
        {"symbol": "BULLUSDT", "priceChangePercent": "8.0", "quoteVolume": "90000000"},
        {"symbol": "BEARUSDT", "priceChangePercent": "-7.0", "quoteVolume": "80000000"},
        {"symbol": "THINUSDT", "priceChangePercent": "-30.0", "quoteVolume": "1000000"},
    ]

    ranked = _rank_short_scan_candidates(tickers, symbol_meta, {"BEARUSDT": -0.0002}, config)

    assert [item["symbol"] for item in ranked] == ["BEARUSDT"]


def test_merge_scan_candidates_preserves_long_candidates_and_adds_unique_shorts() -> None:
    long_ranked = [
        {"symbol": "AUSDT", "priceChangePercent": "12.0", "quoteVolume": "100000000"},
        {"symbol": "BUSDT", "priceChangePercent": "8.0", "quoteVolume": "90000000"},
    ]
    short_ranked = [
        {"symbol": "BUSDT", "priceChangePercent": "-6.0", "quoteVolume": "90000000"},
        {"symbol": "CUSDT", "priceChangePercent": "-9.0", "quoteVolume": "110000000"},
    ]

    merged = _merge_scan_candidates(long_ranked, short_ranked)

    assert [item["symbol"] for item in merged] == ["AUSDT", "BUSDT", "CUSDT"]
