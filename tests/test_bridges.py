from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_system.bridges import BinanceTopMoversBridge, CoinGeckoTrendingBridge, ManualSentimentBridge, OpenNewsBridge, OpenTwitterBridge, OptionsDeliveryBridge, SquareFeedBridge
from trading_system.issues import build_default_issue_registry, sync_live_access_issue


def test_manual_sentiment_bridge_reads_csv(tmp_path: Path) -> None:
    path = tmp_path / "sentiment.csv"
    path.write_text(
        "symbol,hotness_score,mention_count,source,updated_at,notes\nAIXBTUSDT,90,10,manual_distill,2026-04-12T00:00:00+00:00,test\n",
        encoding="utf-8",
    )
    result = ManualSentimentBridge().load(path)
    assert result.metadata["status"] == "ok"
    assert len(result.records) == 1
    assert result.records[0].symbol == "AIXBTUSDT"


def test_square_feed_bridge_derives_symbol_scores(tmp_path: Path) -> None:
    path = tmp_path / "square_feed_analysis.json"
    path.write_text(
        """
{
  "posts": [
    {
      "trading_pairs": ["BONKUSDT", "SHIBUSDT"],
      "view_count": 1000,
      "bot_features": {"engagement_total": 30, "engagement_view_ratio": 0.03}
    },
    {
      "trading_pairs": ["BONKUSDT"],
      "view_count": 500,
      "bot_features": {"engagement_total": 10, "engagement_view_ratio": 0.02}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    result = SquareFeedBridge().load(path)
    scores = {item.symbol: item.hotness_score for item in result.records}
    assert scores["BONKUSDT"] >= scores["SHIBUSDT"]


def test_square_feed_bridge_downweights_stale_report(tmp_path: Path) -> None:
    path = tmp_path / "square_feed_analysis.json"
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    path.write_text(
        f'''{{"summary": {{"captured_at": "{old_ts}", "status": "ok"}}, "posts": [{{"trading_pairs": ["BONKUSDT"], "view_count": 1000, "bot_features": {{"engagement_total": 30}}}}]}}''',
        encoding="utf-8",
    )

    result = SquareFeedBridge(max_age_minutes=30, stale_hotness_multiplier=0.25).load(path)

    assert result.metadata["status"] == "stale"
    assert result.records[0].hotness_score == 25.0
    assert "stale_square_feed" in result.records[0].notes


def test_sync_live_access_issue_only_exists_on_failure() -> None:
    registry = build_default_issue_registry()
    assert all(item.code != "BINANCE_FUTURES_LIVE_BLOCKED" for item in registry.items())

    sync_live_access_issue(registry, RuntimeError("HTTP 451 restricted location"))
    codes_after_error = {item.code for item in registry.items()}
    assert "BINANCE_FUTURES_LIVE_BLOCKED" in codes_after_error

    sync_live_access_issue(registry)
    codes_after_success = {item.code for item in registry.items()}
    assert "BINANCE_FUTURES_LIVE_BLOCKED" not in codes_after_success


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_opennews_bridge_normalizes_records(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "data": [
            {
                "ts": "2026-04-18T14:17:00Z",
                "engineType": "listing",
                "aiRating": {"score": 88, "summary": "AIXBT 上线催化"},
                "coins": [{"symbol": "AIXBT", "market_type": "cex"}, {"symbol": "USDT", "market_type": "cex"}],
            }
        ]
    }

    monkeypatch.setattr("trading_system.bridges.requests.post", lambda *args, **kwargs: _Resp(payload))
    result = OpenNewsBridge(token="x", snapshot_path=tmp_path / "opennews.json").fetch_for_symbols(["AIXBTUSDT"])
    assert result.metadata["status"] == "ok"
    assert result.records[0].symbol == "AIXBTUSDT"
    assert result.records[0].source == "opennews_listing"
    assert result.records[0].mention_count == 1


def test_opentwitter_bridge_extracts_targeted_mentions(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "data": [
            {
                "id": "1",
                "text": "$AIXBT breakout setting up now",
                "createdAt": "Sat Apr 18 14:30:45 +0000 2026",
                "favoriteCount": 40,
                "retweetCount": 5,
                "replyCount": 2,
                "userFollowers": 12000,
                "userScreenName": "trader_a",
            }
        ]
    }

    monkeypatch.setattr("trading_system.bridges.requests.post", lambda *args, **kwargs: _Resp(payload))
    result = OpenTwitterBridge(token="x", snapshot_path=tmp_path / "opentwitter.json").fetch_for_symbols(["AIXBTUSDT"])
    assert result.metadata["status"] == "ok"
    assert result.records[0].symbol == "AIXBTUSDT"
    assert result.records[0].source == "opentwitter_search"
    assert result.records[0].mention_count == 1


def test_options_delivery_bridge_collects_binance_okx_as_news_source(tmp_path: Path, monkeypatch) -> None:
    def fake_get(url, **kwargs):
        if "eapi.binance.com" in url:
            return _Resp(
                [
                    {
                        "symbol": "BTC-260625-68000-P",
                        "strikePrice": "68000",
                        "realStrikePrice": "61878.579",
                        "expiryDate": 1782374400000,
                        "strikeResult": "REALISTIC_VALUE_STRICKEN",
                    },
                    {
                        "symbol": "BTC-260625-70000-C",
                        "strikePrice": "70000",
                        "realStrikePrice": "61878.579",
                        "expiryDate": 1782374400000,
                        "strikeResult": "EXTRINSIC_VALUE_EXPIRED",
                    },
                ]
            )
        if "okx.com" in url:
            return _Resp(
                {
                    "code": "0",
                    "data": [
                        {
                            "ts": "1782374400000",
                            "details": [
                                {"insId": "BTC-USD-260625-60000-C", "px": "61788.32", "type": "exercised"},
                                {"insId": "BTC-USD-260625-62000-P", "px": "61788.32", "type": "expired_otm"},
                            ],
                        }
                    ],
                }
            )
        raise AssertionError(f"unexpected url {url}")

    snapshot = tmp_path / "options_delivery.json"
    monkeypatch.setattr("trading_system.bridges.requests.get", fake_get)

    result = OptionsDeliveryBridge(snapshot_path=snapshot, underlyings=("BTCUSDT",), limit=5).load()

    assert result.metadata["status"] == "ok"
    assert result.metadata["count"] == 2
    assert {record.source for record in result.records} == {"options_delivery_binance", "options_delivery_okx"}
    assert all(record.symbol == "BTCUSDT" for record in result.records)
    assert {record.updated_at.date().isoformat() for record in result.records} == {"2026-06-25"}
    assert any("Binance期权交割" in record.notes for record in result.records)
    assert any("OKX期权交割" in record.notes for record in result.records)
    assert snapshot.exists()
    payload = snapshot.read_text(encoding="utf-8")
    assert "options_delivery" in payload
    assert "BTCUSDT" in payload
