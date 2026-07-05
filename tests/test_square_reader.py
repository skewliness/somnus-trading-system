from pathlib import Path
import json

from trading_system.square_reader import (
    _extract_post_id,
    _extract_ticker_mentions,
    _normalize_feed_post,
    _normalize_square_post,
    build_reader_from_export,
    parse_curl_export,
    parse_har_export,
    parse_request_export,
)


def test_extract_post_id_from_url() -> None:
    assert _extract_post_id("https://www.binance.com/en/square/post/24584125889874") == "24584125889874"


def test_normalize_square_post_extracts_pairs() -> None:
    post = _normalize_square_post(
        {
            "id": 123,
            "username": "tester",
            "displayName": "Tester",
            "bodyTextOnly": "hello {spot}(BONKUSDT)",
            "title": "",
            "lan": "en",
            "tradingPairs": [{"symbol": "BONKUSDT"}],
            "hashtagList": [{"name": "meme"}],
            "likeCount": 10,
            "commentCount": 2,
            "quoteCount": 1,
            "viewCount": 100,
            "bookmarkCount": 3,
            "firstReleaseTime": 1747907070000,
            "latestReleaseTime": 1747907092000,
            "webLink": "https://www.binance.com/en/square/post/123",
            "highQuality": True,
            "recommend": True,
            "isCreatedByAI": False,
        }
    )
    assert post.post_id == "123"
    assert post.trading_pairs == ["BONKUSDT"]
    assert post.bot_features["engagement_total"] == 16


def test_normalize_feed_post_extracts_content() -> None:
    post = _normalize_feed_post(
        {
            "id": "321",
            "username": "creator",
            "authorName": "Creator",
            "content": "btc is moving",
            "tradingPairs": [{"symbol": "BTCUSDT"}],
            "hashtagList": [{"name": "btc"}],
            "likeCount": 5,
            "commentCount": 2,
            "quoteCount": 1,
            "viewCount": 100,
            "date": 1775935117,
            "webLink": "https://www.binance.com/en/square/post/321",
            "cardType": "BUZZ_SHORT",
            "extraFeature": "NONE",
        }
    )
    assert post.post_id == "321"
    assert post.body_text == "btc is moving"
    assert post.trading_pairs == ["BTCUSDT"]


def test_extract_ticker_mentions_filters_stablecoin_noise() -> None:
    mentions = _extract_ticker_mentions("关注 $BTC、$ETH 和 $USDT，另外还有 $BUSD / $DOGE。")
    assert mentions == ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]


def test_normalize_feed_post_adds_regex_ticker_mentions() -> None:
    post = _normalize_feed_post(
        {
            "id": "654",
            "username": "creator",
            "authorName": "Creator",
            "content": "今晚重点看 $DOGE 和 $XRP，$USDC 过滤掉",
            "tradingPairs": [],
            "hashtagList": [],
            "likeCount": 5,
            "commentCount": 2,
            "quoteCount": 1,
            "viewCount": 100,
            "date": 1775935117,
            "webLink": "https://www.binance.com/en/square/post/654",
            "cardType": "BUZZ_SHORT",
            "extraFeature": "NONE",
        }
    )
    assert post.trading_pairs == ["DOGEUSDT", "XRPUSDT"]
    assert post.bot_features["regex_ticker_mentions"] == ["DOGEUSDT", "XRPUSDT"]


def test_parse_curl_export_reads_payload_and_cookie(tmp_path: Path) -> None:
    path = tmp_path / "curl.txt"
    path.write_text(
        "curl 'https://www.binance.com/test' -H 'accept: */*' -b 'a=1; b=2' --data-raw '{\"pageIndex\":1}'",
        encoding="utf-8",
    )
    url, headers, payload = parse_curl_export(path)
    assert url == "https://www.binance.com/test"
    assert headers["Cookie"] == "a=1; b=2"
    assert payload["pageIndex"] == 1


def test_parse_har_export_reads_headers_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "square.har"
    path.write_text(
        json.dumps(
            {
                "log": {
                    "entries": [
                        {
                            "request": {
                                "method": "POST",
                                "url": "https://www.binance.com/bapi/composite/v9/friendly/pgc/feed/feed-recommend/list",
                                "headers": [
                                    {"name": "accept", "value": "application/json"},
                                    {"name": "cookie", "value": "a=1; b=2"},
                                ],
                                "postData": {
                                    "text": json.dumps(
                                        {"pageIndex": 1, "pageSize": 20, "scene": "web-homepage", "contentIds": []}
                                    )
                                },
                            }
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    url, headers, payload = parse_har_export(path)
    assert url.endswith("/feed/feed-recommend/list")
    assert headers["cookie"] == "a=1; b=2"
    assert payload == {"pageIndex": 1, "pageSize": 20, "scene": "web-homepage", "contentIds": []}


def test_parse_request_export_and_build_reader_support_har(tmp_path: Path) -> None:
    path = tmp_path / "square.har"
    path.write_text(
        json.dumps(
            {
                "log": {
                    "entries": [
                        {
                            "request": {
                                "method": "POST",
                                "url": "https://www.binance.com/bapi/composite/v9/friendly/pgc/feed/feed-recommend/list",
                                "headers": [
                                    {"name": "accept", "value": "application/json"},
                                    {"name": "cookie", "value": "session=abc"},
                                ],
                                "postData": {
                                    "text": json.dumps(
                                        {"pageIndex": 2, "pageSize": 21, "scene": "web-homepage", "contentIds": []}
                                    )
                                },
                            }
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    url, headers, payload = parse_request_export(path)
    reader = build_reader_from_export(path)
    assert url.endswith("/feed/feed-recommend/list")
    assert headers["cookie"] == "session=abc"
    assert payload["pageIndex"] == 2
    assert reader.base_url == "https://www.binance.com"
    assert reader.session.headers["cookie"] == "session=abc"
