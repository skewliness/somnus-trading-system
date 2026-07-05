from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import requests


STABLECOIN_NOISE = {"USD", "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDE"}


class SquareAccessError(RuntimeError):
    """Raised when Binance Square content cannot be fetched with current access."""


@dataclass(slots=True)
class SquarePost:
    post_id: str
    username: str
    display_name: str
    body_text: str
    title: str
    lang: str
    trading_pairs: list[str]
    hashtag_list: list[str]
    like_count: int
    comment_count: int
    quote_count: int
    view_count: int
    bookmark_count: int
    first_release_time: str | None
    latest_release_time: str | None
    web_link: str
    high_quality: bool
    recommend: bool
    is_created_by_ai: bool
    raw_path: str | None = None
    bot_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BinanceSquareReader:
    def __init__(self, cookie_header: str | None = None, csrf_token: str | None = None) -> None:
        self.base_url = "https://www.binance.com"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://www.binance.com/en/square",
                "Origin": "https://www.binance.com",
            }
        )
        if cookie_header:
            self.session.headers["Cookie"] = cookie_header
        if csrf_token:
            self.session.headers["x-csrf-token"] = csrf_token

    def fetch_post_detail(self, post_id_or_url: str, save_raw_dir: Path | None = None) -> SquarePost:
        post_id = _extract_post_id(post_id_or_url)
        url = f"{self.base_url}/bapi/composite/v2/friendly/pgc/special/content/detail/{post_id}"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise SquareAccessError(f"detail lookup failed for post {post_id}: {payload.get('message')}")
        data = payload["data"]
        raw_path = None
        if save_raw_dir is not None:
            save_raw_dir.mkdir(parents=True, exist_ok=True)
            path = save_raw_dir / f"{post_id}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            raw_path = str(path)
        return _normalize_square_post(data, raw_path)

    def fetch_feed_recommend(
        self,
        page_index: int = 1,
        page_size: int = 20,
        scene: int | str = "web-homepage",
        content_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/bapi/composite/v9/friendly/pgc/feed/feed-recommend/list"
        payload = {
            "pageIndex": page_index,
            "pageSize": page_size,
            "scene": scene,
            "contentIds": content_ids or [],
        }
        response = self.session.post(url, json=payload, timeout=30)
        if response.status_code == 403:
            raise SquareAccessError(
                "feed-recommend returned 403. This usually means Binance WAF or session validation blocked the request. "
                "Provide fresh BINANCE_COOKIE_HEADER and BINANCE_CSRF_TOKEN, or pass a recent browser curl export file."
            )
        response.raise_for_status()
        data = response.json()
        if not data.get("success"):
            raise SquareAccessError(f"feed-recommend failed: {data.get('message')}")
        return data

    def fetch_feed_posts(
        self,
        page_index: int = 1,
        page_size: int = 20,
        scene: int | str = "web-homepage",
        content_ids: list[str] | None = None,
    ) -> list[SquarePost]:
        payload = self.fetch_feed_recommend(
            page_index=page_index,
            page_size=page_size,
            scene=scene,
            content_ids=content_ids,
        )
        return [_normalize_feed_post(item) for item in payload.get("data", {}).get("vos", [])]

    @classmethod
    def from_curl_export(cls, path: Path) -> "BinanceSquareReader":
        url, headers, _payload = parse_curl_export(path)
        parsed = urlparse(url)
        reader = cls()
        reader.base_url = f"{parsed.scheme}://{parsed.netloc}"
        reader.session.headers.clear()
        reader.session.headers.update(headers)
        return reader

    @classmethod
    def from_har_export(cls, path: Path) -> "BinanceSquareReader":
        url, headers, _payload = parse_har_export(path)
        parsed = urlparse(url)
        reader = cls()
        reader.base_url = f"{parsed.scheme}://{parsed.netloc}"
        reader.session.headers.clear()
        reader.session.headers.update(headers)
        return reader


def analyze_posts(posts: list[SquarePost]) -> dict[str, Any]:
    symbols: dict[str, int] = {}
    authors: dict[str, int] = {}
    for post in posts:
        authors[post.username] = authors.get(post.username, 0) + 1
        for symbol in post.trading_pairs:
            symbols[symbol] = symbols.get(symbol, 0) + 1

    ranked_symbols = sorted(symbols.items(), key=lambda item: item[1], reverse=True)
    ranked_authors = sorted(authors.items(), key=lambda item: item[1], reverse=True)
    return {
        "post_count": len(posts),
        "top_symbols": [{"symbol": symbol, "mentions": count} for symbol, count in ranked_symbols[:20]],
        "top_authors": [{"username": username, "posts": count} for username, count in ranked_authors[:20]],
        "high_quality_posts": sum(1 for post in posts if post.high_quality),
        "ai_generated_posts": sum(1 for post in posts if post.is_created_by_ai),
        "avg_views": round(sum(post.view_count for post in posts) / len(posts), 2) if posts else 0.0,
        "avg_engagement": round(
            sum(post.like_count + post.comment_count + post.quote_count + post.bookmark_count for post in posts) / len(posts),
            2,
        ) if posts else 0.0,
        "avg_engagement_view_ratio": round(
            sum(float(post.bot_features.get("engagement_view_ratio", 0.0)) for post in posts) / len(posts),
            6,
        ) if posts else 0.0,
    }


def _normalize_square_post(data: dict[str, Any], raw_path: str | None = None) -> SquarePost:
    trading_pairs = []
    for item in data.get("tradingPairs") or []:
        symbol = item.get("symbol")
        if symbol:
            trading_pairs.append(symbol)
    regex_mentions = _extract_ticker_mentions(f"{data.get('title') or ''} {data.get('bodyTextOnly') or ''}")
    trading_pairs = list(dict.fromkeys([*trading_pairs, *regex_mentions]))
    hashtag_list = []
    for item in data.get("hashtagList") or []:
        if isinstance(item, str):
            name = item
        else:
            name = item.get("name") or item.get("hashtagName")
        if name:
            hashtag_list.append(str(name))
    body_text = (data.get("bodyTextOnly") or "").strip()
    body_text = " ".join(body_text.split())
    return SquarePost(
        post_id=str(data.get("id")),
        username=str(data.get("username") or ""),
        display_name=str(data.get("displayName") or ""),
        body_text=body_text,
        title=str(data.get("title") or ""),
        lang=str(data.get("lan") or ""),
        trading_pairs=trading_pairs,
        hashtag_list=hashtag_list,
        like_count=int(data.get("likeCount") or 0),
        comment_count=int(data.get("commentCount") or 0),
        quote_count=int(data.get("quoteCount") or 0),
        view_count=int(data.get("viewCount") or 0),
        bookmark_count=int(data.get("bookmarkCount") or 0),
        first_release_time=_to_iso(data.get("firstReleaseTime")),
        latest_release_time=_to_iso(data.get("latestReleaseTime")),
        web_link=str(data.get("webLink") or ""),
        high_quality=bool(data.get("highQuality")),
        recommend=bool(data.get("recommend")),
        is_created_by_ai=bool(data.get("isCreatedByAI")),
        raw_path=raw_path,
        bot_features={
            "engagement_total": int(data.get("likeCount") or 0)
            + int(data.get("commentCount") or 0)
            + int(data.get("quoteCount") or 0)
            + int(data.get("bookmarkCount") or 0),
            "engagement_view_ratio": _ratio(
                int(data.get("likeCount") or 0)
                + int(data.get("commentCount") or 0)
                + int(data.get("quoteCount") or 0)
                + int(data.get("bookmarkCount") or 0),
                int(data.get("viewCount") or 0),
            ),
            "text_length": len(body_text),
            "pair_count": len(trading_pairs),
            "regex_ticker_mentions": regex_mentions,
            "hashtag_count": len(hashtag_list),
        },
    )


def _normalize_feed_post(data: dict[str, Any]) -> SquarePost:
    trading_pairs = []
    for item in data.get("tradingPairs") or []:
        symbol = item.get("symbol")
        if symbol:
            trading_pairs.append(symbol)
    hashtag_list = []
    for item in data.get("hashtagList") or []:
        if isinstance(item, str):
            name = item
        else:
            name = item.get("name") or item.get("hashtagName")
        if name:
            hashtag_list.append(str(name))
    body_text = str(data.get("content") or "").strip()
    body_text = " ".join(body_text.split())
    regex_mentions = _extract_ticker_mentions(f"{data.get('title') or ''} {body_text}")
    trading_pairs = list(dict.fromkeys([*trading_pairs, *regex_mentions]))
    return SquarePost(
        post_id=str(data.get("id")),
        username=str(data.get("username") or ""),
        display_name=str(data.get("authorName") or ""),
        body_text=body_text,
        title=str(data.get("title") or ""),
        lang="",
        trading_pairs=trading_pairs,
        hashtag_list=hashtag_list,
        like_count=int(data.get("likeCount") or 0),
        comment_count=int(data.get("commentCount") or 0),
        quote_count=int(data.get("quoteCount") or 0),
        view_count=int(data.get("viewCount") or 0),
        bookmark_count=int(data.get("bookmarkCount") or 0),
        first_release_time=_from_seconds_to_iso(data.get("date")),
        latest_release_time=_from_seconds_to_iso(data.get("date")),
        web_link=str(data.get("webLink") or ""),
        high_quality=bool(data.get("highQuality")),
        recommend=True,
        is_created_by_ai=bool(data.get("isCreatedByAI")),
        raw_path=None,
        bot_features={
            "engagement_total": int(data.get("likeCount") or 0)
            + int(data.get("commentCount") or 0)
            + int(data.get("quoteCount") or 0)
            + int(data.get("bookmarkCount") or 0),
            "engagement_view_ratio": _ratio(
                int(data.get("likeCount") or 0)
                + int(data.get("commentCount") or 0)
                + int(data.get("quoteCount") or 0)
                + int(data.get("bookmarkCount") or 0),
                int(data.get("viewCount") or 0),
            ),
            "text_length": len(body_text),
            "pair_count": len(trading_pairs),
            "regex_ticker_mentions": regex_mentions,
            "hashtag_count": len(hashtag_list),
            "card_type": str(data.get("cardType") or ""),
            "extra_feature": str(data.get("extraFeature") or ""),
        },
    )


def _extract_post_id(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value
    parsed = urlparse(value)
    parts = [part for part in parsed.path.split("/") if part]
    if "post" in parts:
        idx = parts.index("post")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError(f"cannot extract Square post id from: {value}")


def _extract_ticker_mentions(text: str) -> list[str]:
    symbols: list[str] = []
    for raw in re.findall(r"\$([A-Za-z][A-Za-z0-9]{1,14})\b", text or ""):
        symbol = raw.upper()
        if symbol in STABLECOIN_NOISE:
            continue
        normalized = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
        if normalized not in symbols:
            symbols.append(normalized)
    return symbols


def _to_iso(value: Any) -> str | None:
    if not value:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat()


def _from_seconds_to_iso(value: Any) -> str | None:
    if not value:
        return None
    return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def load_auth_from_env() -> tuple[str | None, str | None]:
    return os.getenv("BINANCE_COOKIE_HEADER"), os.getenv("BINANCE_CSRF_TOKEN")


def parse_request_export(path: Path) -> tuple[str, dict[str, str], dict[str, Any]]:
    if path.suffix.lower() == ".har":
        return parse_har_export(path)
    return parse_curl_export(path)


def build_reader_from_export(path: Path) -> BinanceSquareReader:
    if path.suffix.lower() == ".har":
        return BinanceSquareReader.from_har_export(path)
    return BinanceSquareReader.from_curl_export(path)


def parse_curl_export(path: Path) -> tuple[str, dict[str, str], dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    url_match = re.search(r"curl '([^']+)'", text)
    if not url_match:
        raise ValueError("curl export file does not contain a curl URL")
    url = url_match.group(1)
    headers: dict[str, str] = {}
    for match in re.finditer(r"-H '([^:']+):\s*([^']*)'", text):
        headers[match.group(1)] = match.group(2)
    cookie_match = re.search(r"-b '([^']+)'", text, re.S)
    if cookie_match:
        headers["Cookie"] = cookie_match.group(1)
    payload_match = re.search(r"--data-raw '(.+)'", text, re.S)
    payload = json.loads(payload_match.group(1)) if payload_match else {}
    return url, headers, payload


def parse_har_export(path: Path) -> tuple[str, dict[str, str], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("log", {}).get("entries", [])
    target_path = "/bapi/composite/v9/friendly/pgc/feed/feed-recommend/list"
    for entry in entries:
        request = entry.get("request") or {}
        if request.get("method") != "POST":
            continue
        url = str(request.get("url") or "")
        if target_path not in url:
            continue
        headers = {
            str(item.get("name")): str(item.get("value") or "")
            for item in request.get("headers") or []
            if item.get("name")
        }
        post_data = request.get("postData") or {}
        text = post_data.get("text") or ""
        payload = json.loads(text) if text else {}
        return url, headers, payload
    raise ValueError("HAR file does not contain Binance Square feed-recommend POST request")
