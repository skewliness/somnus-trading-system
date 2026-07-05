#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_JINA_URLS = [
    "https://r.jina.ai/http://r.jina.ai/http://https://www.binance.com/en/markets/ai-select",
    "https://r.jina.ai/http://r.jina.ai/http://https://www.binance.com/zh-CN/markets/ai-select",
]
DEFAULT_SENTIMENT_API = "https://www.binance.com/bapi/apex/v1/friendly/apex/web/opportunity/assets?interval=1h&type=sentiment"
DEFAULT_TECHNICAL_API = "https://www.binance.com/bapi/apex/v1/friendly/apex/web/opportunity/assets?interval=1h&type=technical"
DEFAULT_ALPHA_TOKEN_LIST_API = "https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/cex/alpha/all/token/list"
DEFAULT_OUTPUT = Path("data/trading_system/skill_bridges/binance-ai-select.json")

IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
SPACE_RE = re.compile(r"\s+")
SYMBOL_RE = re.compile(r"\b[0-9A-Z]{2,}(?:/USDT|USDT)?\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 SomnusAISelectBridge/1.0",
            "Accept": "text/markdown,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "clienttype": "web",
            "lang": "zh-CN",
            "referer": "https://www.binance.com/zh-CN/markets/ai-select",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def clean_cell(value: str) -> str:
    value = IMAGE_RE.sub(" ", value)
    value = value.replace("`", " ").replace("*", " ")
    return SPACE_RE.sub(" ", value).strip()


def symbol_from_name(value: str) -> str:
    cleaned = clean_cell(value).replace("-", " ").replace("_", " ")
    cleaned = re.sub(r"\bAlpha\b", " ", cleaned, flags=re.IGNORECASE)
    matches = [m.group(0).replace("/", "") for m in SYMBOL_RE.finditer(cleaned.upper())]
    matches = [m for m in matches if m not in {"USDT", "USD", "AI"}]
    if not matches:
        return ""
    raw = matches[0]
    if raw.endswith("USDT"):
        return raw
    if raw.endswith("USD"):
        return f"{raw[:-3]}USDT"
    return f"{raw}USDT"


def parse_rank(value: str) -> int | None:
    text = re.sub(r"[^0-9]", "", value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def parse_ai_select_markdown(text: str, source_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = text.splitlines()
    header: list[str] | None = None
    for line in lines:
        cells = split_markdown_row(line)
        if not cells:
            continue
        normalized = [clean_cell(cell).lower() for cell in cells]
        if "name" in normalized and "rank" in normalized and any("social" in cell for cell in normalized):
            header = normalized
            continue
        if header is None or is_separator_row(cells):
            continue
        if len(cells) < len(header):
            continue
        row = {header[i]: clean_cell(cells[i]) for i in range(min(len(header), len(cells)))}
        symbol = symbol_from_name(row.get("name", ""))
        if not symbol:
            continue
        signal = {
            "symbol": symbol,
            "name": clean_cell(row.get("name", "")),
            "rank": parse_rank(row.get("rank", "")),
            "ai_rating": row.get("ai rating") or row.get("ai评分") or row.get("rating") or "",
            "price": row.get("price") or row.get("价格") or "",
            "social_vol_24h": row.get("24h social vol") or row.get("24h social volume") or "",
            "community": row.get("social") or row.get("social sentiment") or row.get("community") or "",
            "news": row.get("news") or "",
            "kol": row.get("kol") or "",
            "source_section": "sentiment",
            "source_url": source_url,
        }
        rows.append(signal)
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row["symbol"]] = row
    return sorted(deduped.values(), key=lambda item: item.get("rank") or 999999)


def metric_label(metrics: dict[str, Any], key: str) -> str:
    item = metrics.get(key)
    if not isinstance(item, dict):
        return ""
    return str(item.get("valueLabel") or item.get("value") or "").strip()


def metric_value(metrics: dict[str, Any], key: str) -> str:
    item = metrics.get(key)
    if not isinstance(item, dict):
        return ""
    return str(item.get("value") or item.get("valueLabel") or "").strip()


def alpha_token_maps(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if str(payload.get("code") or "") not in {"000000", "0"}:
        raise RuntimeError(f"Binance Alpha token list returned code={payload.get('code')} message={payload.get('message')}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Binance Alpha token list response missing data list")
    by_alpha_id: dict[str, dict[str, Any]] = {}
    by_contract_chain: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or item.get("cexCoinName") or "").upper().strip()
        if not symbol:
            continue
        alpha_id = str(item.get("alphaId") or "").upper().strip()
        contract = str(item.get("contractAddress") or "").lower().strip()
        chain_id = str(item.get("chainId") or "").strip()
        mapped = {
            "symbol": symbol,
            "name": item.get("name") or symbol,
            "alpha_id": item.get("alphaId"),
            "contract_address": item.get("contractAddress"),
            "chain_id": item.get("chainId"),
        }
        if alpha_id:
            by_alpha_id[alpha_id] = mapped
        if contract and chain_id:
            by_contract_chain[f"{contract}@{chain_id}"] = mapped
    return by_alpha_id, by_contract_chain


def resolve_item_asset(item: dict[str, Any], alpha_by_id: dict[str, dict[str, Any]], alpha_by_contract: dict[str, dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    raw_base = str(item.get("baseAsset") or item.get("asset") or "").strip()
    asset_type = str(item.get("assetType") or "").upper().strip()
    alpha_id = str(item.get("alphaId") or "").upper().strip()
    alpha_meta: dict[str, Any] = {}
    if asset_type == "ALPHA":
        alpha_meta = alpha_by_id.get(alpha_id, {})
        if not alpha_meta and "@" in raw_base:
            contract, chain_id = raw_base.lower().split("@", 1)
            alpha_meta = alpha_by_contract.get(f"{contract}@{chain_id}", {})
        if alpha_meta.get("symbol"):
            return str(alpha_meta["symbol"]).upper(), str(alpha_meta.get("name") or alpha_meta["symbol"]), alpha_meta
    return raw_base.upper(), raw_base.upper(), alpha_meta


def parse_api_items(
    payload: dict[str, Any],
    source_url: str,
    section: str,
    limit: int,
    alpha_by_id: dict[str, dict[str, Any]],
    alpha_by_contract: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if str(payload.get("code") or "") not in {"000000", "0"}:
        raise RuntimeError(f"Binance AI Select API returned code={payload.get('code')} message={payload.get('message')}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Binance AI Select API response missing data object")
    items = data.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Binance AI Select API response missing data.items list")
    rows: list[dict[str, Any]] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        base, display_name, alpha_meta = resolve_item_asset(item, alpha_by_id, alpha_by_contract)
        quote = str(item.get("quoteAsset") or "USDT").upper().strip()
        if not base:
            continue
        symbol = f"{base}{quote}" if quote else symbol_from_name(base)
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        common = {
            "symbol": symbol,
            "asset": base,
            "name": display_name,
            "asset_type": item.get("assetType"),
            "alpha_id": item.get("alphaId"),
            "alpha_contract_asset": item.get("asset") if item.get("assetType") == "ALPHA" else "",
            "alpha_name": alpha_meta.get("name") or "",
            "rank": item.get("rank"),
            "source_url": source_url,
            "source_section": section,
            "timestamp": item.get("timestamp"),
        }
        if section == "sentiment":
            common.update(
                {
                    "sentiment_rank": item.get("rank"),
                    "ai_rating": metric_label(metrics, "sentiment_score"),
                    "ai_rating_score": metric_value(metrics, "sentiment_score"),
                    "community": metric_label(metrics, "sentiment_score_social"),
                    "community_score": metric_value(metrics, "sentiment_score_social"),
                    "news": metric_label(metrics, "sentiment_score_news"),
                    "news_score": metric_value(metrics, "sentiment_score_news"),
                    "kol": metric_label(metrics, "sentiment_score_kol"),
                    "kol_score": metric_value(metrics, "sentiment_score_kol"),
                    "social_vol_24h": metric_value(metrics, "sentiment_24h_social_volume"),
                    "social_volume_signal": metric_label(metrics, "sentiment_score_volume"),
                    "social_volume_score": metric_value(metrics, "sentiment_score_volume"),
                }
            )
        else:
            common.update(
                {
                    "technical_rank": item.get("rank"),
                    "ai_rating": metric_label(metrics, "technical_score_1h"),
                    "ai_rating_score": metric_value(metrics, "technical_score_1h"),
                    "technical_ai_rating": metric_label(metrics, "technical_score_1h"),
                    "technical_ai_score": metric_value(metrics, "technical_score_1h"),
                    "ma50_ma200": metric_label(metrics, "technical_ind_ma_signal_1h"),
                    "ma50_ma200_score": metric_value(metrics, "technical_ind_ma_signal_1h"),
                    "adx": metric_label(metrics, "technical_ind_adx_signal_1h"),
                    "adx_score": metric_value(metrics, "technical_ind_adx_signal_1h"),
                    "macd": metric_label(metrics, "technical_ind_macd_signal_1h"),
                    "macd_score": metric_value(metrics, "technical_ind_macd_signal_1h"),
                    "rsi": metric_label(metrics, "technical_ind_rsi_signal_1h"),
                    "rsi_score": metric_value(metrics, "technical_ind_rsi_signal_1h"),
                    "bbands": metric_label(metrics, "technical_ind_bbands_signal_1h"),
                    "bbands_score": metric_value(metrics, "technical_ind_bbands_signal_1h"),
                    "volume_signal": metric_label(metrics, "technical_ind_volume_signal_1h"),
                    "volume_score": metric_value(metrics, "technical_ind_volume_signal_1h"),
                }
            )
        rows.append(common)
    return rows


def merge_signals(sentiment_rows: list[dict[str, Any]], technical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in sentiment_rows:
        symbol = str(row.get("symbol") or "").upper()
        if symbol:
            merged[symbol] = dict(row)
    for row in technical_rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        current = merged.setdefault(symbol, {"symbol": symbol, "asset": row.get("asset"), "name": row.get("name")})
        had_sentiment = current.get("source_section") == "sentiment"
        current.update({k: v for k, v in row.items() if v not in (None, "")})
        if had_sentiment:
            current["source_section"] = "sentiment+technical"
    return sorted(
        merged.values(),
        key=lambda item: min(
            int(item.get("sentiment_rank") or item.get("rank") or 999999),
            int(item.get("technical_rank") or 999999),
        ),
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def write_error(output: Path, error: str, *, errors: list[dict[str, str]] | None = None) -> None:
    atomic_write_json(
        output,
        {
            "updated_at": utc_now(),
            "source": "binance_ai_select",
            "status": "error",
            "row_count": 0,
            "signals": [],
            "error": error,
            "errors": errors or [],
        },
    )


def refresh_api(
    sentiment_url: str,
    technical_url: str,
    alpha_token_list_url: str,
    output: Path,
    timeout: float,
    limit: int,
    min_rows: int,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    sentiment_rows: list[dict[str, Any]] = []
    technical_rows: list[dict[str, Any]] = []
    alpha_by_id: dict[str, dict[str, Any]] = {}
    alpha_by_contract: dict[str, dict[str, Any]] = {}
    try:
        alpha_by_id, alpha_by_contract = alpha_token_maps(fetch_json(alpha_token_list_url, timeout))
    except Exception as exc:
        errors.append({"url": alpha_token_list_url, "error": f"{type(exc).__name__}: {exc}"})
    try:
        sentiment_rows = parse_api_items(fetch_json(sentiment_url, timeout), sentiment_url, "sentiment", limit, alpha_by_id, alpha_by_contract)
    except Exception as exc:
        errors.append({"url": sentiment_url, "error": f"{type(exc).__name__}: {exc}"})
    try:
        technical_rows = parse_api_items(fetch_json(technical_url, timeout), technical_url, "technical", limit, alpha_by_id, alpha_by_contract)
    except Exception as exc:
        errors.append({"url": technical_url, "error": f"{type(exc).__name__}: {exc}"})
    rows = merge_signals(sentiment_rows, technical_rows)
    if len(rows) < min_rows:
        error = f"AI Select API rows below minimum: rows={len(rows)} min_rows={min_rows}"
        write_error(output, error, errors=errors)
        raise RuntimeError(f"{error} errors={errors}")
    payload = {
        "updated_at": utc_now(),
        "source": "binance_ai_select",
        "mode": "binance_opportunity_assets_api",
        "status": "ok",
        "limit": limit,
        "row_count": len(rows),
        "sentiment_count": len(sentiment_rows),
        "technical_count": len(technical_rows),
        "alpha_map_count": len(alpha_by_id),
        "source_urls": {
            "sentiment": sentiment_url,
            "technical": technical_url,
            "alpha_token_list": alpha_token_list_url,
        },
        "signals": rows,
        "errors": errors,
    }
    atomic_write_json(output, payload)
    return payload


def refresh_jina(urls: list[str], output: Path, timeout: float, min_rows: int) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    best_rows: list[dict[str, Any]] = []
    best_url = ""
    for url in urls:
        try:
            text = fetch_text(url, timeout)
            rows = parse_ai_select_markdown(text, url)
        except Exception as exc:
            errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if len(rows) > len(best_rows):
            best_rows = rows
            best_url = url
    if len(best_rows) < min_rows:
        error = f"AI Select rows below minimum: rows={len(best_rows)} min_rows={min_rows}"
        write_error(output, error, errors=errors)
        raise RuntimeError(f"{error} errors={errors}")
    payload = {
        "updated_at": utc_now(),
        "source": "binance_ai_select",
        "source_url": best_url,
        "mode": "jina_rendered_sentiment_page",
        "status": "ok",
        "technical_status": "unavailable_from_rendered_page",
        "row_count": len(best_rows),
        "signals": best_rows,
        "errors": errors,
    }
    atomic_write_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh Binance AI Select bridge data for V5 final filtering.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=50, help="Read at least this many rows from each AI Select board where available.")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--sentiment-url", default=DEFAULT_SENTIMENT_API)
    parser.add_argument("--technical-url", default=DEFAULT_TECHNICAL_API)
    parser.add_argument("--alpha-token-list-url", default=DEFAULT_ALPHA_TOKEN_LIST_API)
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--source", choices=("api", "jina"), default="api")
    args = parser.parse_args(argv)
    if args.source == "api":
        payload = refresh_api(args.sentiment_url, args.technical_url, args.alpha_token_list_url, args.output, args.timeout, args.limit, args.min_rows)
    else:
        urls = args.url or DEFAULT_JINA_URLS
        payload = refresh_jina(urls, args.output, args.timeout, args.min_rows)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "row_count": payload["row_count"],
                "sentiment_count": payload.get("sentiment_count"),
                "technical_count": payload.get("technical_count"),
                "alpha_map_count": payload.get("alpha_map_count"),
                "symbols": [row["symbol"] for row in payload["signals"]],
                "mode": payload.get("mode"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
