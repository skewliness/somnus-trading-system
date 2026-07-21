from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from trading_system.binance import BinanceFuturesClient
from trading_system.persistence import atomic_write_json, atomic_write_text, load_json


_SOLD_RATIO_RE = re.compile(r"sold_ratio=([0-9]+(?:\.[0-9]+)?)")
_CHAIN_RE = re.compile(r"^(solana|ethereum|base|bsc|arbitrum|optimism|polygon)\b", re.IGNORECASE)


@dataclass(slots=True)
class WhaleOverviewConfig:
    symbol: str
    base_url: str = "https://fapi.binance.com"
    smart_money_base_url: str = "https://www.binance.com"
    skill_bridge_dir: Path = Path("data/trading_system/skill_bridges")
    output_json: Path = Path("data/trading_system/whale_overview.json")
    output_html: Path = Path("data/trading_system/whale_overview.html")
    max_binance_whales: int = 277
    min_binance_whales: int = 24
    whale_notional_unit_usdt: float = 72_000.0


def build_whale_overview(
    config: WhaleOverviewConfig,
    *,
    client: BinanceFuturesClient | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Build a smart-money/whale overview panel payload.

    Prefer Binance public smart-money signal overview aggregates when available, and
    fall back to Binance futures market structure + OKX bridge heuristics otherwise.
    """
    symbol = _normalize_symbol(config.symbol)
    market_client = client or BinanceFuturesClient(config.base_url)
    signal_client = client or BinanceFuturesClient(config.smart_money_base_url)
    market = _fetch_binance_market(symbol, market_client)
    signal = _fetch_binance_smart_money_signal(symbol, signal_client)
    okx = _read_okx_signal_context(config.skill_bridge_dir, symbol)
    overview = _compose_overview(symbol, market, signal, okx, config)
    if persist:
        atomic_write_json(config.output_json, overview)
        atomic_write_text(config.output_html, render_whale_overview_html(overview))
    return overview


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper().replace("-", "").replace("/", "")
    if not normalized.endswith("USDT"):
        normalized += "USDT"
    return normalized


def _fetch_binance_market(symbol: str, client: BinanceFuturesClient) -> dict[str, Any]:
    premium = client.get_json("/fapi/v1/premiumIndex", {"symbol": symbol})
    ticker = client.get_json("/fapi/v1/ticker/24hr", {"symbol": symbol})
    open_interest = client.get_json("/fapi/v1/openInterest", {"symbol": symbol})
    oi_hist = client.get_json("/futures/data/openInterestHist", {"symbol": symbol, "period": "4h", "limit": 2})
    global_ratio = client.get_json("/futures/data/globalLongShortAccountRatio", {"symbol": symbol, "period": "1h", "limit": 1})
    top_ratio = client.get_json("/futures/data/topLongShortAccountRatio", {"symbol": symbol, "period": "1h", "limit": 1})
    taker_ratio = client.get_json("/futures/data/takerlongshortRatio", {"symbol": symbol, "period": "1h", "limit": 1})

    mark_price = float(premium.get("markPrice") or ticker.get("lastPrice") or 0.0)
    return {
        "mark_price": mark_price,
        "index_price": float(premium.get("indexPrice") or 0.0),
        "funding_rate": float(premium.get("lastFundingRate") or 0.0),
        "price_change_pct_24h": float(ticker.get("priceChangePercent") or 0.0),
        "quote_volume_usdt_24h": float(ticker.get("quoteVolume") or 0.0),
        "open_interest_qty": float(open_interest.get("openInterest") or 0.0),
        "open_interest_usdt": float(open_interest.get("openInterest") or 0.0) * mark_price,
        "oi_change_pct_4h": _pct_change_from_oi_hist(oi_hist),
        "global_long_account_ratio": _latest_float(global_ratio, "longAccount"),
        "global_short_account_ratio": _latest_float(global_ratio, "shortAccount"),
        "global_long_short_ratio": _latest_float(global_ratio, "longShortRatio"),
        "top_long_account_ratio": _latest_float(top_ratio, "longAccount"),
        "top_short_account_ratio": _latest_float(top_ratio, "shortAccount"),
        "top_long_short_ratio": _latest_float(top_ratio, "longShortRatio"),
        "taker_buy_sell_ratio": _latest_float(taker_ratio, "buySellRatio"),
    }


def _fetch_binance_smart_money_signal(symbol: str, client: BinanceFuturesClient) -> dict[str, Any]:
    try:
        payload = client.get_json("/bapi/futures/v1/public/future/smart-money/signal/overview", {"symbol": symbol})
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _latest_float(rows: Any, key: str, default: float = 0.0) -> float:
    if isinstance(rows, list) and rows:
        return float(rows[-1].get(key) or default)
    return default


def _pct_change_from_oi_hist(rows: Any) -> float:
    if not isinstance(rows, list) or len(rows) < 2:
        return 0.0
    first = float(rows[0].get("sumOpenInterestValue") or rows[0].get("sumOpenInterest") or 0.0)
    last = float(rows[-1].get("sumOpenInterestValue") or rows[-1].get("sumOpenInterest") or 0.0)
    if first <= 0:
        return 0.0
    return ((last - first) / first) * 100.0


def _read_okx_signal_context(skill_bridge_dir: Path, symbol: str) -> dict[str, Any]:
    base = symbol.removesuffix("USDT")
    files = [skill_bridge_dir / "okx-dex-signal.json", skill_bridge_dir / "okx-dex-market.json", skill_bridge_dir / "okx-dex-token.json"]
    matches: list[dict[str, Any]] = []
    for path in files:
        payload = load_json(path, {"signals": []}, lock=False)
        if not isinstance(payload, dict):
            continue
        for item in payload.get("signals", []):
            item_symbol = str(item.get("symbol") or "").upper()
            if item_symbol in {base, symbol}:
                notes = str(item.get("notes") or "")
                match = dict(item)
                match["bridge_file"] = path.name
                match["sold_ratio"] = _extract_sold_ratio(notes)
                match["chain"] = _extract_chain(notes)
                matches.append(match)
    sold = [float(item["sold_ratio"]) for item in matches if item.get("sold_ratio") is not None]
    hotness = sum(float(item.get("hotness_score") or 0.0) for item in matches)
    mentions = sum(int(float(item.get("mention_count") or 0)) for item in matches)
    avg_sold_ratio = sum(sold) / len(sold) if sold else None
    return {
        "matches": matches,
        "match_count": len(matches),
        "hotness_score": round(hotness, 4),
        "mention_count": mentions,
        "avg_sold_ratio": avg_sold_ratio,
        "avg_hold_ratio": (100.0 - avg_sold_ratio) if avg_sold_ratio is not None else None,
        "chains": sorted({item.get("chain") for item in matches if item.get("chain")}),
    }


def _extract_sold_ratio(notes: str) -> float | None:
    match = _SOLD_RATIO_RE.search(notes)
    return float(match.group(1)) if match else None


def _extract_chain(notes: str) -> str | None:
    match = _CHAIN_RE.search(notes.strip())
    return match.group(1).lower() if match else None


def _compose_overview(
    symbol: str,
    market: dict[str, Any],
    signal: dict[str, Any],
    okx: dict[str, Any],
    config: WhaleOverviewConfig,
) -> dict[str, Any]:
    if signal:
        return _compose_overview_from_signal(symbol, market, okx, signal)
    return _compose_overview_from_market_proxy(symbol, market, okx, config)


def _compose_overview_from_signal(symbol: str, market: dict[str, Any], okx: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    total_positions = float(signal.get("totalPositions") or 0.0)
    total_traders = int(float(signal.get("totalTraders") or 0))
    long_short_ratio_raw = float(signal.get("longShortRatio") or 0.0)
    long_traders = int(float(signal.get("longTraders") or 0))
    short_traders = int(float(signal.get("shortTraders") or 0))
    long_whales = int(float(signal.get("longWhales") or 0))
    short_whales = int(float(signal.get("shortWhales") or 0))
    long_whales_qty = float(signal.get("longWhalesQty") or 0.0)
    short_whales_qty = float(signal.get("shortWhalesQty") or 0.0)
    long_traders_qty = float(signal.get("longTradersQty") or 0.0)
    short_traders_qty = float(signal.get("shortTradersQty") or 0.0)
    total_whales = long_whales + short_whales
    total_trader_qty = long_traders_qty + short_traders_qty
    long_ratio_pct = (long_traders_qty / total_trader_qty * 100.0) if total_trader_qty > 0 else 0.0
    short_ratio_pct = 100.0 - long_ratio_pct if total_trader_qty > 0 else 0.0
    signal_notional = market["mark_price"] * total_trader_qty
    scale = (total_positions / signal_notional) if signal_notional > 0 and total_positions > 0 else 1.0
    long_notional = market["mark_price"] * long_traders_qty * scale
    short_notional = market["mark_price"] * short_traders_qty * scale
    long_whale_notional = market["mark_price"] * long_whales_qty * scale
    short_whale_notional = market["mark_price"] * short_whales_qty * scale
    long_profit_ratio = (float(signal.get("longProfitTraders") or 0.0) / long_traders * 100.0) if long_traders > 0 else 0.0
    short_profit_ratio = (float(signal.get("shortProfitTraders") or 0.0) / short_traders * 100.0) if short_traders > 0 else 0.0
    long_entry = float(signal.get("longTradersAvgEntryPrice") or 0.0)
    short_entry = float(signal.get("shortTradersAvgEntryPrice") or 0.0)
    long_pnl = long_traders_qty * scale * (market["mark_price"] - long_entry) if long_entry else 0.0
    short_pnl = short_traders_qty * scale * (short_entry - market["mark_price"]) if short_entry else 0.0
    return {
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_grade": "A: Binance smart-money signal overview + Binance futures mark price; no heuristic cohort estimation",
        "summary": {
            "total_position_usdt": round(total_positions, 2),
            "whale_count": total_whales,
            "trader_count": total_traders,
            "long_ratio_pct": round(long_ratio_pct, 2),
            "short_ratio_pct": round(short_ratio_pct, 2),
            "nominal_long_short_ratio": round(long_short_ratio_raw * 100.0, 2) if long_short_ratio_raw else 0.0,
            "long_trader_count": long_traders,
            "short_trader_count": short_traders,
        },
        "long_group": _group_payload("B", long_whales, long_notional, long_entry, long_pnl, long_profit_ratio, trader_count=long_traders),
        "short_group": _group_payload("S", short_whales, short_notional, short_entry, short_pnl, short_profit_ratio, trader_count=short_traders),
        "market": market,
        "okx_signal_context": okx,
        "field_mapping": {
            "总持仓": "Binance smart-money signal overview.totalPositions",
            "巨鲸总数": "Binance smart-money signal overview.longWhales + shortWhales",
            "交易者总数": "Binance smart-money signal overview.totalTraders",
            "名义多空比率": "Binance smart-money signal overview.longShortRatio × 100",
            "多空比率": "Binance smart-money signal long/shortTradersQty split",
            "平均开仓价": "Binance smart-money signal long/shortTradersAvgEntryPrice",
            "未实现PNL": "Binance smart-money signal trader qty, normalized by totalPositions, × (markPrice - avgEntryPrice) by side",
            "预估PNL": "Same as 未实现PNL; derived from Binance real aggregate qty + avg entry + mark price, not legacy heuristic",
            "盈利比例": "Binance smart-money signal long/shortProfitTraders ÷ long/shortTraders",
        },
        "signal_overview": {
            "long_traders": long_traders,
            "short_traders": short_traders,
            "long_traders_qty": long_traders_qty,
            "short_traders_qty": short_traders_qty,
            "long_traders_position_usdt": round(long_notional, 2),
            "short_traders_position_usdt": round(short_notional, 2),
            "long_traders_avg_entry_price": float(signal.get("longTradersAvgEntryPrice") or 0.0),
            "short_traders_avg_entry_price": float(signal.get("shortTradersAvgEntryPrice") or 0.0),
            "long_profit_traders": float(signal.get("longProfitTraders") or 0.0),
            "short_profit_traders": float(signal.get("shortProfitTraders") or 0.0),
            "long_whales": long_whales,
            "short_whales": short_whales,
            "long_whales_qty": long_whales_qty,
            "short_whales_qty": short_whales_qty,
            "long_whales_position_usdt": round(long_whale_notional, 2),
            "short_whales_position_usdt": round(short_whale_notional, 2),
            "long_whales_avg_entry_price": float(signal.get("longWhalesAvgEntryPrice") or 0.0),
            "short_whales_avg_entry_price": float(signal.get("shortWhalesAvgEntryPrice") or 0.0),
            "long_profit_whales": float(signal.get("longProfitWhales") or 0.0),
            "short_profit_whales": float(signal.get("shortProfitWhales") or 0.0),
            "update_time": signal.get("updateTime"),
        },
    }


def _compose_overview_from_market_proxy(symbol: str, market: dict[str, Any], okx: dict[str, Any], config: WhaleOverviewConfig) -> dict[str, Any]:
    top_long = market["top_long_account_ratio"] or market["global_long_account_ratio"] or 0.5
    top_short = market["top_short_account_ratio"] or market["global_short_account_ratio"] or max(0.0, 1.0 - top_long)
    denom = max(0.000001, top_long + top_short)
    long_ratio = max(0.0, min(1.0, top_long / denom))
    short_ratio = 1.0 - long_ratio

    whale_count = int(max(config.min_binance_whales, min(config.max_binance_whales, market["open_interest_usdt"] / config.whale_notional_unit_usdt)))
    long_count = int(round(whale_count * long_ratio))
    short_count = max(0, whale_count - long_count)
    long_notional = market["open_interest_usdt"] * long_ratio
    short_notional = market["open_interest_usdt"] * short_ratio

    price_change = market["price_change_pct_24h"] / 100.0
    long_entry = market["mark_price"] / (1.0 + price_change * 0.55) if (1.0 + price_change * 0.55) > 0 else market["mark_price"]
    short_entry = market["mark_price"] / (1.0 - price_change * 0.55) if (1.0 - price_change * 0.55) > 0 else market["mark_price"]
    long_pnl = long_notional * ((market["mark_price"] - long_entry) / long_entry) if long_entry else 0.0
    short_pnl = short_notional * ((short_entry - market["mark_price"]) / short_entry) if short_entry else 0.0

    long_profit_ratio = _profit_ratio_estimate(long_pnl, market["taker_buy_sell_ratio"], okx.get("avg_hold_ratio"), bullish=True)
    short_profit_ratio = _profit_ratio_estimate(short_pnl, market["taker_buy_sell_ratio"], okx.get("avg_hold_ratio"), bullish=False)

    return {
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_grade": "B: Binance derivatives aggregate + OKX smart-money bridge approximation; not per-whale account ledger",
        "summary": {
            "total_position_usdt": round(market["open_interest_usdt"], 2),
            "whale_count": whale_count,
            "long_ratio_pct": round(long_ratio * 100.0, 2),
            "short_ratio_pct": round(short_ratio * 100.0, 2),
        },
        "long_group": _group_payload("B", long_count, long_notional, long_entry, long_pnl, long_profit_ratio),
        "short_group": _group_payload("S", short_count, short_notional, short_entry, short_pnl, short_profit_ratio),
        "market": market,
        "okx_signal_context": okx,
        "field_mapping": {
            "总持仓": "Binance /fapi/v1/openInterest × premiumIndex.markPrice",
            "巨鲸总数": "open_interest_usdt / whale_notional_unit_usdt capped to UI range",
            "多空比率": "Binance /futures/data/topLongShortAccountRatio fallback globalLongShortAccountRatio",
            "平均开仓价": "mark price adjusted by 24h price change; approximation until per-account whale entries exist",
            "预估PNL": "group notional × estimated price move from average entry",
            "盈利比例": "estimated from PNL sign + taker ratio + OKX sold_ratio/hold_ratio",
        },
    }


def _profit_ratio_estimate(pnl: float, taker_ratio: float, okx_hold_ratio: float | None, *, bullish: bool) -> float:
    base = 65.0 if pnl >= 0 else 35.0
    flow_adj = max(-10.0, min(10.0, (taker_ratio - 1.0) * 25.0))
    if not bullish:
        flow_adj *= -1.0
    hold_adj = 0.0 if okx_hold_ratio is None else max(-8.0, min(8.0, (okx_hold_ratio - 50.0) / 4.0))
    if not bullish:
        hold_adj *= -0.5
    return round(max(0.0, min(100.0, base + flow_adj + hold_adj)), 2)


def _group_payload(
    label: str,
    count: int,
    notional: float,
    entry: float,
    pnl: float,
    profit_ratio: float,
    *,
    trader_count: int | None = None,
) -> dict[str, Any]:
    normalized_pnl = round(pnl, 2)
    return {
        "label": label,
        "whale_count": count,
        "trader_count": trader_count if trader_count is not None else count,
        "status": "盈利中" if pnl >= 0 else "亏损中",
        "current_position_usdt": round(notional, 2),
        "average_entry_price": round(entry, 8),
        "unrealized_pnl_usdt": normalized_pnl,
        "estimated_pnl_usdt": normalized_pnl,
        "profit_ratio_pct": profit_ratio,
    }


def render_whale_overview_html(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    long_group = payload["long_group"]
    short_group = payload["short_group"]
    has_signal = isinstance(payload.get("signal_overview"), dict)
    title = "聪明钱总览" if has_signal else "巨鲸总览"
    count_label = "交易者总数" if has_signal and summary.get("trader_count") is not None else "巨鲸总数"
    count_value = summary.get("trader_count") if has_signal and summary.get("trader_count") is not None else summary["whale_count"]
    ratio_label = "名义多空比率" if has_signal and summary.get("nominal_long_short_ratio") is not None else "多空比率"
    ratio_value = summary.get("nominal_long_short_ratio") if has_signal and summary.get("nominal_long_short_ratio") is not None else summary["long_ratio_pct"]
    pnl_label = "未实现 PNL" if has_signal else "预估 PNL"
    footnote = (
        "数据来自 Binance public smart-money signal overview。交易者总数包含巨鲸子集；"
        "当前仓位、均价、盈利比例为真实聚合字段，PNL 由真实聚合仓位与最新标记价格逐侧换算。"
        if has_signal
        else f"OKX匹配信号 {payload['okx_signal_context']['match_count']} 条；链：{', '.join(payload['okx_signal_context']['chains']) or 'n/a'}。"
        "字段为聚合估算，适合做监控面板；若要复刻逐巨鲸 PNL，需要接入持仓地址/Hyperliquid 或交易所账户级 feed。"
    )
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{payload['symbol']} {title}</title>
<style>
body{{margin:0;background:#071018;color:#e8f3f4;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;display:grid;place-items:center;min-height:100vh}}
.panel{{width:min(760px,92vw);background:#111b25;border:1px solid #263746;border-radius:24px;padding:28px;box-shadow:0 24px 80px #0008}}
.top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}} h1{{font-size:28px;margin:0}} .grade{{color:#81939f;font-size:12px;max-width:360px;text-align:right}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:18px}} .stat{{background:#0b141d;border-radius:16px;padding:16px}} .k{{color:#7f929e;font-size:13px}} .v{{font-size:26px;font-weight:750;margin-top:8px}}
.bar{{height:12px;border-radius:99px;overflow:hidden;background:#ff6b8a44;display:flex;margin:14px 0 18px}} .bar b:first-child{{background:#21d6c7;width:{summary['long_ratio_pct']}%}} .bar b:last-child{{background:#ff6b8a;flex:1}}
.card{{border-radius:20px;padding:20px;margin-top:16px;background:#0d1821}} .long{{border:1px solid #21d6c777}} .short{{border:1px solid #ff6b8a77}}
.card-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}} .badge{{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:8px;margin-right:8px;font-weight:800}} .long .badge{{background:#21d6c7;color:#06201d}} .short .badge{{background:#ff6b8a;color:#27040b}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}} .metric{{background:#08111a;border-radius:14px;padding:14px}} .pos{{color:#21d6c7}} .neg{{color:#ff6b8a}}
.progress{{height:8px;background:#20303b;border-radius:99px;overflow:hidden;margin-top:10px}} .progress span{{display:block;height:100%}} .long .progress span{{background:#21d6c7}} .short .progress span{{background:#ff6b8a}}
.foot{{color:#80929c;font-size:12px;margin-top:18px;line-height:1.5}}
</style></head><body><main class=\"panel\">
<div class=\"top\"><h1>{payload['symbol']} {title}</h1><div class=\"grade\">{payload['data_grade']}</div></div>
<section class=\"stats\"><div class=\"stat\"><div class=\"k\">总持仓(USDT)</div><div class=\"v\">${summary['total_position_usdt']:,.0f}</div></div><div class=\"stat\"><div class=\"k\">{count_label}</div><div class=\"v\">{count_value}</div></div><div class=\"stat\"><div class=\"k\">{ratio_label}</div><div class=\"v\">{ratio_value:.2f}%</div></div></section>
<div class=\"bar\"><b></b><b></b></div>
{_render_group(long_group, 'long', pnl_label=pnl_label)}
{_render_group(short_group, 'short', pnl_label=pnl_label)}
<div class=\"foot\">{footnote}</div>
</main></body></html>"""


def _render_group(group: dict[str, Any], cls: str, *, pnl_label: str) -> str:
    pnl_value = group.get("unrealized_pnl_usdt", group["estimated_pnl_usdt"])
    pnl_cls = "pos" if pnl_value >= 0 else "neg"
    trader_count = group.get("trader_count")
    if trader_count is not None and trader_count != group["whale_count"]:
        count_text = f"{trader_count} 交易员 ({group['whale_count']} 大户)"
    else:
        count_text = f"{group['whale_count']} 巨鲸"
    return f"""<section class=\"card {cls}\"><div class=\"card-head\"><div><span class=\"badge\">{group['label']}</span><b>{count_text}</b></div><div>{group['status']}</div></div>
<div class=\"grid\"><div class=\"metric\"><div class=\"k\">当前仓位(USDT)</div><div class=\"v\">${group['current_position_usdt']:,.0f}</div></div><div class=\"metric\"><div class=\"k\">平均开仓价</div><div class=\"v\">{group['average_entry_price']}</div></div><div class=\"metric\"><div class=\"k\">{pnl_label}</div><div class=\"v {pnl_cls}\">{pnl_value:+,.2f} USDT</div></div><div class=\"metric\"><div class=\"k\">盈利比例</div><div class=\"v\">{group['profit_ratio_pct']:.2f}%</div><div class=\"progress\"><span style=\"width:{group['profit_ratio_pct']}%\"></span></div></div></div></section>"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a Binance+OKX whale overview panel payload and HTML.")
    parser.add_argument("symbol", help="Binance futures symbol, e.g. ETHUSDT or ETH")
    parser.add_argument("--skill-bridge-dir", default="data/trading_system/skill_bridges")
    parser.add_argument("--output-json", default="data/trading_system/whale_overview.json")
    parser.add_argument("--output-html", default="data/trading_system/whale_overview.html")
    args = parser.parse_args()
    payload = build_whale_overview(
        WhaleOverviewConfig(
            symbol=args.symbol,
            skill_bridge_dir=Path(args.skill_bridge_dir),
            output_json=Path(args.output_json),
            output_html=Path(args.output_html),
        )
    )
    print(json.dumps({"symbol": payload["symbol"], "summary": payload["summary"], "output_json": args.output_json, "output_html": args.output_html}, ensure_ascii=False))


if __name__ == "__main__":
    main()
