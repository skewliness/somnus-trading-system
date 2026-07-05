#!/usr/bin/env python3
"""Hermes/Xuejian bridge for Somnus intelligence commands.

This helper lets Hermes answer position, chart, macro, and news requests by
reusing the existing Somnus Telegram monitor functions without sending through
V5/V6 trading bots. It intentionally does not submit orders or mutate ledgers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_cert_file = ROOT / "venv/lib/python3.12/site-packages/certifi/cacert.pem"
if _cert_file.exists():
    os.environ.setdefault("SSL_CERT_FILE", str(_cert_file))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_cert_file))
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("ALL_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

from scripts import telegram_monitor_v5_v6 as tg  # noqa: E402


def _load_env() -> argparse.Namespace:
    tg.load_dotenv(ROOT / ".env.local")
    # The V5 telegram env only contains the V5 bot identity. Loading it is safe
    # because this helper never sends through Telegram.
    tg.load_dotenv(ROOT / ".env.telegram_v5")
    return tg.parse_args([])


def _emit_text(text: str) -> int:
    sys.stdout.write(text.rstrip() + "\n")
    return 0


def _positions(args: argparse.Namespace, targets: list[str]) -> int:
    status = tg.collect_status(args)
    if not targets:
        targets = ["v5", "v6"]
    chunks = []
    for target in targets:
        target = target.lower().strip()
        if target not in {"v5", "v6"}:
            raise SystemExit(f"unsupported position target: {target}")
        chunks.append(tg.build_position_response(target, status))
    return _emit_text("\n\n".join(chunks))


def _chart(symbol: str, interval: str, focus: str | None, *, as_json: bool) -> int:
    normalized_symbol = tg.normalize_symbol(symbol)
    normalized_interval = tg.normalize_interval(interval) or "15m"
    focus_period = tg.parse_focus_period(focus) if focus else None
    payload = tg.build_chart_image(normalized_symbol, normalized_interval, focus_period)
    result: dict[str, Any] = {
        "symbol": normalized_symbol,
        "interval": normalized_interval,
        "focus_period": focus_period,
        "image": str(Path(payload["output"]).resolve()),
        "caption": tg.format_chart_caption(payload),
    }
    if as_json:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"图片={result['image']}\n{result['caption']}\n")
    return 0


def _report(symbols: list[str]) -> int:
    normalized = [tg.normalize_symbol(item) for item in symbols if tg.normalize_symbol(item)]
    if not normalized:
        normalized = ["BTCUSDT", "ETHUSDT"]
    return _emit_text(tg.build_report_response(normalized))


def _news_wss_summary(limit: int = 6) -> str:
    bridge = ROOT / "data/trading_system/skill_bridges/opennews_wss.json"
    if not bridge.exists():
        return ""
    try:
        payload = json.loads(bridge.read_text(encoding="utf-8"))
    except Exception:
        return ""
    signals = payload.get("signals") or []
    prefs = payload.get("display_preferences") or {}
    policy = payload.get("primary_source_policy") or {}
    lines = [
        "",
        "⚡【OpenNews WSS实时过滤】",
        f"显示=原文｜雪见翻译=中文｜密度={prefs.get('content_density', 'professional')}｜优先={','.join(policy.get('prefer') or ['binance','okx'])}，6551补齐",
    ]
    if not signals:
        lines.append("暂无通过过滤的实时事件。")
        return "\n".join(lines)
    for item in signals[-limit:]:
        meta = item.get("metadata") or {}
        category = meta.get("category") or "unknown"
        symbol = item.get("symbol") or "--"
        score = item.get("hotness_score")
        source = meta.get("matched_source") or item.get("source") or "--"
        note = str(item.get("notes") or meta.get("text_original") or "").strip()
        if len(note) > 220:
            note = note[:219].rstrip() + "…"
        lines.append(f"▫️ {category}｜{symbol}｜分数={score}｜源={source}｜{note}")
    return "\n".join(lines)


def _refresh_options_delivery_report() -> None:
    try:
        from trading_system.bridges import OptionsDeliveryBridge
        from trading_system.config import StrategyConfig

        config = StrategyConfig()
        OptionsDeliveryBridge(
            underlyings=config.options_delivery_underlyings,
            limit=config.options_delivery_limit,
            snapshot_path=config.options_delivery_report_file,
        ).load()
    except Exception:
        # Keep the last snapshot usable; news should degrade, not fail.
        return


def _options_delivery_summary(limit: int = 6) -> str:
    report = ROOT / "data/trading_system/options_delivery_report.json"
    if not report.exists():
        return ""
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return ""
    records = payload.get("records") or []
    if not records:
        return ""
    lines = ["", "🧾【官方期权交割】", "来源=Binance/OKX官方API｜用途=BTC/ETH交割压力新闻源，不单独触发下单"]
    for item in records[:limit]:
        symbol = item.get("symbol") or "--"
        score = item.get("hotness_score")
        source = item.get("source") or "--"
        note = str(item.get("notes") or "").strip()
        if len(note) > 220:
            note = note[:219].rstrip() + "…"
        lines.append(f"▫️ {symbol}｜分数={score}｜源={source}｜{note}")
    return "\n".join(lines)


def _news(args: argparse.Namespace, lookback_hours: int | None) -> int:
    if lookback_hours is not None:
        args.daily_news_lookback_hours = lookback_hours
    report = tg.build_daily_news_report(args)
    _refresh_options_delivery_report()
    options_summary = _options_delivery_summary()
    if options_summary:
        report = (report.rstrip() + "\n" + options_summary).strip()
    wss_summary = _news_wss_summary()
    if wss_summary:
        report = (report.rstrip() + "\n" + wss_summary).strip()
    return _emit_text(report[:6000])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Somnus intelligence commands for Hermes/Xuejian.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("p", help="Show V5/V6 positions.")
    p.add_argument("targets", nargs="*", help="v5 and/or v6. Defaults to both.")

    fc = sub.add_parser("fc", help="Render chart image.")
    fc.add_argument("symbol")
    fc.add_argument("interval", nargs="?", default="15m")
    fc.add_argument("focus", nargs="?", default=None, help="Optional BB period, e.g. bb672 or 672.")
    fc.add_argument("--json", action="store_true", help="Print JSON with image path and caption.")

    report = sub.add_parser("report", help="Build BTC/ETH macro report.")
    report.add_argument("symbols", nargs="*")

    news = sub.add_parser("news", help="Build opennews/opentwitter daily intelligence report.")
    news.add_argument("--lookback-hours", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _load_env()
    ns = build_parser().parse_args(argv)
    if ns.command == "p":
        return _positions(args, ns.targets)
    if ns.command == "fc":
        return _chart(ns.symbol, ns.interval, ns.focus, as_json=ns.json)
    if ns.command == "report":
        return _report(ns.symbols)
    if ns.command == "news":
        return _news(args, ns.lookback_hours)
    raise SystemExit(f"unsupported command: {ns.command}")


if __name__ == "__main__":
    raise SystemExit(main())


