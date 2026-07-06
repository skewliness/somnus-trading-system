#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

try:
    from trading_system.coin_scout_4h_labeler import CoinScout4HLabeler, INTERVAL_MS
except ModuleNotFoundError:
    from trading_system.optional_integrations import CoinScout4HLabeler, INTERVAL_MS


ASSET_DIR = ROOT / "data/trading_system/telegram_monitor/assets"
CHART_KLINE_CACHE_DIR = ROOT / os.getenv("TS_CHART_KLINE_CACHE_DIR", "data/trading_system/cache/chart_klines")
CHART_JS = ASSET_DIR / "lightweight-charts-4.2.3.standalone.production.js"
CHART_JS_URL = "https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"
V6_SUMMARY_FILE = ROOT / "data/trading_system/v6_execution/latest_run_summary.json"
V6_ORDERS_FILE = ROOT / "data/trading_system/v6_execution/orders.jsonl"
V5_V6_ORDERS_FILE = ROOT / "data/trading_system/v5_v6_execution/orders.jsonl"
V5_PAPER_STATE_FILE = ROOT / "data/trading_system/paper_state.json"
CHART_WIDTH = 1080
CHART_HEIGHT = 720
CHART_HEADER_HEIGHT = 74
CHART_BODY_HEIGHT = CHART_HEIGHT - CHART_HEADER_HEIGHT
SYMBOL_ALIASES = {
    "FTX": "FTTUSDT",
    "FTXUSDT": "FTTUSDT",
}

PERIODS_BY_INTERVAL = {
    "1m": [105, 300, 672, 1920],
    "5m": [96, 105, 120, 210, 300, 672, 1440],
    "15m": [96, 105, 120, 210, 300, 672, 1440, 1920, 2880, 4707],
    "1h": [96, 105, 120, 210, 300, 672, 1440],
    "4h": [96, 105, 120, 210, 300, 672, 1440, 1920, 2880, 4707],
    "1d": [96, 105, 120, 210, 300, 672, 1440, 1920, 2880, 4707],
}

LINE_COLORS = {
    "focus_upper": "#86efac",
    "focus_mid": "#f8fafc",
    "focus_lower": "#86efac",
    "bb96": "#38bdf8",
    "bb105": "#fb7185",
    "bb120": "#f97316",
    "bb210": "#a78bfa",
    "bb300": "#2dd4bf",
    "bb672": "#facc15",
    "bb1440": "#22c55e",
    "bb1920": "#14b8a6",
    "bb2880": "#e879f9",
    "bb4707": "#f8fafc",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ChartUserError(RuntimeError):
    pass


def normalize_symbol(raw: str) -> str:
    symbol = str(raw or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if symbol in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[symbol]
    if symbol and not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    symbol = SYMBOL_ALIASES.get(symbol, symbol)
    return symbol


def compute_bb(closes: list[float], period: int, std_dev: float = 2.0) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    rolling_sum = 0.0
    rolling_sq_sum = 0.0
    window: list[float] = []
    for close in closes:
        window.append(close)
        rolling_sum += close
        rolling_sq_sum += close * close
        if len(window) > period:
            old = window.pop(0)
            rolling_sum -= old
            rolling_sq_sum -= old * old
        if len(window) < period:
            rows.append({})
            continue
        mean = rolling_sum / period
        variance = max(0.0, rolling_sq_sum / period - mean * mean)
        sd = math.sqrt(variance)
        rows.append({"upper": mean + std_dev * sd, "mid": mean, "lower": mean - std_dev * sd})
    return rows


def format_price(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "--"
    if numeric >= 1000:
        return f"{numeric:.2f}"
    if numeric >= 10:
        return f"{numeric:.3f}"
    return f"{numeric:.5f}"


def zh_side(value: Any) -> str:
    return {"long": "多头", "short": "空头", "neutral": "中性", "wait": "等待"}.get(str(value or ""), str(value or "--"))


def parse_iso_timestamp(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def marker_price_from_order(order: dict[str, Any], intent: dict[str, Any]) -> float:
    response = order.get("response") if isinstance(order.get("response"), dict) else {}
    for key in ("avgPrice", "price"):
        try:
            value = float(response.get(key) or 0.0)
        except Exception:
            value = 0.0
        if value > 0:
            return value
    try:
        return float(intent.get("entry_price") or 0.0)
    except Exception:
        return 0.0


def marker_side_from_v5_action(action: str) -> str:
    value = str(action or "").lower()
    if value in {"open_long", "buy", "long_entry"}:
        return "BUY"
    if value in {"close_short", "cover_short"}:
        return "BUY"
    if value in {"open_short", "sell_short"}:
        return "SELL"
    if value in {"exit", "close_long", "sell", "take_profit", "stop_loss"}:
        return "SELL"
    return ""


def marker_text_from_action(action: str, side: str) -> str:
    value = str(action or "").lower()
    normalized_side = str(side or "").upper()
    if value in {"open_long", "buy", "long_entry"}:
        return "开多"
    if value in {"close_long", "exit", "sell", "take_profit", "stop_loss"}:
        return "平多"
    if value in {"open_short", "sell_short"}:
        return "开空"
    if value in {"close_short", "cover_short"}:
        return "平空"
    if normalized_side == "BUY":
        return "B"
    if normalized_side == "SELL":
        return "S"
    return ""


def marker_price_from_v5_row(row: dict[str, Any], side: str) -> float:
    keys = ("entry_price", "price", "mark_price") if side == "BUY" else ("exit_price", "close_price", "current_price", "entry_mark_price", "price")
    for key in keys:
        try:
            value = float(row.get(key) or 0.0)
        except Exception:
            value = 0.0
        if value > 0:
            return value
    if side == "SELL":
        try:
            return float(row.get("take_profit_price") or row.get("stop_loss_price") or 0.0)
        except Exception:
            return 0.0
    return 0.0


def append_side_marker(
    markers: list[dict[str, Any]],
    seen: set[tuple[int, str, str, int]],
    *,
    chart_time: int,
    side: str,
    price: float,
    source: str,
    action: str = "",
) -> None:
    if price <= 0 or side not in {"BUY", "SELL"}:
        return
    text = marker_text_from_action(action, side)
    if not text:
        return
    dedupe_key = (int(chart_time), text, source, int(round(price * 1_000_000_000)))
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    markers.append(
        {
            "time": chart_time,
            "position": "belowBar" if side == "BUY" else "aboveBar",
            "color": "#22c55e" if side == "BUY" else "#ef4444",
            "shape": "arrowUp" if side == "BUY" else "arrowDown",
            "text": text,
            "size": 2,
            "price": price,
            "source": source,
            "action": action,
        }
    )


def candle_time_for_timestamp(timestamp: datetime, interval_ms: int, candle_times: set[int], first_time: int, last_time: int) -> int | None:
    open_ms = int(timestamp.timestamp() * 1000) // interval_ms * interval_ms
    chart_time = open_ms // 1000
    if chart_time < first_time or chart_time > last_time or chart_time not in candle_times:
        return None
    return chart_time


def load_v6_trade_markers(
    symbol: str,
    interval_ms: int,
    candle_times: set[int],
    first_time: int,
    last_time: int,
    markers: list[dict[str, Any]],
    seen: set[tuple[int, str, str, int]],
) -> None:
    if not V6_ORDERS_FILE.exists():
        return
    try:
        lines = V6_ORDERS_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    for line in lines:
        if not line.strip():
            continue
        try:
            order = json.loads(line)
        except Exception:
            continue
        if order.get("event"):
            continue
        if order.get("status") not in {"submitted", "dry_run"}:
            continue
        intent = order.get("intent") if isinstance(order.get("intent"), dict) else {}
        if str(intent.get("symbol") or "").upper() != symbol:
            continue
        action = str(intent.get("action") or "")
        is_entry = action in {"open_long", "open_short"}
        is_take_profit = action in {"close_long", "close_short"} and (
            bool(intent.get("target_name"))
            or "take profit" in str(intent.get("rationale") or "").lower()
            or bool(((intent.get("risk_plan") or {}).get("partial_take_profit") if isinstance(intent.get("risk_plan"), dict) else None))
        )
        if not is_entry and not is_take_profit:
            continue
        timestamp = parse_iso_timestamp(order.get("submitted_at") or intent.get("generated_at"))
        if timestamp is None:
            continue
        chart_time = candle_time_for_timestamp(timestamp, interval_ms, candle_times, first_time, last_time)
        if chart_time is None:
            continue
        price = marker_price_from_order(order, intent)
        append_side_marker(markers, seen, chart_time=chart_time, side=str(intent.get("side") or "").upper(), price=price, source="v6", action=action)


def load_v5_v6_execution_markers(
    symbol: str,
    interval_ms: int,
    candle_times: set[int],
    first_time: int,
    last_time: int,
    markers: list[dict[str, Any]],
    seen: set[tuple[int, str, str, int]],
) -> None:
    if not V5_V6_ORDERS_FILE.exists():
        return
    try:
        lines = V5_V6_ORDERS_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    for line in lines:
        if not line.strip():
            continue
        try:
            order = json.loads(line)
        except Exception:
            continue
        if order.get("event"):
            continue
        if order.get("status") not in {"submitted", "dry_run"}:
            continue
        intent = order.get("intent") if isinstance(order.get("intent"), dict) else {}
        if str(intent.get("symbol") or "").upper() != symbol:
            continue
        action = str(intent.get("action") or "")
        if action not in {"open_long", "open_short", "close_long", "close_short"}:
            continue
        timestamp = parse_iso_timestamp(order.get("submitted_at") or intent.get("generated_at"))
        if timestamp is None:
            continue
        chart_time = candle_time_for_timestamp(timestamp, interval_ms, candle_times, first_time, last_time)
        if chart_time is None:
            continue
        price = marker_price_from_order(order, intent)
        append_side_marker(
            markers,
            seen,
            chart_time=chart_time,
            side=str(intent.get("side") or "").upper(),
            price=price,
            source="v5/v6",
            action=action,
        )


def load_v5_trade_markers(
    symbol: str,
    interval_ms: int,
    candle_times: set[int],
    first_time: int,
    last_time: int,
    markers: list[dict[str, Any]],
    seen: set[tuple[int, str, str, int]],
) -> None:
    state = load_json(V5_PAPER_STATE_FILE)
    rows: list[dict[str, Any]] = []
    for key in ("journal", "learning_journal"):
        rows.extend(row for row in state.get(key) or [] if isinstance(row, dict))
    for row in rows:
        if str(row.get("symbol") or "").upper() != symbol:
            continue
        side = marker_side_from_v5_action(str(row.get("action") or ""))
        if not side:
            continue
        timestamp = parse_iso_timestamp(row.get("timestamp") or row.get("opened_at") or row.get("closed_at"))
        if timestamp is None:
            continue
        chart_time = candle_time_for_timestamp(timestamp, interval_ms, candle_times, first_time, last_time)
        if chart_time is None:
            continue
        price = marker_price_from_v5_row(row, side)
        append_side_marker(markers, seen, chart_time=chart_time, side=side, price=price, source="v5", action=str(row.get("action") or ""))

    for position in state.get("positions") or []:
        if not isinstance(position, dict) or str(position.get("symbol") or "").upper() != symbol:
            continue
        timestamp = parse_iso_timestamp(position.get("opened_at"))
        if timestamp is None:
            continue
        chart_time = candle_time_for_timestamp(timestamp, interval_ms, candle_times, first_time, last_time)
        if chart_time is None:
            continue
        side = "SELL" if str(position.get("side") or "").lower() == "short" else "BUY"
        try:
            price = float(position.get("entry_price") or 0.0)
        except Exception:
            price = 0.0
        action = "open_short" if side == "SELL" else "open_long"
        append_side_marker(markers, seen, chart_time=chart_time, side=side, price=price, source="v5", action=action)


def load_trade_markers(symbol: str, interval: str, candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candles:
        return []
    interval_ms = INTERVAL_MS[interval]
    candle_times = {int(item["time"]) for item in candles}
    first_time = min(candle_times)
    last_time = max(candle_times)
    markers: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str, int]] = set()
    load_v6_trade_markers(symbol, interval_ms, candle_times, first_time, last_time, markers, seen)
    load_v5_v6_execution_markers(symbol, interval_ms, candle_times, first_time, last_time, markers, seen)
    load_v5_trade_markers(symbol, interval_ms, candle_times, first_time, last_time, markers, seen)
    markers.sort(key=lambda item: (int(item["time"]), item["text"]))
    return markers


def ensure_chart_js() -> str:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if not CHART_JS.exists() or CHART_JS.stat().st_size < 100_000:
        with urllib.request.urlopen(CHART_JS_URL, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
        CHART_JS.write_text(body, encoding="utf-8")
    return CHART_JS.read_text(encoding="utf-8")


def fetch_klines(symbol: str, interval: str, max_period: int) -> list[dict[str, Any]]:
    interval_ms = INTERVAL_MS[interval]
    bars = max(max_period + 520, 900)
    start = utc_now() - timedelta(milliseconds=interval_ms * bars)
    labeler = CoinScout4HLabeler(cache_dir=str(CHART_KLINE_CACHE_DIR))
    labeler_output = io.StringIO()
    try:
        with contextlib.redirect_stdout(labeler_output), contextlib.redirect_stderr(labeler_output):
            rows = labeler.fetch_klines(symbol, interval, start, utc_now(), force_refresh=True)
    except Exception as exc:
        message = f"{labeler_output.getvalue()} {exc}"
        if "Invalid symbol" in message or "Bad Request" in message or "API error" in message:
            raise ChartUserError(
                f"{symbol} 不是 Binance U本位合约有效交易对，无法生成图表。"
                "请确认币种代码；例如 FTX 平台币应使用 FTT。"
            ) from exc
        raise
    rows.sort(key=lambda item: int(item.get("open_time") or 0))
    if not rows:
        message = labeler_output.getvalue()
        if "Invalid symbol" in message or "Bad Request" in message or "API error" in message:
            raise ChartUserError(
                f"{symbol} 不是 Binance U本位合约有效交易对，无法生成图表。"
                "请确认币种代码；例如 FTX 平台币应使用 FTT。"
            )
        raise ChartUserError(f"{symbol} {interval.upper()} 没有可用K线数据，无法生成图表。")
    return rows


def find_v6_signal(symbol: str) -> dict[str, Any]:
    summary = load_json(V6_SUMMARY_FILE)
    for signal in summary.get("signals") or []:
        if str(signal.get("symbol") or "").upper() == symbol:
            return signal
    return {}


def build_series(
    klines: list[dict[str, Any]],
    periods: list[int],
    focus_period: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    closes = [float(item["close"]) for item in klines]
    candles = [
        {
            "time": int(item["open_time"]) // 1000,
            "open": float(item["open"]),
            "high": float(item["high"]),
            "low": float(item["low"]),
            "close": float(item["close"]),
        }
        for item in klines
    ]
    volume = [
        {
            "time": int(item["open_time"]) // 1000,
            "value": float(item.get("volume") or 0),
            "color": "rgba(34,197,94,0.25)" if float(item["close"]) >= float(item["open"]) else "rgba(248,113,113,0.25)",
        }
        for item in klines
    ]

    line_defs: list[dict[str, Any]] = []
    band_defs: list[dict[str, Any]] = []
    focus_bb: dict[str, Any] = {}
    for period in periods:
        bb_rows = compute_bb(closes, period)
        if not any(row for row in bb_rows):
            continue
        if period == focus_period:
            upper_data = [{"time": candles[i]["time"], "value": row["upper"]} for i, row in enumerate(bb_rows) if row]
            lower_data = [{"time": candles[i]["time"], "value": row["lower"]} for i, row in enumerate(bb_rows) if row]
            band_defs.append(
                {
                    "title": f"BB{period}布林带",
                    "upper": upper_data,
                    "lower": lower_data,
                    "fillColor": "rgba(34, 197, 94, 0.14)",
                    "borderColor": "rgba(134, 239, 172, 0.34)",
                }
            )
            for band, title, color, width in [
                ("upper", f"BB{period}上轨", LINE_COLORS["focus_upper"], 2),
                ("mid", f"BB{period}中轨", LINE_COLORS["focus_mid"], 3),
                ("lower", f"BB{period}下轨", LINE_COLORS["focus_lower"], 2),
            ]:
                data = [{"time": candles[i]["time"], "value": row[band]} for i, row in enumerate(bb_rows) if row]
                line_defs.append({"title": title, "color": color, "lineWidth": width, "data": data})
            focus_bb = bb_rows[-1] if bb_rows and bb_rows[-1] else {}

    meta = {
        "price": closes[-1],
        "focus_upper": focus_bb.get("upper"),
        "focus_mid": focus_bb.get("mid"),
        "focus_lower": focus_bb.get("lower"),
        "bars": len(candles),
        "from": datetime.fromtimestamp(candles[0]["time"], timezone.utc).isoformat(),
        "to": datetime.fromtimestamp(candles[-1]["time"], timezone.utc).isoformat(),
    }
    return candles, volume, line_defs, band_defs, meta


def make_html(payload: dict[str, Any], chart_js: str) -> str:
    serialized = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
html, body {{
  margin: 0;
  width: {CHART_WIDTH}px;
  height: {CHART_HEIGHT}px;
  background: #070b10;
  color: #dbeafe;
  font-family: Arial, "Noto Sans CJK SC", "WenQuanYi Zen Hei", sans-serif;
}}
#frame {{
  width: {CHART_WIDTH}px;
  height: {CHART_HEIGHT}px;
  background: #070b10;
  position: relative;
  overflow: hidden;
}}
#chart {{
  position: absolute;
  left: 0;
  top: {CHART_HEADER_HEIGHT}px;
  width: {CHART_WIDTH}px;
  height: {CHART_BODY_HEIGHT}px;
}}
#chart canvas.bb-band-overlay {{
  position: absolute;
  left: 0;
  top: 0;
  width: {CHART_WIDTH}px;
  height: {CHART_BODY_HEIGHT}px;
  pointer-events: none;
  z-index: 5;
}}
#top {{
  position: absolute;
  left: 0;
  top: 0;
  right: 0;
  height: {CHART_HEADER_HEIGHT}px;
  box-sizing: border-box;
  padding: 12px 18px 8px 18px;
  background: linear-gradient(180deg, #111827 0%, #0b111a 100%);
  border-bottom: 1px solid rgba(148, 163, 184, 0.25);
}}
.title {{
  font-size: 24px;
  font-weight: 700;
  letter-spacing: 0;
  color: #f8fafc;
}}
.sub {{
  margin-top: 6px;
  font-size: 15px;
  color: #cbd5e1;
  white-space: nowrap;
}}
.pill {{
  display: inline-block;
  margin-left: 10px;
  padding: 2px 8px;
  border-radius: 4px;
  background: rgba(250, 204, 21, 0.15);
  color: #fde68a;
}}
#legend {{
  position: absolute;
  right: 18px;
  top: 16px;
  font-size: 13px;
  color: #cbd5e1;
  text-align: right;
  line-height: 1.5;
}}
</style>
<script>{chart_js}</script>
</head>
<body>
<div id="frame">
  <div id="top">
    <div class="title" id="title"></div>
    <div class="sub" id="subtitle"></div>
    <div id="legend"></div>
  </div>
  <div id="chart"></div>
</div>
<script>
const payload = {serialized};
document.getElementById('title').innerText = payload.title;
document.getElementById('subtitle').innerText = payload.subtitle;
document.getElementById('legend').innerHTML = payload.legend.join('<br>');
const chartNode = document.getElementById('chart');
const chart = LightweightCharts.createChart(chartNode, {{
  width: {CHART_WIDTH},
  height: {CHART_BODY_HEIGHT},
  layout: {{
    background: {{ type: 'solid', color: '#070b10' }},
    textColor: '#cbd5e1',
    fontSize: 12,
  }},
  grid: {{
    vertLines: {{ color: 'rgba(148, 163, 184, 0.12)' }},
    horzLines: {{ color: 'rgba(148, 163, 184, 0.12)' }},
  }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{
    borderColor: 'rgba(148, 163, 184, 0.30)',
    scaleMargins: {{ top: 0.08, bottom: 0.20 }},
  }},
  timeScale: {{
    borderColor: 'rgba(148, 163, 184, 0.30)',
    timeVisible: true,
    secondsVisible: false,
  }},
}});
const candles = chart.addCandlestickSeries({{
  upColor: '#14b8a6',
  downColor: '#f43f5e',
  borderUpColor: '#14b8a6',
  borderDownColor: '#f43f5e',
  wickUpColor: '#5eead4',
  wickDownColor: '#fb7185',
}});
candles.setData(payload.candles);
if (Array.isArray(payload.markers) && payload.markers.length > 0) {{
  candles.setMarkers(payload.markers);
}}
const volume = chart.addHistogramSeries({{
  priceScaleId: '',
  priceFormat: {{ type: 'volume' }},
}});
volume.priceScale().applyOptions({{ scaleMargins: {{ top: 0.82, bottom: 0 }} }});
volume.setData(payload.volume);
const bandOverlay = document.createElement('canvas');
bandOverlay.width = {CHART_WIDTH};
bandOverlay.height = {CHART_BODY_HEIGHT};
bandOverlay.className = 'bb-band-overlay';
chartNode.appendChild(bandOverlay);

function visiblePoints(data) {{
  const points = [];
  for (const item of data || []) {{
    const x = chart.timeScale().timeToCoordinate(item.time);
    const y = candles.priceToCoordinate(item.value);
    if (x === null || y === null || Number.isNaN(x) || Number.isNaN(y)) {{
      continue;
    }}
    if (x < -80 || x > bandOverlay.width + 80) {{
      continue;
    }}
    points.push([x, y]);
  }}
  return points;
}}

function drawBandOverlay() {{
  const ctx = bandOverlay.getContext('2d');
  ctx.clearRect(0, 0, bandOverlay.width, bandOverlay.height);
  for (const band of payload.bands || []) {{
    const upper = visiblePoints(band.upper);
    const lower = visiblePoints(band.lower);
    if (upper.length < 2 || lower.length < 2) {{
      continue;
    }}
    ctx.beginPath();
    ctx.moveTo(upper[0][0], upper[0][1]);
    for (const [x, y] of upper.slice(1)) {{
      ctx.lineTo(x, y);
    }}
    for (const [x, y] of lower.slice().reverse()) {{
      ctx.lineTo(x, y);
    }}
    ctx.closePath();
    ctx.fillStyle = band.fillColor || 'rgba(34, 197, 94, 0.12)';
    ctx.fill();
    ctx.strokeStyle = band.borderColor || 'rgba(134, 239, 172, 0.30)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }}
}}

for (const line of payload.lines) {{
  const series = chart.addLineSeries({{
    color: line.color,
    lineWidth: line.lineWidth,
    title: line.title,
    lastValueVisible: true,
    priceLineVisible: false,
  }});
  series.setData(line.data);
}}
const latest = payload.candles[payload.candles.length - 1];
candles.createPriceLine({{
  price: latest.close,
  color: '#e5e7eb',
  lineWidth: 1,
  lineStyle: LightweightCharts.LineStyle.Dashed,
  axisLabelVisible: true,
  title: '现价',
}});
const from = Math.max(0, payload.candles.length - payload.visibleBars);
chart.timeScale().setVisibleLogicalRange({{ from, to: payload.candles.length + 8 }});
requestAnimationFrame(() => {{
  drawBandOverlay();
  window.__chartReady = true;
}});
</script>
</body>
</html>"""


def render_chart(args: argparse.Namespace) -> dict[str, Any]:
    symbol = normalize_symbol(args.symbol)
    interval = args.interval.lower()
    focus_period = args.focus_period if args.focus_period > 0 else None
    periods = [focus_period] if focus_period else []
    max_period = focus_period or 0
    klines = fetch_klines(symbol, interval, max_period)
    if focus_period and len(klines) < max_period:
        raise RuntimeError(f"{symbol} {interval} K线不足：{len(klines)}/{max_period}")

    candles, volume, lines, bands, meta = build_series(klines, periods, focus_period)
    markers = load_trade_markers(symbol, interval, candles)
    signal = find_v6_signal(symbol)
    v6_text = "V6暂无该币最新分析"
    if signal:
        v6_text = (
            f"V6 {zh_side(signal.get('side_bias'))} "
            f"多={float(signal.get('long_probability') or 0) * 100:.2f}% "
            f"空={float(signal.get('short_probability') or 0) * 100:.2f}% "
            f"等={float(signal.get('wait_probability') or 0) * 100:.2f}%"
        )

    title = f"{symbol} {interval.upper()}"
    subtitle = f"价格={format_price(meta['price'])}  {v6_text}"
    legend = ["K线来源=Binance实时刷新"]
    if markers:
        sources = sorted({str(marker.get("source") or "") for marker in markers if marker.get("source")})
        source_text = f" 来源={','.join(sources).upper()}" if sources else ""
        legend.append(f"标记=开多/平多/开空/平空 绿色=买入 红色=卖出{source_text}")
    if focus_period:
        title = f"{symbol} {interval.upper()} BB{focus_period}"
        subtitle = (
            f"价格={format_price(meta['price'])}  "
            f"BB{focus_period} 上={format_price(meta.get('focus_upper'))} "
            f"中={format_price(meta.get('focus_mid'))} "
            f"下={format_price(meta.get('focus_lower'))}  "
            f"{v6_text}"
        )
        legend.append(f"BB指标=BB{focus_period}")

    payload = {
        "title": title,
        "subtitle": subtitle,
        "legend": legend,
        "candles": candles,
        "volume": volume,
        "lines": lines,
        "bands": bands,
        "markers": markers,
        "visibleBars": args.visible_bars,
    }
    chart_js = ensure_chart_js()
    html = make_html(payload, chart_js)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    html_path = output.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": CHART_WIDTH, "height": CHART_HEIGHT}, device_scale_factor=1)
        page.set_content(html, wait_until="load")
        page.wait_for_function("window.__chartReady === true", timeout=10_000)
        page.locator("#frame").screenshot(path=str(output))
        browser.close()

    return {
        "symbol": symbol,
        "interval": interval,
        "focus_period": focus_period,
        "output": str(output),
        "html": str(html_path),
        "price": meta["price"],
        "bars": meta["bars"],
        "marker_count": len(markers),
        "kline_source": "Binance实时刷新",
        "kline_from": meta.get("from"),
        "kline_to": meta.get("to"),
        "v6": signal,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render V6 chart image with TradingView Lightweight Charts.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", choices=sorted(PERIODS_BY_INTERVAL), required=True)
    parser.add_argument("--focus-period", type=int, default=0)
    parser.add_argument("--visible-bars", type=int, default=360)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = render_chart(args)
    except ChartUserError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    result["requested_symbol"] = args.symbol
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
