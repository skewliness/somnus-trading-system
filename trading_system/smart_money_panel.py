from __future__ import annotations

from datetime import datetime, timezone
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from trading_system.binance import BinanceAccessError
from trading_system.whale_overview import WhaleOverviewConfig, build_whale_overview


DEFAULT_REFRESH_SECONDS = 1


def serve_smart_money_panel(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), _build_handler())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def get_smart_money_panel_payload(symbol: str) -> dict[str, Any]:
    payload = build_whale_overview(
        WhaleOverviewConfig(
            symbol=symbol,
            skill_bridge_dir=Path("data/trading_system/skill_bridges"),
            output_json=Path("/tmp/unused_whale_overview.json"),
            output_html=Path("/tmp/unused_whale_overview.html"),
        ),
        persist=False,
    )
    return _panel_payload_from_overview(payload)


def _build_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_panel_html())
                return
            if parsed.path == "/api/smart-money":
                query = parse_qs(parsed.query)
                symbol = str((query.get("symbol") or ["BTCUSDT"])[0] or "BTCUSDT")
                try:
                    payload = get_smart_money_panel_payload(symbol)
                except BinanceAccessError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_GATEWAY)
                    return
                except Exception as exc:  # pragma: no cover - defensive
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self._send_json({"ok": True, "data": payload})
                return
            self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _panel_payload_from_overview(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    signal = payload.get("signal_overview") or {}
    long_group = payload["long_group"]
    short_group = payload["short_group"]
    updated_at = signal.get("update_time")
    long_whales = int(signal.get("long_whales") or long_group["whale_count"] or 0)
    short_whales = int(signal.get("short_whales") or short_group["whale_count"] or 0)
    long_profit_whales = float(signal.get("long_profit_whales") or 0.0)
    short_profit_whales = float(signal.get("short_profit_whales") or 0.0)
    return {
        "symbol": payload["symbol"],
        "as_of_utc": _timestamp_ms_to_iso(updated_at),
        "data_grade": payload["data_grade"],
        "price": {
            "mark_price": payload["market"]["mark_price"],
            "index_price": payload["market"]["index_price"],
            "funding_rate": payload["market"]["funding_rate"],
        },
        "summary": {
            "total_position_usdt": summary["total_position_usdt"],
            "total_traders": summary.get("trader_count", 0),
            "total_whales": summary["whale_count"],
            "nominal_long_short_ratio_pct": summary.get("nominal_long_short_ratio", 0.0),
        },
        "views": {
            "smart_money": {
                "label": "聪明钱",
                "count_label": "交易员",
                "long_side": {
                    "count": long_group.get("trader_count", 0),
                    "subset_count": long_group["whale_count"],
                    "current_position_usdt": long_group["current_position_usdt"],
                    "average_entry_price": long_group["average_entry_price"],
                    "unrealized_pnl_usdt": long_group["unrealized_pnl_usdt"],
                    "profit_ratio_pct": long_group["profit_ratio_pct"],
                    "quantity": signal.get("long_traders_qty", 0.0),
                },
                "short_side": {
                    "count": short_group.get("trader_count", 0),
                    "subset_count": short_group["whale_count"],
                    "current_position_usdt": short_group["current_position_usdt"],
                    "average_entry_price": short_group["average_entry_price"],
                    "unrealized_pnl_usdt": short_group["unrealized_pnl_usdt"],
                    "profit_ratio_pct": short_group["profit_ratio_pct"],
                    "quantity": signal.get("short_traders_qty", 0.0),
                },
            },
            "whales": {
                "label": "巨鲸",
                "count_label": "巨鲸",
                "long_side": {
                    "count": long_whales,
                    "subset_count": None,
                    "current_position_usdt": signal.get("long_whales_position_usdt", 0.0),
                    "average_entry_price": signal.get("long_whales_avg_entry_price", 0.0),
                    "unrealized_pnl_usdt": _side_unrealized_pnl(
                        quantity=signal.get("long_whales_qty", 0.0),
                        current_position_usdt=signal.get("long_whales_position_usdt", 0.0),
                        mark_price=payload["market"]["mark_price"],
                        average_entry_price=signal.get("long_whales_avg_entry_price", 0.0),
                        is_long=True,
                    ),
                    "profit_ratio_pct": (long_profit_whales / long_whales * 100.0) if long_whales > 0 else 0.0,
                    "quantity": signal.get("long_whales_qty", 0.0),
                },
                "short_side": {
                    "count": short_whales,
                    "subset_count": None,
                    "current_position_usdt": signal.get("short_whales_position_usdt", 0.0),
                    "average_entry_price": signal.get("short_whales_avg_entry_price", 0.0),
                    "unrealized_pnl_usdt": _side_unrealized_pnl(
                        quantity=signal.get("short_whales_qty", 0.0),
                        current_position_usdt=signal.get("short_whales_position_usdt", 0.0),
                        mark_price=payload["market"]["mark_price"],
                        average_entry_price=signal.get("short_whales_avg_entry_price", 0.0),
                        is_long=False,
                    ),
                    "profit_ratio_pct": (short_profit_whales / short_whales * 100.0) if short_whales > 0 else 0.0,
                    "quantity": signal.get("short_whales_qty", 0.0),
                },
            },
        },
    }


def _side_unrealized_pnl(
    *,
    quantity: Any,
    current_position_usdt: Any,
    mark_price: Any,
    average_entry_price: Any,
    is_long: bool,
) -> float:
    qty = float(quantity or 0.0)
    position_usdt = float(current_position_usdt or 0.0)
    price = float(mark_price or 0.0)
    entry = float(average_entry_price or 0.0)
    if qty <= 0 or position_usdt <= 0 or price <= 0 or entry <= 0:
        return 0.0
    scale = position_usdt / (qty * price)
    direction_move = (price - entry) if is_long else (entry - price)
    return round(qty * scale * direction_move, 2)


def _timestamp_ms_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _format_compact_number(value: Any) -> str:
    number = float(value or 0.0)
    sign = "-" if number < 0 else ""
    absolute = abs(number)
    if absolute >= 1_000_000_000:
        return f"{sign}{absolute / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{sign}{absolute / 1_000_000:.2f}M"
    return f"{number:,.2f}"


def _panel_html() -> str:
    return _compact_panel_html()


def _compact_panel_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Binance 合约数据面板</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #070b10;
      --surface: #0c121a;
      --surface-2: #101823;
      --surface-3: #121d29;
      --line: rgba(156, 170, 190, 0.17);
      --line-strong: rgba(156, 170, 190, 0.3);
      --text: #edf4fb;
      --muted: #93a2b5;
      --faint: #657284;
      --buy: #5ee2cc;
      --sell: #ff7894;
      --warn: #f4c35c;
      --accent: #8bb7ff;
      --shadow: 0 22px 70px rgba(0, 0, 0, 0.38);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(180deg, #081019 0%, #05080d 100%);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.035);
      color: var(--text);
      font: inherit;
      font-size: 14px;
    }
    button {
      padding: 0 13px;
      cursor: pointer;
      font-weight: 700;
      transition: border-color 0.16s ease, background 0.16s ease;
    }
    button:hover {
      border-color: var(--line-strong);
      background: rgba(255, 255, 255, 0.06);
    }
    input {
      width: 190px;
      padding: 0 12px;
      font-weight: 650;
      text-transform: uppercase;
      background: rgba(255, 255, 255, 0.045);
    }
    .page {
      width: min(100%, 1680px);
      margin: 0 auto;
      padding: 14px;
    }
    .shell {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(14, 21, 31, 0.98), rgba(8, 12, 18, 0.99));
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .topbar {
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 12px 16px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.018);
    }
    .topbar-meta,
    .topbar-main {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      width: 100%;
      min-width: 0;
    }
    .meta-left,
    .meta-right {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      flex-wrap: wrap;
    }
    .meta-right {
      justify-content: flex-end;
    }
    .title-wrap {
      min-width: 0;
    }
    .eyebrow {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .title-row {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
      flex-wrap: wrap;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.08;
      font-weight: 800;
      letter-spacing: 0;
    }
    .symbol-pill, .data-pill {
      display: inline-flex;
      align-items: center;
      height: 26px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.035);
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }
    .symbol-pill {
      color: #111822;
      border-color: rgba(244, 195, 92, 0.55);
      background: linear-gradient(180deg, #ffd66d, #f2b84d);
    }
    .auto-pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      height: 26px;
      padding: 0 10px;
      border: 1px solid rgba(94, 226, 204, 0.28);
      border-radius: 999px;
      color: var(--buy);
      background: rgba(94, 226, 204, 0.08);
      font-size: 12px;
      font-weight: 780;
      white-space: nowrap;
    }
    .auto-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--buy);
      box-shadow: 0 0 0 3px rgba(94, 226, 204, 0.13);
    }
    .auto-pill.paused {
      color: var(--warn);
      border-color: rgba(244, 195, 92, 0.32);
      background: rgba(244, 195, 92, 0.08);
    }
    .auto-pill.paused .auto-dot {
      background: var(--warn);
      box-shadow: 0 0 0 3px rgba(244, 195, 92, 0.13);
    }
    .auto-toggle {
      height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      font-size: 12px;
    }
    .controls {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .dashboard {
      padding: 12px;
      display: grid;
      gap: 12px;
    }
    .market-grid {
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 8px;
    }
    .mini-stat {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.026);
      padding: 10px 11px;
    }
    .mini-stat .k {
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
      margin-bottom: 6px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .mini-stat .v {
      font-size: 16px;
      line-height: 1.1;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .mini-stat .sub {
      margin-top: 5px;
      color: var(--faint);
      font-size: 11px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .cohort-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .cohort {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(18, 28, 41, 0.94), rgba(10, 15, 22, 0.98));
      overflow: hidden;
    }
    .cohort.smart {
      box-shadow: inset 0 1px 0 rgba(94, 226, 204, 0.12);
    }
    .cohort.whale {
      box-shadow: inset 0 1px 0 rgba(139, 183, 255, 0.12);
    }
    .cohort-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      padding: 13px 14px 11px;
      border-bottom: 1px solid var(--line);
    }
    .cohort-title {
      display: flex;
      align-items: center;
      gap: 9px;
      min-width: 0;
    }
    .cohort-mark {
      width: 28px;
      height: 28px;
      display: grid;
      place-items: center;
      border-radius: 9px;
      font-size: 14px;
      font-weight: 850;
      color: #081016;
      background: var(--buy);
      flex: 0 0 auto;
    }
    .cohort.whale .cohort-mark {
      background: var(--accent);
    }
    .cohort-title h2 {
      margin: 0;
      font-size: 18px;
      line-height: 1.1;
      font-weight: 820;
      letter-spacing: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .cohort-title small {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .cohort-kpis {
      display: grid;
      grid-template-columns: repeat(3, minmax(92px, 1fr));
      gap: 8px;
      min-width: 330px;
    }
    .kpi {
      min-width: 0;
      padding-left: 10px;
      border-left: 1px solid var(--line);
    }
    .kpi .k {
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 5px;
      white-space: nowrap;
    }
    .kpi .v {
      font-size: 15px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .dominance {
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.015);
    }
    .dominance-top {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
      white-space: nowrap;
    }
    .dominance-bar {
      display: flex;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.08);
    }
    .dominance-long {
      width: 50%;
      background: linear-gradient(90deg, rgba(94, 226, 204, 0.95), rgba(94, 226, 204, 0.72));
    }
    .dominance-short {
      width: 50%;
      background: linear-gradient(90deg, rgba(255, 120, 148, 0.72), rgba(255, 120, 148, 0.95));
    }
    .side-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 10px;
    }
    .side-card {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: rgba(255, 255, 255, 0.024);
      overflow: hidden;
    }
    .side-card.long {
      box-shadow: inset 0 0 0 1px rgba(94, 226, 204, 0.13);
    }
    .side-card.short {
      box-shadow: inset 0 0 0 1px rgba(255, 120, 148, 0.12);
    }
    .side-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 10px 11px;
      border-bottom: 1px solid var(--line);
    }
    .side-name {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .side-tag {
      width: 24px;
      height: 24px;
      display: grid;
      place-items: center;
      border-radius: 7px;
      font-size: 13px;
      font-weight: 850;
      flex: 0 0 auto;
    }
    .long .side-tag {
      color: #071014;
      background: var(--buy);
    }
    .short .side-tag {
      color: #16080d;
      background: var(--sell);
    }
    .side-name strong {
      font-size: 15px;
      font-weight: 820;
      white-space: nowrap;
    }
    .side-name span {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .pnl-pill {
      max-width: 45%;
      padding: 5px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.035);
    }
    .positive {
      color: var(--buy);
    }
    .negative {
      color: var(--sell);
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .metric {
      min-width: 0;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
    }
    .metric:nth-child(2n) {
      border-left: 1px solid var(--line);
    }
    .metric .k {
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric .v {
      font-size: 14px;
      line-height: 1.15;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric.wide {
      grid-column: span 2;
    }
    .profit-line {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: 8px;
      padding: 9px 10px 10px;
    }
    .profit-bar {
      height: 8px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.08);
    }
    .profit-bar span {
      display: block;
      width: 0%;
      height: 100%;
      border-radius: inherit;
    }
    .long .profit-bar span {
      background: var(--buy);
    }
    .short .profit-bar span {
      background: var(--sell);
    }
    .profit-line b {
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .error {
      min-height: 20px;
      padding: 0 14px 12px;
      color: #ff9fb3;
      font-size: 13px;
    }
    .footer-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 14px 12px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    @media (max-width: 1420px) {
      .market-grid {
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
      .cohort-grid {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 880px) {
      .topbar-main,
      .topbar-meta {
        flex-direction: column;
        align-items: stretch;
      }
      .meta-left,
      .meta-right,
      .controls {
        justify-content: flex-start;
      }
      input {
        flex: 1 1 180px;
      }
      .market-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .cohort-head {
        grid-template-columns: 1fr;
      }
      .cohort-kpis {
        min-width: 0;
      }
      .side-grid {
        grid-template-columns: 1fr;
      }
      .footer-meta {
        align-items: flex-start;
        flex-direction: column;
        white-space: normal;
      }
    }
    @media (max-width: 520px) {
      .page {
        padding: 8px;
      }
      .shell {
        border-radius: 14px;
      }
      h1 {
        font-size: 21px;
      }
      .controls,
      .controls input,
      .controls button {
        width: 100%;
      }
      .market-grid,
      .cohort-kpis,
      .metrics {
        grid-template-columns: 1fr;
      }
      .metric:nth-child(2n) {
        border-left: 0;
      }
      .metric.wide {
        grid-column: span 1;
      }
      .dominance-top {
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="shell">
      <header class="topbar">
        <div class="topbar-meta">
          <div class="meta-left">
            <span class="eyebrow">Binance Futures</span>
            <span class="auto-pill" id="autoState"><span class="auto-dot"></span>自动更新 1s</span>
          </div>
          <div class="meta-right">
            <span class="data-pill" id="updatedAt">等待加载</span>
            <button class="auto-toggle" id="toggleBtn">暂停</button>
          </div>
        </div>
        <div class="topbar-main">
          <div class="title-wrap">
            <div class="title-row">
              <h1>合约数据面板</h1>
              <span class="symbol-pill" id="symbolBadge">BTCUSDT</span>
            </div>
          </div>
          <div class="controls">
            <input id="symbolInput" list="symbolSuggestions" value="" placeholder="搜索合约，如 SOL 或 SOLUSDT" autocomplete="off" />
            <datalist id="symbolSuggestions">
              <option value="BTCUSDT"></option>
              <option value="ETHUSDT"></option>
              <option value="SOLUSDT"></option>
              <option value="BNBUSDT"></option>
              <option value="XRPUSDT"></option>
              <option value="DOGEUSDT"></option>
              <option value="ADAUSDT"></option>
              <option value="AVAXUSDT"></option>
              <option value="LINKUSDT"></option>
              <option value="SUIUSDT"></option>
              <option value="AIXBTUSDT"></option>
              <option value="WIFUSDT"></option>
              <option value="PENDLEUSDT"></option>
              <option value="ENAUSDT"></option>
              <option value="1000PEPEUSDT"></option>
            </datalist>
            <button id="refreshBtn">刷新</button>
          </div>
        </div>
      </header>

      <section class="dashboard">
        <section class="market-grid">
          <div class="mini-stat">
            <div class="k">标记价</div>
            <div class="v" id="marketMark">--</div>
            <div class="sub">Binance Futures</div>
          </div>
          <div class="mini-stat">
            <div class="k">指数价</div>
            <div class="v" id="marketIndex">--</div>
            <div class="sub" id="marketSpread">偏离 --</div>
          </div>
          <div class="mini-stat">
            <div class="k">资金费率</div>
            <div class="v" id="marketFunding">--</div>
            <div class="sub">当前周期</div>
          </div>
          <div class="mini-stat">
            <div class="k">名义多空比率</div>
            <div class="v" id="summaryRatio">--</div>
            <div class="sub" id="summaryRatioX">--</div>
          </div>
          <div class="mini-stat">
            <div class="k">聪明钱总持仓</div>
            <div class="v" id="smartTotalPosition">--</div>
            <div class="sub" id="smartTotalCount">--</div>
          </div>
          <div class="mini-stat">
            <div class="k">巨鲸总持仓</div>
            <div class="v" id="whaleTotalPosition">--</div>
            <div class="sub" id="whaleTotalCount">--</div>
          </div>
          <div class="mini-stat">
            <div class="k">聪明钱净倾斜</div>
            <div class="v" id="smartBias">--</div>
            <div class="sub" id="smartBiasSub">--</div>
          </div>
          <div class="mini-stat">
            <div class="k">巨鲸净倾斜</div>
            <div class="v" id="whaleBias">--</div>
            <div class="sub" id="whaleBiasSub">--</div>
          </div>
        </section>

        <section class="cohort-grid">
          <article class="cohort smart" id="smartCohort">
            <div class="cohort-head">
              <div class="cohort-title">
                <span class="cohort-mark">SM</span>
                <div>
                  <h2>聪明钱</h2>
                </div>
              </div>
              <div class="cohort-kpis">
                <div class="kpi"><div class="k">总持仓</div><div class="v" id="smartHeadPosition">--</div></div>
                <div class="kpi"><div class="k">交易员</div><div class="v" id="smartHeadCount">--</div></div>
                <div class="kpi"><div class="k">多头占比</div><div class="v" id="smartHeadLongShare">--</div></div>
              </div>
            </div>
            <div class="dominance">
              <div class="dominance-top">
                <span id="smartDominanceLong">B 多头 --</span>
                <span id="smartDominanceShort">S 空头 --</span>
              </div>
              <div class="dominance-bar">
                <span class="dominance-long" id="smartDominanceLongBar"></span>
                <span class="dominance-short" id="smartDominanceShortBar"></span>
              </div>
            </div>
            <div class="side-grid">
              <section class="side-card long" data-side="long" data-view="smart"></section>
              <section class="side-card short" data-side="short" data-view="smart"></section>
            </div>
          </article>

          <article class="cohort whale" id="whaleCohort">
            <div class="cohort-head">
              <div class="cohort-title">
                <span class="cohort-mark">W</span>
                <div>
                  <h2>巨鲸</h2>
                </div>
              </div>
              <div class="cohort-kpis">
                <div class="kpi"><div class="k">总持仓</div><div class="v" id="whaleHeadPosition">--</div></div>
                <div class="kpi"><div class="k">巨鲸</div><div class="v" id="whaleHeadCount">--</div></div>
                <div class="kpi"><div class="k">多头占比</div><div class="v" id="whaleHeadLongShare">--</div></div>
              </div>
            </div>
            <div class="dominance">
              <div class="dominance-top">
                <span id="whaleDominanceLong">B 多头 --</span>
                <span id="whaleDominanceShort">S 空头 --</span>
              </div>
              <div class="dominance-bar">
                <span class="dominance-long" id="whaleDominanceLongBar"></span>
                <span class="dominance-short" id="whaleDominanceShortBar"></span>
              </div>
            </div>
            <div class="side-grid">
              <section class="side-card long" data-side="long" data-view="whale"></section>
              <section class="side-card short" data-side="short" data-view="whale"></section>
            </div>
          </article>
        </section>
      </section>

      <div class="error" id="errorBox"></div>
      <div class="footer-meta">
        <span id="metaText">每 __REFRESH_SECONDS__ 秒自动更新</span>
      </div>
    </section>
  </main>

  <script>
    const refreshSeconds = __REFRESH_SECONDS__;
    const state = {
      timer: null,
      enabled: true,
      latest: null,
      symbol: 'BTCUSDT',
      requestSeq: 0,
      inFlight: false,
    };

    const $ = (id) => document.getElementById(id);

    function clamp(value, min = 0, max = 100) {
      return Math.max(min, Math.min(max, Number(value || 0)));
    }

    function compactNumber(value) {
      const number = Number(value || 0);
      const abs = Math.abs(number);
      const sign = number < 0 ? '-' : '';
      if (abs >= 1_000_000_000) return `${sign}${(abs / 1_000_000_000).toFixed(2)}B`;
      if (abs >= 1_000_000) return `${sign}${(abs / 1_000_000).toFixed(2)}M`;
      if (abs >= 1_000) return `${sign}${(abs / 1_000).toFixed(2)}K`;
      return number.toLocaleString('en-US', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
    }

    function formatCount(value) {
      return Number(value || 0).toLocaleString('en-US');
    }

    function formatUsdFull(value) {
      return Number(value || 0).toLocaleString('en-US', {
        maximumFractionDigits: 2,
        minimumFractionDigits: 2,
      });
    }

    function formatPrice(value) {
      const number = Number(value || 0);
      const abs = Math.abs(number);
      let digits = 8;
      if (abs >= 1_000) digits = 2;
      else if (abs >= 1) digits = 4;
      return number.toLocaleString('en-US', {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      });
    }

    function formatPct(value) {
      return `${Number(value || 0).toFixed(2)}%`;
    }

    function formatRatioX(value) {
      return `${(Number(value || 0) / 100).toFixed(2)}x`;
    }

    function formatUtc(value) {
      if (!value) return '--';
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleString('zh-CN', {
        hour12: false,
        timeZone: 'America/Toronto',
      });
    }

    function spreadBps(mark, index) {
      return index > 0 ? Math.abs((mark - index) / index) * 10000 : 0;
    }

    function totalPosition(view) {
      return Number(view.long_side.current_position_usdt || 0) + Number(view.short_side.current_position_usdt || 0);
    }

    function totalCount(view) {
      return Number(view.long_side.count || 0) + Number(view.short_side.count || 0);
    }

    function ratioShare(longValue, shortValue) {
      const total = Number(longValue || 0) + Number(shortValue || 0);
      if (total <= 0) return { longPct: 50, shortPct: 50 };
      const longPct = (Number(longValue || 0) / total) * 100;
      return { longPct, shortPct: 100 - longPct };
    }

    function signedClass(value) {
      return Number(value || 0) >= 0 ? 'positive' : 'negative';
    }

    function setSignedText(id, value, suffix = '') {
      const node = $(id);
      const number = Number(value || 0);
      node.classList.toggle('positive', number >= 0);
      node.classList.toggle('negative', number < 0);
      node.textContent = `${number >= 0 ? '+' : ''}${compactNumber(number)}${suffix}`;
    }

    function sideTemplate(viewName, sideName, side, data, isWhale) {
      const isLong = sideName === 'long';
      const tag = isLong ? 'B' : 'S';
      const sideTitle = isLong ? '多头' : '空头';
      const pnl = Number(side.unrealized_pnl_usdt || 0);
      const countUnit = isWhale ? '巨鲸' : '交易员';
      const subsetText = isWhale
        ? formatCount(side.count)
        : formatCount(side.subset_count || 0);
      const fundingPct = Number(data.price.funding_rate || 0) * 100;
      const pnlText = `${pnl >= 0 ? '+' : ''}${compactNumber(pnl)}`;
      const pnlFull = `${pnl >= 0 ? '+' : ''}${formatUsdFull(pnl)}`;
      const profit = clamp(side.profit_ratio_pct);
      return `
        <div class="side-head">
          <div class="side-name">
            <span class="side-tag">${tag}</span>
            <div>
              <strong>${sideTitle}</strong>
              <span>${formatCount(side.count)} ${countUnit}</span>
            </div>
          </div>
          <div class="pnl-pill ${signedClass(pnl)}" title="${pnlFull} USDT">${pnlText}</div>
        </div>
        <div class="metrics">
          <div class="metric">
            <div class="k">仓位</div>
            <div class="v">${compactNumber(side.current_position_usdt)}</div>
          </div>
          <div class="metric">
            <div class="k">均价</div>
            <div class="v">${formatPrice(side.average_entry_price)}</div>
          </div>
          <div class="metric">
            <div class="k">数量</div>
            <div class="v">${Number(side.quantity || 0).toLocaleString('en-US', { maximumFractionDigits: 3 })}</div>
          </div>
          <div class="metric">
            <div class="k">${isWhale ? '样本' : '巨鲸子集'}</div>
            <div class="v">${subsetText}</div>
          </div>
          <div class="metric wide">
            <div class="k">未实现盈亏</div>
            <div class="v ${signedClass(pnl)}">${pnlFull}</div>
          </div>
          <div class="metric">
            <div class="k">资金费率</div>
            <div class="v ${signedClass(fundingPct)}">${fundingPct >= 0 ? '+' : ''}${formatPct(fundingPct)}</div>
          </div>
          <div class="metric">
            <div class="k">标记价</div>
            <div class="v">${formatPrice(data.price.mark_price)}</div>
          </div>
        </div>
        <div class="profit-line">
          <div class="profit-bar"><span style="width: ${profit}%"></span></div>
          <b>${formatPct(side.profit_ratio_pct)} 盈利</b>
        </div>
      `;
    }

    function renderCohort(key, view, data) {
      const isWhale = key === 'whale';
      const longSide = view.long_side;
      const shortSide = view.short_side;
      const shares = ratioShare(longSide.current_position_usdt, shortSide.current_position_usdt);
      const total = totalPosition(view);
      const count = totalCount(view);
      const bias = Number(longSide.current_position_usdt || 0) - Number(shortSide.current_position_usdt || 0);
      const prefix = key === 'smart' ? 'smart' : 'whale';

      $(`${prefix}HeadPosition`).textContent = compactNumber(total);
      $(`${prefix}HeadCount`).textContent = `${formatCount(count)} ${isWhale ? '巨鲸' : '交易员'}`;
      $(`${prefix}HeadLongShare`).textContent = formatPct(shares.longPct);
      $(`${prefix}DominanceLong`).textContent = `B 多头 ${formatPct(shares.longPct)} · ${compactNumber(longSide.current_position_usdt)}`;
      $(`${prefix}DominanceShort`).textContent = `S 空头 ${formatPct(shares.shortPct)} · ${compactNumber(shortSide.current_position_usdt)}`;
      $(`${prefix}DominanceLongBar`).style.width = `${clamp(shares.longPct)}%`;
      $(`${prefix}DominanceShortBar`).style.width = `${clamp(shares.shortPct)}%`;
      setSignedText(`${prefix}Bias`, bias);
      $(`${prefix}BiasSub`).textContent = bias >= 0 ? '多头净占优' : '空头净占优';

      document.querySelector(`[data-view="${prefix}"][data-side="long"]`).innerHTML = sideTemplate(prefix, 'long', longSide, data, isWhale);
      document.querySelector(`[data-view="${prefix}"][data-side="short"]`).innerHTML = sideTemplate(prefix, 'short', shortSide, data, isWhale);
    }

    function render(data) {
      state.latest = data;
      state.symbol = data.symbol;
      const smart = data.views.smart_money;
      const whale = data.views.whales;
      const mark = Number(data.price.mark_price || 0);
      const index = Number(data.price.index_price || 0);
      const spread = spreadBps(mark, index);
      const fundingPct = Number(data.price.funding_rate || 0) * 100;

      $('errorBox').textContent = '';
      $('symbolBadge').textContent = data.symbol;
      $('updatedAt').textContent = `更新时间 ${formatUtc(data.as_of_utc)}`;
      $('marketMark').textContent = formatPrice(mark);
      $('marketIndex').textContent = formatPrice(index);
      $('marketSpread').textContent = `偏离 ${spread.toFixed(1)} bps`;
      $('marketFunding').textContent = `${fundingPct >= 0 ? '+' : ''}${formatPct(fundingPct)}`;
      $('marketFunding').className = `v ${signedClass(fundingPct)}`;
      $('summaryRatio').textContent = formatPct(data.summary.nominal_long_short_ratio_pct);
      $('summaryRatioX').textContent = formatRatioX(data.summary.nominal_long_short_ratio_pct);
      $('smartTotalPosition').textContent = compactNumber(totalPosition(smart));
      $('smartTotalCount').textContent = `${formatCount(data.summary.total_traders)} 交易员`;
      $('whaleTotalPosition').textContent = compactNumber(totalPosition(whale));
      $('whaleTotalCount').textContent = `${formatCount(data.summary.total_whales)} 巨鲸`;

      renderCohort('smart', smart, data);
      renderCohort('whale', whale, data);
      $('metaText').textContent = `${data.symbol} · 标记价 ${formatPrice(mark)} · 指数价 ${formatPrice(index)} · 每 ${refreshSeconds} 秒自动更新`;
    }

    function normalizeSymbol(raw) {
      const value = String(raw || '').trim().toUpperCase();
      if (!value) return state.symbol;
      return value.endsWith('USDT') ? value : `${value}USDT`;
    }

    async function load(symbol = state.symbol, options = {}) {
      if (state.inFlight && !options.force) return;
      const normalizedSymbol = normalizeSymbol(symbol);
      const requestSeq = ++state.requestSeq;
      state.inFlight = true;
      state.symbol = normalizedSymbol;
      if (!state.latest) {
        $('updatedAt').textContent = '加载中';
        $('metaText').textContent = `${normalizedSymbol} · 正在请求实时数据`;
      }
      try {
        const res = await fetch(`/api/smart-money?symbol=${encodeURIComponent(normalizedSymbol)}`, { cache: 'no-store' });
        const payload = await res.json();
        if (!payload.ok) throw new Error(payload.error || 'request failed');
        if (requestSeq !== state.requestSeq) return;
        render(payload.data);
      } catch (err) {
        if (requestSeq !== state.requestSeq) return;
        $('errorBox').textContent = String(err.message || err);
      } finally {
        if (requestSeq === state.requestSeq) {
          state.inFlight = false;
        }
      }
    }

    function submitSymbolSearch() {
      const raw = $('symbolInput').value.trim();
      if (raw) {
        $('symbolInput').value = '';
      }
      load(raw || state.symbol, { force: true });
    }

    function startTimer() {
      stopTimer();
      if (!state.enabled) return;
      state.timer = setInterval(() => load(state.symbol), refreshSeconds * 1000);
    }

    function stopTimer() {
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
    }

    function renderAutoState() {
      const node = $('autoState');
      node.classList.toggle('paused', !state.enabled);
      node.innerHTML = `<span class="auto-dot"></span>${state.enabled ? `自动更新 ${refreshSeconds}s` : '已暂停'}`;
      $('toggleBtn').textContent = state.enabled ? '暂停' : '恢复';
    }

    $('refreshBtn').addEventListener('click', submitSymbolSearch);
    $('symbolInput').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') submitSymbolSearch();
    });
    $('toggleBtn').addEventListener('click', () => {
      state.enabled = !state.enabled;
      renderAutoState();
      startTimer();
    });

    renderAutoState();
    load();
    startTimer();
  </script>
</body>
</html>""".replace("__REFRESH_SECONDS__", str(DEFAULT_REFRESH_SECONDS))



def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a local Binance smart-money live panel.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(json.dumps({"status": "starting", "url": f"http://{args.host}:{args.port}"}, ensure_ascii=False))
    serve_smart_money_panel(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
