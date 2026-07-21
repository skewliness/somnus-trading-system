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
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
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


V6_DEMO_STATE_FILE = ROOT / "data/trading_system/v6_execution/state.json"
V6_DEMO_SUMMARY_FILE = ROOT / "data/trading_system/v6_execution/latest_run_summary.json"
V6_LIVE_STATE_FILE = ROOT / "data/trading_system/v6_execution_live/state.json"
V6_LIVE_SUMMARY_FILE = ROOT / "data/trading_system/v6_execution_live/latest_run_summary.json"
V6_LIVE_ENV_FILE = ROOT / ".env.v6_live"
V6_LIVE_MOMO2_STATE_FILE = ROOT / "data/trading_system/v6_execution_live_momo2/state.json"
V6_LIVE_MOMO2_SUMMARY_FILE = ROOT / "data/trading_system/v6_execution_live_momo2/latest_run_summary.json"
V6_LIVE_MOMO2_ENV_FILE = ROOT / ".env.v6_live_momo2"
TELEGRAM_WHALE_BRIDGE_FILE = ROOT / "data/trading_system/skill_bridges/telegram_whale_crypto.json"
ZERO = Decimal("0")
BEIJING_TZ = timezone(timedelta(hours=8))


def _load_env() -> argparse.Namespace:
    tg.load_dotenv(ROOT / ".env.local")
    # The V5 telegram env only contains the V5 bot identity. Loading it is safe
    # because this helper never sends through Telegram.
    tg.load_dotenv(ROOT / ".env.telegram_v5")
    return tg.parse_args([])


def _emit_text(text: str) -> int:
    sys.stdout.write(text.rstrip() + "\n")
    return 0


def _read_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _v6_status_from_files(state_file: Path, summary_file: Path) -> dict[str, Any]:
    state = tg.load_json(state_file)
    summary = tg.load_json(summary_file)
    if isinstance(state, dict) and "positions" in state:
        positions = state.get("positions") or []
    else:
        positions = summary.get("positions") or []
    return {"v6": {"positions": positions}}


def _live_v6_config(account: str = "momo") -> tuple[tg.V6ExecutionConfig, str]:
    env_file = V6_LIVE_MOMO2_ENV_FILE if account == "momo2" else V6_LIVE_ENV_FILE
    values = _read_env_values(env_file)
    api_key = values.get("BINANCE_V6_API_KEY", "")
    api_secret = values.get("BINANCE_V6_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("V6 Live凭证未配置")

    config = tg.V6ExecutionConfig.from_env()
    config.order_mode = tg.canonical_order_mode("live_order")
    config.live_api_key = api_key
    config.live_api_secret = api_secret
    config.live_base_url = values.get("TS_V6_BINANCE_LIVE_BASE_URL", config.live_base_url)
    label = values.get("BINANCE_V6_API_LABEL", "momo").strip() or "momo"
    return config, label


def _live_v6_label(account: str = "momo") -> str:
    env_file = V6_LIVE_MOMO2_ENV_FILE if account == "momo2" else V6_LIVE_ENV_FILE
    fallback = "momo2" if account == "momo2" else "momo"
    values = _read_env_values(env_file)
    return values.get("BINANCE_V6_API_LABEL", fallback).strip() or fallback


def _live_v6_balance_text(config: tg.V6ExecutionConfig) -> str:
    client = tg.V6BinanceClient(config)
    try:
        account = client.signed_request("GET", "/fapi/v2/account", client.signed_params())
    finally:
        client.close()
    if not isinstance(account, dict):
        raise RuntimeError("Binance账户余额返回格式无效")

    raw_balance = account.get("totalMarginBalance")
    if raw_balance in (None, ""):
        raw_balance = account.get("totalWalletBalance")
    if raw_balance in (None, ""):
        assets = account.get("assets") or []
        usdt = next(
            (row for row in assets if isinstance(row, dict) and str(row.get("asset") or "").upper() == "USDT"),
            {},
        )
        raw_balance = usdt.get("marginBalance") or usdt.get("walletBalance")
    if raw_balance in (None, ""):
        raise RuntimeError("Binance账户余额字段缺失")
    return f"💰 当前余额={_decimal(raw_balance):,.2f}U"


def _prepend_position_balance(response: str, balance_text: str) -> str:
    lines = response.splitlines()
    if not lines:
        return balance_text
    return "\n".join([lines[0], balance_text, *lines[1:]])


def _positions(args: argparse.Namespace, targets: list[str]) -> int:
    if not targets:
        targets = ["v6"]
    demo_status: dict[str, Any] | None = None
    chunks = []
    for target in targets:
        target = target.lower().strip()
        if target in {"momo", "live", "momo2"}:
            account = "momo2" if target == "momo2" else "momo"
            config, label = _live_v6_config(account) if account == "momo2" else _live_v6_config()
            state_file = V6_LIVE_MOMO2_STATE_FILE if account == "momo2" else V6_LIVE_STATE_FILE
            summary_file = V6_LIVE_MOMO2_SUMMARY_FILE if account == "momo2" else V6_LIVE_SUMMARY_FILE
            live_status = _v6_status_from_files(state_file, summary_file)
            response = tg.build_position_response(
                "v6",
                live_status,
                v6_config=config,
                v6_title=f"[Somnus持仓] V6-{label}",
                v6_exchange_source="binance_live_positionRisk",
            )
            try:
                balance_text = _live_v6_balance_text(config)
            except Exception:
                balance_text = "💰 当前余额=读取失败"
            chunks.append(_prepend_position_balance(response, balance_text))
            continue
        if target not in {"v5", "v6"}:
            raise SystemExit(f"unsupported position target: {target}")
        if demo_status is None:
            demo_status = tg.collect_status(args)
        title = "[Somnus持仓] V6 Demo" if target == "v6" else None
        chunks.append(tg.build_position_response(target, demo_status, v6_title=title))
    return _emit_text("\n\n".join(chunks))


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value or "0"))
    except (InvalidOperation, TypeError, ValueError):
        return ZERO


def _live_history_hours(targets: list[str]) -> int | None:
    if len(targets) < 2 or targets[0].lower().strip() not in {"momo", "live"}:
        return None
    token = targets[1].lower().strip()
    if token in {"history", "pnl", "历史", "盈亏"}:
        token = targets[2].lower().strip() if len(targets) >= 3 else "24h"
    elif len(targets) > 2:
        raise SystemExit("历史盈亏格式：p momo 24h")
    match = re.fullmatch(r"(\d{1,3})(?:h|小时)", token)
    if not match:
        raise SystemExit("历史盈亏格式：p momo 24h")
    hours = int(match.group(1))
    if not 1 <= hours <= 168:
        raise SystemExit("历史盈亏窗口仅支持1至168小时")
    return hours


def _fetch_signed_history_rows(
    client: tg.V6BinanceClient,
    endpoint: str,
    *,
    start_ms: int,
    end_ms: int,
    max_pages: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    cursor = start_ms
    for _ in range(max_pages):
        batch = client.signed_request(
            "GET",
            endpoint,
            {
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
                **client.signed_params(),
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        max_time = cursor
        for item in batch:
            if not isinstance(item, dict):
                continue
            item_time = int(item.get("time") or 0)
            if item_time < start_ms or item_time > end_ms:
                continue
            if endpoint.endswith("userTrades"):
                identity = (item.get("symbol"), item.get("id"))
            else:
                identity = (
                    item.get("symbol"),
                    item.get("incomeType"),
                    item.get("tranId"),
                    item.get("tradeId"),
                    item_time,
                )
            if identity not in seen:
                seen.add(identity)
                rows.append(item)
            max_time = max(max_time, item_time)
        if len(batch) < 1000 or max_time < cursor:
            break
        cursor = max_time + 1
        if cursor > end_ms:
            break
    rows.sort(key=lambda item: int(item.get("time") or 0))
    return rows


def _trade_direction(row: dict[str, Any], *, closing: bool = False) -> str:
    position_side = str(row.get("positionSide") or "").upper()
    side = str(row.get("side") or "").upper()
    if position_side == "LONG":
        return "long"
    if position_side == "SHORT":
        return "short"
    if closing:
        return "long" if side == "SELL" else "short" if side == "BUY" else ""
    return ""


def _is_close_trade(row: dict[str, Any]) -> bool:
    position_side = str(row.get("positionSide") or "").upper()
    side = str(row.get("side") or "").upper()
    if position_side == "LONG":
        return side == "SELL"
    if position_side == "SHORT":
        return side == "BUY"
    return abs(_decimal(row.get("realizedPnl"))) > ZERO


def _format_usdt(value: Decimal) -> str:
    return f"{value:+.3f}U"


def _format_live_history_report(
    trades: list[dict[str, Any]],
    income_rows: list[dict[str, Any]],
    *,
    hours: int,
    label: str,
    generated_at: datetime,
) -> str:
    close_groups: dict[tuple[str, str], dict[str, Any]] = {}
    trade_fees: dict[tuple[str, str], Decimal] = defaultdict(lambda: ZERO)
    for row in trades:
        symbol = str(row.get("symbol") or "").upper()
        direction = _trade_direction(row, closing=_is_close_trade(row))
        if not symbol or not direction:
            continue
        if str(row.get("commissionAsset") or "USDT").upper() == "USDT":
            trade_fees[(symbol, direction)] += _decimal(row.get("commission"))
        if not _is_close_trade(row):
            continue
        price = _decimal(row.get("price"))
        quantity = _decimal(row.get("qty"))
        realized = _decimal(row.get("realizedPnl"))
        if price <= ZERO or quantity <= ZERO:
            continue
        implied_entry = price - (realized / quantity) if direction == "long" else price + (realized / quantity)
        group = close_groups.setdefault(
            (symbol, direction),
            {
                "quantity": ZERO,
                "entry_quote": ZERO,
                "exit_quote": ZERO,
                "realized": ZERO,
                "order_ids": set(),
            },
        )
        group["quantity"] += quantity
        group["entry_quote"] += implied_entry * quantity
        group["exit_quote"] += price * quantity
        group["realized"] += realized
        group["order_ids"].add(str(row.get("orderId") or row.get("id") or ""))

    income_by_symbol: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: {"realized": ZERO, "commission": ZERO, "funding": ZERO}
    )
    for row in income_rows:
        if str(row.get("asset") or "USDT").upper() != "USDT":
            continue
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        income_type = str(row.get("incomeType") or "").upper()
        value = _decimal(row.get("income"))
        if income_type == "REALIZED_PNL":
            income_by_symbol[symbol]["realized"] += value
        elif income_type == "COMMISSION":
            income_by_symbol[symbol]["commission"] += value
        elif income_type == "FUNDING_FEE":
            income_by_symbol[symbol]["funding"] += value

    symbols = sorted({symbol for symbol, _ in close_groups} | {symbol for symbol, values in income_by_symbol.items() if values["realized"]})
    start_at = generated_at - timedelta(hours=hours)
    lines = [
        f"📒 [Somnus历史盈亏] V6 LIVE · {label}",
        f"窗口={start_at.astimezone(BEIJING_TZ).strftime('%m-%d %H:%M')} 至 "
        f"{generated_at.astimezone(BEIJING_TZ).strftime('%m-%d %H:%M')}｜Binance LIVE真实成交",
        "口径=已实现盈亏-手续费+资金费｜不含当前持仓浮盈亏",
    ]
    if not symbols:
        lines.append(f"最近{hours}小时没有已平仓成交。")
        return "\n".join(lines)

    totals = {"realized": ZERO, "commission": ZERO, "funding": ZERO, "net": ZERO}
    symbol_values: dict[str, dict[str, Decimal]] = {}
    for symbol in symbols:
        income = income_by_symbol.get(symbol) or {"realized": ZERO, "commission": ZERO, "funding": ZERO}
        if not income_rows:
            income = {
                "realized": sum((group["realized"] for (name, _), group in close_groups.items() if name == symbol), ZERO),
                "commission": -sum((trade_fees[(name, direction)] for name, direction in close_groups if name == symbol), ZERO),
                "funding": ZERO,
            }
        net = income["realized"] + income["commission"] + income["funding"]
        symbol_values[symbol] = {**income, "net": net}
        for key in totals:
            totals[key] += symbol_values[symbol][key]

    for symbol in sorted(symbols, key=lambda name: (-abs(symbol_values[name]["net"]), name)):
        lines.extend(["", f"【{symbol}】"])
        for direction in ("long", "short"):
            group = close_groups.get((symbol, direction))
            if not group or group["quantity"] <= ZERO:
                continue
            entry_price = group["entry_quote"] / group["quantity"]
            exit_price = group["exit_quote"] / group["quantity"]
            icon = "💹 多" if direction == "long" else "📉 空"
            lines.append(
                f"{icon}｜开仓均价={tg.format_price(entry_price)}｜平仓均价={tg.format_price(exit_price)}｜"
                f"平仓订单={len(group['order_ids'])}"
            )
        values = symbol_values[symbol]
        lines.append(
            f"已实现={_format_usdt(values['realized'])}｜手续费={abs(values['commission']):.3f}U｜"
            f"资金费={_format_usdt(values['funding'])}｜币种总盈利={_format_usdt(values['net'])}"
        )

    lines.extend(
        [
            "",
            "合计："
            f"已实现={_format_usdt(totals['realized'])}｜手续费={abs(totals['commission']):.3f}U｜"
            f"资金费={_format_usdt(totals['funding'])}｜净盈利={_format_usdt(totals['net'])}",
        ]
    )
    return "\n".join(lines)


def _live_history(hours: int) -> int:
    config, label = _live_v6_config()
    generated_at = datetime.now(timezone.utc)
    start_ms = int((generated_at - timedelta(hours=hours)).timestamp() * 1000)
    end_ms = int(generated_at.timestamp() * 1000)
    client = tg.V6BinanceClient(config)
    try:
        trades = _fetch_signed_history_rows(client, "/fapi/v1/userTrades", start_ms=start_ms, end_ms=end_ms)
        try:
            income_rows = _fetch_signed_history_rows(client, "/fapi/v1/income", start_ms=start_ms, end_ms=end_ms)
        except Exception:
            income_rows = []
    finally:
        client.close()
    return _emit_text(
        _format_live_history_report(
            trades,
            income_rows,
            hours=hours,
            label=label,
            generated_at=generated_at,
        )
    )


def _chart(
    symbol: str,
    interval: str,
    focus: str | None,
    *,
    as_json: bool,
    execution_scope: str = "default",
    account_label: str = "",
) -> int:
    normalized_symbol = tg.normalize_symbol(symbol)
    normalized_interval = tg.normalize_interval(interval) or "15m"
    focus_period = tg.parse_focus_period(focus) if focus else None
    payload = tg.build_chart_image(
        normalized_symbol,
        normalized_interval,
        focus_period,
        execution_scope=execution_scope,
        account_label=account_label,
    )
    result: dict[str, Any] = {
        "symbol": normalized_symbol,
        "interval": normalized_interval,
        "focus_period": focus_period,
        "execution_scope": execution_scope,
        "account_label": account_label,
        "image": str(Path(payload["output"]).resolve()),
        "caption": tg.format_chart_caption(payload),
    }
    if as_json:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"图片={result['image']}\n{result['caption']}\n")
    return 0


def _parse_chart_tokens(tokens: list[str]) -> tuple[str, str, str | None, str, str]:
    values = [str(item or "").strip() for item in tokens if str(item or "").strip()]
    execution_scope = "default"
    account_label = ""
    if values and values[0].lower() in {"momo", "live"}:
        values.pop(0)
        execution_scope = "live"
        account_label = _live_v6_label()
    if not values:
        raise SystemExit("fc momo 需要指定币种，例如：fc momo BTC")
    if len(values) > 3:
        raise SystemExit("fc 参数过多；格式：fc [momo] <币种> [周期] [BB周期]")
    symbol = values[0]
    interval = values[1] if len(values) >= 2 else "15m"
    focus = values[2] if len(values) >= 3 else None
    return symbol, interval, focus, execution_scope, account_label


def _format_usd_compact(value: Any) -> str:
    try:
        amount = float(value or 0.0)
    except (TypeError, ValueError):
        amount = 0.0
    absolute = abs(amount)
    if absolute >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{amount / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{amount / 1_000:.2f}K"
    return f"{amount:.0f}"


def _telegram_whale_summary(limit: int = 5) -> str:
    if not TELEGRAM_WHALE_BRIDGE_FILE.exists():
        return ""
    try:
        payload = json.loads(TELEGRAM_WHALE_BRIDGE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    windows = payload.get("windows") or {}
    window_label = "近60分钟"
    window = (windows.get("60m") or {})
    if int(window.get("event_count") or 0) <= 0 and int((windows.get("24h") or {}).get("event_count") or 0) > 0:
        window_label = "近24小时"
        window = windows.get("24h") or {}
    totals = window.get("totals") or {}
    symbols = window.get("symbols") or []
    event_count = int(window.get("event_count") or 0)
    lines = ["", "🐋【Hyperdash 加密资金流】"]
    if event_count <= 0:
        lines.append("近24小时暂无通过加密资产校验的强平事件。")
        return "\n".join(lines)

    long_usd = float(totals.get("liquidated_long_usd") or 0.0)
    short_usd = float(totals.get("liquidated_short_usd") or 0.0)
    net_usd = float(totals.get("net_liquidation_pressure_usd") or 0.0)
    direction = "偏多" if net_usd > 0 else "偏空" if net_usd < 0 else "中性"
    lines.append(
        f"{window_label}={event_count}条｜多单强平={_format_usd_compact(long_usd)}"
        f"｜空单强平={_format_usd_compact(short_usd)}｜净压力={direction} {_format_usd_compact(abs(net_usd))}"
    )
    for item in symbols[:limit]:
        symbol = str(item.get("symbol") or "--")
        symbol_long = float(item.get("liquidated_long_usd") or 0.0)
        symbol_short = float(item.get("liquidated_short_usd") or 0.0)
        imminent_long = float(item.get("imminent_long_usd") or 0.0)
        imminent_short = float(item.get("imminent_short_usd") or 0.0)
        detail = [
            f"多爆={_format_usd_compact(symbol_long)}",
            f"空爆={_format_usd_compact(symbol_short)}",
        ]
        if imminent_long:
            detail.append(f"临近多爆={_format_usd_compact(imminent_long)}")
        if imminent_short:
            detail.append(f"临近空爆={_format_usd_compact(imminent_short)}")
        lines.append(f"• {symbol}：{'｜'.join(detail)}")
    lines.append("用途=宏观与仓位辅助，不单独触发下单。")
    return "\n".join(lines)


def _report(symbols: list[str]) -> int:
    normalized = [tg.normalize_symbol(item) for item in symbols if tg.normalize_symbol(item)]
    if not normalized:
        normalized = ["BTCUSDT", "ETHUSDT"]
    report = tg.build_report_response(normalized)
    whale_summary = _telegram_whale_summary()
    if whale_summary:
        report = (report.rstrip() + "\n" + whale_summary).strip()
    return _emit_text(report[:6000])


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
    whale_summary = _telegram_whale_summary()
    if whale_summary:
        report = (report.rstrip() + "\n" + whale_summary).strip()
    return _emit_text(report[:6000])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Somnus intelligence commands for Hermes/Xuejian.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("p", help="Show Demo or Live positions.")
    p.add_argument("targets", nargs="*", help="Defaults to V6 Demo. Use momo or momo2 for V6 Live; v5/v6 remain supported.")

    fc = sub.add_parser("fc", help="Render chart image.")
    fc.add_argument("tokens", nargs="+", help="[momo] <symbol> [interval] [bb period]")
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
        history_hours = _live_history_hours(ns.targets)
        if history_hours is not None:
            return _live_history(history_hours)
        return _positions(args, ns.targets)
    if ns.command == "fc":
        symbol, interval, focus, execution_scope, account_label = _parse_chart_tokens(ns.tokens)
        return _chart(
            symbol,
            interval,
            focus,
            as_json=ns.json,
            execution_scope=execution_scope,
            account_label=account_label,
        )
    if ns.command == "report":
        return _report(ns.symbols)
    if ns.command == "news":
        return _news(args, ns.lookback_hours)
    raise SystemExit(f"unsupported command: {ns.command}")


if __name__ == "__main__":
    raise SystemExit(main())
