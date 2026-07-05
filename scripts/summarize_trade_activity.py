#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTION_LABELS = {
    "open_long": "开多",
    "open_short": "开空",
    "close_long": "平多",
    "close_short": "平空",
    "exit": "平仓",
    "take_profit_partial": "部分止盈",
}

REASON_LABELS = {
    "shadow_stop_loss_hit": "影子止损",
    "shadow_take_profit_hit": "影子止盈",
    "shadow_partial_take_profit": "影子部分止盈",
    "shadow_trailing_stop": "影子跟踪止盈",
    "shadow_runner_max_tp": "影子趋势止盈",
    "shadow_max_hold_reached": "影子超时平仓",
    "shadow_market_flash_crash_protection": "影子闪崩保护",
    "stop_loss_hit": "止损",
    "take_profit_hit": "止盈",
    "score_faded": "分数衰退",
    "max_hold_reached": "持仓超时",
    "market_flash_crash_protection": "闪崩保护",
    "market_snapshot_missing_timeout": "行情缺失超时",
    "oi_divergence_signal_fade": "OI 背离信号衰退",
}

STATUS_LABELS = {
    "submitted": "已提交",
    "submitted_unprotected": "已开仓-保护单失败",
    "failed": "失败",
    "partial": "部分失败",
    "empty": "空请求",
    "placeholder": "占位",
    "unknown": "未知",
}

DIRECTION_LABELS = {
    "long": "多",
    "short": "空",
}

SIDE_LABELS = {
    "BUY": "买入",
    "SELL": "卖出",
}

REQUEST_MODE_LABELS = {
    "test_order": "本地测试单",
    "testnet_order": "币安",
    "live_order": "真实下单",
    "disabled": "关闭",
}

SIGNAL_SOURCE_LABELS = {
    "paper": "旧 paper 交易",
    "shadow_paper": "shadow 虚拟实盘",
}


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def _fmt_num(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except Exception:
        return "--"
    return f"{number:.{digits}f}{suffix}"


def _fmt_pct_from_fraction(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "--"


def _label_action(value: Any) -> str:
    raw = str(value or "")
    return ACTION_LABELS.get(raw, raw or "--")


def _label_reason(value: Any) -> str:
    raw = str(value or "")
    return REASON_LABELS.get(raw, raw or "--")


def _label_direction(value: Any) -> str:
    raw = str(value or "")
    return DIRECTION_LABELS.get(raw, raw or "--")


def _label_side(value: Any) -> str:
    raw = str(value or "")
    return SIDE_LABELS.get(raw, raw or "--")


def _label_request_mode(value: Any) -> str:
    raw = str(value or "")
    return REQUEST_MODE_LABELS.get(raw, raw or "--")


def _label_signal_source(value: Any) -> str:
    raw = str(value or "")
    return SIGNAL_SOURCE_LABELS.get(raw, raw or "--")


def _macro_line(report: dict[str, Any]) -> str:
    macro = report.get("macro_trend") if isinstance(report.get("macro_trend"), dict) else {}
    if not macro:
        return "宏观方向: --"
    return (
        "宏观方向: "
        f"BTC {macro.get('btc_trend', '--')} "
        f"MA7距离={_fmt_num(macro.get('btc_distance_pct'), 2)}% "
        f"斜率3日={_fmt_num(macro.get('btc_ma7_slope_pct_3d'), 2)}%；"
        f"ETH {macro.get('eth_trend', '--')} "
        f"MA7距离={_fmt_num(macro.get('eth_distance_pct'), 2)}% "
        f"斜率3日={_fmt_num(macro.get('eth_ma7_slope_pct_3d'), 2)}%；"
        f"方向={macro.get('macro_direction', '--')} "
        f"置信={macro.get('macro_confidence', '--')} "
        f"来源={macro.get('source', '--')} "
        f"口径=UTC日线"
    )


def _line_for_entry(item: dict[str, Any]) -> str:
    trigger = item.get("shadow_trigger") if isinstance(item.get("shadow_trigger"), dict) else {}
    trigger_text = ""
    if trigger:
        trigger_text = (
            f"｜触发={trigger.get('model', '--')} "
            f"moon={_fmt_num(trigger.get('prob_moon'), 3)}"
        )
    return (
        f"【{_label_action(item.get('action'))}】{item.get('symbol', '--')} "
        f"入场={_fmt_num(item.get('entry_price'), 8)} "
        f"止损={_fmt_num(item.get('stop_loss_price'), 8)} "
        f"止盈={_fmt_num(item.get('take_profit_price'), 8)} "
        f"名义={_fmt_num(item.get('notional_usdt'), 2)}U "
        f"score={_fmt_num(item.get('entry_score', item.get('score')), 2)}"
        f"{trigger_text}"
    )


def _line_for_exit(item: dict[str, Any]) -> str:
    return (
        f"【{_label_reason(item.get('reason'))}】{item.get('symbol', '--')} "
        f"入场={_fmt_num(item.get('entry_price'), 8)} "
        f"出场={_fmt_num(item.get('exit_price'), 8)} "
        f"收益={_fmt_pct_from_fraction(item.get('pnl_pct'))} "
        f"PnL={_fmt_num(item.get('pnl_usdt'), 4)}U"
    )


def _line_for_position(item: dict[str, Any], label: str = "持仓") -> str:
    trigger = item.get("shadow_trigger") if isinstance(item.get("shadow_trigger"), dict) else {}
    trigger_text = f"｜触发={trigger.get('model', '--')} moon={_fmt_num(trigger.get('prob_moon'), 3)}" if trigger else ""
    pnl_usdt = item.get("unrealized_pnl_usdt")
    if pnl_usdt is None:
        try:
            pnl_usdt = float(item.get("pnl_pct")) * float(item.get("notional_usdt"))
        except Exception:
            pnl_usdt = None
    return (
        f"【{label}】{item.get('symbol', '--')} "
        f"方向={_label_direction(item.get('side', 'long'))} "
        f"入场={_fmt_num(item.get('entry_price'), 8)} "
        f"现价={_fmt_num(item.get('current_price'), 8)} "
        f"浮盈={_fmt_pct_from_fraction(item.get('pnl_pct'))} "
        f"PnL={_fmt_num(pnl_usdt, 4)}U "
        f"名义={_fmt_num(item.get('notional_usdt'), 2)}U"
        f"{trigger_text}"
    )


def _latest_execution_results(results_dir: Path, limit: int) -> list[dict[str, Any]]:
    if not results_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)[-limit * 3:]:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        for result in payload.get("results", []) or []:
            if isinstance(result, dict):
                rows.append({**result, "_file": str(path), "_request_mode": payload.get("request_mode")})
    return rows[-limit:]


def _current_execution_payload(report: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    bridge = report.get("execution_bridge_summary") if isinstance(report.get("execution_bridge_summary"), dict) else {}
    request_file = str(bridge.get("request_file") or "")
    if not request_file:
        return {}
    expected = results_dir / f"{Path(request_file).stem}_result.json"
    payload = _load_json(expected)
    return payload if isinstance(payload, dict) else {}


def _current_execution_results(report: dict[str, Any], results_dir: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _current_execution_payload(report, results_dir)
    rows: list[dict[str, Any]] = []
    for result in payload.get("results", []) or []:
        if isinstance(result, dict):
            rows.append({**result, "_file": str(results_dir / f"{Path(str((report.get('execution_bridge_summary') or {}).get('request_file') or '')).stem}_result.json"), "_request_mode": payload.get("request_mode")})
    return rows[-limit:], payload


def _line_for_execution_result(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    request = item.get("request") if isinstance(item.get("request"), dict) else {}
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    response = item.get("response") if isinstance(item.get("response"), dict) else {}
    status = str(item.get("status") or "unknown")
    side = params.get("side") or response.get("side") or ""
    quantity = params.get("quantity") or response.get("executedQty") or response.get("origQty") or ""
    mode = item.get("request_mode") or item.get("_request_mode") or meta.get("request_mode") or "--"
    base = (
        f"【币安执行-{STATUS_LABELS.get(status, status)}】{item.get('symbol', '--')} "
        f"{_label_action(meta.get('action'))}/{_label_side(side)} "
        f"模式={_label_request_mode(mode)} qty={quantity or '--'} "
        f"名义_hint={_fmt_num(meta.get('quantity_hint_usdt'), 2)}U"
    )
    if status == "failed":
        return base + f"｜错误={str(item.get('error') or '')[:220]}"
    follow_up_orders = item.get("follow_up_orders") if isinstance(item.get("follow_up_orders"), list) else []
    failed_follow_up = next((order for order in follow_up_orders if isinstance(order, dict) and order.get("status") == "failed"), None)
    if failed_follow_up:
        return base + f"｜保护单失败={str(failed_follow_up.get('error') or '')[:180]}"
    return base


def summarize(report: dict[str, Any], state: dict[str, Any], results_dir: Path, limit: int = 8) -> str:
    now = datetime.now(timezone.utc).isoformat()
    paper = report.get("paper_rebalance") if isinstance(report.get("paper_rebalance"), dict) else {}
    bridge = report.get("execution_bridge_summary") if isinstance(report.get("execution_bridge_summary"), dict) else {}
    shadow = paper.get("shadow_paper") if isinstance(paper.get("shadow_paper"), dict) else {}

    paper_entries = paper.get("entries") if isinstance(paper.get("entries"), list) else []
    paper_exits = paper.get("exits") if isinstance(paper.get("exits"), list) else []
    shadow_entries = shadow.get("entries") if isinstance(shadow.get("entries"), list) else []
    shadow_exits = shadow.get("exits") if isinstance(shadow.get("exits"), list) else []
    shadow_positions = state.get("shadow_positions") if isinstance(state.get("shadow_positions"), list) else []
    normal_positions = state.get("positions") if isinstance(state.get("positions"), list) else []
    shadow_journal = state.get("shadow_journal") if isinstance(state.get("shadow_journal"), list) else []
    latest_exec, current_exec_payload = _current_execution_results(report, results_dir, limit)

    lines = [
        "===== 交易操作日志 =====",
        f"时间={now} run_id={report.get('run_id', '--')}",
        (
            f"模式: 下单={_label_request_mode(bridge.get('request_mode'))} "
            f"信号={_label_signal_source(bridge.get('signal_source'))} "
            f"候选操作={bridge.get('intent_count', 0)} "
            f"拦截={bridge.get('blocked_intent_count', 0)}"
        ),
        (
            f"账户: shadow持仓={len(shadow_positions)} paper持仓={len(normal_positions)} "
            f"shadow浮动PnL={_fmt_num(shadow.get('unrealized_pnl_usdt'), 4)}U "
            f"shadow本轮已实现={_fmt_num(shadow.get('realized_pnl_usdt'), 4)}U"
        ),
        _macro_line(report),
        "",
        "本轮开仓:",
    ]
    entries = [*paper_entries, *shadow_entries]
    lines.extend(_line_for_entry(item) for item in entries[:limit] if isinstance(item, dict))
    if not entries:
        lines.append("无")

    lines.append("")
    lines.append("本轮平仓:")
    exits = [*paper_exits, *shadow_exits]
    lines.extend(_line_for_exit(item) for item in exits[:limit] if isinstance(item, dict))
    if not exits:
        lines.append("无")

    lines.append("")
    lines.append("当前持仓:")
    position_lines = [
        *(_line_for_position(item, "paper持仓") for item in normal_positions[:limit] if isinstance(item, dict)),
        *(_line_for_position(item, "shadow持仓") for item in shadow_positions[:limit] if isinstance(item, dict)),
    ]
    lines.extend(position_lines[:limit])
    if not position_lines:
        lines.append("无")

    lines.append("")
    lines.append("本轮币安执行:")
    lines.extend(_line_for_execution_result(item) for item in latest_exec)
    if not latest_exec:
        current_status = str(current_exec_payload.get("status") or "empty")
        if current_status == "empty":
            lines.append("无，本轮没有生成需要发送到币安的订单")
        else:
            lines.append(f"无，本轮执行结果状态={STATUS_LABELS.get(current_status, current_status)}")

    latest_errors = [item for item in latest_exec if item.get("status") == "failed"]
    if latest_errors:
        lines.append("")
        lines.append("需要注意:")
        for item in latest_errors[:3]:
            lines.append(f"- {item.get('symbol', '--')}: {str(item.get('error') or '')[:300]}")

    lines.append("===== 交易操作日志结束 =====")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a Chinese trading activity summary.")
    parser.add_argument("--report", default="data/trading_system/pipeline_run_report.json")
    parser.add_argument("--state", default="data/trading_system/paper_state.json")
    parser.add_argument("--results-dir", default="data/trading_system/execution_results")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    report = _load_json(Path(args.report))
    state = _load_json(Path(args.state))
    if not isinstance(report, dict):
        report = {}
    if not isinstance(state, dict):
        state = {}
    print(summarize(report, state, Path(args.results_dir), limit=max(1, args.limit)))


if __name__ == "__main__":
    main()
