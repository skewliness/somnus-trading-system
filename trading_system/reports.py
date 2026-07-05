from __future__ import annotations

from pathlib import Path

from trading_system.persistence import atomic_write_text


def _fmt_bool(value: object) -> str:
    return "on" if value else "off"


def _fmt_coverage_item(category: str, item: dict) -> str:
    mode = item.get("mode", "missing")
    if mode == "parallel":
        return f"- {category}: Binance + OKX"
    if mode == "okx_only":
        return f"- {category}: OKX"
    if mode == "binance_only":
        return f"- {category}: Binance"
    return f"- {category}: missing"


def _fmt_signal_item(item: dict, rank: int) -> str:
    symbol = item.get("symbol", "?")
    score = item.get("score", 0)
    mentions = item.get("mentions", 0)
    parallel = "+".join(item.get("parallel_sources", [])) or "single"
    return f"{rank}. {symbol}｜热度 {score}｜提及 {mentions}｜来源 {parallel}"


def _fmt_candidate_item(item: dict, rank: int) -> str:
    base = f"{rank}. {item['symbol']}｜评分 {item['score']}｜风险 {item['risk_band']}"
    blocked = item.get("blocked_reasons") or []
    if blocked and item.get("execution_status") == "watch_only":
        return base + f"｜未执行 {'/'.join(blocked[:2])}"
    return base


def _fmt_meme_radar_item(item: dict, rank: int) -> str:
    symbol = item.get("symbol", "?")
    radar_score = item.get("radar_score", 0)
    tags = "/".join((item.get("tags") or [])[:2]) or "watch"
    status = item.get("execution_status", "watch_only")
    next_step = (item.get("next_validation") or [""])[0]
    return f"{rank}. {symbol}｜雷达 {radar_score}｜{status}｜{tags}｜验 {next_step[:24]}"


def _fmt_exec_line(bridge: dict) -> str:
    intent_count = bridge.get("intent_count", 0)
    request_mode = bridge.get("request_mode", "unknown")
    if intent_count:
        return f"🟢 执行：{request_mode}｜intent={intent_count}"
    return f"⚪ 执行：{request_mode}｜本轮无下单"


def _fmt_exec_request(item: dict, rank: int) -> str:
    symbol = item.get("symbol") or "?"
    action = item.get("action") or "unknown"
    side = item.get("side") or "?"
    request_mode = item.get("request_mode") or "unknown"
    qty = item.get("quantity_hint_usdt") or 0
    leverage = item.get("leverage") or 0
    status = item.get("status") or "unknown"
    rationale = str(item.get("rationale") or "")
    detail = f"{rank}. {symbol}｜{action}/{side}｜{request_mode}｜名义 {qty:g}U｜杠杆 {leverage}x｜状态 {status}"
    if rationale:
        detail += f"｜理由 {rationale[:60]}"
    return detail


def _fmt_exec_error(item: dict, rank: int) -> str:
    symbol = item.get("symbol") or "?"
    action = item.get("action") or "unknown"
    params = item.get("params") or {}
    side = params.get("side") or "?"
    err = str(item.get("error") or "")
    return f"{rank}. {symbol}｜{action}/{side}｜报错 {err[:120]}"


def write_markdown_report(report: dict, path: Path) -> None:
    analysis_mode = report.get("analysis_mode", report.get("mode"))
    bridge = report.get("execution_bridge_summary", {})
    execution_results_summary = report.get("execution_results_summary", {})
    issues = report.get("issues", [])
    alerts = report.get("alerts", [])
    top_signals = report.get("unified_signal_summary", {}).get("top_symbols", [])[:3]
    meme_radar = report.get("meme_radar", {}).get("candidates", [])[:3]
    candidates = report.get("candidate_basket", [])[:3]
    coverage = report.get("unified_coverage", {})

    market_mode = coverage.get("market_data", {}).get("mode", "missing")
    signal_mode = coverage.get("signal", {}).get("mode", "missing")
    square_state = "已启用" if report.get("square_feed_used") else "关闭"

    lines = [
        "┌─ 📡 推土机播报",
        f"├ 状态：{analysis_mode}",
        f"├ 覆盖：市场 {market_mode}｜信号 {signal_mode}",
        f"├ Square：{square_state}｜问题 {report.get('issue_count', 0)}｜告警 {len(alerts)}",
    ]

    if bridge:
        lines.append(f"├ {_fmt_exec_line(bridge)}")
    else:
        lines.append("├ ⚪ 执行：无摘要")

    performance = report.get("performance_dashboard") or {}
    if performance:
        lines.append("├─ 💰 绩效看板")
        if performance.get("status") == "ok":
            lines.append(
                "├ "
                f"持仓 {performance.get('open_positions', 0)}｜"
                f"已实现 {performance.get('realized_pnl_usdt', 0):+g}U｜"
                f"未实现 {performance.get('unrealized_pnl_usdt', 0):+g}U｜"
                f"累计 {performance.get('total_pnl_usdt', 0):+g}U"
            )
            lines.append(
                "├ "
                f"胜率 {performance.get('win_rate', 0)}% "
                f"({performance.get('wins', 0)}/{performance.get('closed_trades', 0)})｜"
                f"期望 {performance.get('expectancy_pct_per_closed_trade', 0):+g}%｜"
                f"诊断 {performance.get('diagnosis', '')}"
            )
        else:
            lines.append("├ 暂无绩效状态")

    latest_requests = execution_results_summary.get("latest_requests", [])
    latest_errors = execution_results_summary.get("latest_errors", [])
    if latest_requests:
        lines.append("├─ 🧾 最近交易内容")
        for idx, item in enumerate(latest_requests[:3], start=1):
            lines.append(f"├ {_fmt_exec_request(item, idx)}")
    if latest_errors:
        lines.append("├─ ❌ 最近报错")
        for idx, item in enumerate(latest_errors[:3], start=1):
            lines.append(f"├ {_fmt_exec_error(item, idx)}")

    lines.extend([
        "├─ 🔥 热点Top3",
    ])
    if top_signals:
        for idx, item in enumerate(top_signals, start=1):
            prefix = "├" if idx < len(top_signals) or candidates or issues else "└"
            lines.append(f"{prefix} {_fmt_signal_item(item, idx)}")
    else:
        lines.append("├ 无")

    lines.append("├─ 🐶 妖币雷达Top3")
    if meme_radar:
        for idx, item in enumerate(meme_radar, start=1):
            lines.append(f"├ {_fmt_meme_radar_item(item, idx)}")
    else:
        lines.append("├ 无")

    lines.append("├─ 🎯 候选Top3")
    if candidates:
        for idx, item in enumerate(candidates, start=1):
            lines.append(f"├ {_fmt_candidate_item(item, idx)}")
    else:
        lines.append("├ 无")

    lines.append("├─ ⚠️ 风险")
    if issues:
        for issue in issues[:2]:
            lines.append(f"├ {issue['summary']}")
    else:
        lines.append("├ 无")

    lines.append("└─ 结束")

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(lines) + "\n")
