#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _fmt_num(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except Exception:
        return "--"
    return f"{number:.{digits}f}{suffix}"


def _fmt_pct(value: Any) -> str:
    return _fmt_num(value, 2, "%")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc), "_path": str(path)}
    return payload if isinstance(payload, dict) else {"_load_error": "report is not a json object", "_path": str(path)}


def _current_execution_payload(report: dict[str, Any]) -> dict[str, Any]:
    bridge = report.get("execution_bridge_summary") or {}
    if not isinstance(bridge, dict):
        return {}
    request_file = str(bridge.get("request_file") or "")
    if not request_file:
        return {}
    result_path = Path("data/trading_system/execution_results") / f"{Path(request_file).stem}_result.json"
    return _load_json(result_path)


def _top_candidates(report: dict[str, Any], limit: int = 5) -> list[str]:
    rows = report.get("candidate_basket") or []
    if not isinstance(rows, list):
        return []
    lines: list[str] = []
    for index, item in enumerate(rows[:limit], start=1):
        if not isinstance(item, dict):
            continue
        blocked = item.get("blocked_reasons") or []
        blocked_text = ",".join(str(x) for x in blocked[:3]) if blocked else "none"
        lines.append(
            f"{index}. {item.get('symbol', '--')} score={_fmt_num(item.get('score'))} "
            f"risk={item.get('risk_band', '--')} status={item.get('execution_status', '--')} "
            f"24h={_fmt_pct(item.get('price_change_pct_24h'))} oi4h={_fmt_pct(item.get('oi_change_pct_4h'))} "
            f"block={blocked_text}"
        )
    return lines


def _blocked_reason_summary(report: dict[str, Any]) -> str:
    rows = report.get("candidate_basket") or []
    counter: Counter[str] = Counter()
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, dict):
                for reason in item.get("blocked_reasons") or []:
                    counter[str(reason)] += 1
    if not counter:
        return "none"
    return ", ".join(f"{reason}:{count}" for reason, count in counter.most_common(5))


def _shadow_summary(report: dict[str, Any], limit: int = 5) -> list[str]:
    shadow = report.get("shadow_candidate_scores") or {}
    if not isinstance(shadow, dict):
        return []
    scores = shadow.get("scores") or {}
    if not isinstance(scores, dict):
        return []

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    disagreements: list[str] = []
    for symbol, per_model in scores.items():
        if not isinstance(per_model, dict):
            continue
        moons = []
        labels = []
        for model_name, score in per_model.items():
            if not isinstance(score, dict):
                continue
            moons.append(float(score.get("prob_moon") or 0.0))
            labels.append(str(score.get("model_label", "--")))
        if moons:
            ranked.append((max(moons), str(symbol), per_model))
        if len(set(labels)) > 1:
            disagreements.append(str(symbol))

    lines: list[str] = []
    for max_moon, symbol, per_model in sorted(ranked, reverse=True)[:limit]:
        parts = []
        for model_name, score in sorted(per_model.items()):
            if isinstance(score, dict):
                parts.append(f"{model_name}:L{score.get('model_label')} moon={_fmt_num(score.get('prob_moon'), 3)}")
        lines.append(f"{symbol} max_moon={max_moon:.3f} | " + " ; ".join(parts))
    if disagreements:
        lines.append("分歧: " + ", ".join(disagreements[:8]))
    errors = shadow.get("errors") or {}
    if errors:
        lines.append("errors: " + json.dumps(errors, ensure_ascii=False))
    return lines


def _execution_summary(report: dict[str, Any]) -> list[str]:
    bridge = report.get("execution_bridge_summary") or {}
    paper = report.get("paper_rebalance") or {}
    results = report.get("execution_results_summary") or {}
    if not isinstance(bridge, dict):
        bridge = {}
    if not isinstance(paper, dict):
        paper = {}
    if not isinstance(results, dict):
        results = {}
    current_execution = _current_execution_payload(report)
    current_results = current_execution.get("results") if isinstance(current_execution.get("results"), list) else []
    current_status = str(current_execution.get("status") or "pending")

    entries = paper.get("entries") if isinstance(paper.get("entries"), list) else []
    exits = paper.get("exits") if isinstance(paper.get("exits"), list) else []
    positions = paper.get("positions") if isinstance(paper.get("positions"), list) else []
    shadow = paper.get("shadow_paper") if isinstance(paper.get("shadow_paper"), dict) else {}
    shadow_entries = shadow.get("entries") if isinstance(shadow.get("entries"), list) else []
    shadow_exits = shadow.get("exits") if isinstance(shadow.get("exits"), list) else []
    shadow_positions = shadow.get("positions") if isinstance(shadow.get("positions"), list) else []
    risk_controls = bridge.get("risk_controls") if isinstance(bridge.get("risk_controls"), dict) else {}

    lines = [
        f"request_mode={bridge.get('request_mode', '--')} intents={bridge.get('intent_count', 0)} "
        f"blocked={bridge.get('blocked_intent_count', 0)} signal_source={bridge.get('signal_source', '--')} "
        f"binance_current_status={current_status} binance_current_results={len(current_results)}",
        f"paper entries={len(entries)} exits={len(exits)} positions={len(positions)} risk={risk_controls.get('status', '--')}",
    ]
    if entries:
        lines.append("entries: " + ", ".join(str(item.get("symbol", "--")) for item in entries[:5] if isinstance(item, dict)))
    if exits:
        lines.append("exits: " + ", ".join(str(item.get("symbol", "--")) for item in exits[:5] if isinstance(item, dict)))
    if shadow:
        lines.append(
            f"shadow_paper entries={len(shadow_entries)} exits={len(shadow_exits)} positions={len(shadow_positions)} "
            f"realized={_fmt_num(shadow.get('realized_pnl_usdt'), 4)}U "
            f"unrealized={_fmt_num(shadow.get('unrealized_pnl_usdt'), 4)}U "
            f"threshold={_fmt_num(shadow.get('trigger_threshold'), 2)}"
        )
    if shadow_entries:
        lines.append("shadow entries: " + ", ".join(str(item.get("symbol", "--")) for item in shadow_entries[:5] if isinstance(item, dict)))
    if shadow_exits:
        lines.append("shadow exits: " + ", ".join(
            f"{item.get('symbol', '--')}:{_fmt_pct(float(item.get('pnl_pct') or 0) * 100)}"
            for item in shadow_exits[:5]
            if isinstance(item, dict)
        ))
    latest_errors = [item for item in current_results if isinstance(item, dict) and item.get("status") == "failed"]
    if latest_errors:
        lines.append("execution_errors: " + json.dumps(latest_errors[:2], ensure_ascii=False)[:500])
    return lines


def _performance_summary(report: dict[str, Any]) -> list[str]:
    perf = report.get("performance_dashboard") or {}
    if not isinstance(perf, dict):
        return []
    rolling_10 = perf.get("rolling_10") if isinstance(perf.get("rolling_10"), dict) else {}
    rolling_20 = perf.get("rolling_20") if isinstance(perf.get("rolling_20"), dict) else {}
    return [
        f"open={perf.get('open_positions', 0)} closed={perf.get('closed_trades', 0)} "
        f"WR={_fmt_pct(perf.get('win_rate'))} PnL={_fmt_num(perf.get('total_pnl_usdt'), 4)}U "
        f"expectancy={_fmt_pct(perf.get('expectancy_pct_per_closed_trade'))}",
        f"rolling10 WR={_fmt_pct(rolling_10.get('win_rate'))} exp={_fmt_pct(rolling_10.get('expectancy_pct_per_closed_trade'))} | "
        f"rolling20 WR={_fmt_pct(rolling_20.get('win_rate'))} exp={_fmt_pct(rolling_20.get('expectancy_pct_per_closed_trade'))}",
        f"diagnosis={perf.get('diagnosis', '--')}",
    ]


def _issues_summary(report: dict[str, Any]) -> list[str]:
    issues = report.get("issues") or []
    if not isinstance(issues, list):
        return []
    lines = []
    for item in issues[:5]:
        if isinstance(item, dict):
            lines.append(f"{item.get('severity', '--')} {item.get('code', '--')}: {item.get('summary', '--')}")
    return lines


def summarize(report: dict[str, Any]) -> str:
    if report.get("_load_error"):
        return f"===== 本轮摘要生成失败 =====\nreport={report.get('_path')} error={report.get('_load_error')}\n===== 摘要结束 ====="

    lines = [
        "===== 本轮自动摘要 =====",
        f"run_id={report.get('run_id', '--')} mode={report.get('mode', '--')} "
        f"analysis={report.get('analysis_mode', '--')} execution={report.get('execution_mode', '--')}",
        f"coverage: candidates={len(report.get('candidate_basket') or [])} "
        f"coin_scout={len(report.get('coin_scout_evaluations') or [])} "
        f"whale_scout={len(report.get('whale_scout_evaluations') or [])} issues={report.get('issue_count', 0)}",
        "",
        "候选 Top:",
    ]
    top_candidates = _top_candidates(report)
    lines.extend(top_candidates or ["无候选"])
    lines.append(f"阻塞原因统计: {_blocked_reason_summary(report)}")

    lines.append("")
    lines.append("执行:")
    lines.extend(_execution_summary(report))

    lines.append("")
    lines.append("Shadow 模型:")
    lines.extend(_shadow_summary(report) or ["无 shadow scores"])

    lines.append("")
    lines.append("绩效:")
    lines.extend(_performance_summary(report) or ["无绩效数据"])

    issues = _issues_summary(report)
    if issues:
        lines.append("")
        lines.append("风险/问题:")
        lines.extend(issues)

    bridge = report.get("execution_bridge_summary") or {}
    intent_count = bridge.get("intent_count", 0) if isinstance(bridge, dict) else 0
    paper = report.get("paper_rebalance") or {}
    shadow = paper.get("shadow_paper") if isinstance(paper, dict) and isinstance(paper.get("shadow_paper"), dict) else {}
    shadow_entry_count = len(shadow.get("entries") or []) if isinstance(shadow, dict) else 0
    shadow_position_count = len(shadow.get("positions") or []) if isinstance(shadow, dict) else 0
    if intent_count:
        verdict = f"本轮有 {intent_count} 个执行意图，请重点检查 execution request。"
    elif shadow_entry_count or shadow_position_count:
        verdict = f"本轮无真实执行意图；shadow paper 正在跑虚拟实盘，新增 {shadow_entry_count} 个信号，当前 {shadow_position_count} 个虚拟持仓。"
    else:
        verdict = "本轮无下单；主要用于刷新市场快照、shadow 打分和风控观察。"
    lines.extend(["", f"结论: {verdict}", "===== 摘要结束 ====="])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the latest trading pipeline report.")
    parser.add_argument("--report", default="data/trading_system/pipeline_run_report.json")
    args = parser.parse_args()
    print(summarize(_load_json(Path(args.report))))


if __name__ == "__main__":
    main()
