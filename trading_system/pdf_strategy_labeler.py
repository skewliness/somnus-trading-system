"""PDF 交易策略蒸馏器：将 PDF 方法论编码为可量化的训练信号。

将 PDF 中的交易策略规则（趋势确认、仓位管理、冷却期、让利润跑）
编码为 Python 函数，对每笔交易输出策略对齐度分数和违规标记。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class RuleResult:
    name: str
    passed: bool
    weight: float
    detail: str = ""


@dataclass(slots=True)
class PdfStrategyLabel:
    pdf_entry_score: float = 0.0
    pdf_risk_score: float = 0.0
    pdf_hold_score: float = 0.0
    pdf_overall_score: float = 0.0
    violations: list[str] = field(default_factory=list)
    pdf_label: int = -1
    rule_results: list[RuleResult] = field(default_factory=list)


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Entry Quality Rules
# ---------------------------------------------------------------------------

def _rule_trend_alignment(ctx: dict) -> RuleResult:
    """做多时要求趋势向上：kline_direction > 0 且 MA20 距离 > -2%"""
    direction = _safe_float(ctx.get("kline_direction"))
    ma20_dist = _safe_float(ctx.get("kline_ma20_distance_pct"))
    passed = direction > 0 and ma20_dist > -2.0
    return RuleResult("trend_alignment", passed, 0.20,
                      f"dir={direction:.2f} ma20_dist={ma20_dist:.2f}")


def _rule_pullback_entry(ctx: dict) -> RuleResult:
    """趋势回调入场：price_position_7d 在 [0.15, 0.50] 区间（不追高不抄底）"""
    pp7d = _safe_float(ctx.get("price_position_7d"), -1.0)
    if pp7d < 0:
        return RuleResult("pullback_entry", True, 0.15, "no data, pass")
    passed = 0.15 <= pp7d <= 0.50
    return RuleResult("pullback_entry", passed, 0.15, f"pp7d={pp7d:.3f}")


def _rule_volume_confirmation(ctx: dict) -> RuleResult:
    """成交量确认：volume >= 1.2x MA5"""
    vol_ratio = _safe_float(ctx.get("kline_volume_vs_ma5"))
    passed = vol_ratio >= 1.2
    return RuleResult("volume_confirmation", passed, 0.10, f"vol_ratio={vol_ratio:.2f}")


def _rule_pattern_confirmation(ctx: dict) -> RuleResult:
    """K线形态确认：至少 1 个看涨形态"""
    bullish = _safe_float(ctx.get("kline_pattern_bullish_count"))
    passed = bullish >= 1
    return RuleResult("pattern_confirmation", passed, 0.10, f"bullish={bullish:.0f}")


def _rule_no_exhaustion(ctx: dict) -> RuleResult:
    """不在衰竭位置入场：price_position_7d < 0.85 且上影线 < 0.5"""
    pp7d = _safe_float(ctx.get("price_position_7d"), 0.5)
    upper_shadow = _safe_float(ctx.get("kline_upper_shadow_pct"))
    passed = pp7d < 0.85 or upper_shadow < 0.4
    return RuleResult("no_exhaustion", passed, 0.10,
                      f"pp7d={pp7d:.3f} upper={upper_shadow:.3f}")


def _rule_adequate_score(ctx: dict) -> RuleResult:
    """等待高质量机会：entry_score >= 70"""
    score = _safe_float(ctx.get("score") or ctx.get("entry_score"))
    passed = score >= 70
    return RuleResult("adequate_score", passed, 0.15, f"score={score:.1f}")


def _rule_risk_reward_valid(ctx: dict) -> RuleResult:
    """盈亏比 >= 2.0"""
    entry = _safe_float(ctx.get("entry_price"))
    sl = _safe_float(ctx.get("stop_loss_price"))
    tp = _safe_float(ctx.get("take_profit_price"))
    if entry <= 0 or sl <= 0 or tp <= 0:
        return RuleResult("risk_reward_valid", True, 0.20, "no data, pass")
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / max(risk, 1e-9)
    passed = rr >= 2.0
    return RuleResult("risk_reward_valid", passed, 0.20, f"RR={rr:.2f}")


def _rule_close_confirmation(ctx: dict) -> RuleResult:
    """入场 K 线收盘确认：close_position >= 0.6（收在上方 40%）"""
    close_pos = _safe_float(ctx.get("kline_close_position"), -1.0)
    if close_pos < 0:
        return RuleResult("close_confirmation", True, 0.10, "no data, pass")
    passed = close_pos >= 0.6
    return RuleResult("close_confirmation", passed, 0.10, f"close_pos={close_pos:.3f}")


ENTRY_RULES = [
    _rule_trend_alignment,
    _rule_pullback_entry,
    _rule_volume_confirmation,
    _rule_pattern_confirmation,
    _rule_no_exhaustion,
    _rule_adequate_score,
    _rule_risk_reward_valid,
    _rule_close_confirmation,
]

# ---------------------------------------------------------------------------
# Risk Management Rules
# ---------------------------------------------------------------------------

def _rule_stop_loss_tight(ctx: dict) -> RuleResult:
    """单笔风险 <= 4%"""
    risk = _safe_float(ctx.get("initial_risk_pct"))
    if risk <= 0:
        return RuleResult("stop_loss_tight", True, 0.25, "no data, pass")
    passed = risk <= 0.04
    return RuleResult("stop_loss_tight", passed, 0.25, f"risk={risk*100:.2f}%")


def _rule_no_revenge_entry(ctx: dict) -> RuleResult:
    """止损后同币冷却 >= 240 分钟"""
    minutes_since_sl = _safe_float(ctx.get("minutes_since_last_sl"), 9999)
    last_exit_reason = ctx.get("last_exit_reason", "")
    if last_exit_reason != "stop_loss_hit":
        return RuleResult("no_revenge_entry", True, 0.30, "no prior SL")
    passed = minutes_since_sl >= 240
    return RuleResult("no_revenge_entry", passed, 0.30,
                      f"min_since_sl={minutes_since_sl:.0f}")


def _rule_no_downtrend_long(ctx: dict) -> RuleResult:
    """不在下降趋势中做多（日线 MA20 以下且连续向下）"""
    direction = _safe_float(ctx.get("kline_direction"))
    ma20_dist = _safe_float(ctx.get("kline_ma20_distance_pct"))
    consecutive = _safe_float(ctx.get("kline_consecutive_direction"))
    is_downtrend = direction < 0 and ma20_dist < -1.5 and consecutive <= -1
    passed = not is_downtrend
    return RuleResult("no_downtrend_long", passed, 0.25,
                      f"dir={direction:.1f} ma20={ma20_dist:.1f} consec={consecutive:.0f}")


def _rule_position_sizing(ctx: dict) -> RuleResult:
    """仓位不超过总资金 5%"""
    notional = _safe_float(ctx.get("notional_usdt"))
    capital = _safe_float(ctx.get("total_capital_usdt"), 1000.0)
    if notional <= 0 or capital <= 0:
        return RuleResult("position_sizing", True, 0.20, "no data, pass")
    ratio = notional / capital
    passed = ratio <= 0.05
    return RuleResult("position_sizing", passed, 0.20, f"ratio={ratio*100:.1f}%")


def _rule_daily_loss_discipline(ctx: dict) -> RuleResult:
    """当日累计亏损未超限（默认 -100 USDT）"""
    daily_loss = _safe_float(ctx.get("daily_loss_usdt"))
    if daily_loss == 0.0 and "daily_loss_usdt" not in ctx:
        return RuleResult("daily_loss_discipline", True, 0.20, "no data")
    passed = daily_loss > -100.0
    return RuleResult("daily_loss_discipline", passed, 0.20, f"daily_pnl={daily_loss:.1f}")


def _rule_position_after_loss(ctx: dict) -> RuleResult:
    """连续亏损后降低仓位"""
    cons_losses = int(_safe_float(ctx.get("consecutive_losses")))
    notional = _safe_float(ctx.get("notional_usdt"))
    capital = _safe_float(ctx.get("total_capital_usdt"), 1000.0)
    if cons_losses == 0:
        return RuleResult("position_after_loss", True, 0.20, "no streak")
    max_ratio = 0.025 if cons_losses >= 1 else 0.05
    passed = capital <= 0 or notional <= capital * max_ratio
    return RuleResult("position_after_loss", passed, 0.20,
                      f"cons={cons_losses} ratio={notional/max(capital,1)*100:.1f}%")


RISK_RULES = [
    _rule_stop_loss_tight,
    _rule_no_revenge_entry,
    _rule_no_downtrend_long,
    _rule_position_sizing,
    _rule_daily_loss_discipline,
    _rule_position_after_loss,
]


# ---------------------------------------------------------------------------
# Hold Management Rules
# ---------------------------------------------------------------------------

def _rule_dont_cut_winners(ctx: dict) -> RuleResult:
    """浮盈 > 2% 时不因 score_faded 退出"""
    pnl_pct = _safe_float(ctx.get("pnl_pct"))
    exit_reason = str(ctx.get("exit_reason") or "")
    if "score_faded" not in exit_reason:
        return RuleResult("dont_cut_winners", True, 0.30, "not score_faded")
    passed = pnl_pct <= 0.02
    return RuleResult("dont_cut_winners", passed, 0.30,
                      f"pnl={pnl_pct*100:.2f}% reason={exit_reason}")


def _rule_trailing_stop_logic(ctx: dict) -> RuleResult:
    """浮盈 > 5% 后应有止损保护（不应全亏回吐）"""
    pnl_pct = _safe_float(ctx.get("pnl_pct"))
    max_pnl = _safe_float(ctx.get("max_unrealized_pnl_pct"), pnl_pct)
    if max_pnl < 0.05:
        return RuleResult("trailing_stop_logic", True, 0.25, "never reached +5%")
    gave_back = max_pnl - pnl_pct
    passed = gave_back < 0.04
    return RuleResult("trailing_stop_logic", passed, 0.25,
                      f"max={max_pnl*100:.1f}% final={pnl_pct*100:.1f}% gave_back={gave_back*100:.1f}%")


def _rule_time_frame_consistency(ctx: dict) -> RuleResult:
    """中线信号持仓 >= 2h"""
    hold_minutes = _safe_float(ctx.get("hold_minutes"))
    if hold_minutes <= 0:
        return RuleResult("time_frame_consistency", True, 0.20, "no data")
    passed = hold_minutes >= 120
    return RuleResult("time_frame_consistency", passed, 0.20,
                      f"hold={hold_minutes:.0f}min")


def _rule_partial_tp(ctx: dict) -> RuleResult:
    """到达 TP 后分批平仓而非全平"""
    size_fraction = _safe_float(ctx.get("size_fraction"), 1.0)
    exit_reason = str(ctx.get("exit_reason") or "")
    if "take_profit" not in exit_reason:
        return RuleResult("partial_tp", True, 0.25, "not TP exit")
    passed = size_fraction < 1.0
    return RuleResult("partial_tp", passed, 0.25, f"fraction={size_fraction:.2f}")


HOLD_RULES = [
    _rule_dont_cut_winners,
    _rule_trailing_stop_logic,
    _rule_time_frame_consistency,
    _rule_partial_tp,
]


# ---------------------------------------------------------------------------
# Main Labeling Function
# ---------------------------------------------------------------------------

def _compute_category_score(rules: list, ctx: dict) -> tuple[float, list[RuleResult]]:
    results = [rule(ctx) for rule in rules]
    total_weight = sum(r.weight for r in results)
    if total_weight <= 0:
        return 0.0, results
    weighted_sum = sum(r.weight for r in results if r.passed)
    return weighted_sum / total_weight, results


def label_trade(ctx: dict) -> PdfStrategyLabel:
    """
    对一笔交易评估 PDF 策略对齐度。

    ctx 应包含入场时的 MarketSnapshot 特征 + 交易结果信息：
      - kline_direction, kline_ma20_distance_pct, price_position_7d, ...
      - entry_price, stop_loss_price, take_profit_price
      - score / entry_score
      - pnl_pct, exit_reason, hold_minutes, size_fraction
      - minutes_since_last_sl, last_exit_reason (冷却期判断)
      - notional_usdt, total_capital_usdt
      - kline_pattern_bullish_count
    """
    entry_score, entry_results = _compute_category_score(ENTRY_RULES, ctx)
    risk_score, risk_results = _compute_category_score(RISK_RULES, ctx)
    hold_score, hold_results = _compute_category_score(HOLD_RULES, ctx)

    all_results = entry_results + risk_results + hold_results
    violations = [r.name for r in all_results if not r.passed]

    overall = entry_score * 0.35 + risk_score * 0.35 + hold_score * 0.30

    # Determine label
    has_critical_violation = any(
        r.name in ("no_revenge_entry", "no_downtrend_long") and not r.passed
        for r in all_results
    )
    pnl_pct = _safe_float(ctx.get("pnl_pct"))

    if has_critical_violation:
        pdf_label = 0
    elif not _find_result(all_results, "stop_loss_tight").passed and pnl_pct < -0.08:
        pdf_label = 0
    elif not _find_result(all_results, "adequate_score").passed and pnl_pct < 0:
        pdf_label = 0
    elif overall >= 0.6 and not has_critical_violation:
        pdf_label = 1
    else:
        pdf_label = -1

    return PdfStrategyLabel(
        pdf_entry_score=round(entry_score, 4),
        pdf_risk_score=round(risk_score, 4),
        pdf_hold_score=round(hold_score, 4),
        pdf_overall_score=round(overall, 4),
        violations=violations,
        pdf_label=pdf_label,
        rule_results=all_results,
    )


def _find_result(results: list[RuleResult], name: str) -> RuleResult:
    for r in results:
        if r.name == name:
            return r
    return RuleResult(name, True, 0.0)


# ---------------------------------------------------------------------------
# Batch labeling for historical trades
# ---------------------------------------------------------------------------

def label_journal_trades(journal: list[dict], recent_exits: dict | None = None) -> list[dict]:
    """
    对 journal 中的所有 exit 交易进行 PDF 策略标注。

    Args:
        journal: paper_state journal (entry + exit records)
        recent_exits: {symbol: {timestamp, reason, pnl_pct}} 最近退出记录

    Returns:
        list of {symbol, timestamp, pnl_pct, pdf_label, pdf_overall_score, violations}
    """
    recent_exits = recent_exits or {}
    exit_history: dict[str, dict] = {}
    results = []

    for item in journal:
        action = str(item.get("action") or "")
        symbol = str(item.get("symbol") or "")

        if "open" in action:
            pass
        elif action in ("exit", "take_profit_partial"):
            ctx = _build_context_from_journal_item(item, exit_history.get(symbol))
            label = label_trade(ctx)
            results.append({
                "symbol": symbol,
                "timestamp": item.get("timestamp"),
                "pnl_pct": _safe_float(item.get("pnl_pct")),
                "exit_reason": str(item.get("reason") or ""),
                "pdf_label": label.pdf_label,
                "pdf_entry_score": label.pdf_entry_score,
                "pdf_risk_score": label.pdf_risk_score,
                "pdf_hold_score": label.pdf_hold_score,
                "pdf_overall_score": label.pdf_overall_score,
                "violations": label.violations,
            })
            exit_history[symbol] = {
                "timestamp": item.get("timestamp"),
                "reason": str(item.get("reason") or ""),
                "pnl_pct": _safe_float(item.get("pnl_pct")),
            }

    return results


def _build_context_from_journal_item(item: dict, prev_exit: dict | None) -> dict:
    """从 journal exit item 构建 label_trade 所需的 context dict。"""
    cf = item.get("coin_features") or {}

    minutes_since_sl = 9999.0
    last_exit_reason = ""
    if prev_exit:
        last_exit_reason = prev_exit.get("reason", "")
        prev_ts = prev_exit.get("timestamp", "")
        curr_ts = item.get("entry_timestamp") or item.get("timestamp") or ""
        if prev_ts and curr_ts:
            try:
                t1 = datetime.fromisoformat(str(prev_ts).replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(str(curr_ts).replace("Z", "+00:00"))
                minutes_since_sl = max(0, (t2 - t1).total_seconds() / 60)
            except (ValueError, TypeError):
                pass

    hold_minutes = 0.0
    entry_ts = item.get("entry_timestamp", "")
    exit_ts = item.get("timestamp", "")
    if entry_ts and exit_ts:
        try:
            t1 = datetime.fromisoformat(str(entry_ts).replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(str(exit_ts).replace("Z", "+00:00"))
            hold_minutes = max(0, (t2 - t1).total_seconds() / 60)
        except (ValueError, TypeError):
            pass

    return {
        "kline_direction": _safe_float(cf.get("kline_direction") or item.get("kline_direction")),
        "kline_ma20_distance_pct": _safe_float(cf.get("kline_ma20_distance_pct") or item.get("kline_ma20_distance_pct")),
        "kline_consecutive_direction": _safe_float(cf.get("kline_consecutive_direction") or item.get("kline_consecutive_direction")),
        "kline_volume_vs_ma5": _safe_float(cf.get("kline_volume_vs_ma5") or item.get("kline_volume_vs_ma5")),
        "kline_upper_shadow_pct": _safe_float(cf.get("kline_upper_shadow_pct") or item.get("kline_upper_shadow_pct")),
        "kline_pattern_bullish_count": _safe_float(item.get("kline_pattern_bullish_count") or cf.get("kline_pattern_bullish_count")),
        "price_position_7d": _safe_float(cf.get("price_position_7d") or item.get("price_position_7d")),
        "score": _safe_float(item.get("entry_score") or item.get("score")),
        "entry_price": _safe_float(item.get("entry_price")),
        "stop_loss_price": _safe_float(item.get("stop_loss_price")),
        "take_profit_price": _safe_float(item.get("take_profit_price")),
        "initial_risk_pct": _safe_float(item.get("initial_risk_pct")),
        "notional_usdt": _safe_float(item.get("notional_usdt")),
        "total_capital_usdt": 1000.0,
        "pnl_pct": _safe_float(item.get("pnl_pct")),
        "exit_reason": str(item.get("reason") or ""),
        "hold_minutes": hold_minutes,
        "size_fraction": _safe_float(item.get("size_fraction"), 1.0),
        "minutes_since_last_sl": minutes_since_sl,
        "last_exit_reason": last_exit_reason,
    }
