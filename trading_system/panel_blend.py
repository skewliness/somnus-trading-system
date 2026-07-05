"""聪明钱面板分数融合 (置信度浮动权重).

核心设计 (基于 122 笔真实平仓单实测):
- 面板 single_side_pct 在正常区间(50-70%)与真实盈亏 几乎零相关 (Pearson r=0.0279)。
  故面板不能拿固定权重 — 模糊时必须闭嘴, 让 BB 技术面 100% 说了算。
- 只有极端档 (single_side_pct >= 90% 且方向与持仓/候选同向) 才有真信号 (85% 胜率)。
- 因此面板权重随它自己的"看清程度"(single_side_pct)浮动, 上限 max_weight(默认0.30):
    ss < conf_floor(70)        -> weight = 0       技术面独裁
    conf_floor <= ss < conf_ceil(90) -> weight 线性 0 -> max_weight
    ss >= conf_ceil(90)        -> weight = max_weight 面板满话语权

融合后 final_score 仍箝制在 0-100, 无缝衔接既有的 entry_threshold / exit_threshold,
所以同一个覆写既驱动开仓选币(score排序+过门槛)又驱动持仓退出(score_faded)。
反转退出自然发生: 持仓时主力翻空且极端 -> panel_score 拉低 -> final 跌破 exit_threshold。

纯函数, 无副作用, 无 IO, 可独立单测。
"""
from __future__ import annotations

from typing import Any


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def panel_confidence_weight(
    single_side_pct: float,
    *,
    max_weight: float,
    conf_floor: float,
    conf_ceil: float,
) -> float:
    """面板权重 = max_weight × 线性斜坡(single_side_pct: floor->ceil 映射 0->1)。"""
    if conf_ceil <= conf_floor:
        return 0.0
    ramp = _clamp((single_side_pct - conf_floor) / (conf_ceil - conf_floor), 0.0, 1.0)
    return max_weight * ramp


def panel_directional_score(single_side_pct: float, dominant_side: str, position_side: str) -> float:
    """面板方向分 (0-100, 以 50 为中性):
    主力方向与持仓/候选方向 同向 -> 50 拉向 single_side_pct (看多越极端越接近100);
    反向 -> 50 拉向 (100 - single_side_pct) (看空越极端越接近0)。
    """
    aligned = (str(dominant_side).lower() == str(position_side).lower())
    deviation = single_side_pct - 50.0
    return _clamp(50.0 + deviation * (1.0 if aligned else -1.0), 0.0, 100.0)


def blend_panel_score(
    technical_score: float,
    panel_ctx: dict[str, Any] | None,
    position_side: str,
    config: Any,
) -> tuple[float, dict[str, Any]]:
    """把面板信号按置信度浮动权重融进技术分。

    返回 (final_score, debug) — final 始终箝制 0-100。
    面板不可用 / 未开启 / 模糊(权重0) 时, final == technical_score (无副作用)。
    debug 供 journal/日志留痕, 不参与下游逻辑。
    """
    technical_score = _clamp(float(technical_score), 0.0, 100.0)
    debug: dict[str, Any] = {"panel_applied": False, "technical_score": technical_score}

    if not getattr(config, "panel_blend_enabled", False):
        debug["skip"] = "disabled"
        return technical_score, debug
    if not panel_ctx or not panel_ctx.get("available"):
        debug["skip"] = "panel_unavailable"
        return technical_score, debug

    single_side_pct = float(panel_ctx.get("single_side_pct") or 0.0)
    dominant_side = str(panel_ctx.get("dominant_side") or "")
    if not dominant_side:
        debug["skip"] = "no_dominant_side"
        return technical_score, debug

    max_weight = float(getattr(config, "panel_blend_max_weight", 0.30))
    conf_floor = float(getattr(config, "panel_blend_conf_floor", 70.0))
    conf_ceil = float(getattr(config, "panel_blend_conf_ceil", 90.0))

    weight = panel_confidence_weight(
        single_side_pct, max_weight=max_weight, conf_floor=conf_floor, conf_ceil=conf_ceil
    )
    if weight <= 0.0:
        debug.update({"skip": "muddy_below_floor", "single_side_pct": single_side_pct, "weight": 0.0})
        return technical_score, debug

    panel_score = panel_directional_score(single_side_pct, dominant_side, position_side)
    final = _clamp((1.0 - weight) * technical_score + weight * panel_score, 0.0, 100.0)
    debug.update(
        {
            "panel_applied": True,
            "single_side_pct": single_side_pct,
            "dominant_side": dominant_side,
            "position_side": position_side,
            "weight": round(weight, 4),
            "panel_score": round(panel_score, 2),
            "final_score": round(final, 2),
            "delta": round(final - technical_score, 2),
        }
    )
    return final, debug
