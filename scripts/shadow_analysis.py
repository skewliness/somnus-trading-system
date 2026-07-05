#!/usr/bin/env python3
"""
Shadow 模型打分分析
==================
从 paper_closed_trades.jsonl 读取带 shadow_scores 的交易记录，分析：

1. prob_moon 分桶 vs win_rate / avg_pnl
2. 两模型 disagreement 样本分析
3. prob_moon 阈值收益曲线（>0.4/0.5/0.6）
4. stop_loss_hit 里模型是否提前判低分

用法:
    python3 scripts/shadow_analysis.py [--trades-file PATH] [--min-samples N]

Author: OWL for snow
Date: 2026-05-09
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TRADES_FILE = ROOT / "data/trading_system/model_training/paper_replay_autolearn/paper_closed_trades.jsonl"


def load_trades(path: Path) -> list[dict]:
    if not path.exists():
        print(f"文件不存在: {path}")
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def extract_shadow_scores(trade: dict) -> dict | None:
    """从 trade 提取 shadow scores，兼容多种格式"""
    ss = trade.get("shadow_scores")
    if not ss:
        return None
    # 新格式: {"scores": {...}, "errors": {...}, "model_meta": {...}}
    if "scores" in ss:
        return ss
    # 旧格式（直接是 {model_name: score}）直接包一下
    return {"scores": ss, "errors": {}, "model_meta": {}}


def analyze_prob_moon_buckets(trades: list[dict], model_name: str, buckets: list[float]):
    """prob_moon 分桶分析"""
    print(f"\n{'=' * 70}")
    print(f"prob_moon 分桶分析 — {model_name}")
    print(f"{'=' * 70}")

    bucket_data = {b: {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0} for b in buckets}
    bucket_data["other"] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0}

    for t in trades:
        ss = extract_shadow_scores(t)
        if not ss or model_name not in ss.get("scores", {}):
            continue
        score = ss["scores"][model_name]
        prob_moon = score.get("prob_moon", 0)
        pnl = t.get("pnl_pct", 0) or 0
        is_win = pnl > 0

        assigned = False
        for b in sorted(buckets, reverse=True):
            if prob_moon >= b:
                bucket_data[b]["count"] += 1
                bucket_data[b]["total_pnl"] += pnl
                if is_win:
                    bucket_data[b]["wins"] += 1
                else:
                    bucket_data[b]["losses"] += 1
                assigned = True
                break
        if not assigned:
            bucket_data["other"]["count"] += 1
            bucket_data["other"]["total_pnl"] += pnl
            if is_win:
                bucket_data["other"]["wins"] += 1
            else:
                bucket_data["other"]["losses"] += 1

    print(f"{'分桶':<15} {'笔数':>6} {'WinRate':>8} {'AvgPnL%':>10} {'总PnL%':>10}")
    print(f"{'-' * 55}")
    for b in sorted(buckets, reverse=True):
        d = bucket_data[b]
        if d["count"] == 0:
            print(f"prob>={b:<4}       0")
            continue
        wr = d["wins"] / d["count"] * 100
        avg_pnl = d["total_pnl"] / d["count"]
        print(f"prob>={b:<4}  {d['count']:>6} {wr:>7.1f}% {avg_pnl:>9.4f}% {d['total_pnl']:>9.4f}%")
    d = bucket_data["other"]
    if d["count"] > 0:
        wr = d["wins"] / d["count"] * 100
        avg_pnl = d["total_pnl"] / d["count"]
        print(f"{'other':<15} {d['count']:>6} {wr:>7.1f}% {avg_pnl:>9.4f}% {d['total_pnl']:>9.4f}%")


def analyze_model_disagreement(trades: list[dict], model_names: list[str]):
    """两模型 disagreement 分析"""
    if len(model_names) < 2:
        return
    m1, m2 = model_names[0], model_names[1]

    print(f"\n{'=' * 70}")
    print(f"模型 disagreement 分析 — {m1} vs {m2}")
    print(f"{'=' * 70}")

    agree = 0
    disagree = 0
    disagree_wins = 0
    disagree_losses = 0
    agree_wins = 0
    agree_losses = 0
    m1_higher_wins = 0
    m1_higher_losses = 0
    m2_higher_wins = 0
    m2_higher_losses = 0

    for t in trades:
        ss = extract_shadow_scores(t)
        if not ss or m1 not in ss.get("scores", {}) or m2 not in ss.get("scores", {}):
            continue
        s1 = ss["scores"][m1]
        s2 = ss["scores"][m2]
        l1 = s1.get("model_label", 0)
        l2 = s2.get("model_label", 0)
        pnl = t.get("pnl_pct", 0) or 0
        is_win = pnl > 0

        if l1 == l2:
            agree += 1
            if is_win:
                agree_wins += 1
            else:
                agree_losses += 1
        else:
            disagree += 1
            if is_win:
                disagree_wins += 1
            else:
                disagree_losses += 1
            p1 = s1.get("prob_moon", 0)
            p2 = s2.get("prob_moon", 0)
            if p1 > p2:
                if is_win:
                    m1_higher_wins += 1
                else:
                    m1_higher_losses += 1
            elif p2 > p1:
                if is_win:
                    m2_higher_wins += 1
                else:
                    m2_higher_losses += 1

    total = agree + disagree
    if total == 0:
        print("  无 disagreement 数据")
        return

    print(f"总样本: {total}")
    print(f"  一致: {agree} ({agree/total*100:.1f}%) — WinRate: {agree_wins/max(agree,1)*100:.1f}%")
    print(f"  分歧: {disagree} ({disagree/total*100:.1f}%) — WinRate: {disagree_wins/max(disagree,1)*100:.1f}%")
    if disagree > 0:
        tag1 = m1.split("_")[0]
        tag2 = m2.split("_")[0]
        print(f"\n  分歧时 {tag1} 分更高: {m1_higher_wins + m1_higher_losses} 次 — WinRate: {m1_higher_wins/max(m1_higher_wins+m1_higher_losses,1)*100:.1f}%")
        print(f"  分歧时 {tag2} 分更高: {m2_higher_wins + m2_higher_losses} 次 — WinRate: {m2_higher_wins/max(m2_higher_wins+m2_higher_losses,1)*100:.1f}%")


def analyze_threshold_curve(trades: list[dict], model_name: str, thresholds: list[float]):
    """prob_moon 阈值收益曲线"""
    print(f"\n{'=' * 70}")
    print(f"prob_moon 阈值收益曲线 — {model_name}")
    print(f"{'=' * 70}")

    for thresh in thresholds:
        above = []
        below = []
        for t in trades:
            ss = extract_shadow_scores(t)
            if not ss or model_name not in ss.get("scores", {}):
                continue
            prob_moon = ss["scores"][model_name].get("prob_moon", 0)
            pnl = t.get("pnl_pct", 0) or 0
            if prob_moon >= thresh:
                above.append(pnl)
            else:
                below.append(pnl)

        if not above:
            print(f"  >= {thresh}: 无样本")
            continue

        above_wr = sum(1 for p in above if p > 0) / len(above) * 100
        above_avg = sum(above) / len(above)
        below_wr = sum(1 for p in below if p > 0) / max(len(below), 1) * 100
        below_avg = sum(below) / max(len(below), 1)

        print(f"  >= {thresh}: {len(above):>4} 笔  WR={above_wr:.1f}%  AvgPnL={above_avg:.4f}%  "
              f"|  < {thresh}: {len(below):>4} 笔  WR={below_wr:.1f}%  AvgPnL={below_avg:.4f}%")


def analyze_stop_loss_shadow(trades: list[dict], model_names: list[str]):
    """stop_loss_hit 里模型是否提前判低分"""
    print(f"\n{'=' * 70}")
    print(f"stop_loss_hit 中的 shadow 打分")
    print(f"{'=' * 70}")

    sl_trades = [t for t in trades if t.get("reason") == "stop_loss_hit" or t.get("reason_family") == "stop_loss_hit"]
    print(f"stop_loss_hit 总笔数: {len(sl_trades)}")

    for model_name in model_names:
        scored = []
        for t in sl_trades:
            ss = extract_shadow_scores(t)
            if not ss or model_name not in ss.get("scores", {}):
                continue
            scored.append(ss["scores"][model_name].get("prob_moon", 0))

        if not scored:
            print(f"  {model_name}: 无 shadow 打分数据")
            continue

        avg = sum(scored) / len(scored)
        low = sum(1 for s in scored if s < 0.3)
        mid = sum(1 for s in scored if 0.3 <= s < 0.5)
        high = sum(1 for s in scored if s >= 0.5)
        print(f"  {model_name}: avg_prob_moon={avg:.4f}  "
              f"low(<0.3)={low}  mid(0.3-0.5)={mid}  high(>=0.5)={high}  total={len(scored)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shadow 模型打分分析")
    parser.add_argument("--trades-file", type=str, default=str(DEFAULT_TRADES_FILE))
    parser.add_argument("--min-samples", type=int, default=10, help="最少样本数")
    args = parser.parse_args()

    trades_file = Path(args.trades_file)
    print(f"加载交易数据: {trades_file}")
    trades = load_trades(trades_file)
    print(f"总交易笔数: {len(trades)}")

    has_shadow = 0
    model_names = set()
    for t in trades:
        ss = extract_shadow_scores(t)
        if ss:
            has_shadow += 1
            model_names.update(ss.get("scores", {}).keys())

    print(f"有 shadow_scores 的笔数: {has_shadow}")
    print(f"模型列表: {sorted(model_names)}")

    if has_shadow < args.min_samples:
        print(f"\n⚠️  有 shadow_scores 的样本不足 {args.min_samples} 笔，分析结果可能不可靠")
        print("   继续跑 paper 积累更多带 shadow 打分的交易...")

    if not model_names:
        print("\n❌ 未找到任何 shadow 打分数据")
        print("   确认 pipeline 已运行且 shadow 模型已加载")
        return

    model_names = sorted(model_names)
    buckets = [0.6, 0.5, 0.4, 0.3, 0.2]
    thresholds = [0.6, 0.5, 0.4, 0.3]

    for model_name in model_names:
        analyze_prob_moon_buckets(trades, model_name, buckets)
        analyze_threshold_curve(trades, model_name, thresholds)

    if len(model_names) >= 2:
        analyze_model_disagreement(trades, model_names)

    analyze_stop_loss_shadow(trades, model_names)

    # 输出模型元数据
    print(f"\n{'=' * 70}")
    print("模型元数据")
    print(f"{'=' * 70}")
    for t in trades:
        ss = extract_shadow_scores(t)
        if ss and ss.get("model_meta"):
            for name, meta in ss["model_meta"].items():
                print(f"  {name}:")
                for k, v in meta.items():
                    print(f"    {k}: {v}")
            break


if __name__ == "__main__":
    main()
