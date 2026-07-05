"""Out-of-Time predictive model validation.

Reads exit journal entries that contain predictive_snapshot from
paper_state.json (journal + shadow_journal), buckets by predicted_win_probability,
and reports actual PnL discrimination to validate model generalization on unseen trades.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

DEFAULT_STATE_FILE = Path("data/trading_system/paper_state.json")

EXIT_ACTIONS = {"close_long", "close_short", "exit", "shadow_close_long", "shadow_close_short", "shadow_stop_loss", "shadow_take_profit", "shadow_partial_take_profit"}


def _load_exits_with_predictions(state_path: Path) -> list[dict]:
    if not state_path.exists():
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    all_entries = list(state.get("journal", [])) + list(state.get("shadow_journal", []))
    results = []
    for row in all_entries:
        action = str(row.get("action") or "").lower()
        if not any(exit_kw in action for exit_kw in ("close", "stop_loss", "take_profit", "exit")):
            continue
        snap = row.get("predictive_snapshot")
        if not snap or not isinstance(snap, dict):
            continue
        if snap.get("predicted_win_probability") is None and snap.get("hist_win_rate") is None:
            continue
        results.append(row)
    return results


def _bucket_analysis(rows: list[dict]) -> dict:
    buckets = {"high_prob": [], "mid_prob": [], "low_prob": []}
    for row in rows:
        snap = row["predictive_snapshot"]
        prob = snap.get("predicted_win_probability") or snap.get("hist_win_rate")
        if prob is None:
            continue
        prob = float(prob)
        pnl = float(row.get("pnl_pct") or 0)
        entry = {"symbol": row.get("symbol"), "pnl_pct": pnl, "prob": prob}
        if prob >= 0.6:
            buckets["high_prob"].append(entry)
        elif prob >= 0.4:
            buckets["mid_prob"].append(entry)
        else:
            buckets["low_prob"].append(entry)

    summary = {}
    for name, items in buckets.items():
        if not items:
            summary[name] = {"count": 0, "avg_pnl": 0, "win_rate": 0}
            continue
        pnls = [i["pnl_pct"] for i in items]
        wins = sum(1 for p in pnls if p > 0)
        summary[name] = {
            "count": len(items),
            "avg_pnl": round(mean(pnls) * 100, 4),
            "win_rate": round(wins / len(items) * 100, 2),
        }
    return summary


def _stop_loss_analysis(rows: list[dict]) -> dict:
    high_risk = []
    low_risk = []
    for row in rows:
        snap = row["predictive_snapshot"]
        sl_risk = snap.get("predicted_stop_loss_risk")
        if sl_risk is None:
            continue
        reason = str(row.get("reason") or "").lower()
        is_sl = "stop_loss" in reason
        entry = {"symbol": row.get("symbol"), "sl_risk": float(sl_risk), "actual_sl": is_sl}
        if float(sl_risk) >= 0.5:
            high_risk.append(entry)
        else:
            low_risk.append(entry)

    def _rate(items):
        if not items:
            return {"count": 0, "actual_sl_rate": 0}
        actual = sum(1 for i in items if i["actual_sl"])
        return {"count": len(items), "actual_sl_rate": round(actual / len(items) * 100, 2)}

    return {"high_sl_risk": _rate(high_risk), "low_sl_risk": _rate(low_risk)}


def _oot_integrity_check(rows: list[dict]) -> dict:
    """Verify model_trained_until < entry_timestamp (no lookahead)."""
    violations = 0
    checked = 0
    for row in rows:
        snap = row["predictive_snapshot"]
        trained_until = snap.get("model_trained_until", "")
        entry_ts = row.get("entry_timestamp", "")
        if not trained_until or not entry_ts:
            continue
        checked += 1
        if trained_until >= entry_ts:
            violations += 1
    return {"checked": checked, "lookahead_violations": violations}


def run(state_path: Path | None = None) -> None:
    path = state_path or DEFAULT_STATE_FILE
    rows = _load_exits_with_predictions(path)

    print(f"=== OOT Predictive Model Validation ===")
    print(f"Journal: {path}")
    print(f"Exit records with predictive_snapshot: {len(rows)}")
    print()

    if len(rows) < 10:
        print(f"数据不足（当前 {len(rows)} 笔），需积累 50+ 笔带 predictive_snapshot 的平仓记录。")
        print("模型预测快照会在新开仓时自动记录，平仓后继承到 exit journal。")
        return

    prob_buckets = _bucket_analysis(rows)
    print("--- Win Probability Buckets ---")
    for name, stats in prob_buckets.items():
        label = {"high_prob": "prob >= 0.6", "mid_prob": "0.4 <= prob < 0.6", "low_prob": "prob < 0.4"}[name]
        print(f"  {label}: n={stats['count']}, avg_pnl={stats['avg_pnl']:.2f}%, win_rate={stats['win_rate']:.1f}%")

    discrimination = (prob_buckets["high_prob"]["avg_pnl"] - prob_buckets["low_prob"]["avg_pnl"])
    print(f"\n  PnL discrimination (high - low): {discrimination:.2f}%")
    print()

    sl_stats = _stop_loss_analysis(rows)
    print("--- Stop Loss Risk Buckets ---")
    for name, stats in sl_stats.items():
        label = {"high_sl_risk": "sl_risk >= 0.5", "low_sl_risk": "sl_risk < 0.5"}[name]
        print(f"  {label}: n={stats['count']}, actual_sl_rate={stats['actual_sl_rate']:.1f}%")
    print()

    integrity = _oot_integrity_check(rows)
    print("--- OOT Integrity ---")
    print(f"  Checked: {integrity['checked']}, Lookahead violations: {integrity['lookahead_violations']}")
    if integrity["lookahead_violations"] > 0:
        print("  WARNING: 存在前视偏差！model_trained_until >= entry_timestamp")
    else:
        print("  OK: 所有预测均为 out-of-time（无前视）")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run(path)
