#!/usr/bin/env python3
"""PDF 方法论回测复盘脚本。

对 V4 历史 paper 交易拉取入场时刻的日线 K 线，
用 PDF 方法论评估入场/持仓/退出质量，
生成高质量训练样本和复盘报告。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Binance kline fetcher
# ---------------------------------------------------------------------------

@dataclass
class _SimpleClient:
    base_url: str
    session: requests.Session

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=20)
                if resp.status_code not in {429, 500, 502, 503, 504}:
                    break
            except requests.RequestException:
                resp = None
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
        if resp is None or resp.status_code != 200:
            return []
        return resp.json()


def _make_client() -> _SimpleClient:
    base_url = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")
    session = requests.Session()
    session.headers.update({"User-Agent": "pdf-backtest-review/1.0"})
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if proxy:
        session.proxies = {"https": proxy, "http": proxy}
    return _SimpleClient(base_url=base_url, session=session)


def fetch_historical_klines(client: _SimpleClient, symbol: str, end_time_ms: int, interval: str = "1d", limit: int = 20) -> list:
    params = {"symbol": symbol, "interval": interval, "endTime": end_time_ms, "limit": limit}
    return client.get_json("/fapi/v1/klines", params)


# ---------------------------------------------------------------------------
# K-line analysis (reuse from trading_system.binance)
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading_system.binance import _kline_statistics, _detect_kline_patterns, _market_structure_from_klines
from trading_system.kline_pattern import detect_patterns
from trading_system.pdf_strategy_labeler import label_trade


# ---------------------------------------------------------------------------
# PDF review logic
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _compute_hold_minutes(entry_ts: str | None, exit_ts: str | None) -> float:
    entry = _parse_ts(entry_ts)
    exit_ = _parse_ts(exit_ts)
    if not entry or not exit_:
        return 0.0
    return max(0.0, (exit_ - entry).total_seconds() / 60.0)


def _is_polluted(trade: dict, hold_minutes: float) -> str | None:
    """Returns reason string if trade should be excluded, else None."""
    reason = trade.get("reason", "")
    if reason == "market_snapshot_missing_timeout":
        return "system_fault"
    if hold_minutes < 5 and reason == "stop_loss_hit":
        return "execution_noise"
    return None


def _sample_weight(trade: dict, hold_minutes: float, pdf_overall: float) -> float:
    reason = trade.get("reason", "")
    pnl = float(trade.get("pnl_pct", 0))
    if reason == "score_faded" and -0.01 <= pnl <= 0.01:
        return 0.3
    score = float(trade.get("entry_score") or 0)
    if score > 0 and score < 40:
        return 0.5
    return 1.0


def review_single_trade(client: _SimpleClient, trade: dict) -> dict | None:
    """Review a single closed trade using PDF methodology. Returns enriched sample or None."""
    symbol = trade.get("symbol", "")
    entry_ts_str = trade.get("entry_timestamp") or trade.get("timestamp")
    exit_ts_str = trade.get("timestamp")
    entry_price = float(trade.get("entry_price", 0))
    exit_price = float(trade.get("exit_price", 0))
    pnl_pct = float(trade.get("pnl_pct", 0))
    reason = trade.get("reason", "")

    hold_minutes = _compute_hold_minutes(trade.get("entry_timestamp"), exit_ts_str)

    pollution_reason = _is_polluted(trade, hold_minutes)
    if pollution_reason:
        return {"_excluded": True, "_exclusion_reason": pollution_reason, "symbol": symbol}

    entry_dt = _parse_ts(trade.get("entry_timestamp"))
    if not entry_dt:
        return {"_excluded": True, "_exclusion_reason": "no_entry_timestamp", "symbol": symbol}

    end_time_ms = int(entry_dt.timestamp() * 1000)
    klines = fetch_historical_klines(client, symbol, end_time_ms, "1d", 20)

    if not klines or len(klines) < 5:
        return {"_excluded": True, "_exclusion_reason": "insufficient_klines", "symbol": symbol}

    kline_stats = _kline_statistics(klines, mark_price=entry_price)
    pattern_result = _detect_kline_patterns(klines)
    structure = _market_structure_from_klines(klines, mark_price=entry_price)

    # Build context for PDF labeler
    initial_risk_pct = float(trade.get("initial_risk_pct") or 0.04)
    notional = float(trade.get("notional_usdt", 15))

    labeler_ctx = {
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": pnl_pct,
        "notional_usdt": notional,
        "total_capital_usdt": 1000.0,
        "initial_risk_pct": initial_risk_pct,
        "hold_minutes": hold_minutes,
        "exit_reason": reason,
        "entry_score": float(trade.get("entry_score") or 50),
        "kline_direction": kline_stats.get("kline_direction", 0),
        "kline_ma20_distance_pct": kline_stats.get("kline_ma20_distance_pct", 0),
        "kline_close_position": kline_stats.get("kline_close_position", 0.5),
        "kline_volume_vs_ma5": kline_stats.get("kline_volume_vs_ma5", 1.0),
        "kline_atr_pct": kline_stats.get("kline_atr_pct", 0),
        "kline_consecutive_direction": kline_stats.get("kline_consecutive_direction", 0),
        "price_position_7d": structure.get("price_position_7d"),
        "kline_upper_shadow_pct": kline_stats.get("kline_upper_shadow_pct", 0),
        "kline_lower_shadow_pct": kline_stats.get("kline_lower_shadow_pct", 0),
        "kline_pattern_bullish_count": pattern_result.get("bullish_count", 0),
        "kline_pattern_bearish_count": pattern_result.get("bearish_count", 0),
        "size_fraction": float(trade.get("size_fraction", 1.0)),
        "prev_exit": None,
        "daily_loss_usdt": 0.0,
        "consecutive_losses": 0,
    }

    pdf_result = label_trade(labeler_ctx)

    # Determine review quality
    entry_quality = "good" if pdf_result.pdf_entry_score >= 0.7 else ("bad" if pdf_result.pdf_entry_score < 0.3 else "neutral")

    # Exit quality assessment
    exit_quality = "neutral"
    if pnl_pct > 0.02 and reason == "score_faded":
        exit_quality = "bad"  # cut winner early
    elif reason == "take_profit_hit":
        exit_quality = "good"
    elif reason == "stop_loss_hit" and abs(pnl_pct) <= initial_risk_pct * 1.2:
        exit_quality = "good"  # stop loss worked as designed
    elif reason == "stop_loss_hit" and abs(pnl_pct) > initial_risk_pct * 2:
        exit_quality = "bad"  # stop loss too wide or slipped

    # Review notes
    notes = []
    if kline_stats.get("kline_direction", 0) < 0 and kline_stats.get("kline_ma20_distance_pct", 0) < -1.5:
        notes.append("逆势入场：MA20下方做多")
    if (structure.get("price_position_7d") or 0) > 0.85:
        notes.append("追高入场：7日价格位置过高")
    if pattern_result.get("bullish_count", 0) == 0 and pattern_result.get("bearish_count", 0) == 0:
        notes.append("无K线形态确认")
    if pnl_pct > 0.02 and reason == "score_faded":
        notes.append("过早退出盈利仓")
    if hold_minutes < 120 and reason != "stop_loss_hit":
        notes.append("持仓时间不足2小时")

    weight = _sample_weight(trade, hold_minutes, pdf_result.pdf_overall_score)

    is_unlucky = pdf_result.pdf_entry_score >= 0.7 and pnl_pct < 0
    is_exemplar = pdf_result.pdf_entry_score >= 0.7 and pnl_pct > 0
    is_anti_exemplar = pdf_result.pdf_entry_score < 0.3 and pnl_pct < 0

    # Label strategy:
    # - pdf_unlucky (good process, bad outcome): use pdf_label=1 as target, low weight
    #   so model doesn't learn "good entry features → lose"
    # - pdf_exemplar: boost weight, confirms good entry → good outcome
    # - pdf_anti_exemplar: keep weight, confirms bad entry → bad outcome
    if is_unlucky:
        effective_is_win = True
        weight *= 0.3
    elif is_exemplar:
        effective_is_win = True
        weight *= 1.5
    elif is_anti_exemplar:
        effective_is_win = False
        weight *= 1.2
    else:
        effective_is_win = pnl_pct > 0

    sample = {
        "sample_key": f"pdf_backtest|{symbol}|{trade.get('entry_timestamp', '')}",
        "sample_type": "pdf_backtest_review",
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": pnl_pct,
        "reason": reason,
        "reason_family": reason,
        "action": "exit",
        "entry_timestamp": trade.get("entry_timestamp"),
        "exit_timestamp": exit_ts_str,
        "hold_minutes": hold_minutes,
        "notional_usdt": notional,
        "entry_score": float(trade.get("entry_score") or 0),
        "setup_type": trade.get("setup_type", "trend"),
        # Kline features (backfilled)
        "kline_direction": kline_stats.get("kline_direction"),
        "kline_ma20_distance_pct": kline_stats.get("kline_ma20_distance_pct"),
        "kline_atr_pct": kline_stats.get("kline_atr_pct"),
        "kline_close_position": kline_stats.get("kline_close_position"),
        "kline_volume_vs_ma5": kline_stats.get("kline_volume_vs_ma5"),
        "kline_consecutive_direction": kline_stats.get("kline_consecutive_direction"),
        "kline_lower_shadow_pct": kline_stats.get("kline_lower_shadow_pct"),
        "kline_upper_shadow_pct": kline_stats.get("kline_upper_shadow_pct"),
        "kline_body_pct": kline_stats.get("kline_body_pct"),
        "kline_volume_trend": kline_stats.get("kline_volume_trend"),
        "kline_max_drawdown_5d": kline_stats.get("kline_max_drawdown_5d"),
        "kline_range_vs_ma5_range": kline_stats.get("kline_range_vs_ma5_range"),
        "kline_pattern_bullish_count": pattern_result.get("bullish_count", 0),
        "kline_pattern_bearish_count": pattern_result.get("bearish_count", 0),
        "kline_pattern_flags": pattern_result.get("flags", ""),
        # Price structure
        "price_position_7d": structure.get("price_position_7d"),
        "price_position_14d": structure.get("price_position_14d"),
        "price_position_20d": structure.get("price_position_20d"),
        "breakout_distance_7d": structure.get("breakout_distance_7d"),
        "compression_score": structure.get("compression_score"),
        # PDF review scores
        "pdf_entry_score": pdf_result.pdf_entry_score,
        "pdf_risk_score": pdf_result.pdf_risk_score,
        "pdf_hold_score": pdf_result.pdf_hold_score,
        "pdf_overall_score": pdf_result.pdf_overall_score,
        "pdf_label": pdf_result.pdf_label,
        "pdf_violations": pdf_result.violations,
        # Review metadata
        "pdf_review_entry_quality": entry_quality,
        "pdf_review_exit_quality": exit_quality,
        "pdf_review_notes": "；".join(notes) if notes else "",
        "pdf_review_weight": weight,
        # Training fields (sample_weight is what the trainer reads)
        "sample_weight": weight,
        "is_win": effective_is_win,
        "real_pnl_label": 1 if pnl_pct > 0 else 0,
        "label": pdf_result.pdf_label if pdf_result.pdf_label in (0, 1) else (1 if effective_is_win else 0),
        # Exemplar flags
        "pdf_exemplar": 1 if is_exemplar else 0,
        "pdf_anti_exemplar": 1 if is_anti_exemplar else 0,
        "pdf_unlucky": 1 if is_unlucky else 0,
    }
    return sample


# ---------------------------------------------------------------------------
# Batch processing & report generation
# ---------------------------------------------------------------------------

def _generate_report(samples: list[dict], excluded: list[dict]) -> dict:
    """Generate human-readable review report from processed samples."""
    wins = [s for s in samples if s.get("pnl_pct", 0) > 0]
    losses = [s for s in samples if s.get("pnl_pct", 0) <= 0]
    exemplars = [s for s in samples if s.get("pdf_exemplar") == 1]
    anti_exemplars = [s for s in samples if s.get("pdf_anti_exemplar") == 1]
    unlucky = [s for s in samples if s.get("pdf_unlucky") == 1]

    note_counter: Counter = Counter()
    for s in samples:
        for note in (s.get("pdf_review_notes") or "").split("；"):
            if note.strip():
                note_counter[note.strip()] += 1

    loss_notes: list[dict] = []
    for s in losses:
        for note in (s.get("pdf_review_notes") or "").split("；"):
            if note.strip():
                loss_notes.append({"note": note.strip(), "pnl_pct": s["pnl_pct"]})

    loss_reason_agg: dict[str, list[float]] = {}
    for item in loss_notes:
        loss_reason_agg.setdefault(item["note"], []).append(item["pnl_pct"])

    top_loss_reasons = sorted(
        [{"reason": k, "count": len(v), "avg_loss_pct": sum(v) / len(v)}
         for k, v in loss_reason_agg.items()],
        key=lambda x: x["count"], reverse=True,
    )[:10]

    exit_quality_counter = Counter(s.get("pdf_review_exit_quality", "neutral") for s in samples)
    entry_quality_counter = Counter(s.get("pdf_review_entry_quality", "neutral") for s in samples)
    exclusion_counter = Counter(e.get("_exclusion_reason", "unknown") for e in excluded)

    avg_pdf_entry = sum(s.get("pdf_entry_score", 0) for s in samples) / max(len(samples), 1)
    avg_pdf_overall = sum(s.get("pdf_overall_score", 0) for s in samples) / max(len(samples), 1)
    win_avg_entry = sum(s.get("pdf_entry_score", 0) for s in wins) / max(len(wins), 1)
    loss_avg_entry = sum(s.get("pdf_entry_score", 0) for s in losses) / max(len(losses), 1)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_trades_reviewed": len(samples) + len(excluded),
        "valid_samples": len(samples),
        "excluded_as_polluted": len(excluded),
        "exclusion_reasons": dict(exclusion_counter),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": len(wins) / max(len(samples), 1),
        "exemplars": len(exemplars),
        "anti_exemplars": len(anti_exemplars),
        "unlucky_good_process": len(unlucky),
        "avg_pdf_entry_score": round(avg_pdf_entry, 3),
        "avg_pdf_overall_score": round(avg_pdf_overall, 3),
        "win_avg_pdf_entry_score": round(win_avg_entry, 3),
        "loss_avg_pdf_entry_score": round(loss_avg_entry, 3),
        "entry_quality": dict(entry_quality_counter),
        "exit_quality": dict(exit_quality_counter),
        "top_loss_reasons": top_loss_reasons,
        "common_notes": note_counter.most_common(10),
    }


def _match_entry_timestamps(journal: list[dict]) -> list[dict]:
    """Match each exit with its corresponding entry to get entry_timestamp.

    Strategy: for each exit, find the most recent entry for the same symbol
    that occurred before the exit timestamp.
    """
    entries_by_symbol: dict[str, list[dict]] = {}
    for item in journal:
        if item.get("action") in ("open_long", "open_short"):
            sym = item.get("symbol", "")
            entries_by_symbol.setdefault(sym, []).append(item)

    enriched_exits: list[dict] = []
    for item in journal:
        if item.get("action") != "exit":
            continue
        sym = item.get("symbol", "")
        exit_ts = _parse_ts(item.get("timestamp"))
        entry_price = float(item.get("entry_price", 0))

        best_entry = None
        candidates = entries_by_symbol.get(sym, [])
        for entry in reversed(candidates):
            entry_dt = _parse_ts(entry.get("timestamp"))
            if not entry_dt or not exit_ts:
                continue
            if entry_dt >= exit_ts:
                continue
            if abs(float(entry.get("entry_price", 0)) - entry_price) < entry_price * 0.001:
                best_entry = entry
                break
            if best_entry is None:
                best_entry = entry

        enriched = dict(item)
        if best_entry:
            enriched["entry_timestamp"] = best_entry.get("timestamp")
            enriched["entry_score"] = best_entry.get("entry_score") or best_entry.get("score")
            if best_entry in candidates:
                candidates.remove(best_entry)
        enriched_exits.append(enriched)

    return enriched_exits


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF methodology backtest review for V4 paper trades.")
    parser.add_argument("--state", default="data/trading_system/paper_state.json",
                        help="Path to paper_state.json")
    parser.add_argument("--output-samples", default="data/trading_system/model_training/paper_replay_autolearn/paper_closed_trades.jsonl",
                        help="Path to training JSONL (append)")
    parser.add_argument("--output-report", default="data/trading_system/pdf_backtest_review_report.json",
                        help="Path to review report JSON")
    parser.add_argument("--limit", type=int, default=0, help="Max trades to review (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to training JSONL")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    state_path = base_dir / args.state
    output_samples_path = base_dir / args.output_samples
    output_report_path = base_dir / args.output_report

    print(f"[PDF Backtest Review] Loading state from {state_path}")
    if not state_path.exists():
        print(f"ERROR: state file not found: {state_path}")
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    journal = state.get("journal", [])
    exits = _match_entry_timestamps(journal)
    print(f"[PDF Backtest Review] Found {len(exits)} exit trades in journal")

    if args.limit > 0:
        exits = exits[:args.limit]
        print(f"[PDF Backtest Review] Limited to {len(exits)} trades")

    client = _make_client()
    samples: list[dict] = []
    excluded: list[dict] = []

    for i, trade in enumerate(exits):
        symbol = trade.get("symbol", "?")
        print(f"  [{i+1}/{len(exits)}] Reviewing {symbol}...", end=" ", flush=True)
        try:
            result = review_single_trade(client, trade)
        except Exception as exc:
            print(f"ERROR: {exc}")
            excluded.append({"_excluded": True, "_exclusion_reason": f"exception:{type(exc).__name__}", "symbol": symbol})
            continue

        if result is None:
            print("skipped (no result)")
            continue
        if result.get("_excluded"):
            print(f"excluded ({result.get('_exclusion_reason')})")
            excluded.append(result)
        else:
            pnl = result.get("pnl_pct", 0)
            pdf_entry = result.get("pdf_entry_score", 0)
            print(f"OK pnl={pnl*100:.2f}% pdf_entry={pdf_entry:.2f}")
            samples.append(result)

        if args.delay > 0 and i < len(exits) - 1:
            time.sleep(args.delay)

    print(f"\n[PDF Backtest Review] Results: {len(samples)} valid, {len(excluded)} excluded")

    report = _generate_report(samples, excluded)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PDF Backtest Review] Report saved to {output_report_path}")

    if samples and not args.dry_run:
        from trading_system.self_train_dataset import append_new_samples
        added = append_new_samples(output_samples_path, samples)
        print(f"[PDF Backtest Review] Appended {added} new samples to {output_samples_path}")
    elif args.dry_run:
        print(f"[PDF Backtest Review] Dry run — {len(samples)} samples NOT written")

    print("\n===== 复盘摘要 =====")
    print(f"总交易: {report['total_trades_reviewed']} | 有效样本: {report['valid_samples']} | 排除: {report['excluded_as_polluted']}")
    print(f"胜率: {report['win_rate']*100:.1f}% ({report['win_count']}W / {report['loss_count']}L)")
    print(f"PDF入场评分: 均值={report['avg_pdf_entry_score']:.3f} | 盈利={report['win_avg_pdf_entry_score']:.3f} | 亏损={report['loss_avg_pdf_entry_score']:.3f}")
    print(f"正面范例: {report['exemplars']} | 反面教材: {report['anti_exemplars']} | 好过程坏结果: {report['unlucky_good_process']}")
    if report["top_loss_reasons"]:
        print("\n亏损主因:")
        for item in report["top_loss_reasons"][:5]:
            print(f"  - {item['reason']}: {item['count']}次, 平均亏损 {item['avg_loss_pct']*100:.2f}%")
    print("===== 复盘结束 =====")


if __name__ == "__main__":
    main()
