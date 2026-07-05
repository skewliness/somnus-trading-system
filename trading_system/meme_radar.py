from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from trading_system.models import MarketSnapshot

SQUARE_SOURCE = "square_feed"


def build_meme_radar(
    ranked: list[MarketSnapshot],
    candidate_basket: list[dict],
    hotness_priority: list[dict],
    square_enriched_report_file: Path | None = None,
    limit: int = 10,
) -> dict:
    """Build a trader-facing meme radar: tags, evidence chain, validation steps.

    This is intentionally deterministic and JSON-serializable so it can feed reports,
    daily scan archives, and model samples without adding UI dependencies.
    """
    square_context = _load_square_context(square_enriched_report_file)
    hotness_by_symbol = {str(item.get("symbol") or "").upper(): item for item in hotness_priority}
    basket_by_symbol = {str(item.get("symbol") or "").upper(): item for item in candidate_basket}

    rows: list[dict] = []
    for snapshot in ranked[: max(limit * 2, limit)]:
        if snapshot.score <= 0:
            continue
        symbol = snapshot.symbol.upper()
        basket = basket_by_symbol.get(symbol, {})
        hotness = hotness_by_symbol.get(symbol, {})
        sq = square_context["by_symbol"].get(symbol, {})
        tags = _build_tags(snapshot, hotness, sq)
        evidence_chain = _build_evidence_chain(snapshot, basket, hotness, sq)
        validation_steps = _build_validation_steps(snapshot, basket, sq)
        invalidation = _build_invalidation(snapshot, basket)
        radar_score = _radar_score(snapshot, hotness, sq, tags)
        rows.append(
            {
                "symbol": symbol,
                "radar_score": round(radar_score, 2),
                "score": round(snapshot.score, 2),
                "execution_status": basket.get("execution_status", "watch_only"),
                "risk_band": basket.get("risk_band", "high"),
                "setup_type": snapshot.setup_type,
                "tags": tags,
                "evidence_chain": evidence_chain,
                "next_validation": validation_steps,
                "invalidation": invalidation,
                "square_rank": sq.get("best_rank"),
                "square_post_count": sq.get("post_count", 0),
                "square_still_on_feed": bool(sq.get("post_count", 0)),
                "price_change_pct_5m": round(snapshot.price_change_pct_5m, 3),
                "price_change_pct_24h": round(snapshot.price_change_pct_24h, 3),
                "oi_change_pct_4h": round(snapshot.oi_change_pct_4h, 3),
                "funding_rate": round(snapshot.funding_rate, 8),
                "taker_buy_sell_ratio": round(snapshot.taker_buy_sell_ratio, 4),
                "sentiment_mentions": snapshot.sentiment_mentions,
                "sentiment_source": snapshot.sentiment_source,
            }
        )
    rows.sort(key=lambda item: (item["radar_score"], item["score"], item["square_post_count"]), reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "nicjiang_meme_radar_v1",
        "summary": {
            "candidate_count": len(rows[:limit]),
            "square_tracked_count": sum(1 for item in rows[:limit] if item.get("square_post_count", 0) > 0),
            "executable_count": sum(1 for item in rows[:limit] if item.get("execution_status") == "executable_now"),
        },
        "candidates": rows[:limit],
    }


def write_daily_scan_archive(radar: dict, archive_dir: Path, captured_at: str | None = None) -> Path:
    dt = _parse_dt(captured_at) or datetime.now(timezone.utc)
    path = archive_dir / f"{dt.date().isoformat()}.md"
    lines = [
        f"# 推土机妖币扫描 {dt.date().isoformat()}",
        "",
        f"生成时间: {radar.get('generated_at', '')}",
        f"候选数: {radar.get('summary', {}).get('candidate_count', 0)}｜Square跟踪: {radar.get('summary', {}).get('square_tracked_count', 0)}｜可执行: {radar.get('summary', {}).get('executable_count', 0)}",
        "",
        "## Top candidates",
    ]
    for idx, item in enumerate(radar.get("candidates", [])[:10], start=1):
        lines.extend(
            [
                f"### {idx}. {item.get('symbol')}｜Radar {item.get('radar_score')}｜系统评分 {item.get('score')}",
                f"状态: {item.get('execution_status')}｜风险: {item.get('risk_band')}｜结构: {item.get('setup_type')}",
                f"标签: {', '.join(item.get('tags') or []) or 'none'}",
                f"证据链: {'; '.join(item.get('evidence_chain') or []) or 'none'}",
                f"下一步验证: {'; '.join(item.get('next_validation') or []) or 'none'}",
                f"失效: {item.get('invalidation') or 'fixed_pct'}",
                "",
            ]
        )
    archive_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _load_square_context(path: Path | None) -> dict:
    by_symbol: dict[str, dict] = defaultdict(lambda: {"post_count": 0, "best_rank": None, "authors": [], "first_seen_at": "", "latest_seen_at": ""})
    if path is None or not path.exists():
        return {"by_symbol": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"by_symbol": {}}
    for idx, post in enumerate(payload.get("posts", []), start=1):
        if not (post.get("bot_filter", {}).get("keep", True)):
            continue
        ts = str(post.get("created_at") or post.get("update_time") or post.get("time") or "")
        author = str(post.get("username") or post.get("author") or "")
        for raw_symbol in post.get("trading_pairs", []) or []:
            symbol = str(raw_symbol).upper().strip()
            if not symbol:
                continue
            slot = by_symbol[symbol]
            slot["post_count"] += 1
            slot["best_rank"] = idx if slot["best_rank"] is None else min(slot["best_rank"], idx)
            if author and author not in slot["authors"]:
                slot["authors"].append(author)
            if ts:
                if not slot["first_seen_at"] or ts < slot["first_seen_at"]:
                    slot["first_seen_at"] = ts
                if not slot["latest_seen_at"] or ts > slot["latest_seen_at"]:
                    slot["latest_seen_at"] = ts
    return {"by_symbol": dict(by_symbol)}


def _build_tags(snapshot: MarketSnapshot, hotness: dict, sq: dict) -> list[str]:
    tags: list[str] = []
    source = (snapshot.sentiment_source or "").lower()
    if sq.get("post_count", 0) > 0 or SQUARE_SOURCE in source:
        tags.append("binance_square_hot")
    if snapshot.setup_type in {"mania_followthrough", "base_breakout_confirmed", "base_breakout_weak"}:
        tags.append(snapshot.setup_type)
    if snapshot.oi_change_pct_4h >= 12:
        tags.append("oi_expansion")
    if snapshot.funding_rate < 0 and snapshot.price_change_pct_24h > 0:
        tags.append("short_fuel")
    if snapshot.quote_volume_usdt_24h >= 100_000_000:
        tags.append("liquid_futures")
    if snapshot.price_change_pct_24h >= 18 or snapshot.price_change_pct_5m >= 3:
        tags.append("wide_range")
    if hotness.get("priority_score", 0) >= 80:
        tags.append("hotness_priority")
    return list(dict.fromkeys(tags))


def _build_evidence_chain(snapshot: MarketSnapshot, basket: dict, hotness: dict, sq: dict) -> list[str]:
    evidence = [f"score={snapshot.score:.1f}"]
    if sq.get("post_count", 0):
        evidence.append(f"square_rank={sq.get('best_rank')} posts={sq.get('post_count')}")
    if hotness:
        evidence.append(f"hotness_priority={hotness.get('priority_score')} source={hotness.get('source')}")
    evidence.append(f"24h={snapshot.price_change_pct_24h:.2f}% 5m={snapshot.price_change_pct_5m:.2f}%")
    evidence.append(f"oi4h={snapshot.oi_change_pct_4h:.2f}% funding={snapshot.funding_rate:.6f}")
    evidence.append(f"sentiment={snapshot.sentiment_score:.1f}/{snapshot.sentiment_mentions} src={snapshot.sentiment_source or 'none'}")
    if basket.get("blocked_reasons"):
        evidence.append("blocked=" + ",".join(basket.get("blocked_reasons")[:3]))
    return evidence


def _build_validation_steps(snapshot: MarketSnapshot, basket: dict, sq: dict) -> list[str]:
    steps: list[str] = []
    if not sq.get("post_count", 0) and "square_feed" not in (snapshot.sentiment_source or ""):
        steps.append("确认是否进入Binance Square热搜/广场流")
    else:
        steps.append("确认Square是否仍在榜且排名未快速衰退")
    if snapshot.setup_type in {"base_breakout_confirmed", "base_breakout_weak"}:
        steps.append("等回踩不破结构低点再执行")
    elif snapshot.setup_type == "mania_followthrough":
        steps.append("只追二次放量，不追单根急拉尾端")
    elif snapshot.setup_type == "oi_divergence_long_candidate":
        steps.append("确认OI继续扩张且价格不再横盘失速")
    else:
        steps.append("等待价格/OI/成交主动买入三者同向")
    if basket.get("execution_status") != "executable_now":
        steps.append("保持watch，补足可信来源或提及数后再升级")
    return steps


def _build_invalidation(snapshot: MarketSnapshot, basket: dict) -> str:
    stop_ref = basket.get("stop_reference") or "fixed_pct"
    if stop_ref == "structure_low_7d":
        return "跌回7日结构低点/突破位下方"
    if stop_ref == "range_low_4h":
        return "跌破4小时震荡下沿或OI扩张失败"
    if stop_ref == "tight_mania_stop":
        return "5m动能熄火或回撤超过mania紧止损"
    if stop_ref == "squeeze_high":
        return "挤压结构高点被重新收复"
    return "固定止损或评分跌破退出阈值"


def _radar_score(snapshot: MarketSnapshot, hotness: dict, sq: dict, tags: list[str]) -> float:
    score = snapshot.score * 0.55
    score += min(22.0, float(hotness.get("priority_score") or 0) * 0.22)
    score += min(12.0, float(sq.get("post_count") or 0) * 4.0)
    if sq.get("best_rank"):
        score += max(0.0, 8.0 - float(sq["best_rank"]) * 0.4)
    score += min(10.0, max(0.0, snapshot.oi_change_pct_4h) * 0.35)
    score += min(8.0, max(0.0, snapshot.price_change_pct_24h) * 0.12)
    if "short_fuel" in tags:
        score += 4.0
    return min(100.0, score)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
