from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_USDT_SYMBOL = __import__("re").compile(r"^[A-Z0-9]{2,20}USDT$")


@dataclass(slots=True)
class BotFilterDecision:
    keep: bool
    suspicion_score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep": self.keep,
            "suspicion_score": round(self.suspicion_score, 3),
            "reasons": self.reasons,
        }


def decide_square_post(post: dict[str, Any], author_post_count: int = 1) -> BotFilterDecision:
    reasons: list[str] = []
    score = 0.0

    username = str(post.get("username") or "").strip()
    display_name = str(post.get("display_name") or "").strip()
    body_text = str(post.get("body_text") or "").strip()
    view_count = int(post.get("view_count") or 0)
    pair_count = len(post.get("trading_pairs") or [])
    ratio = float(post.get("bot_features", {}).get("engagement_view_ratio", 0.0))
    engagement = int(post.get("bot_features", {}).get("engagement_total", 0))
    repeated_text_count = int(post.get("bot_features", {}).get("repeated_text_count", 1))
    author_unique_text_ratio = float(post.get("bot_features", {}).get("author_unique_text_ratio", 1.0))

    if not username:
        score += 3.0
        reasons.append("missing_username")
    if username.startswith("Square-Creator-"):
        score += 1.5
        reasons.append("generic_square_creator_username")
    if len(body_text) < 8:
        score += 2.0
        reasons.append("very_short_text")
    if view_count >= 5000 and ratio < 0.0015:
        score += 2.5
        reasons.append("high_views_low_engagement")
    if pair_count >= 4:
        score += 1.5
        reasons.append("too_many_pairs")
    if author_post_count >= 3:
        score += 1.5
        reasons.append("high_author_frequency")
    if repeated_text_count >= 2:
        score += 2.0
        reasons.append("duplicate_body_pattern")
    if author_post_count >= 3 and author_unique_text_ratio < 0.7:
        score += 1.5
        reasons.append("low_text_variety")
    if display_name and display_name == username:
        score += 0.5
        reasons.append("display_name_same_as_username")
    if engagement == 0 and view_count > 1000:
        score += 1.5
        reasons.append("views_without_engagement")

    keep = score < 3.5
    return BotFilterDecision(keep=keep, suspicion_score=score, reasons=reasons)


def is_valid_tradable_symbol(symbol: str) -> bool:
    return bool(VALID_USDT_SYMBOL.match(symbol))
