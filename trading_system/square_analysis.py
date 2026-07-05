from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from trading_system.bot_filter import decide_square_post, is_valid_tradable_symbol


def enrich_square_feed(raw_path: Path, output_path: Path) -> dict:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    posts = payload.get("posts", [])
    author_counts = Counter(str(post.get("username") or "").strip() for post in posts)
    normalized_texts = [_normalize_text(str(post.get("body_text") or "")) for post in posts]
    global_text_counts = Counter(item for item in normalized_texts if item)
    author_text_sets: dict[str, set[str]] = {}
    for post, normalized in zip(posts, normalized_texts, strict=False):
        username = str(post.get("username") or "").strip()
        if not username or not normalized:
            continue
        author_text_sets.setdefault(username, set()).add(normalized)

    enriched_posts = []
    kept = 0
    dropped = 0
    invalid_symbol_drops = 0

    for post, normalized in zip(posts, normalized_texts, strict=False):
        normalized_symbols = []
        for symbol in post.get("trading_pairs", []):
            symbol_text = str(symbol).upper().strip()
            if is_valid_tradable_symbol(symbol_text):
                normalized_symbols.append(symbol_text)
            else:
                invalid_symbol_drops += 1
        post["trading_pairs"] = list(dict.fromkeys(normalized_symbols))
        username = str(post.get("username") or "").strip()
        unique_text_count = len(author_text_sets.get(username, set()))
        author_post_count = author_counts.get(username, 0)
        post.setdefault("bot_features", {})
        post["bot_features"]["repeated_text_count"] = global_text_counts.get(normalized, 1) if normalized else 1
        post["bot_features"]["author_unique_text_ratio"] = round(
            unique_text_count / max(1, author_post_count),
            3,
        )

        decision = decide_square_post(post, author_post_count=author_post_count)
        post["bot_filter"] = decision.to_dict()
        if decision.keep and post["trading_pairs"]:
            kept += 1
        else:
            dropped += 1
        enriched_posts.append(post)

    summary = {
        "raw_post_count": len(posts),
        "kept_post_count": kept,
        "dropped_post_count": dropped,
        "invalid_symbol_drop_count": invalid_symbol_drops,
        "unique_authors": len([key for key in author_counts if key]),
    }
    output = {"summary": summary, "posts": enriched_posts}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _normalize_text(text: str) -> str:
    collapsed = re.sub(r"https?://\S+", " ", text.lower())
    collapsed = re.sub(r"@[a-z0-9_]+", " ", collapsed)
    collapsed = re.sub(r"[^a-z0-9$# ]+", " ", collapsed)
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return collapsed
