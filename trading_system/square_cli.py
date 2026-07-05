from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_system.square_reader import (
    BinanceSquareReader,
    SquarePost,
    analyze_posts,
    build_reader_from_export,
    load_auth_from_env,
    parse_request_export,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance Square reader bridge")
    sub = parser.add_subparsers(dest="command", required=True)

    detail = sub.add_parser("detail")
    detail.add_argument("post_id_or_url")
    detail.add_argument("--save-raw-dir", default="data/trading_system/square_raw")

    batch = sub.add_parser("batch")
    batch.add_argument("--input", required=True, help="Text file with one post id or post URL per line")
    batch.add_argument("--save-raw-dir", default="data/trading_system/square_raw")
    batch.add_argument("--output", default="data/trading_system/square_posts_analysis.json")

    feed = sub.add_parser("feed")
    feed.add_argument("--page-index", type=int, default=1)
    feed.add_argument("--page-size", type=int, default=20)
    feed.add_argument("--scene", default="web-homepage")
    feed.add_argument("--curl-file", help="Path to browser-exported request file (.txt curl export or .har)")
    feed.add_argument("--output", default="data/trading_system/square_feed_analysis.json")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    cookie_header, csrf_token = load_auth_from_env()
    reader = BinanceSquareReader(cookie_header=cookie_header, csrf_token=csrf_token)

    if args.command == "detail":
        post = reader.fetch_post_detail(args.post_id_or_url, Path(args.save_raw_dir))
        print(json.dumps(post.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "batch":
        input_path = Path(args.input)
        lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        posts: list[SquarePost] = [reader.fetch_post_detail(line, Path(args.save_raw_dir)) for line in lines]
        report = {
            "summary": analyze_posts(posts),
            "posts": [post.to_dict() for post in posts],
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return

    if args.curl_file:
        request_file = Path(args.curl_file)
        _url, _headers, payload = parse_request_export(request_file)
        reader = build_reader_from_export(request_file)
        posts = reader.fetch_feed_posts(
            page_index=int(payload.get("pageIndex", args.page_index)),
            page_size=int(payload.get("pageSize", args.page_size)),
            scene=payload.get("scene", args.scene),
            content_ids=payload.get("contentIds", []),
        )
    else:
        posts = reader.fetch_feed_posts(page_index=args.page_index, page_size=args.page_size, scene=args.scene)
    report = {
        "summary": analyze_posts(posts),
        "posts": [post.to_dict() for post in posts],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
