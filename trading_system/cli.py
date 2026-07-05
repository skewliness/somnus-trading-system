from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from trading_system.config import StrategyConfig
from trading_system.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meme/AI Binance trading system")
    parser.add_argument(
        "command",
        choices=[
            "scan",
            "rebalance",
            "init-sentiment",
            "pipeline",
            "collect-square-feed",
            "collect-options-delivery",
            "import-skill-output",
            "import-execution-result",
            "execute-request",
            "execute-pending-requests",
            "build-bitmex-dataset",
            "build-wallet-dataset",
            "factor-backtest",
            "hf-backtest",
            "build-hf-market-data",
            "build-hf-market-data-v2",
            "discover-factors",
            "smart-money-panel",
            "train-coin-scout",
            "train-coin-scout-4h",
            "whale-scout-scan",
            "train-whale-scout",
            "distill-lanaai-square",
            "hf-backtest-v2",
            "news-wss",
        ],
    )
    parser.add_argument("--mode", default="live", choices=["live", "replay"])
    parser.add_argument(
        "--square-curl-file",
        default=os.getenv("TS_SQUARE_CURL_FILE", "data/trading_system/binance_square_request.txt"),
    )
    parser.add_argument("--input-file")
    parser.add_argument("--bridge-name")
    parser.add_argument("--label")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-file")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--hold-bars", type=int, default=3)
    parser.add_argument("--min-momentum-pct", type=float, default=1.2)
    parser.add_argument("--volume-z-min", type=float, default=1.0)
    parser.add_argument("--min-oi-change-pct", type=float, default=0.5)
    parser.add_argument("--max-momentum-pct", type=float)
    parser.add_argument("--volume-z-max", type=float)
    parser.add_argument("--max-oi-change-pct", type=float)
    parser.add_argument("--hf-interval", default="5m")
    parser.add_argument("--hf-limit", type=int, default=120)
    parser.add_argument("--hf-symbol-limit", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-gap-hours", type=float, default=6.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--report-file")
    parser.add_argument("--v2-momentum-threshold", type=float, default=0.3)
    parser.add_argument("--v2-volume-z-threshold", type=float, default=1.0)
    parser.add_argument("--v2-oi-threshold", type=float, default=0.8)
    parser.add_argument("--v2-funding-threshold", type=float, default=0.0005)
    parser.add_argument("--v2-taker-threshold", type=float, default=0.55)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = StrategyConfig()
    configure_logging(config.log_level, json_logs=config.log_format.lower() == "json")

    if args.command == "init-sentiment":
        _init_sentiment(config.sentiment_file)
        print(f"initialized: {config.sentiment_file}")
        return

    if args.command == "collect-square-feed":
        from trading_system.pipeline import TradingPipeline

        pipeline = TradingPipeline(config)
        report = pipeline.collect_square_feed_only(Path(args.square_curl_file))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "collect-options-delivery":
        from trading_system.pipeline import TradingPipeline

        pipeline = TradingPipeline(config)
        report = pipeline.collect_options_delivery_only()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return


    if args.command == "import-skill-output":
        from trading_system.skill_adapters import import_skill_output

        if not args.input_file or not args.bridge_name:
            raise SystemExit("--input-file and --bridge-name are required")
        report = import_skill_output(
            Path(args.input_file),
            args.bridge_name,
            config.skill_raw_dir,
            config.skill_bridge_dir,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "import-execution-result":
        from trading_system.skill_adapters import import_execution_result

        if not args.input_file:
            raise SystemExit("--input-file is required")
        report = import_execution_result(Path(args.input_file), config.execution_result_dir, label=args.label)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "execute-request":
        from trading_system.execution_runner import BinanceExecutionError, BinanceExecutionRunner

        if not args.input_file:
            raise SystemExit("--input-file is required")
        try:
            report = BinanceExecutionRunner(config).execute_request_file(Path(args.input_file))
        except BinanceExecutionError as exc:
            print(json.dumps({"status": "failed", "error": str(exc), "input_file": args.input_file}, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "execute-pending-requests":
        from trading_system.execution_runner import BinanceExecutionError, BinanceExecutionRunner
        from trading_system.persistence import load_json

        try:
            runner = BinanceExecutionRunner(config)
            report = runner.execute_pending_requests()
        except BinanceExecutionError as exc:
            print(json.dumps({"status": "failed", "error": str(exc), "request_dir": str(config.execution_request_dir)}, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        orphan_report: list = []
        try:
            paper_state = load_json(config.state_file) if config.state_file.exists() else {}
            paper_symbols = {
                str(p.get("symbol") or "")
                for p in paper_state.get("positions", []) + paper_state.get("shadow_positions", [])
                if p.get("symbol")
            }
            orphan_report = runner.reconcile_orphan_positions(paper_symbols)
        except Exception:
            pass
        if orphan_report:
            report["orphan_reconcile"] = orphan_report
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "build-bitmex-dataset":
        from trading_system.bitmex_distill import build_episode_dataset

        if not args.input_dir:
            raise SystemExit("--input-dir is required")
        output_dir = Path(args.output_dir) if args.output_dir else Path("data/trading_system/model_training/bitmex_btc_since_2020")
        report = build_episode_dataset(Path(args.input_dir), output_dir, max_gap_hours=args.max_gap_hours)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "build-wallet-dataset":
        from trading_system.wallet_distill import build_wallet_training_dataset

        if not args.input_file:
            raise SystemExit("--input-file is required")
        output_dir = Path(args.output_dir) if args.output_dir else Path("data/trading_system/model_training/gmgn_wallet_distill")
        bridge_path = config.skill_bridge_dir / "gmgn-wallet-distill.json"
        report = build_wallet_training_dataset(Path(args.input_file), output_dir, bridge_path)
        print(json.dumps({**report, "bridge_output": str(bridge_path)}, ensure_ascii=False, indent=2))
        return

    if args.command == "factor-backtest":
        from trading_system.factor_backtest import run_factor_backtest

        input_file = Path(args.input_file) if args.input_file else config.closed_trade_dataset_file
        output_file = Path(args.output_file) if args.output_file else config.factor_backtest_file
        report = run_factor_backtest(input_file, output_file, min_samples=args.min_samples)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "hf-backtest":
        from trading_system.factor_backtest import (
            _symbols_from_replay_snapshot,
            build_binance_hf_market_jsonl,
            run_high_frequency_backtest,
        )

        input_file = Path(args.input_file) if args.input_file else config.replay_file.with_name("hf_market_data.jsonl")
        if not input_file.exists() and config.replay_file.exists():
            symbols = _symbols_from_replay_snapshot(config.replay_file, limit=args.hf_symbol_limit)
            if symbols:
                build_binance_hf_market_jsonl(
                    symbols,
                    input_file,
                    interval=args.hf_interval,
                    limit=args.hf_limit,
                    config=config,
                )
        output_file = Path(args.output_file) if args.output_file else config.factor_backtest_file.with_name("hf_backtest_report.json")
        report = run_high_frequency_backtest(
            input_file,
            output_file,
            lookback=args.lookback,
            hold_bars=args.hold_bars,
            min_momentum_pct=args.min_momentum_pct,
            volume_z_min=args.volume_z_min,
            min_oi_change_pct=args.min_oi_change_pct,
            max_momentum_pct=args.max_momentum_pct,
            volume_z_max=args.volume_z_max,
            max_oi_change_pct=args.max_oi_change_pct,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "build-hf-market-data":
        from trading_system.factor_backtest import _symbols_from_replay_snapshot, build_binance_hf_market_jsonl

        if args.input_file:
            symbols = _symbols_from_replay_snapshot(Path(args.input_file), limit=args.hf_symbol_limit)
        else:
            symbols = _symbols_from_replay_snapshot(config.replay_file, limit=args.hf_symbol_limit)
        if not symbols:
            raise SystemExit("no valid USDT symbols found for hf market-data build")
        output_file = Path(args.output_file) if args.output_file else config.replay_file.with_name("hf_market_data.jsonl")
        report = build_binance_hf_market_jsonl(
            symbols,
            output_file,
            interval=args.hf_interval,
            limit=args.hf_limit,
            config=config,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "build-hf-market-data-v2":
        from trading_system.factor_backtest import _symbols_from_replay_snapshot, build_binance_hf_market_jsonl_v2

        if args.input_file:
            symbols = _symbols_from_replay_snapshot(Path(args.input_file), limit=args.hf_symbol_limit)
        else:
            symbols = _symbols_from_replay_snapshot(config.replay_file, limit=args.hf_symbol_limit)
        if not symbols:
            raise SystemExit("no valid USDT symbols found for hf market-data v2 build")
        output_file = Path(args.output_file) if args.output_file else config.replay_file.with_name("hf_market_data_v2.jsonl")
        report = build_binance_hf_market_jsonl_v2(
            symbols,
            output_file,
            interval=args.hf_interval,
            limit=args.hf_limit,
            config=config,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "hf-backtest-v2":
        from trading_system.factor_backtest import run_hf_backtest_v2

        input_file = Path(args.input_file) if args.input_file else config.replay_file.with_name("hf_market_data_v2.jsonl")
        output_file = Path(args.output_file) if args.output_file else config.replay_file.with_name("hf_backtest_v2_entries.jsonl")
        report_file = Path(args.report_file) if args.report_file else config.replay_file.with_name("hf_backtest_v2_report.json")
        report = run_hf_backtest_v2(
            input_file,
            output_path=output_file,
            report_path=report_file,
            lookback=args.lookback,
            momentum_threshold_pct=args.v2_momentum_threshold,
            volume_z_threshold=args.v2_volume_z_threshold,
            oi_change_threshold_pct=args.v2_oi_threshold,
            funding_extreme_threshold=args.v2_funding_threshold,
            taker_imbalance_threshold=args.v2_taker_threshold,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "discover-factors":
        from trading_system.factor_backtest import discover_quant_factors

        input_file = Path(args.input_file) if args.input_file else config.closed_trade_dataset_file
        output_file = Path(args.output_file) if args.output_file else config.factor_backtest_file.with_name("quant_factor_discovery.json")
        report = discover_quant_factors(input_file, output_file, min_samples=args.min_samples, top_n=args.top_n)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "news-wss":
        from trading_system.news_wss_listener import run_news_wss

        run_news_wss(config)
        return

    if args.command == "smart-money-panel":
        from trading_system.smart_money_panel import serve_smart_money_panel

        print(json.dumps({"status": "starting", "url": f"http://{args.host}:{args.port}"}, ensure_ascii=False))
        serve_smart_money_panel(host=args.host, port=args.port)
        return

    if args.command == "train-coin-scout":
        # Default: use 4H klines labeler (recommended)
        from trading_system.coin_scout_4h_labeler import build_and_train_4h

        ds_path = args.input_file or "data/trading_system/model_training/paper_replay_autolearn/paper_decision_samples.jsonl"
        model_dir = args.output_dir or "data/trading_system/model_training/coin_scout"
        report = build_and_train_4h(
            decision_samples_path=ds_path,
            model_dir=model_dir,
            max_symbols=None,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return

    if args.command == "train-coin-scout-4h":
        from trading_system.coin_scout_4h_labeler import build_and_train_4h

        ds_path = args.input_file or "data/trading_system/model_training/paper_replay_autolearn/paper_decision_samples.jsonl"
        model_dir = args.output_dir or "data/trading_system/model_training/coin_scout"
        max_sym = args.min_samples if args.min_samples > 0 else None  # reuse --min-samples as max symbols for testing
        report = build_and_train_4h(
            decision_samples_path=ds_path,
            model_dir=model_dir,
            max_symbols=max_sym,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return

    if args.command == "distill-lanaai-square":
        from trading_system.lanaai_square_distiller import run_distill_once, print_result
        result = run_distill_once(dry_run=False)
        print_result(result)
        return

    if args.command == "whale-scout-scan":
        from trading_system.whale_scout_scan import run_whale_scout_scan, print_results

        result = run_whale_scout_scan(
            model_dir=args.output_dir or "data/trading_system/model_training/whale_scout",
            top_n=args.top_n,
            min_score=0,
            min_volume_usdt=20_000_000,
            output_file=args.output_file,
        )
        if result.get("status") != "error":
            print_results(result)
        return

    if args.command == "train-whale-scout":
        from trading_system.train_from_matched import main as train_main
        train_main()
        return

    if args.command == "pipeline":
        from trading_system.pipeline import TradingPipeline

        pipeline = TradingPipeline(config)
        report = pipeline.run(mode=args.mode, square_curl_file=Path(args.square_curl_file), collect_square_feed=True)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    from trading_system.binance import BinanceAccessError
    from trading_system.engine import TradingSystem
    from trading_system.issues import build_default_issue_registry, sync_live_access_issue

    issue_registry = build_default_issue_registry()
    if args.command == "rebalance" and args.mode != "live":
        raise SystemExit("rebalance command requires --mode live; replay is research-only")
    try:
        system = TradingSystem(config=config, mode=args.mode)
        if args.command == "scan":
            snapshots = system.scan()
            if args.mode == "live":
                sync_live_access_issue(issue_registry)
                issue_registry.write(config.issues_json_file, config.issues_md_file)
            print(json.dumps([item.to_dict() for item in snapshots[:10]], ensure_ascii=False, indent=2))
            return

        result = system.rebalance()
        if args.mode == "live":
            sync_live_access_issue(issue_registry)
            issue_registry.write(config.issues_json_file, config.issues_md_file)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BinanceAccessError as exc:
        if args.mode == "live":
            sync_live_access_issue(issue_registry, exc)
            issue_registry.write(config.issues_json_file, config.issues_md_file)
        raise SystemExit(str(exc))


def _init_sentiment(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    rows = [
        ["symbol", "hotness_score", "mention_count", "source", "updated_at", "notes"],
        ["AIXBTUSDT", "91", "164", "manual_distill", "2026-04-12T08:00:00+00:00", "AI topic still hot"],
        ["VIRTUALUSDT", "87", "142", "manual_distill", "2026-04-12T08:00:00+00:00", "Strong AI meme attention"],
        ["TSTUSDT", "74", "96", "manual_distill", "2026-04-12T08:00:00+00:00", "High Square chatter"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
