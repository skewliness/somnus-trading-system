# Trading System

This project turns the earlier Opus architecture into a runnable first version.

## What it does

- Pulls Binance USD-M perpetual market data in `live` mode.
- Supports `replay` mode with local sample data so the full pipeline can run without exchange access.
- Reads a manual sentiment file that you can later replace with distilled X/Twitter or Binance Square data.
- Reads distilled Twitter/X account datasets and converts them into tradable symbol heat.
- Reads Binance Square homepage feed via a browser-exported curl file.
- Filters suspicious Square posts before they enter the sentiment layer, including repeated-template checks.
- Imports raw official Binance skill outputs into normalized bridge files.
- Merges multiple sentiment bridges into one normalized sentiment file.
- Records open implementation issues instead of blocking the framework build.
- Generates Binance futures execution request files for the official `derivatives-trading-usds-futures` skill.
- Imports external futures execution result JSON files back into the reporting layer.
- Scores symbols using hotness, positive funding rate, 4h open-interest expansion, 24h momentum, 24h quote volume, and taker buy/sell ratio.
- Generates paper-trade entries and exits with fixed `200 USDT` notional, stop loss, take profit, and max-hold rules.

## Commands

```bash
python -m trading_system.cli init-sentiment
python -m trading_system.cli collect-square-feed --square-curl-file data/trading_system/binance_square_request.txt
python -m trading_system.cli import-skill-output --bridge-name crypto-market-rank --input-file C:/temp/crypto-market-rank.json
python -m trading_system.cli import-skill-output --bridge-name meme-rush --input-file C:/temp/meme-rush.json
python -m trading_system.cli import-skill-output --bridge-name trading-signal --input-file C:/temp/trading-signal.json
python -m trading_system.cli import-execution-result --input-file C:/temp/futures-result.json --label latest_execution
python -m trading_system.cli execute-request --input-file data/trading_system/execution_requests/execution_request_YYYYMMDDTHHMMSSZ.json
python -m trading_system.cli scan --mode replay
python -m trading_system.cli rebalance --mode replay
python -m trading_system.cli pipeline --mode replay --square-curl-file E:/lujing/binance.txt
python -m trading_system.cli scan --mode live
python -m trading_system.cli rebalance --mode live
```

## Files

- Sentiment input: `data/trading_system/sentiment_signals.csv`
- Merged sentiment output: `data/trading_system/merged_sentiment_signals.csv`
- Twitter distill report: `data/trading_system/twitter_distill_report.json`
- Twitter raw datasets: `data/trading_system/twitter_raw`
- Replay market snapshot: `data/trading_system/replay_market_data.json`
- Square feed report: `data/trading_system/square_feed_analysis.json`
- Enriched Square feed report: `data/trading_system/square_feed_enriched.json`
- Imported skill raw files: `data/trading_system/skill_raw`
- Imported skill bridge files: `data/trading_system/skill_bridges`
- Latest scan report: `data/trading_system/last_scan_report.json`
- Paper portfolio state: `data/trading_system/paper_state.json`
- Pipeline run report: `data/trading_system/pipeline_run_report.json`
- Pipeline markdown report: `data/trading_system/pipeline_run_report.md`
- Execution requests: `data/trading_system/execution_requests`
- Execution results: `data/trading_system/execution_results`
- Open issues: `data/trading_system/open_issues.json` and `data/trading_system/open_issues.md`

## Framework layers

- `trading_system/bridges.py`: file-based bridges for manual sentiment, Square feed, and future Binance skill outputs
- `trading_system/bot_filter.py`: Square post quality heuristics
- `trading_system/square_analysis.py`: Square feed enrichment and filtering
- `trading_system/twitter_bridge.py`: distilled Twitter dataset to sentiment signal bridge
- `trading_system/sentiment_aggregator.py`: normalized sentiment merge layer
- `trading_system/skill_adapters.py`: raw Binance skill payload to bridge conversion
- `trading_system/candidates.py`: candidate basket generation
- `trading_system/execution_bridge.py`: official Binance futures skill request generation
- `trading_system/pipeline.py`: end-to-end orchestration
- `trading_system/issues.py`: unresolved issue registry

## OpenClaw handoff

The intended OpenClaw loop is:

1. OpenClaw or another external orchestrator calls `crypto-market-rank`, `meme-rush`, and `trading-signal`.
2. Save each JSON output locally and import it with `import-skill-output`.
3. Run `pipeline` or `rebalance` to write `execution_requests`.
4. An authenticated executor consumes `execution_requests` and submits them through `derivatives-trading-usds-futures`.
5. Save the returned order result JSON and import it with `import-execution-result`.
6. Run `pipeline` again so the report can include the execution result state.

## Real account execution setup

Set these environment variables before `execute-request`:

- `BINANCE_API_KEY=...`
- `BINANCE_API_SECRET=...`
- `BINANCE_FUTURES_BASE_URL=https://fapi.binance.com`
- `TS_EXECUTION_MODE=live`
- `TS_EXECUTION_TEST_ORDER=false` for real orders (`true` keeps Binance test-order mode)
- `TS_LIVE_EXECUTION_CONFIRMATION=CONFIRM_LIVE_BINANCE_FUTURES` is required for real `/fapi/v1/order` submissions.

Recommended safety defaults:

- Keep withdraw permission disabled on the Binance API key.
- Start with `TS_EXECUTION_TEST_ORDER=true` and verify request/result flow first.
- Use small `TS_POSITION_NOTIONAL_USDT` until live execution is confirmed.
- Keep `TS_LIVE_MAX_ORDER_NOTIONAL_USDT`, `TS_LIVE_MAX_INTENTS_PER_REQUEST`, and `TS_LIVE_MAX_REQUEST_LOSS_USDT` conservative; requests breaching those limits are written as blocked audit files instead of executable live orders.

## Planned bridge directory

Future Binance skills should write normalized JSON files into:

- `data/trading_system/skill_bridges`

Raw skill payloads can also be dropped into:

- `data/trading_system/skill_raw`

Expected payload shape:

```json
{
  "bridge_name": "crypto-market-rank",
  "signals": [
    {
      "symbol": "AIXBTUSDT",
      "hotness_score": 88,
      "mention_count": 42,
      "source": "crypto-market-rank",
      "notes": "social hype rank"
    }
  ]
}
```

## Important limitation

This environment is blocked by Binance Futures `HTTP 451 restricted location`, so `live` mode cannot be verified here. The code handles that case and tells you to run from an allowed network or route traffic through your own compliant proxy setup.
