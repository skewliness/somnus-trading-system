# Trading System Architecture

## Goal

Build a meme and AI trading framework that can keep moving even when some bridges are still incomplete.

## Layers

1. Source bridges
   - Manual sentiment CSV
   - Twitter distill raw datasets
   - Binance Square feed bridge
   - Binance skill raw import directory
   - Binance skill bridge directory
2. Square filtering
   - Heuristic bot and low-quality post filtering
   - Repeated-template and low-text-variety detection
2. Sentiment aggregation
   - Normalize source outputs
   - Merge into one ranked signal file
3. Market scanner
   - Binance futures live data or local replay data
4. Scoring
   - Heat
   - Funding
   - OI expansion
   - Momentum
   - Volume
   - Taker ratio
5. Paper execution
   - Fixed notional
   - Stop loss
   - Take profit
   - Max hold
6. Binance execution bridge
   - Generate official skill requests
   - Import external execution result JSON back into the reporting layer
   - Keep authenticated submission outside core strategy runtime
7. Issue registry
   - Track known gaps without blocking the rest of the framework

## Current state

- Source bridge layer exists
- Square filtering layer exists
- Aggregation layer exists
- Scanner exists
- Paper execution exists
- Binance execution bridge exists
- Issue logging exists
- Binance skill raw-to-bridge sync exists
- Binance skill import commands exist for OpenClaw handoff
- External execution-result import exists for OpenClaw handoff
- Bot-account classifier now includes repetition checks but is still rules-based

## Principle

Do not block the framework because one bridge is incomplete.
Record the gap, keep the pipeline running, and keep the interfaces stable.
