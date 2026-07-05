from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from trading_system.bridges import BridgeLoadResult
from trading_system.models import SentimentSignal


SOURCE_WEIGHTS = {
    "manual_sentiment": 1.0,
    "manual_distill": 1.0,
    "square_feed": 1.05,
    "coingecko_trending": 0.7,
    "binance_top_movers": 0.8,
    "opennews_news": 0.8,
    "opennews_listing": 0.95,
    "opennews_onchain": 0.9,
    "opennews_meme": 0.85,
    "opennews_market": 0.85,
    "opennews_prediction": 0.7,
    "opentwitter_search": 0.85,
    "options_delivery_binance": 0.65,
    "options_delivery_okx": 0.65,
    "crypto-market-rank": 0.9,
    "meme-rush": 0.85,
    "trading-signal": 0.65,
    "lanaaielsa-distill": 0.85,
    "lanaai-square-distill": 0.85,
    "gmgn-wallet-distill": 0.8,
}

SIGNAL_STRONG_THRESHOLD = 70.0
OKX_ONCHAIN_SOURCES = {
    "gmgn-wallet-distill",
    "okx_wallet",
    "okx_wallet_distill",
    "okx-dex-signal",
    "okx_dex_signal",
    "okx_onchain",
    "okx_wallet_signal",
    "opennews_onchain",
}
BINANCE_CEX_SOURCES = {
    "binance_top_movers",
    "options_delivery_binance",
    "square_feed",
    "crypto-market-rank",
    "trading-signal",
}


@dataclass(slots=True)
class AggregationResult:
    output_path: Path
    signals: list[SentimentSignal]
    bridge_summaries: list[dict]


def aggregate_sentiment(results: list[BridgeLoadResult], output_path: Path) -> AggregationResult:
    merged: dict[str, dict] = {}
    summaries: list[dict] = []
    now = datetime.now(timezone.utc)
    for result in results:
        summaries.append({"bridge_name": result.bridge_name, **result.metadata})
        for signal in result.records:
            bucket = merged.setdefault(
                signal.symbol,
                {
                    "symbol": signal.symbol,
                    "score_total": 0.0,
                    "weight_total": 0.0,
                    "mention_count": 0,
                    "sources": set(),
                    "notes": [],
                    "earliest_strong_source": "",
                    "earliest_strong_timestamp": None,
                    "okx_confirmed_at": None,
                    "binance_confirmed_at": None,
                },
            )
            weight = SOURCE_WEIGHTS.get(signal.source, SOURCE_WEIGHTS.get(result.bridge_name, 0.6))
            bucket["score_total"] += signal.hotness_score * weight
            bucket["weight_total"] += weight
            bucket["mention_count"] += signal.mention_count
            bucket["sources"].add(signal.source)
            if signal.notes:
                bucket["notes"].append(signal.notes)
            if signal.hotness_score >= SIGNAL_STRONG_THRESHOLD:
                signal_time = signal.updated_at
                if bucket["earliest_strong_timestamp"] is None or signal_time < bucket["earliest_strong_timestamp"]:
                    bucket["earliest_strong_timestamp"] = signal_time
                    bucket["earliest_strong_source"] = signal.source
                source_parts = {item.strip() for item in signal.source.split(",") if item.strip()}
                if source_parts & OKX_ONCHAIN_SOURCES:
                    if bucket["okx_confirmed_at"] is None or signal_time < bucket["okx_confirmed_at"]:
                        bucket["okx_confirmed_at"] = signal_time
                if source_parts & BINANCE_CEX_SOURCES:
                    if bucket["binance_confirmed_at"] is None or signal_time < bucket["binance_confirmed_at"]:
                        bucket["binance_confirmed_at"] = signal_time

    output_signals: list[SentimentSignal] = []
    for symbol, bucket in sorted(merged.items()):
        if bucket["weight_total"] <= 0:
            continue
        score = round(bucket["score_total"] / bucket["weight_total"], 2)
        source = ",".join(sorted(bucket["sources"]))
        note = "; ".join(dict.fromkeys(bucket["notes"]))[:400]
        okx_confirmed_at = bucket["okx_confirmed_at"]
        binance_confirmed_at = bucket["binance_confirmed_at"]
        signal_sequence, lead_time_minutes = _classify_signal_sequence(okx_confirmed_at, binance_confirmed_at)
        output_signals.append(
            SentimentSignal(
                symbol=symbol,
                hotness_score=score,
                mention_count=bucket["mention_count"],
                source=source,
                updated_at=now,
                notes=note,
                earliest_strong_source=bucket["earliest_strong_source"],
                earliest_strong_timestamp=bucket["earliest_strong_timestamp"],
                okx_confirmed_at=okx_confirmed_at,
                binance_confirmed_at=binance_confirmed_at,
                signal_sequence=signal_sequence,
                lead_time_minutes=lead_time_minutes,
            )
        )

    write_sentiment_csv(output_signals, output_path)
    return AggregationResult(output_path=output_path, signals=output_signals, bridge_summaries=summaries)


def write_sentiment_csv(signals: list[SentimentSignal], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "hotness_score",
                "mention_count",
                "source",
                "updated_at",
                "notes",
                "earliest_strong_source",
                "earliest_strong_timestamp",
                "okx_confirmed_at",
                "binance_confirmed_at",
                "signal_sequence",
                "lead_time_minutes",
            ]
        )
        for signal in sorted(signals, key=lambda item: item.hotness_score, reverse=True):
            writer.writerow(
                [
                    signal.symbol,
                    signal.hotness_score,
                    signal.mention_count,
                    signal.source,
                    signal.updated_at.isoformat(),
                    signal.notes,
                    signal.earliest_strong_source,
                    _format_dt(signal.earliest_strong_timestamp),
                    _format_dt(signal.okx_confirmed_at),
                    _format_dt(signal.binance_confirmed_at),
                    signal.signal_sequence,
                    "" if signal.lead_time_minutes is None else round(signal.lead_time_minutes, 2),
                ]
            )


def _classify_signal_sequence(okx_confirmed_at: datetime | None, binance_confirmed_at: datetime | None) -> tuple[str, float | None]:
    if okx_confirmed_at and binance_confirmed_at:
        lead_time = (binance_confirmed_at - okx_confirmed_at).total_seconds() / 60.0
        if lead_time > 30:
            return "onchain_first", lead_time
        if lead_time < -30:
            return "cex_first", lead_time
        return "synchronized", lead_time
    return "single_source", None


def _format_dt(value: datetime | None) -> str:
    return value.isoformat() if value else ""
