import csv
from datetime import datetime, timezone
from pathlib import Path

from trading_system.bridges import BridgeLoadResult, ManualSentimentBridge
from trading_system.models import SentimentSignal
from trading_system.sentiment import SentimentBook
from trading_system.sentiment_aggregator import aggregate_sentiment


def _signal(symbol: str, score: float, source: str, ts: str) -> SentimentSignal:
    return SentimentSignal(
        symbol=symbol,
        hotness_score=score,
        mention_count=1,
        source=source,
        updated_at=datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
    )


def test_aggregate_sentiment_marks_onchain_first_sequence(tmp_path: Path) -> None:
    output = tmp_path / "merged.csv"
    result = aggregate_sentiment(
        [
            BridgeLoadResult("skill_bridge_directory", [_signal("DOGEUSDT", 82.0, "gmgn-wallet-distill", "2026-04-24T00:00:00")], {}),
            BridgeLoadResult("binance_top_movers", [_signal("DOGEUSDT", 76.0, "binance_top_movers", "2026-04-24T01:05:00")], {}),
        ],
        output,
    )

    assert result.signals[0].signal_sequence == "onchain_first"
    assert result.signals[0].lead_time_minutes == 65.0
    assert result.signals[0].earliest_strong_source == "gmgn-wallet-distill"

    rows = list(csv.DictReader(output.open("r", encoding="utf-8")))
    assert rows[0]["earliest_strong_source"] == "gmgn-wallet-distill"
    assert rows[0]["okx_confirmed_at"] == "2026-04-24T00:00:00+00:00"
    assert rows[0]["binance_confirmed_at"] == "2026-04-24T01:05:00+00:00"
    assert rows[0]["signal_sequence"] == "onchain_first"
    assert rows[0]["lead_time_minutes"] == "65.0"


def test_sentiment_csv_readers_preserve_sequence_fields(tmp_path: Path) -> None:
    path = tmp_path / "sentiment.csv"
    path.write_text(
        "symbol,hotness_score,mention_count,source,updated_at,notes,earliest_strong_source,earliest_strong_timestamp,okx_confirmed_at,binance_confirmed_at,signal_sequence,lead_time_minutes\n"
        "DOGEUSDT,79,2,binance_top_movers,2026-04-24T02:00:00+00:00,,gmgn-wallet-distill,2026-04-24T00:00:00+00:00,2026-04-24T00:00:00+00:00,2026-04-24T01:05:00+00:00,onchain_first,65.0\n",
        encoding="utf-8",
    )

    bridge_record = ManualSentimentBridge().load(path).records[0]
    book_record = SentimentBook.from_csv(path).lookup("DOGEUSDT")

    assert bridge_record.signal_sequence == "onchain_first"
    assert bridge_record.lead_time_minutes == 65.0
    assert book_record is not None
    assert book_record.okx_confirmed_at is not None
    assert book_record.binance_confirmed_at is not None
    assert book_record.signal_sequence == "onchain_first"
