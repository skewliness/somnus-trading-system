from datetime import timezone, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_system.bb_structure import compute_bb_structure, compute_williams_r, williams_zone
from trading_system.binance import KlineCache, _select_momo_symbols
from trading_system.candidates import build_candidate_basket
from trading_system.models import MarketSnapshot, SentimentSignal


class _SentimentBook:
    def __init__(self, signals: dict[str, SentimentSignal]) -> None:
        self._signals = signals

    def lookup(self, symbol: str) -> SentimentSignal | None:
        return self._signals.get(symbol.upper())


def _klines(count: int, start: float = 100.0, step: float = 0.1) -> list[list]:
    rows = []
    for idx in range(count):
        close = start + idx * step
        rows.append([idx, close - 0.02, close + 0.1, close - 0.1, close, 1000])
    return rows


def test_momo_structure_computes_extended_bands_and_williams_r():
    klines_15m = _klines(1500)
    klines_4h = _klines(1500)
    klines_1m = _klines(1000)
    klines_1d = _klines(400)
    mark_price = float(klines_15m[-1][4])

    signal = compute_bb_structure(
        klines_15m,
        klines_4h,
        mark_price,
        klines_1m=klines_1m,
        klines_1d=klines_1d,
    )

    assert signal.momo_15m_bars == 1500
    assert signal.momo_1m_bars == 1000
    assert signal.momo_4h_bars == 1500
    assert signal.momo_1d_bars == 400
    assert signal.bb120_upper_15m > 0
    assert signal.bb300_upper_15m > 0
    assert signal.bb672_upper_15m > 0
    assert signal.bb1440_upper_15m > 0
    assert signal.bb1920_upper_15m == 0.0
    assert signal.bb105_upper_4h > 0
    assert signal.bb1440_upper_4h > 0
    assert signal.bb105_upper_1m > 0
    assert signal.bb672_upper_1m > 0
    assert signal.bb1920_upper_1m == 0.0
    assert signal.daily_ma7 > 0
    assert signal.daily_ma20 > 0
    assert signal.daily_ema21 > 0
    assert signal.daily_bb365_upper > 0
    assert signal.williams_r_15m_zone == "overbought"
    assert signal.williams_r_15m_120_zone == "overbought"
    assert signal.williams_r_1m_zone == "overbought"


def test_williams_r_levels():
    assert williams_zone(-10.0) == "overbought"
    assert williams_zone(-35.0) == "upper_mid"
    assert williams_zone(-65.0) == "lower_mid"
    assert williams_zone(-90.0) == "oversold"
    assert compute_williams_r([]) == -50.0


def test_candidate_basket_exposes_momo_structure_block():
    snapshot = MarketSnapshot(
        symbol="AAAUSDT",
        mark_price=1.0,
        score=88.0,
        sentiment_mentions=12,
        sentiment_source="square_feed",
        bb_price_position_120=0.7,
        bb_price_position_300=0.6,
        bb_price_position_672_15m=0.58,
        bb_price_position_105_4h=0.55,
        bb_price_position_105_1m=0.52,
        williams_r_15m=-18.0,
        williams_r_15m_zone="overbought",
        williams_r_15m_120=-62.0,
        williams_r_15m_120_zone="lower_mid",
    )

    row = build_candidate_basket([snapshot])[0]

    assert row["momo_structure"]["m15"]["bb120_position"] == 0.7
    assert row["momo_structure"]["m15"]["bb672_position"] == 0.58
    assert row["momo_structure"]["h4"]["bb105_position"] == 0.55
    assert row["momo_structure"]["m1"]["bb105_position"] == 0.52
    assert row["momo_structure"]["williams_r"]["15m_zone"] == "overbought"
    assert row["momo_structure"]["williams_r"]["15m_120_zone"] == "lower_mid"


def test_kline_cache_merges_and_sorts_rows():
    with TemporaryDirectory() as tmp:
        cache = KlineCache(Path(tmp))
        cache.save("AAAUSDT", "15m", [[30, 1, 1, 1, 1], [10, 1, 1, 1, 1], [30, 2, 2, 2, 2]])

        rows = cache.load("AAAUSDT", "15m")

    assert [row[0] for row in rows] == [10, 30]
    assert rows[-1][4] == 2


def test_momo_prefilter_prefers_hot_sentiment_symbols():
    ranked = [
        {"symbol": "AAAUSDT", "priceChangePercent": "5", "quoteVolume": "30000000", "lastPrice": "1"},
        {"symbol": "BBBUSDT", "priceChangePercent": "4", "quoteVolume": "30000000", "lastPrice": "1"},
    ]
    sentiment = _SentimentBook({
        "BBBUSDT": SentimentSignal(
            symbol="BBBUSDT",
            hotness_score=95,
            mention_count=80,
            source="manual",
            updated_at=datetime.now(timezone.utc),
        )
    })

    selected = _select_momo_symbols(
        ranked,
        sentiment=sentiment,
        mark_price_map={"AAAUSDT": 1.0, "BBBUSDT": 1.0},
        funding_map={"AAAUSDT": 0.0, "BBBUSDT": 0.0},
        max_symbols=1,
    )

    assert selected == {"BBBUSDT"}
