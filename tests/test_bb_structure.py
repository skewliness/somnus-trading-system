from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_system.bb_structure import (
    BBStructureSignal,
    compute_bb,
    compute_bb_structure,
    _classify_touch_zone,
    _price_position,
)
from trading_system.execution import PaperExecutionEngine
from trading_system.config import StrategyConfig
from trading_system.models import MarketSnapshot


def test_compute_bb_basic():
    closes = [100.0] * 96
    upper, mid, lower = compute_bb(closes, 96)
    assert mid == 100.0
    assert upper == 100.0
    assert lower == 100.0


def test_compute_bb_with_variance():
    closes = list(range(1, 97))
    upper, mid, lower = compute_bb(closes, 96)
    assert mid > 0
    assert upper > mid
    assert lower < mid


def test_compute_bb_insufficient_data():
    closes = [50.0, 60.0]
    upper, mid, lower = compute_bb(closes, 96)
    assert upper == mid == lower == 60.0


def test_price_position():
    assert _price_position(100, 200, 0) == 0.5
    assert _price_position(200, 200, 0) == 1.0
    assert _price_position(0, 200, 0) == 0.0
    assert _price_position(50, 50, 50) == 0.5


def test_classify_touch_zone_none():
    result = _classify_touch_zone(
        price=100.0,
        bb96_upper=110.0,
        bb96_lower=90.0,
        bb120_upper=120.0,
        bb120_lower=80.0,
        bb210_upper=130.0,
        bb210_lower=70.0,
        threshold_pct=0.3,
    )
    assert result == "none"


def test_classify_touch_zone_bb96_upper():
    result = _classify_touch_zone(
        price=110.0,
        bb96_upper=110.0,
        bb96_lower=90.0,
        bb120_upper=120.0,
        bb120_lower=80.0,
        bb210_upper=130.0,
        bb210_lower=70.0,
        threshold_pct=0.3,
    )
    assert result == "bb96_upper"


def test_classify_touch_zone_bb210_lower():
    result = _classify_touch_zone(
        price=70.0,
        bb96_upper=110.0,
        bb96_lower=90.0,
        bb120_upper=120.0,
        bb120_lower=80.0,
        bb210_upper=130.0,
        bb210_lower=70.0,
        threshold_pct=0.3,
    )
    assert result == "bb210_lower"


def test_classify_touch_zone_bb672_lower():
    result = _classify_touch_zone(
        price=70.0,
        bb96_upper=110.0,
        bb96_lower=90.0,
        bb120_upper=120.0,
        bb120_lower=80.0,
        bb210_upper=130.0,
        bb210_lower=60.0,
        bb672_upper=140.0,
        bb672_lower=70.0,
        threshold_pct=0.3,
    )
    assert result == "bb672_lower"


def test_classify_touch_zone_bb4460_15m_lower():
    result = _classify_touch_zone(
        price=55.0,
        bb96_upper=110.0,
        bb96_lower=90.0,
        bb120_upper=120.0,
        bb120_lower=80.0,
        bb210_upper=130.0,
        bb210_lower=70.0,
        bb4460_upper_15m=150.0,
        bb4460_lower_15m=55.0,
        threshold_pct=0.3,
    )
    assert result == "bb4460_lower_15m"


def test_compute_bb_structure_basic():
    klines_15m = [[0, 0, 0, 0, str(100 + i * 0.1)] for i in range(215)]
    klines_4h = [[0, 0, 0, 0, str(100 + i * 0.05)] for i in range(700)]
    signal = compute_bb_structure(klines_15m, klines_4h, 121.4, threshold_pct=0.3)
    assert isinstance(signal, BBStructureSignal)
    assert signal.bb96_mid_15m > 0
    assert signal.bb210_mid_15m > 0
    assert signal.bb672_mid_4h > 0
    assert 0 <= signal.bb_price_position_96 <= 1.5
    assert isinstance(signal.bb_touch_zone, str)


def test_compute_bb_structure_15m_4707_ready():
    klines_15m = [[0, 0, 0, 0, "100", "10"] for _ in range(4707)]
    klines_4h = [[0, 0, 0, 0, "100"] for _ in range(700)]
    signal = compute_bb_structure(klines_15m, klines_4h, 100.0, threshold_pct=0.3)
    assert signal.bb4707_mid_15m == 100.0
    assert signal.bb4707_lower_15m == 100.0
    assert signal.bb_price_position_4707_15m == 0.5
    assert signal.ma105_15m == 100.0
    assert signal.bb105_mid_15m == 100.0
    assert signal.volume_ma5_15m == 10.0



def test_compute_bb_structure_shared_periods_ready_on_15m_4h_1d():
    klines_15m = [[0, 0, 0, 0, "100", "10"] for _ in range(4707)]
    klines_4h = [[0, 0, 0, 0, "100", "20"] for _ in range(4707)]
    klines_1d = [[0, 0, 0, 0, "100", "30"] for _ in range(4707)]
    signal = compute_bb_structure(
        klines_15m,
        klines_4h,
        100.0,
        threshold_pct=0.3,
        klines_1d=klines_1d,
    )
    for timeframe in ("15m", "4h", "1d"):
        for period in (96, 105, 120, 210, 300, 672, 1440, 1920, 2880, 4707):
            assert getattr(signal, f"bb{period}_mid_{timeframe}") == 100.0
            assert getattr(signal, f"bb{period}_lower_{timeframe}") == 100.0
            assert getattr(signal, f"bb{period}_upper_{timeframe}") == 100.0
    assert signal.bb_price_position_4707_4h == 0.5
    assert signal.bb_price_position_4707_1d == 0.5
    assert signal.ma60_4h == 100.0
    assert signal.ma105_1d == 100.0
    assert signal.volume_ma10_4h == 20.0
    assert signal.volume_ma10_1d == 30.0

def test_update_bb_touch_counts():
    config = StrategyConfig()
    config.bb_touch_min_gap_minutes = 0
    engine = PaperExecutionEngine(config)
    snap = MarketSnapshot(symbol="BTCUSDT", mark_price=100.0, bb_touch_zone="bb96_upper")
    counts: dict = {}
    engine._update_bb_touch_counts(counts, [snap])
    assert counts["BTCUSDT:bb96_upper"]["count"] == 1
    engine._update_bb_touch_counts(counts, [snap])
    assert counts["BTCUSDT:bb96_upper"]["count"] == 2


def test_update_consecutive_entries():
    config = StrategyConfig()
    engine = PaperExecutionEngine(config)
    entries: dict = {}
    engine._update_consecutive_entries(entries, "BTCUSDT")
    assert entries["BTCUSDT"]["count"] == 1
    engine._update_consecutive_entries(entries, "BTCUSDT")
    assert entries["BTCUSDT"]["count"] == 2


def test_consecutive_entry_reset_after_timeout():
    config = StrategyConfig()
    config.consecutive_entry_reset_hours = 4
    engine = PaperExecutionEngine(config)
    entries: dict = {}
    engine._update_consecutive_entries(entries, "BTCUSDT")
    entries["BTCUSDT"]["last_entry_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=5)
    ).isoformat()
    engine._update_consecutive_entries(entries, "BTCUSDT")
    assert entries["BTCUSDT"]["count"] == 1
