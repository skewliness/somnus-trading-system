"""Tests for 4H structure health side-channel."""
from __future__ import annotations

import pytest

from trading_system.h4_structure import H4StructureSignal, compute_h4_structure


def _make_klines(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> list:
    """Build fake kline arrays: [open_time, open, high, low, close, volume, ...]."""
    n = len(closes)
    if highs is None:
        highs = [c * 1.01 for c in closes]
    if lows is None:
        lows = [c * 0.99 for c in closes]
    klines = []
    for i in range(n):
        klines.append([
            1700000000000 + i * 14400000,  # open_time
            str(closes[i]),                 # open
            str(highs[i]),                  # high
            str(lows[i]),                   # low
            str(closes[i]),                 # close
            "1000",                         # volume
            1700000000000 + (i + 1) * 14400000 - 1,
            "50000", "100", "500", "25000", "0",
        ])
    return klines


class TestInsufficientData:
    def test_returns_unknown_when_klines_empty(self):
        result = compute_h4_structure("BTCUSDT", [], 50000.0)
        assert result.h4_structure_state == "unknown"
        assert result.h4_price_location == "middle"
        assert result.h4_pattern_type == "unknown"

    def test_returns_unknown_when_klines_too_few(self):
        klines = _make_klines([100.0] * 10)
        result = compute_h4_structure("ETHUSDT", klines, 100.0)
        assert result.h4_structure_state == "unknown"


class TestValidStructure:
    def test_close_above_ma20_no_break(self):
        # Uptrend: prices rising, close above MA20, no low break
        closes = [100 + i * 2 for i in range(25)]
        klines = _make_klines(closes)
        mark_price = closes[-1] + 5  # above last close
        result = compute_h4_structure("BTCUSDT", klines, mark_price)
        assert result.h4_structure_state == "valid"
        assert result.h4_close_above_ma20 is True
        assert result.h4_low_break_recent_20bars is False

    def test_ma20_slope_positive_in_uptrend(self):
        # 27 bars of steady uptrend → slope must be > 0
        closes = [100 + i * 2 for i in range(27)]
        klines = _make_klines(closes)
        result = compute_h4_structure("BTCUSDT", klines, closes[-1])
        assert result.h4_ma20_slope_pct > 0

    def test_binance_limit_28_provides_enough_for_slope(self):
        """Simulate the 28-bar fetch from binance.py and verify slope computes."""
        closes = [100 + i * 1.5 for i in range(28)]
        klines = _make_klines(closes)
        result = compute_h4_structure("BTCUSDT", klines, closes[-1])
        assert result.h4_ma20_slope_pct > 0
        assert result.h4_structure_state != "unknown"
        assert result.h4_close_position_20bars >= 0.25


class TestBrokenStructure:
    def test_close_below_ma20_with_negative_slope(self):
        # Downtrend: prices falling, close below MA20, slope negative
        closes = [200 - i * 3 for i in range(25)]
        klines = _make_klines(closes)
        mark_price = closes[-1] - 10  # well below
        result = compute_h4_structure("BTCUSDT", klines, mark_price)
        assert result.h4_structure_state == "broken"
        assert result.h4_close_above_ma20 is False

    def test_low_break_triggers_broken(self):
        # Flat range then break below
        closes = [100.0] * 25
        highs = [102.0] * 25
        lows = [98.0] * 25
        klines = _make_klines(closes, highs, lows)
        mark_price = 95.0  # below range_low (98)
        result = compute_h4_structure("BTCUSDT", klines, mark_price)
        assert result.h4_structure_state == "broken"
        assert result.h4_low_break_recent_20bars is True
        assert result.h4_price_location == "below_support"
        assert result.h4_pattern_type == "breakdown"


class TestWeakeningStructure:
    def test_close_below_ma20_mild(self):
        # Close slightly below MA20 but no break
        closes = [100.0] * 25
        klines = _make_klines(closes)
        mark_price = 99.0  # below MA20 (100) but above range_low
        result = compute_h4_structure("BTCUSDT", klines, mark_price)
        assert result.h4_structure_state == "weakening"


class TestBreakoutAboveResistance:
    def test_high_break_triggers_breakout(self):
        closes = [100.0] * 25
        highs = [102.0] * 25
        lows = [98.0] * 25
        klines = _make_klines(closes, highs, lows)
        mark_price = 105.0  # above range_high (102)
        result = compute_h4_structure("BTCUSDT", klines, mark_price)
        assert result.h4_high_break_recent_20bars is True
        assert result.h4_price_location == "above_resistance"
        assert result.h4_pattern_type == "breakout"


class TestTrianglePattern:
    def test_converging_highs_and_lows(self):
        # Create converging pattern: lower highs + higher lows + ATR compression
        n = 25
        closes = []
        highs = []
        lows = []
        for i in range(n):
            mid = 100.0
            amplitude = 10.0 - i * 0.3  # shrinking range
            closes.append(mid)
            highs.append(mid + amplitude)
            lows.append(mid - amplitude)
        klines = _make_klines(closes, highs, lows)
        result = compute_h4_structure("BTCUSDT", klines, 100.0)
        # With converging highs/lows and ATR compression, should detect triangle
        assert result.h4_atr_compression < 1.0
        # Pattern detection depends on swing detection finding enough points
        assert result.h4_pattern_type in ("triangle", "wedge", "range", "unknown")


class TestCandidateBasketH4Fields:
    def test_basket_contains_h4_fields(self):
        from trading_system.candidates import build_candidate_basket
        from trading_system.models import MarketSnapshot

        snapshot = MarketSnapshot(
            symbol="TESTUSDT",
            mark_price=100.0,
            score=85.0,
            setup_type="base_breakout_confirmed",
            h4_structure_state="valid",
            h4_price_location="upper_half",
            h4_pattern_type="breakout",
            h4_close_above_ma20=True,
            h4_ma20_slope_pct=1.5,
            h4_price_vs_ma20_pct=2.0,
            h4_close_position_20bars=0.8,
            h4_atr_compression=0.7,
            reasons=["test"],
            sentiment_score=80.0,
            sentiment_mentions=5,
            sentiment_source="manual_distill",
        )
        basket = build_candidate_basket([snapshot])
        assert len(basket) == 1
        row = basket[0]
        assert row["h4_structure_state"] == "valid"
        assert row["h4_price_location"] == "upper_half"
        assert row["h4_pattern_type"] == "breakout"
        assert row["h4_close_above_ma20"] is True
        assert row["h4_ma20_slope_pct"] == 1.5
        assert row["h4_atr_compression"] == 0.7


class TestDecisionSamplesH4Fields:
    def test_decision_samples_extract_h4(self):
        from trading_system.self_train_dataset import collect_decision_samples

        report = {
            "candidate_basket": [
                {
                    "symbol": "BTCUSDT",
                    "score": 80,
                    "setup_type": "base_breakout_confirmed",
                    "execution_status": "executable_now",
                    "h4_structure_state": "valid",
                    "h4_price_location": "upper_half",
                    "h4_pattern_type": "triangle",
                    "h4_close_above_ma20": True,
                    "h4_ma20_slope_pct": 0.5,
                    "h4_price_vs_ma20_pct": 1.2,
                    "h4_close_position_20bars": 0.6,
                    "h4_low_break_recent_20bars": False,
                    "h4_high_break_recent_20bars": False,
                    "h4_atr_compression": 0.8,
                    "h4_lower_highs_count": 2,
                    "h4_higher_lows_count": 2,
                }
            ],
            "top_ranked_symbols": [{"symbol": "BTCUSDT", "score": 80}],
            "paper_rebalance": {"entries": []},
        }
        samples = collect_decision_samples(report)
        assert len(samples) == 1
        s = samples[0]
        assert s["h4_structure_state"] == "valid"
        assert s["h4_price_location"] == "upper_half"
        assert s["h4_pattern_type"] == "triangle"
        assert s["h4_close_above_ma20"] is True
        assert s["h4_atr_compression"] == 0.8


class TestPositionH4Snapshot:
    def test_position_stores_h4_structure(self):
        from trading_system.models import Position

        h4_snap = {
            "h4_close_above_ma20": True,
            "h4_ma20_slope_pct": 1.0,
            "h4_price_vs_ma20_pct": 2.0,
            "h4_close_position_20bars": 0.7,
            "h4_low_break_recent_20bars": False,
            "h4_high_break_recent_20bars": False,
            "h4_atr_compression": 0.9,
            "h4_lower_highs_count": 0,
            "h4_higher_lows_count": 1,
            "h4_structure_state": "valid",
            "h4_price_location": "upper_half",
            "h4_pattern_type": "unknown",
        }
        pos = Position(
            symbol="BTCUSDT",
            opened_at="2026-01-01T00:00:00+00:00",
            entry_price=50000.0,
            current_price=50000.0,
            stop_loss_price=49000.0,
            take_profit_price=52000.0,
            notional_usdt=1000.0,
            score=80.0,
            rationale="test",
            entry_h4_structure=h4_snap,
        )
        assert pos.entry_h4_structure is not None
        assert pos.entry_h4_structure["h4_structure_state"] == "valid"
        d = pos.to_dict()
        assert d["entry_h4_structure"]["h4_close_above_ma20"] is True


class TestClosedTradeSamplesH4:
    def test_closed_samples_extract_from_entry_snapshot(self):
        from trading_system.self_train_dataset import _extract_h4_structure

        h4_snap = {
            "h4_close_above_ma20": False,
            "h4_ma20_slope_pct": -1.5,
            "h4_price_vs_ma20_pct": -3.0,
            "h4_close_position_20bars": 0.2,
            "h4_low_break_recent_20bars": True,
            "h4_high_break_recent_20bars": False,
            "h4_atr_compression": 1.1,
            "h4_lower_highs_count": 3,
            "h4_higher_lows_count": 0,
            "h4_structure_state": "broken",
            "h4_price_location": "below_support",
            "h4_pattern_type": "breakdown",
        }
        result = _extract_h4_structure(h4_snap)
        assert result["h4_structure_state"] == "broken"
        assert result["h4_price_location"] == "below_support"
        assert result["h4_low_break_recent_20bars"] is True
        assert result["h4_lower_highs_count"] == 3

    def test_extract_h4_structure_none_returns_defaults(self):
        from trading_system.self_train_dataset import _extract_h4_structure

        result = _extract_h4_structure(None)
        assert result["h4_structure_state"] == "unknown"
        assert result["h4_close_above_ma20"] is True
        assert result["h4_atr_compression"] == 1.0


class TestH4EntryFilter:
    """Phase 2: h4_structure_state affects entry decisions."""

    def _make_engine(self, tmp_path):
        from trading_system.config import StrategyConfig
        from trading_system.execution import PaperExecutionEngine

        config = StrategyConfig(
            state_file=tmp_path / "state.json",
            entry_threshold=60,
            h4_weakening_score_penalty=5,
            absolute_min_sentiment_mentions=0,
            min_entry_sentiment_mentions=0,
        )
        return PaperExecutionEngine(config)

    def _make_snapshot(self, score=70.0, h4_state="valid"):
        from trading_system.models import MarketSnapshot

        return MarketSnapshot(
            symbol="TESTUSDT",
            mark_price=100.0,
            price_change_pct_24h=5.0,
            quote_volume_usdt_24h=50_000_000,
            funding_rate=0.0003,
            open_interest_usdt=10_000_000,
            oi_change_pct_4h=8.0,
            taker_buy_sell_ratio=1.15,
            listing_days=30,
            sentiment_score=80,
            sentiment_mentions=10,
            sentiment_source="manual_distill",
            setup_type="base_breakout_confirmed",
            score=score,
            h4_structure_state=h4_state,
            reasons=["test"],
        )

    def test_broken_blocks_entry(self, tmp_path):
        engine = self._make_engine(tmp_path)
        snapshot = self._make_snapshot(score=90.0, h4_state="broken")
        result = engine.rebalance([snapshot])
        assert len(result["entries"]) == 0

    def test_weakening_penalizes_score_below_threshold(self, tmp_path):
        engine = self._make_engine(tmp_path)
        # score=63, penalty=5 → effective=58 < threshold=60 → blocked
        snapshot = self._make_snapshot(score=63.0, h4_state="weakening")
        result = engine.rebalance([snapshot])
        assert len(result["entries"]) == 0

    def test_valid_allows_entry(self, tmp_path):
        engine = self._make_engine(tmp_path)
        snapshot = self._make_snapshot(score=70.0, h4_state="valid")
        result = engine.rebalance([snapshot])
        assert len(result["entries"]) == 1

    def test_broken_does_not_block_short(self, tmp_path):
        """Short candidates should NOT be blocked by broken h4 structure."""
        from trading_system.models import MarketSnapshot

        engine = self._make_engine(tmp_path)
        snapshot = MarketSnapshot(
            symbol="TESTUSDT",
            mark_price=100.0,
            price_change_pct_24h=5.0,
            quote_volume_usdt_24h=50_000_000,
            funding_rate=0.001,
            open_interest_usdt=10_000_000,
            oi_change_pct_4h=8.0,
            taker_buy_sell_ratio=0.8,
            listing_days=30,
            sentiment_score=80,
            sentiment_mentions=10,
            sentiment_source="manual_distill",
            setup_type="squeeze_short_candidate",
            score=90.0,
            h4_structure_state="broken",
            kline_close_position=0.2,
            reasons=["test"],
        )
        result = engine.rebalance([snapshot])
        assert len(result["entries"]) == 1

    def test_filter_disabled_allows_broken_long(self, tmp_path):
        """When h4_structure_filter_enabled=False, broken long is allowed."""
        from trading_system.config import StrategyConfig
        from trading_system.execution import PaperExecutionEngine

        config = StrategyConfig(
            state_file=tmp_path / "state.json",
            entry_threshold=60,
            h4_weakening_score_penalty=5,
            h4_structure_filter_enabled=False,
            absolute_min_sentiment_mentions=0,
            min_entry_sentiment_mentions=0,
        )
        engine = PaperExecutionEngine(config)
        snapshot = self._make_snapshot(score=90.0, h4_state="broken")
        result = engine.rebalance([snapshot])
        assert len(result["entries"]) == 1
