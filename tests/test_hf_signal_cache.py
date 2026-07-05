"""Tests for HF signal cache loading, aggregation, and entry blocking."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from trading_system.hf_signal_cache import load_hf_symbol_quality_cache, _aggregate_per_symbol


def _make_entries_file(entries: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for e in entries:
        tmp.write(json.dumps(e) + "\n")
    tmp.close()
    return Path(tmp.name)


class TestAggregatePerSymbol:
    def test_basic_aggregation(self):
        entries = [
            {"symbol": "BTCUSDT", "is_profitable_3bar": True, "mfe_12bar": 0.02, "risk_reward_12bar": 1.5, "return_3bar": 0.01, "triggered_signals": ["momentum"]},
            {"symbol": "BTCUSDT", "is_profitable_3bar": False, "mfe_12bar": 0.005, "risk_reward_12bar": 0.3, "return_3bar": -0.005, "triggered_signals": ["oi_surge"]},
            {"symbol": "BTCUSDT", "is_profitable_3bar": True, "mfe_12bar": 0.015, "risk_reward_12bar": 1.2, "return_3bar": 0.008, "triggered_signals": ["momentum"]},
        ]
        result = _aggregate_per_symbol(entries)
        assert "BTCUSDT" in result
        btc = result["BTCUSDT"]
        assert btc["hf_signal_count_total"] == 3
        assert abs(btc["hf_win_rate_3bar"] - 2 / 3) < 0.01
        assert btc["hf_avg_mfe_12bar"] > 0
        assert btc["hf_avg_risk_reward"] > 0
        assert btc["hf_best_signal_win_rate"] == 1.0  # momentum: 2/2

    def test_empty_entries(self):
        assert _aggregate_per_symbol([]) == {}

    def test_multiple_symbols(self):
        entries = [
            {"symbol": "BTCUSDT", "is_profitable_3bar": True, "mfe_12bar": 0.01, "risk_reward_12bar": 1.0, "return_3bar": 0.005, "triggered_signals": ["momentum"]},
            {"symbol": "ETHUSDT", "is_profitable_3bar": False, "mfe_12bar": 0.002, "risk_reward_12bar": 0.2, "return_3bar": -0.003, "triggered_signals": ["volume_spike"]},
        ]
        result = _aggregate_per_symbol(entries)
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result
        assert result["BTCUSDT"]["hf_win_rate_3bar"] == 1.0
        assert result["ETHUSDT"]["hf_win_rate_3bar"] == 0.0


class TestLoadCache:
    def test_loads_from_file(self):
        entries = [
            {"symbol": "SOLUSDT", "is_profitable_3bar": True, "mfe_12bar": 0.03, "risk_reward_12bar": 2.0, "return_3bar": 0.02, "triggered_signals": ["momentum"]},
        ]
        path = _make_entries_file(entries)
        result = load_hf_symbol_quality_cache(path, report_file=None, ttl_seconds=60)
        assert "SOLUSDT" in result
        assert result["SOLUSDT"]["hf_signal_count_total"] == 1

    def test_missing_file_returns_empty(self):
        result = load_hf_symbol_quality_cache(Path("/nonexistent.jsonl"), report_file=None)
        assert result == {}


class TestHfQualityBlocked:
    """Test the _is_hf_quality_blocked logic via PaperExecutionEngine."""

    def test_blocked_when_low_win_rate_and_enough_samples(self):
        from unittest.mock import MagicMock
        from trading_system.execution import PaperExecutionEngine
        from trading_system.config import StrategyConfig

        config = StrategyConfig()
        config.hf_entry_block_max_win_rate = 0.40
        config.hf_entry_block_min_samples = 20

        engine = PaperExecutionEngine.__new__(PaperExecutionEngine)
        engine.config = config
        engine._hf_quality_lookup = {
            "BADUSDT": {"hf_signal_count_total": 30, "hf_win_rate_3bar": 0.35},
        }
        assert engine._is_hf_quality_blocked("BADUSDT") is True

    def test_not_blocked_when_insufficient_samples(self):
        from trading_system.execution import PaperExecutionEngine
        from trading_system.config import StrategyConfig

        config = StrategyConfig()
        config.hf_entry_block_max_win_rate = 0.40
        config.hf_entry_block_min_samples = 20

        engine = PaperExecutionEngine.__new__(PaperExecutionEngine)
        engine.config = config
        engine._hf_quality_lookup = {
            "NEWUSDT": {"hf_signal_count_total": 5, "hf_win_rate_3bar": 0.20},
        }
        assert engine._is_hf_quality_blocked("NEWUSDT") is False

    def test_not_blocked_when_good_win_rate(self):
        from trading_system.execution import PaperExecutionEngine
        from trading_system.config import StrategyConfig

        config = StrategyConfig()
        config.hf_entry_block_max_win_rate = 0.40
        config.hf_entry_block_min_samples = 20

        engine = PaperExecutionEngine.__new__(PaperExecutionEngine)
        engine.config = config
        engine._hf_quality_lookup = {
            "GOODUSDT": {"hf_signal_count_total": 50, "hf_win_rate_3bar": 0.65},
        }
        assert engine._is_hf_quality_blocked("GOODUSDT") is False

    def test_not_blocked_when_symbol_missing(self):
        from trading_system.execution import PaperExecutionEngine
        from trading_system.config import StrategyConfig

        config = StrategyConfig()
        config.hf_entry_block_max_win_rate = 0.40
        config.hf_entry_block_min_samples = 20

        engine = PaperExecutionEngine.__new__(PaperExecutionEngine)
        engine.config = config
        engine._hf_quality_lookup = {}
        assert engine._is_hf_quality_blocked("UNKNOWNUSDT") is False

    def test_directional_block_uses_side_data(self):
        from trading_system.execution import PaperExecutionEngine
        from trading_system.config import StrategyConfig

        config = StrategyConfig()
        config.hf_entry_block_max_win_rate = 0.40
        config.hf_entry_block_min_samples = 20

        engine = PaperExecutionEngine.__new__(PaperExecutionEngine)
        engine.config = config
        engine._hf_quality_lookup = {
            "MIXUSDT": {
                "hf_signal_count_total": 60,
                "hf_win_rate_3bar": 0.50,
                "by_side": {
                    "long": {"hf_signal_count_total": 30, "hf_win_rate_3bar": 0.35},
                    "short": {"hf_signal_count_total": 30, "hf_win_rate_3bar": 0.65},
                },
            },
        }
        # Long side is bad → blocked
        assert engine._is_hf_quality_blocked("MIXUSDT", side="long") is True
        # Short side is good → not blocked
        assert engine._is_hf_quality_blocked("MIXUSDT", side="short") is False

    def test_aggregation_includes_by_side(self):
        entries = [
            {"symbol": "XUSDT", "side": "long", "is_profitable_3bar": True, "mfe_12bar": 0.02, "risk_reward_12bar": 1.5, "return_3bar": 0.01, "triggered_signals": ["momentum"]},
            {"symbol": "XUSDT", "side": "short", "is_profitable_3bar": False, "mfe_12bar": 0.005, "risk_reward_12bar": 0.3, "return_3bar": -0.005, "triggered_signals": ["oi_surge"]},
        ]
        result = _aggregate_per_symbol(entries)
        assert "XUSDT" in result
        assert "by_side" in result["XUSDT"]
        assert result["XUSDT"]["by_side"]["long"]["hf_win_rate_3bar"] == 1.0
        assert result["XUSDT"]["by_side"]["short"]["hf_win_rate_3bar"] == 0.0


class TestInferEntrySide:
    """Test that _infer_entry_side correctly maps setup_type to side."""

    def test_long_setup_types(self):
        from trading_system.execution import PaperExecutionEngine
        from unittest.mock import MagicMock

        for st in ("trend", "oi_divergence_long_candidate", "base_breakout_confirmed", "mania_followthrough"):
            snap = MagicMock()
            snap.setup_type = st
            assert PaperExecutionEngine._infer_entry_side(snap) == "long"

    def test_short_setup_types(self):
        from trading_system.execution import PaperExecutionEngine
        from unittest.mock import MagicMock

        for st in ("squeeze_short_candidate", "base_breakdown_confirmed"):
            snap = MagicMock()
            snap.setup_type = st
            assert PaperExecutionEngine._infer_entry_side(snap) == "short"


class TestHfBlockDirectionalIntegration:
    """Integration: short candidate should NOT be blocked by bad long-side HF quality."""

    def test_short_candidate_not_blocked_by_long_hf(self):
        from trading_system.execution import PaperExecutionEngine
        from trading_system.config import StrategyConfig
        from unittest.mock import MagicMock

        config = StrategyConfig()
        config.hf_entry_block_max_win_rate = 0.40
        config.hf_entry_block_min_samples = 20

        engine = PaperExecutionEngine.__new__(PaperExecutionEngine)
        engine.config = config
        engine._hf_quality_lookup = {
            "BADUSDT": {
                "hf_signal_count_total": 60,
                "hf_win_rate_3bar": 0.38,
                "by_side": {
                    "long": {"hf_signal_count_total": 30, "hf_win_rate_3bar": 0.30},
                    "short": {"hf_signal_count_total": 30, "hf_win_rate_3bar": 0.60},
                },
            },
        }

        snap = MagicMock()
        snap.symbol = "BADUSDT"
        snap.setup_type = "squeeze_short_candidate"

        side = engine._infer_entry_side(snap)
        assert side == "short"
        assert engine._is_hf_quality_blocked("BADUSDT", side=side) is False

    def test_long_candidate_blocked_by_long_hf(self):
        from trading_system.execution import PaperExecutionEngine
        from trading_system.config import StrategyConfig
        from unittest.mock import MagicMock

        config = StrategyConfig()
        config.hf_entry_block_max_win_rate = 0.40
        config.hf_entry_block_min_samples = 20

        engine = PaperExecutionEngine.__new__(PaperExecutionEngine)
        engine.config = config
        engine._hf_quality_lookup = {
            "BADUSDT": {
                "hf_signal_count_total": 60,
                "hf_win_rate_3bar": 0.38,
                "by_side": {
                    "long": {"hf_signal_count_total": 30, "hf_win_rate_3bar": 0.30},
                    "short": {"hf_signal_count_total": 30, "hf_win_rate_3bar": 0.60},
                },
            },
        }

        snap = MagicMock()
        snap.symbol = "BADUSDT"
        snap.setup_type = "trend"

        side = engine._infer_entry_side(snap)
        assert side == "long"
        assert engine._is_hf_quality_blocked("BADUSDT", side=side) is True


class TestMissingSideFallback:
    """When by_side exists but lacks the requested side, should NOT block and features should be 0."""

    def test_long_not_blocked_when_only_short_data(self):
        from trading_system.execution import PaperExecutionEngine
        from trading_system.config import StrategyConfig

        config = StrategyConfig()
        config.hf_entry_block_max_win_rate = 0.40
        config.hf_entry_block_min_samples = 20

        engine = PaperExecutionEngine.__new__(PaperExecutionEngine)
        engine.config = config
        engine._hf_quality_lookup = {
            "ONLYSHORTUSDT": {
                "hf_signal_count_total": 50,
                "hf_win_rate_3bar": 0.30,
                "by_side": {
                    "short": {"hf_signal_count_total": 50, "hf_win_rate_3bar": 0.30},
                },
            },
        }
        # Long side missing from by_side → should NOT block
        assert engine._is_hf_quality_blocked("ONLYSHORTUSDT", side="long") is False

    def test_extract_hf_quality_returns_zeros_when_side_missing(self):
        from trading_system.self_train_dataset import _extract_hf_quality, _HF_QUALITY_KEYS

        source = {
            "hf_signal_count_total": 50,
            "hf_win_rate_3bar": 0.30,
            "by_side": {
                "short": {"hf_signal_count_total": 50, "hf_win_rate_3bar": 0.30},
            },
        }
        result = _extract_hf_quality(source, side="long")
        assert all(result[k] == 0.0 for k in _HF_QUALITY_KEYS)

    def test_extract_hf_quality_uses_overall_when_no_by_side(self):
        from trading_system.self_train_dataset import _extract_hf_quality

        source = {
            "hf_signal_count_total": 40,
            "hf_win_rate_3bar": 0.55,
        }
        result = _extract_hf_quality(source, side="long")
        assert result["hf_win_rate_3bar"] == 0.55
        assert result["hf_signal_count_total"] == 40.0

    def test_directional_hf_quality_empty_when_side_missing(self):
        from trading_system.execution import PaperExecutionEngine
        from trading_system.config import StrategyConfig
        from unittest.mock import MagicMock

        config = StrategyConfig()
        engine = PaperExecutionEngine.__new__(PaperExecutionEngine)
        engine.config = config
        engine._hf_quality_lookup = {
            "XUSDT": {
                "hf_signal_count_total": 40,
                "hf_win_rate_3bar": 0.25,
                "by_side": {
                    "short": {"hf_signal_count_total": 40, "hf_win_rate_3bar": 0.25},
                },
            },
        }

        snap = MagicMock()
        snap.symbol = "XUSDT"
        snap.setup_type = "trend"  # long

        result = engine._directional_hf_quality(snap)
        assert result["hf_entry_side"] == "long"
        # Should not contain short-side metrics
        assert result.get("hf_win_rate_3bar") is None or result.get("hf_signal_count_total") is None
