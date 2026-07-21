from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from trading_system.decision_replay import ReplayDecisionAdapter
from trading_system.trading_context import TradingContext
from trading_system.v6_execution import V6ExecutionEngine, V6OrderIntent, V6Signal


def make_signal() -> V6Signal:
    indicators = {"ma20_15m": 100.0}
    levels = {"structure_entry_scan": {"long_candidate_count": 1}}
    signal = V6Signal(
        symbol="BTCUSDT",
        computed_at="2026-07-21T00:00:00+00:00",
        price=101.5,
        side_bias="long",
        long_probability=0.7,
        short_probability=0.2,
        wait_probability=0.1,
        model_quality_score=0.8,
        reasons=[],
        h4={"structure_state": "bullish"},
        fifteen_m={},
        one_m={},
        mid_retest={},
        levels=levels,
        indicators=indicators,
    )
    return signal


def test_context_is_frozen_and_preserves_input_identity() -> None:
    signal = make_signal()
    config = object()
    position = {"symbol": "BTCUSDT", "side": "long"}

    context = TradingContext.from_signal(signal, config=config, position=position)

    assert context.symbol == signal.symbol
    assert context.price == signal.price
    assert context.market_regime == signal.side_bias
    assert context.signal is signal
    assert context.position is position
    assert context.indicators is signal.indicators
    assert context.features is signal.levels
    assert context.config is config
    with pytest.raises(FrozenInstanceError):
        context.price = 102.0


def test_context_with_position_keeps_all_market_inputs_unchanged() -> None:
    signal = make_signal()
    context = TradingContext.from_signal(signal, config=object())
    position = {"side": "short"}

    positioned = context.with_position(position)

    assert positioned is not context
    assert positioned.position is position
    assert positioned.signal is context.signal
    assert positioned.indicators is context.indicators
    assert positioned.features is context.features
    assert positioned.config is context.config


def test_engine_exit_boundary_passes_trading_context_when_supported() -> None:
    signal = make_signal()
    position = {"symbol": "BTCUSDT", "side": "long"}
    sentinel = object()
    captured = {}

    class Recorder:
        @staticmethod
        def evaluate_context(engine, context, *, state=None):
            captured["context"] = context
            captured["state"] = state
            return sentinel

    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = object()
    engine.exit_manager = Recorder()

    result = engine.exit_intent_if_needed(signal, position, state={"version": 1})

    assert result is sentinel
    assert isinstance(captured["context"], TradingContext)
    assert captured["context"].signal is signal
    assert captured["context"].position is position
    assert captured["state"] == {"version": 1}


def test_engine_entry_boundary_passes_context_to_strategy_core() -> None:
    signal = make_signal()
    sentinel = object()
    captured = {}
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = object()
    engine.refresh_cross_timeframe_macro_observation = lambda signal, state: None
    engine.find_positions = lambda state, symbol: []

    def capture(context, state, positions):
        captured["context"] = context
        captured["state"] = state
        captured["positions"] = positions
        return sentinel

    engine._entry_intent_for_signal = capture

    result = engine.intent_for_signal(signal, {"positions": []})

    assert result is sentinel
    assert isinstance(captured["context"], TradingContext)
    assert captured["context"].signal is signal
    assert captured["positions"] == []


def test_legacy_signal_input_is_normalized_without_copying() -> None:
    signal = make_signal()
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = SimpleNamespace()

    context = engine.trading_context(signal)
    same_context = engine.trading_context(context)

    assert same_context is context
    assert context.signal is signal


def make_decision_engine() -> V6ExecutionEngine:
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = SimpleNamespace()
    engine.refresh_cross_timeframe_macro_observation = lambda signal, state: None
    engine.find_positions = lambda state, symbol: []

    def deterministic_entry(context, state, positions):
        signal = context.signal
        return V6OrderIntent(
            action="open_long",
            symbol=context.symbol,
            side="BUY",
            quantity_hint_usdt=100.0,
            leverage=8,
            reduce_only=False,
            entry_price=context.price,
            stop_loss_price=99.0,
            take_profit_price=105.0,
            long_probability=signal.long_probability,
            short_probability=signal.short_probability,
            wait_probability=signal.wait_probability,
            rationale="shared-context-test",
            generated_at=signal.computed_at,
        )

    engine._entry_intent_for_signal = deterministic_entry
    return engine


def test_live_dry_run_and_replay_use_identical_context_decision_path() -> None:
    signal = make_signal()
    state = {"positions": []}
    engine = make_decision_engine()
    context = engine.trading_context(signal)

    live_style = [intent.to_dict() for intent in engine.intents_for_signal(signal, state)]
    dry_run_style = [intent.to_dict() for intent in engine.intents_for_context(context, state)]
    replay_style = [intent.to_dict() for intent in ReplayDecisionAdapter(engine).decide(context, state)]

    assert live_style == dry_run_style == replay_style


def test_legacy_single_intent_entrypoint_delegates_to_context_path() -> None:
    signal = make_signal()
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = SimpleNamespace()
    captured = {}
    sentinel = object()

    def decide(context, state):
        captured["context"] = context
        captured["state"] = state
        return sentinel

    engine.intent_for_context = decide
    state = {"positions": []}

    assert engine.intent_for_signal(signal, state) is sentinel
    assert isinstance(captured["context"], TradingContext)
    assert captured["context"].signal is signal
    assert captured["state"] is state


def test_exit_boundary_accepts_prebuilt_context_without_rebuilding() -> None:
    signal = make_signal()
    position = {"symbol": "BTCUSDT", "side": "long"}
    context = TradingContext.from_signal(signal, config=object(), position=position)
    captured = {}
    sentinel = object()

    class Recorder:
        @staticmethod
        def evaluate_context(engine, received, *, state=None):
            captured["context"] = received
            return sentinel

    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.config = context.config
    engine.exit_manager = Recorder()

    assert engine.exit_intent_if_needed(context, state={}) is sentinel
    assert captured["context"] is context
