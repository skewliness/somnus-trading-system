from __future__ import annotations

from types import SimpleNamespace

from trading_system.exit import ExitManager, ExitRuleFamily, ExitSignal, resolve_exit_signals
from trading_system.exit.manager import LONG_EXIT_RULES, SHORT_EXIT_RULES
from trading_system.exit.resolver import resolve_exit_rules
from trading_system.exit.rules import build_exit_signals
from trading_system.v6_execution import V6ExecutionConfig, V6ExecutionEngine, V6OrderIntent, V6Signal, utc_now_iso


def make_signal(price: float = 94.0) -> V6Signal:
    return V6Signal(
        symbol="BTCUSDT",
        computed_at="2026-07-21T00:00:00+00:00",
        price=price,
        side_bias="long",
        long_probability=0.7,
        short_probability=0.2,
        wait_probability=0.1,
        model_quality_score=0.8,
        reasons=[],
        h4={},
        fifteen_m={},
        one_m={},
        mid_retest={},
        levels={},
    )


class StopOnlyEngine:
    def __init__(self) -> None:
        self.config = V6ExecutionConfig(technical_entry_only_symbols=("BTCUSDT",))

    @staticmethod
    def position_max_hold_expired(position):
        return False

    @staticmethod
    def update_position_profit_tracker(signal, position):
        return {}

    @staticmethod
    def parse_utc_datetime(value):
        return None


def test_exit_manager_preserves_hard_stop_intent_contract() -> None:
    manager = ExitManager(intent_factory=V6OrderIntent, utc_now=utc_now_iso)
    position = {
        "symbol": "BTCUSDT",
        "side": "long",
        "notional_usdt": 300.0,
        "leverage": 8,
        "stop_loss_price": 95.0,
        "take_profit_price": 110.0,
        "risk_plan": {},
    }

    intent = manager.evaluate(StopOnlyEngine(), make_signal(), position)

    assert intent.action == "close_long"
    assert intent.side == "SELL"
    assert intent.reduce_only is True
    assert intent.close_fraction == 1.0
    assert intent.rationale == "long stop loss touched"
    assert manager.classify_intent(intent) == ExitRuleFamily.STOP_LOSS


def test_engine_public_exit_method_uses_exit_manager_boundary() -> None:
    sentinel = object()
    recorder = SimpleNamespace(evaluate=lambda engine, signal, position, state=None: sentinel)
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.exit_manager = recorder

    assert engine.exit_intent_if_needed(object(), {}, state={}) is sentinel


def test_rule_families_classify_without_changing_intent() -> None:
    manager = ExitManager(intent_factory=V6OrderIntent, utc_now=utc_now_iso)
    bb_intent = SimpleNamespace(rationale="staged long take profit touched", risk_plan={"partial_take_profit": {"type": "bb_take_profit"}})
    trend_intent = SimpleNamespace(rationale="max hold days reached", risk_plan={})

    assert manager.classify_intent(bb_intent) == ExitRuleFamily.BB_EXIT
    assert manager.classify_intent(trend_intent) == ExitRuleFamily.TREND_EXIT


def test_every_exit_rule_returns_structured_signal() -> None:
    intent = SimpleNamespace(rationale="long stop loss touched", risk_plan={})

    signals = build_exit_signals(intent)

    assert all(isinstance(signal, ExitSignal) for signal in signals)
    assert all(signal.reason == intent.rationale for signal in signals)
    assert {signal.family for signal in signals} == set(ExitRuleFamily)
    selected = resolve_exit_signals(signals)
    assert selected.triggered is True
    assert selected.family == ExitRuleFamily.STOP_LOSS
    assert selected.intent is intent


def test_resolver_preserves_existing_classification_priority_and_identity() -> None:
    intent = SimpleNamespace(
        rationale="profit guard take profit touched",
        risk_plan={"partial_take_profit": {"type": "bb_take_profit"}},
    )

    selected = resolve_exit_signals(build_exit_signals(intent))

    assert selected.family == ExitRuleFamily.PROFIT_GUARD
    assert selected.priority == 300
    assert selected.intent is intent


def test_resolver_rejects_an_empty_or_untriggered_signal_set() -> None:
    untriggered = ExitSignal(
        triggered=False,
        reason="",
        priority=1,
        family=ExitRuleFamily.OTHER,
        intent=None,
    )

    try:
        resolve_exit_signals([untriggered])
    except ValueError as exc:
        assert str(exc) == "at least one triggered exit signal is required"
    else:
        raise AssertionError("resolver accepted an untriggered signal set")


def test_rule_resolver_is_lazy_and_stops_after_first_trigger() -> None:
    calls: list[str] = []

    def miss(context):
        calls.append("miss")
        return ExitSignal(False, "", 1, ExitRuleFamily.OTHER)

    def hit(context):
        calls.append("hit")
        return ExitSignal(True, "selected", 1, ExitRuleFamily.TREND_EXIT)

    def must_not_run(context):
        calls.append("late")
        raise AssertionError("late rule was evaluated")

    selected = resolve_exit_rules(object(), (miss, hit, must_not_run))  # type: ignore[arg-type]

    assert selected is not None and selected.reason == "selected"
    assert calls == ["miss", "hit"]


def test_short_rule_order_matches_v6_baseline_precedence() -> None:
    assert [rule.__name__ for rule in SHORT_EXIT_RULES] == [
        "evaluate_max_hold",
        "evaluate_hard_stop",
        "evaluate_v5_soft_stop",
        "evaluate_daily_reversal",
        "evaluate_logic_weakening",
        "evaluate_smart_signal",
        "evaluate_v5_no_news",
        "evaluate_roi_take_profit",
        "evaluate_staged_take_profit",
        "evaluate_fixed_take_profit",
        "evaluate_opposite_probability",
    ]


def test_long_rule_order_matches_v6_baseline_precedence() -> None:
    assert [rule.__name__ for rule in LONG_EXIT_RULES] == [
        "evaluate_max_hold",
        "evaluate_hard_stop",
        "evaluate_v5_soft_stop",
        "evaluate_daily_reversal",
        "evaluate_v5_no_news",
        "evaluate_fifteen_m_giveback",
        "evaluate_h4_lower_reversal",
        "evaluate_logic_weakening",
        "evaluate_smart_signal",
        "evaluate_roi_take_profit",
        "evaluate_staged_take_profit",
        "evaluate_fixed_take_profit",
        "evaluate_opposite_probability",
    ]
