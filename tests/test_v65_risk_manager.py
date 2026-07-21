from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading_system.risk import RiskAssessment, RiskManager, RiskRecommendation
from trading_system.v6_execution import V6ExecutionConfig, V6ExecutionEngine, V6OrderIntent


def make_manager() -> RiskManager:
    return RiskManager(utc_now=lambda: "2026-07-21T00:00:00+00:00")


def make_intent(**overrides) -> V6OrderIntent:
    values = {
        "action": "open_long",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity_hint_usdt": 300.0,
        "leverage": 8,
        "reduce_only": False,
        "entry_price": 100.0,
        "stop_loss_price": 97.0,
        "take_profit_price": 110.0,
        "long_probability": 0.7,
        "short_probability": 0.2,
        "wait_probability": 0.1,
        "rationale": "test",
        "generated_at": "2026-07-21T00:00:00+00:00",
        "risk_plan": {"existing": True},
    }
    values.update(overrides)
    return V6OrderIntent(**values)


def test_position_capacity_preserves_existing_boundary() -> None:
    manager = make_manager()

    available = manager.assess_position_capacity(active_position_count=1, max_positions=2)
    full = manager.assess_position_capacity(active_position_count=2, max_positions=2)

    assert isinstance(available, RiskAssessment)
    assert available.allowed is True
    assert full.allowed is False
    assert full.reason == "max_positions_reached"
    assert manager.at_position_capacity(active_position_count=0, max_positions=0) is True


def test_scale_in_notional_preserves_existing_epsilon_and_limit() -> None:
    manager = make_manager()

    assert manager.scale_in_notional_allowed(current_notional=100.0, add_notional=50.0, max_total_notional=150.0) is True
    assert manager.scale_in_notional_allowed(current_notional=100.0, add_notional=50.0000000005, max_total_notional=150.0) is True
    assert manager.scale_in_notional_allowed(current_notional=100.0, add_notional=50.000001, max_total_notional=150.0) is False


def test_local_risk_budget_contract_is_unchanged() -> None:
    budget = make_manager().local_risk_budget_for_intent(make_intent())

    assert budget == {
        "action": "open_long",
        "side": "long",
        "notional_budget_usdt": 300.0,
        "initial_margin_budget_usdt": 37.5,
        "leverage": 8,
        "entry_price": 100.0,
        "stop_loss_price": 97.0,
        "stop_distance_pct": 3.0,
        "planned_loss_at_stop_usdt": 9.0,
        "calculation": "notional_budget_usdt * stop_distance_pct",
    }


def test_annotation_preserves_existing_risk_plan_and_account_fields() -> None:
    manager = make_manager()
    intent = make_intent()
    config = SimpleNamespace(
        instance_name="v6_demo",
        account_asset_mode="MULTI_ASSET",
        margin_type="CROSSED",
        virtual_risk_isolation_enabled=True,
        order_mode="testnet",
        exit_control_mode="automatic",
    )

    manager.annotate_intent_execution_risk(config, intent)

    assert intent.risk_plan["existing"] is True
    assert intent.risk_plan["execution_account"]["local_isolation_mode"] == "virtual_subledger"
    assert intent.risk_plan["execution_account"]["exchange_margin_shared"] is True
    assert intent.risk_plan["local_risk_budget"]["planned_loss_at_stop_usdt"] == 9.0
    assert intent.risk_plan["risk_engine"]["advisory_only"] is True


def test_risk_engine_scores_budget_volatility_and_returns_position_advice() -> None:
    low = make_manager().assess_trade_risk(
        planned_loss_at_stop_usdt=2.0,
        max_risk_budget_usdt=10.0,
        volatility_pct=1.0,
        reference_volatility_pct=3.0,
        leverage=3,
        volatility_source="atr_pct",
    )
    high = make_manager().assess_trade_risk(
        planned_loss_at_stop_usdt=15.0,
        max_risk_budget_usdt=10.0,
        volatility_pct=6.0,
        reference_volatility_pct=3.0,
        leverage=12,
        volatility_source="atr_pct",
    )

    assert isinstance(low, RiskRecommendation)
    assert low.advisory_only is True
    assert 0.0 <= low.risk_score < high.risk_score <= 100.0
    assert low.position_size_multiplier == 1.0
    assert high.position_size_multiplier <= 0.5
    assert high.volatility_adjustment == 0.5
    assert high.max_risk_budget_usdt == 10.0
    assert high.budget_utilization_pct == 150.0
    assert "planned_loss_exceeds_reference_budget" in high.reasons
    assert "volatility_above_reference" in high.reasons


def test_risk_engine_annotation_is_observability_only() -> None:
    manager = make_manager()
    intent = make_intent(risk_plan={"entry_context": {"atr_pct": 4.5}})
    config = SimpleNamespace(
        instance_name="v6_demo",
        account_asset_mode="SINGLE_ASSET",
        margin_type="ISOLATED",
        virtual_risk_isolation_enabled=False,
        order_mode="testnet",
        exit_control_mode="automatic",
        base_notional_usdt=300.0,
        stop_loss_pct=0.03,
    )
    before = {
        "action": intent.action,
        "symbol": intent.symbol,
        "side": intent.side,
        "quantity_hint_usdt": intent.quantity_hint_usdt,
        "leverage": intent.leverage,
        "reduce_only": intent.reduce_only,
        "entry_price": intent.entry_price,
        "stop_loss_price": intent.stop_loss_price,
        "take_profit_price": intent.take_profit_price,
        "rationale": intent.rationale,
    }

    manager.annotate_intent_execution_risk(config, intent)

    after = {name: getattr(intent, name) for name in before}
    assert after == before
    recommendation = intent.risk_plan["risk_engine"]
    assert recommendation["advisory_only"] is True
    assert recommendation["max_risk_budget_usdt"] == 9.0
    assert recommendation["metrics"]["volatility_pct"] == 4.5
    assert recommendation["metrics"]["volatility_source"] == "entry_context"


def test_close_preflight_preserves_result_and_exception_behavior() -> None:
    manager = make_manager()
    intent = make_intent(
        action="close_long",
        side="SELL",
        reduce_only=True,
        risk_plan={"close_decision": {"source": "exchange"}},
    )
    client = SimpleNamespace(resolve_close_quantity=lambda candidate: "0.125")

    assert manager.preflight_close_quantity(client, intent) == {
        "quantity": "0.125",
        "decision": {"source": "exchange"},
        "checked_at": "2026-07-21T00:00:00+00:00",
    }

    class MarkerError(RuntimeError):
        pass

    def fail(_intent):
        raise MarkerError("same failure")

    with pytest.raises(MarkerError, match="same failure"):
        manager.preflight_close_quantity(SimpleNamespace(resolve_close_quantity=fail), intent)


def test_exchange_exception_guards_preserve_known_error_matching() -> None:
    manager = make_manager()

    assert manager.is_reduce_only_rejected_error('{"code":-2022,"msg":"ReduceOnly Order is rejected."}')
    assert manager.is_percent_price_filter_error('{"code":-4131,"msg":"PERCENT_PRICE filter"}')
    assert manager.is_status_unknown_error('signed request failed 502: Bad Gateway')
    assert manager.is_status_unknown_error('Timeout waiting for response')
    assert not manager.is_status_unknown_error('invalid symbol')


def test_engine_compatibility_wrapper_lazily_installs_risk_manager() -> None:
    engine = V6ExecutionEngine.__new__(V6ExecutionEngine)
    engine.client = SimpleNamespace(resolve_close_quantity=lambda intent: "1.5")
    intent = make_intent(action="close_long", side="SELL", reduce_only=True, risk_plan={})

    result = engine.preflight_close_quantity(intent)

    assert result["quantity"] == "1.5"
    assert isinstance(engine.risk_manager, RiskManager)


def test_risk_engine_configuration_is_advisory_and_environment_driven(monkeypatch) -> None:
    monkeypatch.setenv("TS_V6_RISK_MAX_BUDGET_USDT", "12.5")
    monkeypatch.setenv("TS_V6_RISK_REFERENCE_VOLATILITY_PCT", "2.75")

    config = V6ExecutionConfig.from_env()

    assert config.risk_max_budget_usdt == 12.5
    assert config.risk_reference_volatility_pct == 2.75
