from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from trading_system.v6_execution import V6ExecutionConfig, V6ExecutionEngine, V6OrderIntent, V6Signal, utc_now_iso


def make_engine(*, oi_expanded=False, panel_aligned=False, strength_enabled=False) -> V6ExecutionEngine:
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        v5_no_news_momentum_tp_enabled=True,
        v5_momentum_strength_tp_enabled=strength_enabled,
        v5_momentum_guard_enabled=True,
        v5_momentum_guard_first_check_minutes=30,
        v5_momentum_guard_first_min_roi_pct=5,
        v5_momentum_guard_first_fraction=0.50,
        v5_momentum_guard_mid_check_minutes=60,
        v5_momentum_guard_mid_min_roi_pct=10,
        v5_momentum_guard_final_check_minutes=120,
        v5_momentum_guard_final_min_roi_pct=18,
        v5_momentum_guard_oi_expand_pct=1,
    )
    engine._v5_sentiment_metadata_cache = {}
    engine._v5_momentum_panel_cache = None
    engine.v5_momentum_oi_context = lambda position: {
        "oi_status": "ready",
        "oi_30m_pct": 2.0 if oi_expanded else -2.0,
        "oi_expanded": oi_expanded,
    }
    engine.v5_momentum_panel_alignment = lambda symbol, side: {
        "aligned": panel_aligned,
        "opposed": not panel_aligned,
        "source_count": 1,
        "sources": [],
    }
    return engine


def make_signal(price: float = 100.0, *, trend_side: str = "long", volume_ratio: float = 0.0, symbol: str = "TESTUSDT") -> V6Signal:
    bullish = trend_side == "long"
    bearish = trend_side == "short"
    return V6Signal(
        symbol=symbol,
        computed_at=datetime.now(timezone.utc).isoformat(),
        price=price,
        side_bias="long",
        long_probability=0.5,
        short_probability=0.2,
        wait_probability=0.3,
        model_quality_score=0.0,
        reasons=[],
        h4={},
        fifteen_m={
            "ma20_trend": {
                "ready": True,
                "state": "bullish_trend_alive" if bullish else "bearish_trend_alive" if bearish else "broken_or_mixed",
                "bullish_trend_alive": bullish,
                "bearish_trend_alive": bearish,
                "price_distance_pct": 1.0 if bullish else -1.0 if bearish else 0.0,
            }
        },
        one_m={},
        mid_retest={},
        levels={},
        indicators={
            "tick_volume_15m": {
                "ready": True,
                "active": volume_ratio >= 1.5,
                "volume_vs_ma5_ratio": volume_ratio,
                "quote_volume_vs_ma5_ratio": volume_ratio,
            }
        },
    )


def make_position(age_minutes: float, *, side="long", remaining=1.0, strategy_id="v5_no_news_momentum") -> dict:
    opened_at = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    return {
        "symbol": "TESTUSDT",
        "side": side,
        "opened_at": opened_at.isoformat(),
        "entry_price": 100.0,
        "leverage": 8,
        "remaining_fraction": remaining,
        "strategy_id": strategy_id,
        "risk_plan": {
            "strategy_id": strategy_id,
            "v5_candidate_metadata": {
                "symbol": "TESTUSDT",
                "price_momentum_only": True,
                "sentiment_source": "binance_top_movers",
                "has_news_confirmation": False,
            }
        },
    }


def test_v5_momentum_guard_requires_explicit_strategy_provenance():
    engine = make_engine()
    legacy_position = make_position(35, strategy_id="")
    legacy_position.pop("strategy_id", None)
    legacy_position["risk_plan"].pop("strategy_id", None)

    assert engine.is_v5_no_news_momentum_position(legacy_position) is False
    assert engine.v5_no_news_momentum_take_profit_candidate(
        make_signal(100.3),
        legacy_position,
        {"current_roi_pct": 2.4, "max_roi_pct": 4.9, "giveback_pct": 0.0},
    ) is None


def test_v6_position_never_inherits_v5_time_guard_from_symbol_metadata():
    engine = make_engine()
    position = make_position(125, strategy_id="v6_execution")

    assert engine.is_v5_no_news_momentum_position(position) is False
    assert engine.v5_no_news_momentum_take_profit_candidate(
        make_signal(101.0),
        position,
        {"current_roi_pct": 1.0, "max_roi_pct": 2.0, "giveback_pct": 1.0},
    ) is None


def test_new_v5_momentum_entry_records_explicit_strategy_provenance():
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        v5_overlay_rules_enabled=True,
        technical_entry_only_symbols=("TESTUSDT",),
        v5_candidate_metadata_by_symbol={
            "TESTUSDT": {
                "symbol": "TESTUSDT",
                "price_momentum_only": True,
                "sentiment_source": "binance_top_movers",
            }
        },
    )
    risk_plan: dict = {}

    engine.apply_candidate_metadata_to_risk_plan("TESTUSDT", risk_plan)

    assert risk_plan["strategy_id"] == "v5_no_news_momentum"
    assert risk_plan["strategy_version"] == "v5"
    assert risk_plan["strategy_provenance"] == "entry_snapshot"


def test_v5_scale_in_preserves_explicit_provenance_through_lots_and_restart():
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        base_notional_usdt=100.0,
        min_notional_usdt=5.0,
        scale_in_enabled=True,
        max_scale_ins_per_position=1,
        scale_in_notional_fraction=0.5,
        scale_in_min_adverse_move_pct=0.5,
        scale_in_max_total_notional_multiplier=2.0,
        v5_staged_entry_enabled=False,
        v5_no_news_momentum_tp_enabled=True,
    )
    engine.market_data_freshness_entry_block = lambda *args, **kwargs: None
    engine.scale_in_recent_price_duplicate = lambda *args, **kwargs: False
    engine.indicator_confluence_hold = lambda *args, **kwargs: False
    engine.ma20_trend_entry_block = lambda *args, **kwargs: None
    engine.entry_context_for_action = lambda *args, **kwargs: {}
    engine.macro_direction_entry_block = lambda *args, **kwargs: None
    engine.smart_signal_entry_block = lambda *args, **kwargs: None
    engine.generic_indicator_entry_block = lambda *args, **kwargs: None
    engine.scale_in_trigger_candidate = lambda *args, **kwargs: {
        "name": "bb96_lower_15m",
        "line_count": 1,
    }
    engine.stop_take_profit = lambda *args, **kwargs: (95.0, 110.0, {"strategy_labels": []})
    engine.scale_in_4h_stop_price = lambda *args, **kwargs: float(kwargs["fallback"])

    position = make_position(10)
    position.update(
        {
            "notional_usdt": 100.0,
            "original_notional_usdt": 100.0,
            "position_qty": 1.0,
            "remaining_quantity": 1.0,
        }
    )
    position["risk_plan"].update(
        {"strategy_version": "v5", "strategy_provenance": "entry_snapshot"}
    )
    entry_intent = V6OrderIntent(
        action="open_long",
        symbol="TESTUSDT",
        side="BUY",
        quantity_hint_usdt=100.0,
        leverage=8,
        reduce_only=False,
        entry_price=100.0,
        stop_loss_price=95.0,
        take_profit_price=110.0,
        long_probability=0.7,
        short_probability=0.1,
        wait_probability=0.2,
        rationale="initial V5 entry",
        generated_at=utc_now_iso(),
        risk_plan=dict(position["risk_plan"]),
    )
    position["lots"] = [
        engine.lot_from_open_result(
            entry_intent,
            {
                "submitted_at": utc_now_iso(),
                "response": {"orderId": 1, "executedQty": "1", "avgPrice": "100", "cumQuote": "100"},
            },
            fill_price=100.0,
            notional=100.0,
        )
    ]

    intent = engine.scale_in_intent_if_needed(make_signal(99.0), position, require_probability=False)

    assert intent is not None
    assert intent.risk_plan["strategy_id"] == "v5_no_news_momentum"
    assert intent.risk_plan["strategy_version"] == "v5"
    assert intent.risk_plan["strategy_provenance"] == "entry_snapshot"
    assert intent.risk_plan["v5_candidate_metadata"]["price_momentum_only"] is True

    result = {
        "status": "submitted",
        "submitted_at": utc_now_iso(),
        "response": {"orderId": 2, "executedQty": "0.5", "avgPrice": "99", "cumQuote": "49.5"},
    }
    lot = engine.lot_from_open_result(intent, result, fill_price=99.0, notional=49.5)
    assert lot["strategy_id"] == "v5_no_news_momentum"
    assert lot["strategy_version"] == "v5"
    assert lot["strategy_provenance"] == "entry_snapshot"
    assert lot["risk_plan"]["strategy_id"] == "v5_no_news_momentum"
    assert lot["risk_plan"]["strategy_version"] == "v5"
    assert lot["risk_plan"]["strategy_provenance"] == "entry_snapshot"
    assert lot["risk_plan"]["v5_candidate_metadata"]["price_momentum_only"] is True

    updated = engine.apply_scale_in_to_position(position, intent, result)
    restarted = json.loads(json.dumps(updated))

    assert restarted["strategy_id"] == "v5_no_news_momentum"
    assert restarted["strategy_version"] == "v5"
    assert restarted["strategy_provenance"] == "entry_snapshot"
    assert restarted["risk_plan"]["strategy_id"] == "v5_no_news_momentum"
    assert {item["strategy_id"] for item in restarted["lots"]} == {"v5_no_news_momentum"}
    assert {item["strategy_version"] for item in restarted["lots"]} == {"v5"}
    assert {item["strategy_provenance"] for item in restarted["lots"]} == {"entry_snapshot"}
    assert engine.is_v5_no_news_momentum_position(restarted) is True


def test_v5_momentum_guard_reduces_half_after_30m_without_support():
    engine = make_engine(oi_expanded=False, panel_aligned=False)
    position = make_position(35)
    result = engine.v5_no_news_momentum_take_profit_candidate(
        make_signal(100.3),
        position,
        {"current_roi_pct": 2.4, "max_roi_pct": 4.9, "giveback_pct": 0.0},
    )
    assert result is not None
    assert result["type"] == "v5_momentum_guard"
    assert result["partial"][0][0]["name"] == "v5_momentum_weak_30m_reduce"
    assert result["partial"][1] == 0.5


def test_v5_momentum_guard_closes_after_60m_unless_oi_and_panel_both_support():
    engine = make_engine(oi_expanded=True, panel_aligned=False)
    position = make_position(65)
    result = engine.v5_no_news_momentum_take_profit_candidate(
        make_signal(100.8),
        position,
        {"current_roi_pct": 6.4, "max_roi_pct": 9.9, "giveback_pct": 0.0},
    )
    assert result is not None
    assert result["partial"][0][0]["name"] == "v5_momentum_weak_60m_exit"
    assert result["partial"][1] == 1.0

    supported_engine = make_engine(oi_expanded=True, panel_aligned=True)
    supported_position = make_position(65)
    assert supported_engine.v5_no_news_momentum_take_profit_candidate(
        make_signal(100.8),
        supported_position,
        {"current_roi_pct": 6.4, "max_roi_pct": 9.9, "giveback_pct": 0.0},
    ) is None


def test_v5_momentum_guard_closes_after_120m_even_with_support():
    engine = make_engine(oi_expanded=True, panel_aligned=True)
    position = make_position(125)
    result = engine.v5_no_news_momentum_take_profit_candidate(
        make_signal(101.0),
        position,
        {"current_roi_pct": 8.0, "max_roi_pct": 17.9, "giveback_pct": 0.0},
    )
    assert result is not None
    assert result["partial"][0][0]["name"] == "v5_momentum_weak_120m_exit"
    assert result["partial"][1] == 1.0


def test_v5_no_news_momentum_tp_marks_profit_guard_stage_executed():
    engine = make_engine()
    position = make_position(20)
    position.update({
        "notional_usdt": 1000.0,
        "original_notional_usdt": 1000.0,
        "position_qty": 10.0,
        "remaining_quantity": 10.0,
        "profit_guard_stages": [
            {
                "name": "v5_no_news_roi_tp_20",
                "status": "pending",
                "fraction_of_initial": 0.70,
                "close_fraction_of_remaining": 0.70,
            }
        ],
    })
    intent = V6OrderIntent(
        action="close_long",
        symbol="TESTUSDT",
        side="SELL",
        quantity_hint_usdt=700.0,
        leverage=8,
        reduce_only=True,
        entry_price=110.0,
        stop_loss_price=90.0,
        take_profit_price=130.0,
        long_probability=0.7,
        short_probability=0.1,
        wait_probability=0.2,
        rationale="test v5 no-news momentum tp",
        generated_at=utc_now_iso(),
        risk_plan={
            "partial_take_profit": {
                "type": "v5_no_news_momentum_tp",
                "fraction_of_initial": 0.70,
                "close_fraction_of_remaining": 0.70,
                "trigger_roi_pct": 25.0,
            }
        },
        close_fraction=0.70,
        target_name="v5_no_news_roi_tp_20",
    )
    result = {
        "status": "submitted",
        "submitted_at": utc_now_iso(),
        "response": {"orderId": 123456, "executedQty": "7", "avgPrice": "110"},
    }

    updated = engine.apply_partial_take_profit_to_position(position, intent, result)

    assert updated is not None
    assert engine.profit_guard_stage_executed(updated, "v5_no_news_roi_tp_20")
    assert engine.v5_no_news_momentum_take_profit_candidate(
        make_signal(110.0),
        updated,
        {"current_roi_pct": 25.0, "max_roi_pct": 25.0, "giveback_pct": 0.0},
    ) is None


def test_v5_roi_program_does_not_fall_through_to_generic_roi_stages():
    engine = make_engine()
    position = make_position(20, remaining=0.1)
    position["profit_guard_stages"] = [
        {"name": "v5_no_news_roi_tp_20", "status": "executed", "fraction_of_initial": 0.7},
        {"name": "v5_no_news_roi_tp_30", "status": "executed", "fraction_of_initial": 0.2},
    ]
    position["profit_guard"] = {"max_roi_pct": 32.0}

    assert engine.next_roi_take_profit(make_signal(104.0), position) is None


def test_v5_momentum_strength_tp_sells_less_when_momentum_is_strong():
    engine = make_engine(oi_expanded=True, panel_aligned=True, strength_enabled=True)
    position = make_position(35)

    result = engine.v5_no_news_momentum_take_profit_candidate(
        make_signal(103.0, trend_side="long", volume_ratio=3.2),
        position,
        {"current_roi_pct": 25.0, "max_roi_pct": 25.0, "giveback_pct": 0.0},
    )

    assert result is not None
    stages, close_fraction = result["partial"]
    assert stages[0]["name"] == "v5_no_news_roi_tp_20"
    assert stages[0]["fraction_of_initial"] == 0.10
    assert close_fraction == 0.10
    assert result["trigger"]["momentum_strength"]["level"] == "strong"


def test_v5_momentum_strength_tp_sells_more_when_momentum_is_exhausted():
    engine = make_engine(oi_expanded=False, panel_aligned=False, strength_enabled=True)
    position = make_position(35)

    result = engine.v5_no_news_momentum_take_profit_candidate(
        make_signal(103.0, trend_side="short", volume_ratio=0.8),
        position,
        {"current_roi_pct": 25.0, "max_roi_pct": 25.0, "giveback_pct": 0.0},
    )

    assert result is not None
    stages, close_fraction = result["partial"]
    assert stages[0]["name"] == "v5_no_news_roi_tp_20"
    assert stages[0]["fraction_of_initial"] == 0.70
    assert close_fraction == 0.70
    assert result["trigger"]["momentum_strength"]["level"] == "exhausted"


def test_v5_momentum_panel_ignores_hyper_by_default(tmp_path):
    smart_path = tmp_path / "smart.json"
    hyper_path = tmp_path / "hyper.json"
    now = utc_now_iso()
    smart_path.write_text(json.dumps({"generated_at": now, "signals": []}), encoding="utf-8")
    hyper_path.write_text(
        json.dumps(
            {
                "generated_at": now,
                "signals": [
                    {
                        "symbol": "TESTUSDT",
                        "direction": "long",
                        "metadata": {"net_ratio": 0.8},
                        "updated_at": now,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        v5_momentum_guard_smart_money_file=smart_path,
        v5_momentum_guard_hyper_whale_file=hyper_path,
        v5_momentum_guard_use_hyper_panel=False,
    )
    engine._v5_momentum_panel_cache = None

    result = engine.v5_momentum_panel_alignment("TESTUSDT", "long")
    assert result["source_count"] == 0
    assert not result["aligned"]

    engine.config.v5_momentum_guard_use_hyper_panel = True
    engine._v5_momentum_panel_cache = None
    enabled = engine.v5_momentum_panel_alignment("TESTUSDT", "long")
    assert enabled["source_count"] == 1
    assert enabled["aligned"]
    assert enabled["sources"][0]["source"] == "hyper"


def test_v5_momentum_panel_reads_binance_smart_signal_bridge(tmp_path):
    smart_path = tmp_path / "smart_signal.json"
    now = utc_now_iso()
    smart_path.write_text(
        json.dumps(
            {
                "generated_at": now,
                "signals": [
                    {
                        "symbol": "TESTSMARTUSDT",
                        "source": "binance-smart-signal",
                        "updated_at": now,
                        "direction": "short",
                        "long_score": 100.0,
                        "short_score": 260.0,
                        "metadata": {
                            "provider": "binance_smart_signal",
                            "side": "SELL",
                            "long_notional": 100.0,
                            "short_notional": 260.0,
                            "long_whales": 20,
                            "short_whales": 50,
                            "short_to_long_notional": 2.6,
                            "short_whale_share": 0.714,
                            "short_avg_entry_price": 101.0,
                            "short_crowded": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(v5_momentum_guard_smart_money_file=smart_path)
    engine._v5_momentum_panel_cache = None

    result = engine.v5_momentum_panel_alignment("TESTSMARTUSDT", "short")

    assert result["source_count"] == 1
    assert result["aligned"]
    assert result["sources"][0]["source"] == "binance_smart_signal"
    assert result["sources"][0]["short_crowded"] is True


def test_v5_short_squeeze_context_flags_crowded_short_reclaim(tmp_path):
    smart_path = tmp_path / "smart_signal.json"
    now = utc_now_iso()
    smart_path.write_text(
        json.dumps(
            {
                "generated_at": now,
                "signals": [
                    {
                        "symbol": "RAVEUSDT",
                        "source": "binance-smart-signal",
                        "updated_at": now,
                        "direction": "short",
                        "long_score": 100.0,
                        "short_score": 240.0,
                        "metadata": {
                            "provider": "binance_smart_signal",
                            "side": "SELL",
                            "long_notional": 100.0,
                            "short_notional": 240.0,
                            "long_whales": 22,
                            "short_whales": 60,
                            "short_to_long_notional": 2.4,
                            "short_whale_share": 0.7317,
                            "short_avg_entry_price": 100.0,
                            "short_crowded": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        v5_momentum_guard_smart_money_file=smart_path,
        v5_momentum_guard_oi_expand_pct=1,
    )
    engine._v5_momentum_panel_cache = None
    engine.v5_momentum_oi_context = lambda position: {
        "oi_status": "ready",
        "oi_30m_pct": 2.0,
        "oi_expanded": True,
    }
    position = make_position(45, side="long")
    position["symbol"] = "RAVEUSDT"

    result = engine.v5_momentum_strength_context(
        make_signal(101.0, trend_side="long", volume_ratio=2.0, symbol="RAVEUSDT"),
        position,
        {"current_roi_pct": 8.0, "max_roi_pct": 12.0, "giveback_pct": 4.0},
    )

    squeeze = result["components"]["short_squeeze"]
    assert squeeze["active"] is True
    assert squeeze["price_reclaimed_ma20"] is True
    assert squeeze["volume_confirmed"] is True
    assert squeeze["short_to_long_notional"] == 2.4


def test_v6_smart_signal_blocks_short_entry_on_squeeze_risk(tmp_path):
    smart_path = tmp_path / "smart_signal.json"
    now = utc_now_iso()
    smart_path.write_text(
        json.dumps(
            {
                "generated_at": now,
                "signals": [
                    {
                        "symbol": "RAVEUSDT",
                        "source": "binance-smart-signal",
                        "updated_at": now,
                        "direction": "short",
                        "long_score": 100.0,
                        "short_score": 310.0,
                        "metadata": {
                            "provider": "binance_smart_signal",
                            "side": "SELL",
                            "long_notional": 100.0,
                            "short_notional": 310.0,
                            "long_whales": 22,
                            "short_whales": 60,
                            "short_to_long_notional": 3.1,
                            "short_whale_share": 0.7317,
                            "short_avg_entry_price": 100.0,
                            "short_crowded": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(smart_signal_file=smart_path)
    engine._smart_signal_cache = None

    signal = make_signal(101.0, trend_side="long", volume_ratio=2.0, symbol="RAVEUSDT")
    block = engine.smart_signal_entry_block(signal, "open_short", {})

    assert block is not None
    assert block["reason"] == "short_blocked_by_smart_signal_short_squeeze"
    assert block["smart_signal"]["short_squeeze_active"] is True


def test_v6_smart_signal_blocks_long_entry_on_long_crowded_flush(tmp_path):
    smart_path = tmp_path / "smart_signal.json"
    now = utc_now_iso()
    smart_path.write_text(
        json.dumps(
            {
                "generated_at": now,
                "signals": [
                    {
                        "symbol": "LABUSDT",
                        "source": "binance-smart-signal",
                        "updated_at": now,
                        "direction": "long",
                        "long_score": 300.0,
                        "short_score": 100.0,
                        "metadata": {
                            "provider": "binance_smart_signal",
                            "side": "BUY",
                            "long_notional": 300.0,
                            "short_notional": 100.0,
                            "long_whales": 70,
                            "short_whales": 20,
                            "long_to_short_notional": 3.0,
                            "long_whale_share": 0.7777,
                            "long_avg_entry_price": 100.0,
                            "long_crowded": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(smart_signal_file=smart_path)
    engine._smart_signal_cache = None

    signal = make_signal(99.0, trend_side="short", volume_ratio=2.0, symbol="LABUSDT")
    block = engine.smart_signal_entry_block(signal, "open_long", {})

    assert block is not None
    assert block["reason"] == "long_blocked_by_smart_signal_long_crowding"
    assert block["smart_signal"]["long_crowded_flush_active"] is True


def test_v6_smart_signal_context_tolerates_missing_fifteen_m(tmp_path):
    smart_path = tmp_path / "smart_signal.json"
    now = utc_now_iso()
    smart_path.write_text(
        json.dumps(
            {
                "generated_at": now,
                "signals": [
                    {
                        "symbol": "RAVEUSDT",
                        "source": "binance-smart-signal",
                        "updated_at": now,
                        "direction": "short",
                        "short_score": 300.0,
                        "long_score": 100.0,
                        "metadata": {"short_crowded": True, "short_to_long_notional": 3.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(smart_signal_file=smart_path)
    engine._smart_signal_cache = None
    signal = type(
        "LegacySignal",
        (),
        {
            "symbol": "RAVEUSDT",
            "price": 101.0,
            "indicators": {},
        },
    )()

    context = engine.v6_smart_signal_context(signal, "short")

    assert context["has_signal"] is True
    assert context["active"] is False
    assert context["price_reclaimed_ma20"] is False


def test_v6_smart_signal_missing_ma20_distance_waits_for_confirmation(tmp_path):
    smart_path = tmp_path / "smart_signal.json"
    now = utc_now_iso()
    smart_path.write_text(
        json.dumps(
            {
                "generated_at": now,
                "signals": [
                    {
                        "symbol": "RAVEUSDT",
                        "source": "binance-smart-signal",
                        "updated_at": now,
                        "direction": "short",
                        "short_score": 300.0,
                        "long_score": 100.0,
                        "metadata": {
                            "short_crowded": True,
                            "short_to_long_notional": 3.0,
                            "short_avg_entry_price": 120.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(smart_signal_file=smart_path)
    engine._smart_signal_cache = None
    signal = make_signal(101.0, trend_side="mixed", volume_ratio=1.0, symbol="RAVEUSDT")
    signal.fifteen_m = {"ma20_trend": {"ready": True}}

    context = engine.v6_smart_signal_context(signal, "short")

    assert context["active"] is False
    assert context["price_reclaimed_ma20"] is False
    assert context["price_lost_ma20"] is False
    assert context["price_distance_ma20_15m_pct"] is None


def test_market_data_freshness_block_prevents_open_entries_only():
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig()
    signal = make_signal(symbol="BTCUSDT")
    signal.market_data_freshness = {
        "stale": True,
        "updated_at": utc_now_iso(),
        "stale_sources": [{"source": "kline_15m", "reason": "network"}],
    }

    block = engine.market_data_freshness_entry_block(signal, "open_long")

    assert block is not None
    assert block["reason"] == "entry_blocked_stale_market_data"
    assert engine.market_data_freshness_entry_block(signal, "close_long") is None


def test_v6_smart_signal_neutral_ma20_zone_is_mutually_exclusive_and_not_risk(tmp_path):
    smart_path = tmp_path / "smart_signal.json"
    now = utc_now_iso()
    smart_path.write_text(
        json.dumps(
            {
                "generated_at": now,
                "signals": [
                    {
                        "symbol": "ZECUSDT",
                        "source": "binance-smart-signal",
                        "updated_at": now,
                        "direction": "long",
                        "long_score": 719.2,
                        "short_score": 100.0,
                        "metadata": {
                            "long_to_short_notional": 7.192,
                            "long_whale_share": 0.7528,
                            "long_avg_entry_price": 500.0,
                            "long_crowded": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        smart_signal_file=smart_path,
        smart_signal_ma20_confirm_pct=0.05,
    )
    engine._smart_signal_cache = None
    signal = make_signal(500.055, trend_side="long", volume_ratio=0.918, symbol="ZECUSDT")
    signal.fifteen_m["ma20_trend"]["price_distance_pct"] = 0.011

    context = engine.v6_smart_signal_context(signal, "long")

    assert context["price_reclaimed_ma20"] is False
    assert context["price_lost_ma20"] is False
    assert context["long_crowded"] is True
    assert context["active"] is False
    assert context["flush_risk_confirmed"] is False


def test_v6_smart_signal_zero_ma20_threshold_keeps_exact_ma20_neutral(tmp_path):
    smart_path = tmp_path / "smart_signal.json"
    now = utc_now_iso()
    smart_path.write_text(
        json.dumps(
            {
                "generated_at": now,
                "signals": [
                    {
                        "symbol": "ZECUSDT",
                        "updated_at": now,
                        "direction": "long",
                        "long_score": 300.0,
                        "short_score": 300.0,
                        "metadata": {
                            "long_crowded": True,
                            "short_crowded": True,
                            "long_avg_entry_price": 500.0,
                            "short_avg_entry_price": 500.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        smart_signal_file=smart_path,
        smart_signal_ma20_confirm_pct=0.0,
    )
    engine._smart_signal_cache = None
    signal = make_signal(500.0, trend_side="mixed", volume_ratio=2.0, symbol="ZECUSDT")
    signal.fifteen_m["ma20_trend"]["price_distance_pct"] = 0.0

    context = engine.v6_smart_signal_context(signal, "long")

    assert context["price_reclaimed_ma20"] is False
    assert context["price_lost_ma20"] is False
    assert context["active"] is False


def test_v6_smart_signal_stale_snapshot_cannot_trigger_exit_risk(tmp_path):
    smart_path = tmp_path / "smart_signal.json"
    stale_at = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    smart_path.write_text(
        json.dumps(
            {
                "generated_at": stale_at,
                "signals": [
                    {
                        "symbol": "RAVEUSDT",
                        "source": "binance-smart-signal",
                        "updated_at": stale_at,
                        "direction": "short",
                        "long_score": 100.0,
                        "short_score": 310.0,
                        "metadata": {
                            "short_to_long_notional": 3.1,
                            "short_whale_share": 0.7317,
                            "short_crowded": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        smart_signal_file=smart_path,
        smart_signal_snapshot_max_age_hours=0.5,
        smart_signal_exit_snapshot_max_age_hours=0.25,
    )
    engine._smart_signal_cache = None

    context = engine.v6_smart_signal_context(
        make_signal(101.0, trend_side="long", volume_ratio=2.0, symbol="RAVEUSDT"),
        "short",
    )

    assert context["has_signal"] is True
    assert context["snapshot_fresh_for_exit"] is False
    assert context["active"] is False


def test_v6_smart_signal_future_snapshot_is_not_fresh():
    future_at = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

    assert V6ExecutionEngine._v5_momentum_iso_is_fresh(future_at, 0.25) is False
    assert V6ExecutionEngine._v5_momentum_iso_is_fresh_seconds(future_at, 900) is False


def test_v6_smart_signal_risk_guard_requires_two_distinct_snapshots(tmp_path):
    smart_path = tmp_path / "smart_signal.json"

    def write_snapshot(updated_at: str) -> None:
        smart_path.write_text(
            json.dumps(
                {
                    "generated_at": updated_at,
                    "signals": [
                        {
                            "symbol": "RAVEUSDT",
                            "source": "binance-smart-signal",
                            "updated_at": updated_at,
                            "direction": "short",
                            "long_score": 100.0,
                            "short_score": 310.0,
                            "metadata": {
                                "provider": "binance_smart_signal",
                                "side": "SELL",
                                "long_notional": 100.0,
                                "short_notional": 310.0,
                                "long_whales": 22,
                                "short_whales": 60,
                                "short_to_long_notional": 3.1,
                                "short_whale_share": 0.7317,
                                "short_avg_entry_price": 100.0,
                                "short_crowded": True,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    first_dt = datetime.now(timezone.utc) - timedelta(seconds=2)
    first_at = first_dt.isoformat()
    write_snapshot(first_at)
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        smart_signal_file=smart_path,
        smart_signal_risk_reduce_fraction=0.30,
        smart_signal_risk_confirmation_snapshots=2,
    )
    engine._smart_signal_cache = None
    position = make_position(45, side="short", strategy_id="v6_execution")
    position["symbol"] = "RAVEUSDT"
    signal = make_signal(101.0, trend_side="long", volume_ratio=2.0, symbol="RAVEUSDT")
    roi_state = {"current_roi_pct": 5.0, "max_roi_pct": 8.0, "giveback_pct": 3.0}

    assert engine.smart_signal_risk_guard_candidate(signal, position, roi_state) is None
    assert position["smart_signal_risk_state"]["confirmation_count"] == 1
    duplicate = engine.smart_signal_risk_guard_candidate(signal, position, roi_state)
    assert duplicate is None
    assert position["smart_signal_risk_state"]["confirmation_count"] == 1

    # 同一时刻的 +00:00 / Z 表示必须规范化为同一快照，不能重复计数。
    write_snapshot(first_at.replace("+00:00", "Z"))
    engine._smart_signal_cache = None
    equivalent = engine.smart_signal_risk_guard_candidate(signal, position, roi_state)
    assert equivalent is None
    assert position["smart_signal_risk_state"]["confirmation_count"] == 1

    # 模拟进程重启：确认状态必须经 JSON 状态文件往返后仍可延续。
    position = json.loads(json.dumps(position))
    second_at = (first_dt + timedelta(seconds=1)).isoformat()
    write_snapshot(second_at)

    engine._smart_signal_cache = None
    result = engine.smart_signal_risk_guard_candidate(signal, position, roi_state)

    assert result is not None
    assert result["type"] == "v6_smart_signal_risk_guard"
    stages, close_fraction = result["partial"]
    assert stages[0]["name"] == "v6_smart_signal_short_squeeze_reduce_30"
    assert stages[0]["fraction_of_initial"] == 0.3
    assert close_fraction == 0.3
    assert position["smart_signal_risk_state"]["confirmation_count"] == 2


def test_v6_smart_signal_risk_guard_recovers_string_float_confirmation_count():
    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        smart_signal_risk_reduce_fraction=0.30,
        smart_signal_risk_confirmation_snapshots=2,
    )
    snapshot_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    engine.v6_smart_signal_context = lambda *args, **kwargs: {
        "enabled": True,
        "has_signal": True,
        "active": True,
        "adverse_to_side": True,
        "snapshot_updated_at": snapshot_at,
    }
    position = make_position(45, side="short", strategy_id="v6_execution")
    position["smart_signal_risk_state"] = {
        "risk_side": "short",
        "confirmation_count": "1.0",
        "last_snapshot_id": "2026-07-20T00:00:00+00:00",
    }

    result = engine.smart_signal_risk_guard_candidate(
        make_signal(101.0),
        position,
        {"current_roi_pct": 5.0, "max_roi_pct": 8.0, "giveback_pct": 3.0},
    )

    assert result is not None
    assert position["smart_signal_risk_state"]["confirmation_count"] == 2

    corrupt_position = make_position(20, side="short", strategy_id="v6_execution")
    corrupt_position["smart_signal_risk_state"] = {
        "risk_side": "short",
        "confirmation_count": "inf",
        "last_snapshot_id": (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
    }
    assert engine.smart_signal_risk_guard_candidate(
        make_signal(100.0, trend_side="short"),
        corrupt_position,
        {"current_roi_pct": -5.0},
    ) is None
    assert corrupt_position["smart_signal_risk_state"]["confirmation_count"] == 1


def test_v5_momentum_oi_context_refreshes_stale_cached_value():
    now = datetime.now(timezone.utc)

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def public_get(self, endpoint, params):
            self.calls += 1
            assert endpoint == "/futures/data/openInterestHist"
            return [
                {
                    "timestamp": int((now - timedelta(minutes=30)).timestamp() * 1000),
                    "sumOpenInterestValue": "100",
                },
                {
                    "timestamp": int(now.timestamp() * 1000),
                    "sumOpenInterestValue": "105",
                },
            ]

    engine = object.__new__(V6ExecutionEngine)
    engine.config = V6ExecutionConfig(
        v5_momentum_guard_first_check_minutes=30,
        v5_momentum_guard_oi_expand_pct=1,
        v5_momentum_guard_oi_refresh_seconds=60,
    )
    engine.client = FakeClient()
    position = make_position(45)
    position["v5_momentum_guard"] = {
        "oi_status": "ready",
        "oi_30m_pct": -8.0,
        "oi_expanded": False,
        "oi_last_checked_at": (now - timedelta(minutes=3)).isoformat(),
    }

    result = engine.v5_momentum_oi_context(position)

    assert engine.client.calls == 1
    assert result["oi_status"] == "ready"
    assert result["oi_30m_pct"] == 5.0
    assert result["oi_expanded"] is True
    assert result["oi_window_minutes"] >= 29.0
