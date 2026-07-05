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


def make_position(age_minutes: float, *, side="long", remaining=1.0) -> dict:
    opened_at = datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
    return {
        "symbol": "TESTUSDT",
        "side": side,
        "opened_at": opened_at.isoformat(),
        "entry_price": 100.0,
        "leverage": 8,
        "remaining_fraction": remaining,
        "risk_plan": {
            "v5_candidate_metadata": {
                "symbol": "TESTUSDT",
                "price_momentum_only": True,
                "sentiment_source": "binance_top_movers",
                "has_news_confirmation": False,
            }
        },
    }


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


def test_v6_smart_signal_risk_guard_reduces_short_position(tmp_path):
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
    engine.config = V6ExecutionConfig(smart_signal_file=smart_path, smart_signal_risk_reduce_fraction=0.30)
    engine._smart_signal_cache = None
    position = make_position(45, side="short")
    position["symbol"] = "RAVEUSDT"

    result = engine.smart_signal_risk_guard_candidate(
        make_signal(101.0, trend_side="long", volume_ratio=2.0, symbol="RAVEUSDT"),
        position,
        {"current_roi_pct": 5.0, "max_roi_pct": 8.0, "giveback_pct": 3.0},
    )

    assert result is not None
    assert result["type"] == "v6_smart_signal_risk_guard"
    stages, close_fraction = result["partial"]
    assert stages[0]["name"] == "v6_smart_signal_short_squeeze_reduce_30"
    assert stages[0]["fraction_of_initial"] == 0.3
    assert close_fraction == 0.3


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
