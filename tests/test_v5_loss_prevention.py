from trading_system.v5_loss_prevention import LABEL_DEFAULT_ACTIONS, classify_v5_loss_risk


def _record(**overrides):
    record = {
        "intent": {
            "action": "open_long",
            "symbol": "TESTUSDT",
            "side": "BUY",
            "entry_price": 125.0,
            "generated_at": "2026-06-19T00:00:00+00:00",
            "rationale": "",
            "risk_plan": {
                "entry_context": {
                    "selected_entry_level": {"name": "ma20_15m", "timeframe": "15m"},
                    "h4_right_side": {"ma20": 100.0, "price_distance_pct": 25.0},
                    "daily_ma7_trend": {
                        "bearish_stack": True,
                        "ma7_reclaimed": True,
                        "ma20_reclaimed": False,
                    },
                },
                "entry_indicators": {
                    "ma20_trend": {"state": "bullish_trend_alive"},
                },
                "structure_entry_scan": {
                    "long_candidate_count": 1,
                    "short_candidate_count": 0,
                },
            },
        }
    }
    _deep_update(record, overrides)
    return record


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def test_classifies_mixed_structure_from_candidate_counts():
    result = classify_v5_loss_risk(
        _record(
            intent={
                "risk_plan": {
                    "structure_entry_scan": {
                        "long_candidate_count": 2,
                        "short_candidate_count": 1,
                    }
                }
            }
        )
    )

    assert "mixed_structure_conflict" in result.risk_labels


def test_classifies_m15_guard_broken_or_mixed():
    result = classify_v5_loss_risk(
        _record(
            intent={
                "risk_plan": {
                    "entry_indicators": {
                        "ma20_trend": {"state": "broken_or_mixed"},
                    }
                }
            }
        )
    )

    assert "m15_guard_broken_or_mixed" in result.risk_labels


def test_classifies_h4_extension_buckets_for_longs():
    extreme = classify_v5_loss_risk(_record())
    moderate = classify_v5_loss_risk(
        _record(intent={"risk_plan": {"entry_context": {"h4_right_side": {"price_distance_pct": 15.0}}}})
    )

    assert "h4_ma20_extension_extreme" in extreme.risk_labels
    assert "h4_ma20_extension_moderate" in moderate.risk_labels
    assert "h4_ma20_extension_extreme" not in moderate.risk_labels


def test_does_not_apply_h4_long_extension_to_shorts():
    result = classify_v5_loss_risk(
        _record(
            intent={
                "action": "open_short",
                "side": "SELL",
                "risk_plan": {"entry_context": {"h4_right_side": {"price_distance_pct": -30.0}}},
            }
        )
    )

    assert "h4_ma20_extension_extreme" not in result.risk_labels


def test_classifies_daily_ma7_only_reclaim():
    result = classify_v5_loss_risk(_record())

    assert "daily_ma7_only_reclaim" in result.risk_labels


def test_classifies_one_minute_upper_pressure_from_rationale():
    result = classify_v5_loss_risk(
        _record(intent={"rationale": "4H above MA20; 1M upper pressure is active"})
    )

    assert "one_m_upper_pressure_active" in result.risk_labels


def test_classifies_long_large_upper_wick_from_entry_indicator():
    result = classify_v5_loss_risk(
        _record(),
        {
            "entry_indicator": {
                "upper_wick_pct": 0.75,
                "candle_body_pct": 0.2,
            }
        },
    )

    assert "long_large_upper_wick" in result.risk_labels
    assert result.details["long_large_upper_wick"]["upper_wick_pct"] == 0.75


def test_classifies_15m_scalp_no_follow_after_first_two_candles():
    result = classify_v5_loss_risk(
        _record(intent={"risk_plan": {"entry_timeframe": "15m"}}),
        {
            "early_15m_path": {
                "n1_favorable_return_pct": -0.12,
                "n2_favorable_return_pct": 0.0,
            }
        },
    )

    assert "m15_scalp_no_follow_1_2" in result.risk_labels


def test_15m_scalp_follow_through_does_not_trigger_no_follow():
    result = classify_v5_loss_risk(
        _record(intent={"risk_plan": {"entry_timeframe": "15m"}}),
        {
            "early_15m_path": {
                "n1_favorable_return_pct": -0.12,
                "n2_favorable_return_pct": 0.08,
            }
        },
    )

    assert "m15_scalp_no_follow_1_2" not in result.risk_labels


def test_default_actions_do_not_promote_first_batch_to_hard_block():
    first_batch = {
        "mixed_structure_conflict",
        "m15_guard_broken_or_mixed",
        "repeat_selected_level_failure",
        "h4_ma20_extension_extreme",
        "long_large_upper_wick",
        "m15_scalp_no_follow_1_2",
    }

    assert all("hard_block" not in LABEL_DEFAULT_ACTIONS[label] for label in first_batch)


def test_repeat_selected_level_uses_recent_failures_context():
    result = classify_v5_loss_risk(
        _record(),
        {
            "opened_at": "2026-06-19T12:00:00+00:00",
            "recent_failures": [
                {
                    "symbol": "TESTUSDT",
                    "side": "long",
                    "selected_level": "ma20_15m",
                    "closed_at": "2026-06-19T06:30:00+00:00",
                    "pnl_usdt": -10.0,
                }
            ],
        },
    )

    assert "repeat_selected_level_failure" in result.risk_labels
    assert result.details["repeat_selected_level_failure"]["symbol_side_failures"]["12h"] == 1


def test_early_health_is_unavailable_unless_context_supplies_it():
    result = classify_v5_loss_risk(_record(), {"early_health_available": False})

    assert "early_health_deterioration" not in result.risk_labels
    assert "early_health_deterioration" in result.unavailable_labels
