"""Tests for scripts/backtest_v6_gann_ab.py — pure engine, no network.

Tiny synthetic klines/candidates fixtures only.
"""

from __future__ import annotations

import json
from datetime import date as _date, timedelta

import pytest

import scripts.backtest_v6_gann_ab as ab
from trading_system.gann_feature import GannFeature


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def make_bars(rows, start: str = "2024-06-01") -> list[dict]:
    """rows = list of (open, high, low, close) on consecutive dates."""
    start_day = _date.fromisoformat(start)
    bars = []
    for i, (o, h, l, c) in enumerate(rows):
        day = (start_day + timedelta(days=i)).isoformat()
        bars.append({"date": day, "time": day, "open": float(o), "high": float(h), "low": float(l), "close": float(c)})
    return bars


def cand(date: str, symbol: str = "AAA", action: str = "open_long", score: float = 1.0, **kw) -> ab.Candidate:
    return ab.Candidate(
        date=date,
        symbol=symbol,
        action=action,
        score=score,
        entry_ref_price=kw.get("entry_ref_price"),
        stop_loss_pct=kw.get("stop_loss_pct", 3.0),
        take_profit_pct=kw.get("take_profit_pct", 6.0),
        max_hold_days=kw.get("max_hold_days", 14),
    )


NO_COST = ab.CostConfig(fee_bps=0.0, slippage_bps=0.0)
PORT = ab.PortfolioConfig()
BASELINE = ab.ARM_CONFIGS["v6_baseline"]


def make_feature(bias: str, confidence: float = 0.8, stale: bool = False) -> GannFeature:
    return GannFeature(
        symbol="AAA",
        zone="above_1x1" if bias == "support_long" else "below_1x2",
        bias=bias,
        confidence=confidence,
        anchor_high=200.0,
        anchor_low=100.0,
        anchor_high_time="2024-01-01",
        anchor_low_time="2024-02-01",
        cycle_days=31,
        scale="log",
        line_values={"1x2": 110.0, "1x1": 120.0, "2x1": 140.0},
        anchor_stability=1.0,
        stale=stale,
    )


# ---------------------------------------------------------------------------
# next-bar execution
# ---------------------------------------------------------------------------


def test_next_bar_fill_at_t_plus_1_open():
    bars = make_bars([(100, 101, 99, 100), (105, 106, 104, 105), (105, 106, 104, 105)])
    candidates = [cand("2024-06-01", max_hold_days=2)]
    result = ab.simulate_portfolio(candidates, {"AAA": bars}, BASELINE, NO_COST, PORT)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date == "2024-06-02"
    assert trade.entry_price == 105.0  # exactly T+1 open, zero slippage
    assert trade.exit_reason == "time_exit"


# ---------------------------------------------------------------------------
# stop-loss priority
# ---------------------------------------------------------------------------


def test_stop_loss_fires_before_take_profit_when_both_touched():
    # fill at 100; stop = 97, tp = 106; same bar touches both extremes
    bars = make_bars([(100, 100, 100, 100), (100, 110, 95, 100)])
    candidates = [cand("2024-06-01", stop_loss_pct=3.0, take_profit_pct=6.0)]
    result = ab.simulate_portfolio(candidates, {"AAA": bars}, BASELINE, NO_COST, PORT)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(97.0)
    assert trade.return_pct == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# same-day allocation: max_concurrent + score priority + tiebreak
# ---------------------------------------------------------------------------


def test_max_concurrent_score_priority_and_skipped_full():
    bars = make_bars([(100, 101, 99, 100)] * 3)
    klines = {"AAA": bars, "BBB": bars, "CCC": bars}
    candidates = [
        cand("2024-06-01", symbol="AAA", score=2.0),
        cand("2024-06-01", symbol="CCC", score=3.0),
        cand("2024-06-01", symbol="BBB", score=3.0),
    ]
    port = ab.PortfolioConfig(max_concurrent=2)
    result = ab.simulate_portfolio(candidates, klines, BASELINE, NO_COST, port)
    # equal top scores 3.0 -> symbol asc tiebreak: BBB then CCC; AAA (score 2) skipped_full
    assert sorted(t.symbol for t in result.trades) == ["BBB", "CCC"]
    assert result.skip_count("skipped_full") == 1
    metrics = ab.compute_metrics(result, "all")
    assert metrics["skipped_full"] == 1
    assert metrics["max_simultaneous_positions"] == 2
    assert metrics["trade_count"] == 2


# ---------------------------------------------------------------------------
# same-symbol overlap
# ---------------------------------------------------------------------------


def test_same_symbol_overlap_ignored_and_counted():
    bars = make_bars([(100, 101, 99, 100)] * 4)
    candidates = [
        cand("2024-06-01", max_hold_days=10),
        cand("2024-06-02", max_hold_days=10),  # fills 06-03 while first still open
    ]
    result = ab.simulate_portfolio(candidates, {"AAA": bars}, BASELINE, NO_COST, PORT)
    assert len(result.trades) == 1
    assert result.skip_count("skipped_same_symbol") == 1
    assert ab.compute_metrics(result, "all")["skipped_same_symbol"] == 1


# ---------------------------------------------------------------------------
# fee + slippage arithmetic (hand-computed round trip)
# ---------------------------------------------------------------------------


def test_fee_and_slippage_arithmetic_on_known_round_trip():
    # equity 10000, risk 1% -> budget 100; stop 5% -> notional 2000; qty 20 @ base 100
    # fee 10 bps, slip 10 bps:
    #   entry_fill = 100 * 1.001 = 100.1        entry_fee = 20 * 100.1 * 0.001 = 2.002
    #   tp raw 110 -> exit_fill = 110 * 0.999 = 109.89
    #   exit_fee = 20 * 109.89 * 0.001 = 2.1978
    #   pnl = 20 * (109.89 - 100.1) - 2.002 - 2.1978 = 191.6002
    #   return_pct = 191.6002 / 2000 * 100 = 9.58001
    bars = make_bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 111, 100, 110)])
    candidates = [cand("2024-06-01", stop_loss_pct=5.0, take_profit_pct=10.0)]
    cost = ab.CostConfig(fee_bps=10.0, slippage_bps=10.0)
    result = ab.simulate_portfolio(candidates, {"AAA": bars}, BASELINE, cost, PORT)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.qty == pytest.approx(20.0)
    assert trade.notional == pytest.approx(2000.0)
    assert trade.entry_price == pytest.approx(100.1)
    assert trade.exit_price == pytest.approx(109.89)
    assert trade.exit_reason == "take_profit"
    assert trade.pnl == pytest.approx(191.6002)
    assert trade.return_pct == pytest.approx(9.58001)


# ---------------------------------------------------------------------------
# A3 hard filter
# ---------------------------------------------------------------------------


def test_a3_hard_filter_skips_conflicting_candidate():
    bars = make_bars([(100, 101, 99, 100)] * 3)
    candidates = [cand("2024-06-01", action="open_long")]
    conflicting = make_feature("support_short", confidence=0.8)

    def provider(symbol, bars_, asof_index):
        return conflicting

    arm = ab.ARM_CONFIGS["gann_hard_filter"]
    result = ab.simulate_portfolio(candidates, {"AAA": bars}, arm, NO_COST, PORT, gann_provider=provider)
    assert result.trades == []
    assert result.skip_count("skipped_gann_filter") == 1
    assert ab.compute_metrics(result, "all")["skipped_gann_filter"] == 1
    # baseline arm ignores Gann entirely -> takes the trade
    base = ab.simulate_portfolio(candidates, {"AAA": bars}, BASELINE, NO_COST, PORT, gann_provider=provider)
    assert len(base.trades) == 1


def test_a3_hard_filter_ignores_low_confidence_conflict():
    bars = make_bars([(100, 101, 99, 100)] * 3)
    candidates = [cand("2024-06-01", action="open_long")]
    weak_conflict = make_feature("support_short", confidence=0.2)  # below 0.40 gate -> unusable
    arm = ab.ARM_CONFIGS["gann_hard_filter"]
    result = ab.simulate_portfolio(
        candidates, {"AAA": bars}, arm, NO_COST, PORT, gann_provider=lambda *a: weak_conflict
    )
    assert len(result.trades) == 1
    assert result.skip_count("skipped_gann_filter") == 0


# ---------------------------------------------------------------------------
# A2 position sizing
# ---------------------------------------------------------------------------


def test_a2_sizing_multiplies_notional_supportive_and_conflicting():
    bars = make_bars([(100, 101, 99, 100)] * 3)
    candidates = [cand("2024-06-01", action="open_long", stop_loss_pct=5.0)]
    arm = ab.ARM_CONFIGS["gann_position_sizing"]
    base = ab.simulate_portfolio(candidates, {"AAA": bars}, BASELINE, NO_COST, PORT)
    base_notional = base.trades[0].notional  # 10000 * 1% / 5% = 2000

    supportive = make_feature("support_long", confidence=0.8)  # mult = 1 + 0.25*0.8 = 1.2
    sized_up = ab.simulate_portfolio(
        candidates, {"AAA": bars}, arm, NO_COST, PORT, gann_provider=lambda *a: supportive
    )
    assert sized_up.trades[0].notional == pytest.approx(base_notional * 1.2)
    assert sized_up.trades[0].qty == pytest.approx(base.trades[0].qty * 1.2)
    assert sized_up.trades[0].gann_multiplier == pytest.approx(1.2)

    conflicting = make_feature("support_short", confidence=0.8)  # mult = 1 - 0.5*0.8 = 0.6
    sized_down = ab.simulate_portfolio(
        candidates, {"AAA": bars}, arm, NO_COST, PORT, gann_provider=lambda *a: conflicting
    )
    assert sized_down.trades[0].notional == pytest.approx(base_notional * 0.6)
    assert sized_down.trades[0].gann_multiplier == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# A1 low-weight scoring reorders same-day allocation
# ---------------------------------------------------------------------------


def test_a1_scoring_alignment_reorders_allocation():
    bars = make_bars([(100, 101, 99, 100)] * 3)
    klines = {"AAA": bars, "BBB": bars}
    # AAA raw score 1.00, BBB raw 0.95; supportive Gann on BBB lifts it: 0.95*1.1=1.045 > 1.0
    candidates = [
        cand("2024-06-01", symbol="AAA", score=1.00),
        cand("2024-06-01", symbol="BBB", score=0.95),
    ]

    def provider(symbol, bars_, asof_index):
        return make_feature("support_long", confidence=0.8) if symbol == "BBB" else None

    port = ab.PortfolioConfig(max_concurrent=1)
    arm = ab.ARM_CONFIGS["gann_low_weight_scoring"]
    result = ab.simulate_portfolio(candidates, klines, arm, NO_COST, port, gann_provider=provider)
    assert [t.symbol for t in result.trades] == ["BBB"]
    assert result.skip_count("skipped_full") == 1
    # baseline keeps the raw ordering
    base = ab.simulate_portfolio(candidates, klines, BASELINE, NO_COST, port)
    assert [t.symbol for t in base.trades] == ["AAA"]


# ---------------------------------------------------------------------------
# OOS window split
# ---------------------------------------------------------------------------


def test_oos_split_by_entry_date():
    bars = make_bars([(100, 101, 99, 100)] * 8, start="2024-12-29")
    klines = {"AAA": bars, "BBB": bars}
    candidates = [
        cand("2024-12-29", symbol="AAA", max_hold_days=2),  # fills 2024-12-30 (train)
        cand("2025-01-02", symbol="BBB", max_hold_days=2),  # fills 2025-01-03 (validation)
    ]
    result = ab.simulate_portfolio(candidates, klines, BASELINE, NO_COST, PORT)
    train = ab.compute_metrics(result, "train", None, "2024-12-31")
    validation = ab.compute_metrics(result, "validation", "2025-01-01", None)
    assert train["trade_count"] == 1
    assert validation["trade_count"] == 1
    train_symbols = [t.symbol for t in result.trades if t.entry_date <= "2024-12-31"]
    assert train_symbols == ["AAA"]
    assert train["funding_omitted"] is True
    assert validation["funding_omitted"] is True


# ---------------------------------------------------------------------------
# demo candidate generator
# ---------------------------------------------------------------------------


def test_demo_generator_sma_cross():
    # 21 flat closes at 100 (close == sma, no cross), then a jump to 110 -> open_long,
    # later a drop to 90 -> open_short
    closes = [100.0] * 21 + [110.0] * 5 + [90.0] * 3
    bars = make_bars([(c, c, c, c) for c in closes], start="2024-01-01")
    candidates = ab.generate_demo_candidates({"AAA": bars}, sma_period=20)
    assert candidates, "expected demo candidates from SMA crosses"
    first = candidates[0]
    assert first.action == "open_long"
    assert first.date == bars[21]["date"]  # the jump bar
    # sma at the jump bar = (19*100 + 110) / 20 = 100.5 -> score = 9.5 / 100.5
    assert first.score == pytest.approx(9.5 / 100.5)
    assert any(c.action == "open_short" for c in candidates)


# ---------------------------------------------------------------------------
# candidates file schema validation
# ---------------------------------------------------------------------------


def test_load_candidates_validates_schema(tmp_path):
    good = {"date": "2025-01-02", "symbol": "btcusdt", "action": "open_long", "score": 1.5}
    path = tmp_path / "cands.jsonl"
    invalid = {"date": "2025-01-03", "symbol": "ETHUSDT", "action": "buy", "score": 1}
    path.write_text(
        "%s\n%s\n" % (json.dumps(good), json.dumps(invalid)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2.*'action'"):
        ab.load_candidates(path)

    path.write_text(json.dumps(good) + "\n", encoding="utf-8")
    loaded = ab.load_candidates(path, default_stop_pct=2.5, default_tp_pct=5.0, default_max_hold_days=7)
    assert loaded[0].symbol == "BTCUSDT"
    assert loaded[0].stop_loss_pct == 2.5
    assert loaded[0].take_profit_pct == 5.0
    assert loaded[0].max_hold_days == 7
