from __future__ import annotations

from pathlib import Path

from scripts.summarize_trade_activity import summarize


def test_trade_activity_summary_includes_macro_trend_line(tmp_path: Path) -> None:
    report = {
        "run_id": "run-1",
        "execution_bridge_summary": {"request_mode": "testnet_order", "signal_source": "shadow_paper"},
        "paper_rebalance": {"shadow_paper": {"entries": [], "exits": [], "unrealized_pnl_usdt": 0.0, "realized_pnl_usdt": 0.0}},
        "macro_trend": {
            "btc_trend": "bearish",
            "btc_distance_pct": -2.3,
            "btc_ma7_slope_pct_3d": -1.2,
            "eth_trend": "neutral",
            "eth_distance_pct": 0.4,
            "eth_ma7_slope_pct_3d": 0.1,
            "macro_direction": "bearish",
            "macro_confidence": "medium",
            "source": "live",
        },
    }

    text = summarize(report, {"positions": [], "shadow_positions": [], "shadow_journal": []}, tmp_path)

    assert "宏观方向: BTC bearish" in text
    assert "方向=bearish" in text
    assert "置信=medium" in text
    assert "口径=UTC日线" in text
