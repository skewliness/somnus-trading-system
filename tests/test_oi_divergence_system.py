from pathlib import Path

from trading_system.config import StrategyConfig
from trading_system.models import MarketSnapshot
from trading_system.oi_divergence_system import OIDivergencePaperSystem


def _snapshot(symbol: str = "PEPEUSDT", price: float = 1.0, score: float = 84.0, setup_type: str = "oi_divergence_long_candidate") -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        mark_price=price,
        price_change_pct_4h=1.0,
        price_change_pct_24h=8.0,
        quote_volume_usdt_24h=80_000_000,
        funding_rate=0.0002,
        open_interest_usdt=25_000_000,
        oi_change_pct_4h=18.0,
        taker_buy_sell_ratio=1.08,
        listing_days=40,
        sentiment_score=80,
        sentiment_mentions=24,
        sentiment_source="square_feed,manual_distill",
        setup_type=setup_type,
        setup_quality=88.0,
        oi_price_divergence_score=210.0,
        score=score,
        reasons=["oi_divergence_probe"],
    )


def test_oi_divergence_system_uses_independent_state_and_only_trades_oi_setups(tmp_path: Path) -> None:
    config = StrategyConfig(
        state_file=tmp_path / "main_paper_state.json",
        oi_divergence_state_file=tmp_path / "oi_state.json",
        oi_divergence_closed_trade_dataset_file=tmp_path / "oi_closed.jsonl",
        oi_divergence_decision_dataset_file=tmp_path / "oi_decisions.jsonl",
    )
    system = OIDivergencePaperSystem(config)

    report = system.rebalance([
        _snapshot("PEPEUSDT"),
        _snapshot("AIXBTUSDT", setup_type="base_breakout_confirmed"),
    ])

    assert config.state_file.exists() is False
    assert config.oi_divergence_state_file.exists() is True
    assert [entry["symbol"] for entry in report["entries"]] == ["PEPEUSDT"]
    assert [position["symbol"] for position in report["positions"]] == ["PEPEUSDT"]
    assert report["strategy_id"] == "oi_divergence_independent"


def test_oi_divergence_system_writes_replay_samples_with_strategy_id(tmp_path: Path) -> None:
    config = StrategyConfig(
        oi_divergence_state_file=tmp_path / "oi_state.json",
        oi_divergence_closed_trade_dataset_file=tmp_path / "oi_closed.jsonl",
        oi_divergence_decision_dataset_file=tmp_path / "oi_decisions.jsonl",
        oi_divergence_take_profit_pct=0.05,
    )
    system = OIDivergencePaperSystem(config)

    system.rebalance([_snapshot("PEPEUSDT", price=1.0)])
    report = system.rebalance([_snapshot("PEPEUSDT", price=1.06, score=90.0)])

    assert len(report["exits"]) == 1
    assert report["positions"] == []
    closed_rows = config.oi_divergence_closed_trade_dataset_file.read_text(encoding="utf-8").strip().splitlines()
    decision_rows = config.oi_divergence_decision_dataset_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(closed_rows) == 1
    assert '"strategy_id": "oi_divergence_independent"' in closed_rows[0]
    assert '"setup_type": "oi_divergence_long_candidate"' in closed_rows[0]
    assert len(decision_rows) >= 1
    assert all('"strategy_id": "oi_divergence_independent"' in row for row in decision_rows)
