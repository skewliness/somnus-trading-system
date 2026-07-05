from __future__ import annotations

from trading_system.binance import BinanceMarketDataProvider, ReplayMarketDataProvider
from trading_system.config import StrategyConfig
from trading_system.execution import PaperExecutionEngine
from trading_system.models import MarketSnapshot
from trading_system.persistence import atomic_write_json
from trading_system.scoring import score_snapshot
from trading_system.sentiment import SentimentBook


class TradingSystem:
    def __init__(self, config: StrategyConfig, mode: str = "live") -> None:
        self.config = config
        self.mode = mode
        self.sentiment = SentimentBook.from_csv(config.sentiment_file)
        if mode == "live":
            self.provider = BinanceMarketDataProvider(config)
        elif mode == "replay":
            self.provider = ReplayMarketDataProvider(config.replay_file, config)
        else:
            raise ValueError(f"unsupported mode: {mode}")

    def scan(self, regime: dict | None = None, predictive_model: dict | None = None, hf_quality_lookup: dict | None = None) -> list[MarketSnapshot]:
        snapshots = self.provider.collect(self.sentiment)
        scored = [score_snapshot(
            snapshot, self.config, regime=regime, predictive_model=predictive_model,
            hf_quality=(hf_quality_lookup or {}).get(snapshot.symbol.upper()),
        ) for snapshot in snapshots]
        scored.sort(key=lambda item: item.score, reverse=True)
        self._write_report(scored)
        return scored

    def rebalance(self, regime: dict | None = None, predictive_model: dict | None = None, hf_quality_lookup: dict | None = None) -> dict[str, list[dict]]:
        scored = self.scan(regime=regime, predictive_model=predictive_model, hf_quality_lookup=hf_quality_lookup)
        executor = PaperExecutionEngine(self.config, mark_price_fetcher=self._fetch_mark_price)
        return executor.rebalance(scored, regime=regime, hf_quality_lookup=hf_quality_lookup, predictive_model=predictive_model)

    def _fetch_mark_price(self, symbol: str) -> float | None:
        payload = self.provider.client.get_json("/fapi/v1/premiumIndex", {"symbol": symbol})
        if isinstance(payload, dict):
            return float(payload.get("markPrice") or 0.0) or None
        return None

    def _write_report(self, snapshots: list[MarketSnapshot]) -> None:
        self.config.report_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": self.mode,
            "symbols": [snapshot.to_dict() for snapshot in snapshots],
        }
        atomic_write_json(self.config.report_file, payload)
