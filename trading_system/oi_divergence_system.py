from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from trading_system.config import StrategyConfig
from trading_system.execution import PaperExecutionEngine
from trading_system.models import MarketSnapshot
from trading_system.persistence import atomic_write_state, load_state
from trading_system.self_train_dataset import append_new_samples, collect_closed_trade_samples

STRATEGY_ID = "oi_divergence_independent"


class OIDivergencePaperSystem:
    """Independent paper engine for OI/price divergence probes.

    It deliberately uses a separate state file and separate replay datasets so the
    OI setup can be scored, blocked, and optimized without contaminating the main
    paper execution journal.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.engine = PaperExecutionEngine(replace(config, state_file=config.oi_divergence_state_file))

    def rebalance(self, snapshots: list[MarketSnapshot], captured_at: str | None = None) -> dict:
        captured_at = captured_at or datetime.now(timezone.utc).isoformat()
        oi_snapshots = [snapshot for snapshot in snapshots if snapshot.setup_type == "oi_divergence_long_candidate"]
        result = self.engine.rebalance(oi_snapshots)
        result = self._suppress_same_cycle_reentry(result)
        report = {
            "strategy_id": STRATEGY_ID,
            "captured_at": captured_at,
            "entries": result.get("entries", []),
            "exits": result.get("exits", []),
            "positions": result.get("positions", []),
            "candidate_count": len(oi_snapshots),
            "top_ranked_symbols": [snapshot.to_dict() for snapshot in oi_snapshots],
            "candidate_basket": [self._candidate_row(snapshot, result) for snapshot in oi_snapshots],
            "oi_divergence_candidates": [self._oi_row(snapshot) for snapshot in oi_snapshots],
            "paper_rebalance": result,
        }
        sample_summary = self._write_training_samples(report, captured_at)
        report["sample_collection_summary"] = sample_summary
        return report

    def _write_training_samples(self, report: dict, captured_at: str) -> dict:
        state = self._load_state()
        closed_samples, _context = collect_closed_trade_samples(state, report, captured_at=captured_at)
        new_closed = [self._with_strategy_id(row) for row in closed_samples]
        added_closed = append_new_samples(self.config.oi_divergence_closed_trade_dataset_file, new_closed) if new_closed else 0
        decision_samples = [self._decision_sample(row, idx, report, captured_at) for idx, row in enumerate(report["candidate_basket"])]
        added_decisions = append_new_samples(self.config.oi_divergence_decision_dataset_file, decision_samples) if decision_samples else 0
        return {
            "strategy_id": STRATEGY_ID,
            "closed_samples_added": added_closed,
            "decision_samples_added": added_decisions,
            "closed_trade_dataset_file": str(self.config.oi_divergence_closed_trade_dataset_file),
            "decision_dataset_file": str(self.config.oi_divergence_decision_dataset_file),
        }

    def _load_state(self) -> dict:
        if not self.config.oi_divergence_state_file.exists():
            return {"schema_version": self.config.state_schema_version, "positions": [], "journal": [], "learning_journal": []}
        return load_state(self.config.oi_divergence_state_file)

    def _suppress_same_cycle_reentry(self, result: dict) -> dict:
        exited_symbols = {str(item.get("symbol") or "") for item in result.get("exits", [])}
        if not exited_symbols:
            return result
        entries = [item for item in result.get("entries", []) if str(item.get("symbol") or "") not in exited_symbols]
        positions = [item for item in result.get("positions", []) if str(item.get("symbol") or "") not in exited_symbols]
        result = {**result, "entries": entries, "positions": positions}
        state = self._load_state()
        state["positions"] = [item for item in state.get("positions", []) if str(item.get("symbol") or "") not in exited_symbols]
        state["journal"] = [
            item
            for item in state.get("journal", [])
            if not (str(item.get("symbol") or "") in exited_symbols and str(item.get("action") or "") == "open_long")
        ]
        self.config.oi_divergence_state_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_state(self.config.oi_divergence_state_file, state)
        return result

    @staticmethod
    def _with_strategy_id(row: dict) -> dict:
        return {**row, "strategy_id": STRATEGY_ID}

    @staticmethod
    def _oi_row(snapshot: MarketSnapshot) -> dict:
        return {
            "symbol": snapshot.symbol,
            "setup_type": snapshot.setup_type,
            "oi_price_divergence_score": snapshot.oi_price_divergence_score,
            "oi_change_pct_4h": snapshot.oi_change_pct_4h,
            "price_change_pct_4h": snapshot.price_change_pct_4h,
            "taker_buy_sell_ratio": snapshot.taker_buy_sell_ratio,
            "timing_label": "oi_up_price_flat",
        }

    @staticmethod
    def _candidate_row(snapshot: MarketSnapshot, result: dict) -> dict:
        entry_symbols = {str(item.get("symbol") or "") for item in result.get("entries", [])}
        return {
            "symbol": snapshot.symbol,
            "score": snapshot.score,
            "risk_band": "oi_divergence",
            "risk_grade": "probe",
            "execution_status": "entered" if snapshot.symbol in entry_symbols else "watch_only",
            "setup_type": snapshot.setup_type,
            "setup_quality": snapshot.setup_quality,
            "oi_price_divergence_score": snapshot.oi_price_divergence_score,
            "oi_change_pct_4h": snapshot.oi_change_pct_4h,
            "price_change_pct_4h": snapshot.price_change_pct_4h,
            "sentiment_score": snapshot.sentiment_score,
        }

    @staticmethod
    def _decision_sample(row: dict, index: int, report: dict, captured_at: str) -> dict:
        entry_symbols = {str(item.get("symbol") or "") for item in report.get("entries", [])}
        symbol = str(row.get("symbol") or "")
        selected = symbol in entry_symbols
        return {
            "sample_key": f"{STRATEGY_ID}|{captured_at}|{index}|{symbol}|{row.get('execution_status') or ''}",
            "captured_at": captured_at,
            "sample_type": "oi_divergence_decision",
            "strategy_id": STRATEGY_ID,
            "symbol": symbol,
            "rank": index + 1,
            "score": float(row.get("score") or 0.0),
            "setup_type": row.get("setup_type") or "",
            "execution_status": row.get("execution_status") or "watch_only",
            "decision_label": "entered" if selected else "blocked",
            "selected_for_entry": selected,
            "risk_band": row.get("risk_band") or "oi_divergence",
            "setup_quality": float(row.get("setup_quality") or 0.0),
            "oi_price_divergence_score": float(row.get("oi_price_divergence_score") or 0.0),
            "oi_change_pct_4h": float(row.get("oi_change_pct_4h") or 0.0),
            "price_change_pct_4h": float(row.get("price_change_pct_4h") or 0.0),
            "sentiment_score": float(row.get("sentiment_score") or 0.0),
        }
