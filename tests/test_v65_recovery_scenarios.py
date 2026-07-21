from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

from scripts.run_v6_ledger_reconcile import run_reconcile
from trading_system.v6_execution import V6ExecutionConfig, V6ExecutionEngine


def exchange_row(symbol: str = "BTCUSDT", quantity: str = "1.5") -> dict[str, object]:
    return {
        "symbol": symbol,
        "positionAmt": quantity,
        "positionSide": "BOTH",
        "entryPrice": "100",
        "markPrice": "101",
        "notional": str(abs(float(quantity)) * 101.0),
        "leverage": "8",
        "marginType": "cross",
    }


class ReconcileClient:
    def __init__(self, *, open_stops: list[dict[str, object]] | None = None) -> None:
        self.open_stops = list(open_stops or [])
        self.submitted: list[tuple[str, str | None]] = []

    @staticmethod
    def is_dust_exchange_snapshot(snapshot: dict[str, object]) -> bool:
        return False

    @staticmethod
    def exchange_snapshot_notional(snapshot: dict[str, object]) -> float:
        return abs(float(snapshot.get("notional") or 0.0))

    @staticmethod
    def quantity_rules(symbol: str) -> tuple[float, float]:
        return 0.001, 0.001

    @staticmethod
    def resolve_price(symbol: str, price: float) -> str:
        return f"{price:g}"

    def signed_request(self, method: str, endpoint: str, params: dict[str, object]) -> list[dict[str, object]]:
        assert method == "GET"
        assert endpoint == "/fapi/v1/openAlgoOrders"
        return self.open_stops

    @staticmethod
    def signed_params() -> dict[str, object]:
        return {}

    @staticmethod
    def protective_stop_precheck(intent) -> dict[str, object]:
        return {"checked": True, "would_immediately_trigger": False}

    def submit_protective_stop(self, intent, quantity: str | None = None) -> dict[str, object]:
        self.submitted.append((intent.symbol, quantity))
        return {
            "status": "submitted",
            "response": {
                "algoId": 9001,
                "clientAlgoId": "recovered-stop",
                "algoStatus": "NEW",
            },
        }

    @staticmethod
    def close() -> None:
        return None


def make_engine(tmp_path: Path) -> V6ExecutionEngine:
    config = V6ExecutionConfig(
        symbols=["BTCUSDT"],
        order_mode="testnet_order",
        state_dir=tmp_path,
        state_file=tmp_path / "state.json",
        signals_file=tmp_path / "signals.jsonl",
        orders_file=tmp_path / "orders.jsonl",
        wal_file=tmp_path / "execution_wal.jsonl",
        latest_summary_file=tmp_path / "latest_run_summary.json",
        closed_trade_review_queue_file=tmp_path / "closed_trade_review_queue.jsonl",
        post_trade_training_samples_file=tmp_path / "post_trade_training_samples.jsonl",
    )
    engine = V6ExecutionEngine(config)
    engine.client.close()
    return engine


def test_exchange_position_missing_locally_is_recovered_with_a_lot(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine.client = ReconcileClient()  # type: ignore[assignment]
    state = {"positions": [], "journal": []}

    result = engine.reconcile_state_with_exchange(
        state,
        actual_rows=[exchange_row()],
        sync_exchange_side_effects=False,
    )

    assert result is not None
    assert result["status"] == "ok"
    assert result["recovered"] == ["BTCUSDT:long"]
    assert len(state["positions"]) == 1
    position = state["positions"][0]
    assert position["risk_status"] == "missing_local_lot_recovered_from_binance_positionRisk"
    assert position["position_qty"] == 1.5
    assert len(position["lots"]) == 1
    assert position["lots"][0]["remaining_quantity"] == 1.5


def test_missing_exchange_protective_stop_is_recreated_and_recorded(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    client = ReconcileClient(open_stops=[])
    engine.client = client  # type: ignore[assignment]
    state = {
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "entry_price": 100.0,
                "current_price": 101.0,
                "stop_loss_price": 97.0,
                "take_profit_price": 110.0,
                "position_qty": 1.5,
                "remaining_quantity": 1.5,
                "notional_usdt": 151.5,
                "leverage": 8,
                "risk_plan": {},
            }
        ],
        "journal": [],
    }

    result = engine.reconcile_state_with_exchange(state, actual_rows=[exchange_row()])

    assert result is not None
    assert result["status"] == "ok"
    assert client.submitted == [("BTCUSDT", "1.5")]
    assert result["protective_stop_sync"][0]["action"] == "submit_protective_stop"
    assert state["positions"][0]["protective_stop_order_id"] == 9001
    assert state["positions"][0]["protective_stop_submit_status"] == "submitted"


def test_ledger_persists_final_mismatch_status_instead_of_premature_ok(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state_file = state_dir / "state.json"
    state_file.write_text('{"positions":[]}', encoding="utf-8")
    config = V6ExecutionConfig(
        symbols=["BTCUSDT"],
        order_mode="testnet_order",
        state_dir=state_dir,
        state_file=state_file,
    )

    class ClientStub:
        @staticmethod
        def actual_position_rows() -> list[dict[str, object]]:
            return []

    class EngineStub:
        client = ClientStub()

        @staticmethod
        def reconcile_state_with_exchange(state, *, actual_rows, sync_exchange_side_effects):
            assert sync_exchange_side_effects is False
            return {"status": "ok"}

        @staticmethod
        def close() -> None:
            return None

    @contextmanager
    def execution_lock(_config):
        yield

    unresolved = [{"key": "BTCUSDT:long", "type": "quantity_mismatch"}]
    with (
        patch("scripts.run_v6_ledger_reconcile.V6ExecutionConfig.from_env", return_value=config),
        patch("scripts.run_v6_ledger_reconcile.V6ExecutionEngine", return_value=EngineStub()),
        patch("scripts.run_v6_ledger_reconcile.v6_execution_lock_context", execution_lock),
        patch("scripts.run_v6_ledger_reconcile.summarize_exchange_rows", return_value=[]),
        patch("scripts.run_v6_ledger_reconcile.summarize_local_positions", return_value=[]),
        patch("scripts.run_v6_ledger_reconcile.compare_local_exchange", side_effect=[unresolved, unresolved]),
    ):
        report = run_reconcile(check_only=False, tolerance=0.0)

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert report["status"] == "mismatch"
    assert persisted["last_ledger_reconcile"]["status"] == "mismatch"
    assert persisted["last_ledger_reconcile"]["after_mismatch_count"] == 1
