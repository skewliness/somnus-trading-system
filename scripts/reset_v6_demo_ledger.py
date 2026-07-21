from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.v6_execution import V6BinanceClient, V6ExecutionConfig


STATE_DIR = ROOT / "data" / "trading_system" / "v6_execution"
STATE_FILE = STATE_DIR / "state.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def signed_params() -> dict[str, str]:
    return {"timestamp": str(int(time.time() * 1000)), "recvWindow": "300000"}


def nonzero_positions(client: V6BinanceClient) -> list[dict[str, Any]]:
    payload = client.signed_request("GET", "/fapi/v2/positionRisk", signed_params())
    rows = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            quantity = abs(float(row.get("positionAmt") or 0.0))
        except (TypeError, ValueError):
            quantity = 0.0
        if quantity <= 0:
            continue
        result.append(
            {
                "symbol": str(row.get("symbol") or ""),
                "position_side": str(row.get("positionSide") or ""),
                "position_amt": str(row.get("positionAmt") or "0"),
                "entry_price": str(row.get("entryPrice") or "0"),
            }
        )
    return result


def regular_open_orders(client: V6BinanceClient) -> list[dict[str, Any]]:
    payload = client.signed_request("GET", "/fapi/v1/openOrders", signed_params())
    rows = payload if isinstance(payload, list) else []
    return [
        {
            "symbol": str(row.get("symbol") or ""),
            "order_id": str(row.get("orderId") or ""),
            "type": str(row.get("type") or ""),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def algo_open_orders(client: V6BinanceClient) -> tuple[list[dict[str, Any]], list[str]]:
    found: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for endpoint in ("/fapi/v1/openAlgoOrders", "/fapi/v1/algoOrders/openOrders"):
        try:
            payload = client.signed_request("GET", endpoint, signed_params())
        except Exception as exc:
            errors.append(f"{endpoint}: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        rows = payload.get("orders", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            identity = str(row.get("algoId") or row.get("orderId") or json.dumps(row, sort_keys=True))
            found[identity] = {
                "symbol": str(row.get("symbol") or ""),
                "algo_id": str(row.get("algoId") or row.get("orderId") or ""),
                "type": str(row.get("type") or row.get("orderType") or ""),
            }
    return list(found.values()), errors


def state_shape(state: dict[str, Any]) -> dict[str, Any]:
    shape: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, (list, dict, str)):
            shape[key] = {"type": type(value).__name__, "size": len(value)}
        else:
            shape[key] = {"type": type(value).__name__, "value": value}
    return shape


def read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".reset.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def backup_runtime_files(stamp: str) -> Path:
    backup_dir = STATE_DIR / "reset_backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "state.json",
        "ledger_reconcile.jsonl",
        "ledger_reconcile_latest.json",
        "closed_trade_review_queue.jsonl",
        "latest_run_summary.json",
    ):
        source = STATE_DIR / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)
    return backup_dir


def reset_state(state: dict[str, Any], stamp: str) -> dict[str, Any]:
    reset = dict(state)
    reset["schema_version"] = int(reset.get("schema_version") or 1)
    reset["engine"] = "v6_execution"
    reset["positions"] = []
    reset["position_lots"] = []
    reset["journal"] = []
    reset["entry_cooldowns"] = {}
    reset["manual_flatten_history"] = []
    reset.pop("updated_at_manual_flatten", None)
    reset["open_skipped_history"] = []
    reset["v5_profit_reentry_guards"] = {}
    reset["mixed_structure_probability_fallback_guards"] = {}
    reset["cross_timeframe_macro_observations"] = {}
    reset["last_ledger_reconcile"] = {
        "status": "account_reset_flat",
        "checked_at": now_iso(),
        "actions": [],
    }
    reset["updated_at"] = now_iso()
    reset["account_reset"] = {
        "reset_at": now_iso(),
        "reason": "binance_demo_account_reset",
        "backup_stamp": stamp,
    }
    return reset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    config = V6ExecutionConfig.from_env()
    client = V6BinanceClient(config)
    try:
        positions = nonzero_positions(client)
        regular = regular_open_orders(client)
        algos, algo_errors = algo_open_orders(client)
    finally:
        client.close()

    state = read_state()
    report: dict[str, Any] = {
        "checked_at": now_iso(),
        "order_mode": config.order_mode,
        "order_host": V6BinanceClient._url_host(client.order_base_url()),
        "exchange_positions": positions,
        "regular_open_orders": regular,
        "algo_open_orders": algos,
        "algo_query_errors": algo_errors,
        "state_before": state_shape(state),
        "reset_requested": bool(args.reset),
    }

    if not args.reset:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if positions or regular or algos:
        report["reset_status"] = "blocked_exchange_not_empty"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_runtime_files(stamp)
    reset = reset_state(state, stamp)
    write_json_atomic(STATE_FILE, reset)

    (STATE_DIR / "ledger_reconcile.jsonl").write_text("", encoding="utf-8")
    write_json_atomic(
        STATE_DIR / "ledger_reconcile_latest.json",
        {
            "status": "account_reset_flat",
            "checked_at": now_iso(),
            "exchange_positions": [],
            "local_positions": [],
            "actions": [],
            "reason": "binance_demo_account_reset",
        },
    )
    (STATE_DIR / "closed_trade_review_queue.jsonl").write_text("", encoding="utf-8")

    report["reset_status"] = "complete"
    report["backup_dir"] = str(backup_dir)
    report["state_after"] = state_shape(reset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
