from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.v6_execution import V6BinanceClient, V6ExecutionConfig, canonical_order_mode


def parse_multi_assets_mode(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    value = payload.get("multiAssetsMargin")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def margin_mode_is_compatible(*, multi_assets_mode: bool, configured_margin_type: str) -> bool:
    margin_type = str(configured_margin_type or "").strip().upper()
    return not (multi_assets_mode and margin_type == "ISOLATED")


def account_asset_mode_matches(*, multi_assets_mode: bool, configured_account_asset_mode: str) -> bool:
    raw = str(configured_account_asset_mode or "SINGLE_ASSET").strip().upper().replace("-", "_")
    configured_multi_assets = raw in {"MULTI", "MULTI_ASSET", "MULTI_ASSETS"}
    return configured_multi_assets == multi_assets_mode


def main() -> int:
    config = V6ExecutionConfig.from_env()
    if canonical_order_mode(config.order_mode) != "live_order":
        print(json.dumps({"status": "blocked", "reason": "TS_V6_ORDER_MODE must be live_order"}))
        return 2

    try:
        config.validate_runtime_isolation()
        client = V6BinanceClient(config)
        try:
            account = client.signed_request("GET", "/fapi/v2/account", client.signed_params())
            position_mode = client.signed_request("GET", "/fapi/v1/positionSide/dual", client.signed_params())
            open_orders = client.signed_request("GET", "/fapi/v1/openOrders", client.signed_params())
            multi_assets = client.signed_request("GET", "/fapi/v1/multiAssetsMargin", client.signed_params())
        finally:
            client.close()
    except Exception as exc:
        print(json.dumps({"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2

    positions = account.get("positions") if isinstance(account, dict) else []
    positions = positions if isinstance(positions, list) else []
    nonzero_positions = [
        row
        for row in positions
        if isinstance(row, dict) and abs(float(row.get("positionAmt") or 0.0)) > 0.0
    ]
    assets = account.get("assets") if isinstance(account, dict) else []
    assets = assets if isinstance(assets, list) else []
    usdt = next((row for row in assets if isinstance(row, dict) and row.get("asset") == "USDT"), {})
    available_balance = float(usdt.get("availableBalance") or 0.0)
    required_initial_margin = float(config.base_notional_usdt) / max(1, int(config.default_leverage))
    hedge_mode = bool(position_mode.get("dualSidePosition")) if isinstance(position_mode, dict) else False
    expected_mode = str(config.position_mode or "auto").strip().lower().replace("-", "_")
    mode_matches = expected_mode in {"", "auto"} or (expected_mode in {"hedge", "dual"} and hedge_mode) or (
        expected_mode in {"oneway", "one_way", "both"} and not hedge_mode
    )
    can_trade = bool(account.get("canTrade")) if isinstance(account, dict) else False
    multi_assets_mode = parse_multi_assets_mode(multi_assets)
    configured_margin_type = str(config.margin_type or "").strip().upper()
    configured_account_asset_mode = str(config.account_asset_mode or "SINGLE_ASSET").strip().upper().replace("-", "_")
    asset_mode_matches = account_asset_mode_matches(
        multi_assets_mode=multi_assets_mode,
        configured_account_asset_mode=configured_account_asset_mode,
    )
    margin_mode_compatible = margin_mode_is_compatible(
        multi_assets_mode=multi_assets_mode,
        configured_margin_type=configured_margin_type,
    )
    blocked_reasons = []
    if not can_trade:
        blocked_reasons.append("Binance account is not allowed to trade")
    if not mode_matches:
        blocked_reasons.append("Binance position mode does not match TS_V6_POSITION_MODE")
    if available_balance < required_initial_margin:
        blocked_reasons.append("Available balance is below the estimated initial margin")
    if not margin_mode_compatible:
        blocked_reasons.append("ISOLATED margin is unavailable while Binance Multi-Assets mode is enabled")
    if not asset_mode_matches:
        blocked_reasons.append("Binance account asset mode does not match TS_V6_ACCOUNT_ASSET_MODE")
    ready = not blocked_reasons
    report = {
        "status": "ready" if ready else "blocked",
        "order_mode": config.order_mode,
        "state_dir": str(config.state_dir),
        "can_trade": can_trade,
        "position_mode": "hedge" if hedge_mode else "one_way",
        "configured_position_mode": config.position_mode,
        "position_mode_matches": mode_matches,
        "multi_assets_mode": multi_assets_mode,
        "configured_account_asset_mode": configured_account_asset_mode,
        "account_asset_mode_matches": asset_mode_matches,
        "configured_margin_type": configured_margin_type,
        "margin_mode_compatible": margin_mode_compatible,
        "nonzero_position_count": len(nonzero_positions),
        "open_order_count": len(open_orders) if isinstance(open_orders, list) else 0,
        "available_balance_usdt": round(available_balance, 8),
        "base_notional_usdt": float(config.base_notional_usdt),
        "leverage": int(config.default_leverage),
        "estimated_initial_margin_usdt": round(required_initial_margin, 8),
        "blocked_reasons": blocked_reasons,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
