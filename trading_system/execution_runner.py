from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from functools import lru_cache
import hashlib
import hmac
import json
import logging
from pathlib import Path
import ssl
from typing import Any
from urllib import error, parse, request

import certifi

from trading_system.config import StrategyConfig
from trading_system.persistence import append_text_line, atomic_write_json, atomic_write_text, load_json
from trading_system.redaction import redact_sensitive_text


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _https_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


class BinanceExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class BinanceExecutionRunner:
    config: StrategyConfig
    _active_request_mode: str = field(default="live_order", init=False, repr=False)
    _position_mode_cache: str | None = field(default=None, init=False, repr=False)
    _exchange_symbols_cache: set[str] | None = field(default=None, init=False, repr=False)
    _invalid_symbols: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.config.execution_result_dir.mkdir(parents=True, exist_ok=True)
        self.config.execution_archive_dir.mkdir(parents=True, exist_ok=True)
        self.config.execution_wal_file.parent.mkdir(parents=True, exist_ok=True)
        self._require_credentials(self._config_request_mode())

    def execute_request_file(self, request_file: Path) -> dict[str, Any]:
        if not request_file.exists():
            raise BinanceExecutionError(f"request file not found: {request_file}")
        payload = load_json(request_file)
        request_mode = self._payload_request_mode(payload)
        self._active_request_mode = request_mode
        self._require_credentials(request_mode)
        self._validate_request_payload(payload, request_mode)
        requests_payload = payload.get("binance_requests") or []
        generated_at = payload.get("generated_at") or datetime.now(timezone.utc).isoformat()
        results: list[dict[str, Any]] = []
        for index, item in enumerate(requests_payload, start=1):
            try:
                results.append(self._execute_single(item, request_file, generated_at, index))
            except BinanceExecutionError as exc:
                error_msg = str(exc)
                if "-1121" in error_msg or "Invalid symbol" in error_msg:
                    sym = str((item.get("params") or {}).get("symbol") or "")
                    if sym:
                        self._invalid_symbols.add(sym)
                results.append(self._build_failed_result(item, request_file, generated_at, index, exc))

        summary = {
            "request_file": str(request_file),
            "generated_at": generated_at,
            "request_mode": request_mode,
            "skill_name": payload.get("skill_name", "derivatives-trading-usds-futures"),
            "status": self._rollup_status(results),
            "result_count": len(results),
            "results": results,
        }
        result_path = self.config.execution_result_dir / f"{request_file.stem}_result.json"
        atomic_write_json(result_path, summary)
        archived = self.config.execution_archive_dir / request_file.name
        atomic_write_text(archived, request_file.read_text(encoding="utf-8"))
        request_file.unlink()
        summary["result_file"] = str(result_path)
        summary["archived_request_file"] = str(archived)
        logger.info("executed binance request file: %s status=%s", request_file, summary["status"])
        return summary

    def execute_pending_requests(self, request_dir: Path | None = None) -> dict[str, Any]:
        queue_dir = request_dir or self.config.execution_request_dir
        queue_dir.mkdir(parents=True, exist_ok=True)
        request_files = sorted(queue_dir.glob("execution_request_*.json"), key=lambda item: item.stat().st_mtime)
        summaries = [self.execute_request_file(path) for path in request_files]
        return {
            "request_dir": str(queue_dir),
            "request_count": len(request_files),
            "executed_count": len(summaries),
            "summaries": summaries,
        }

    def _execute_single(self, item: dict[str, Any], request_file: Path, generated_at: str, index: int) -> dict[str, Any]:
        endpoint = str(item.get("endpoint") or "")
        method = str(item.get("method") or "POST").upper()
        params = dict(item.get("params") or {})
        meta = dict(item.get("meta") or {})
        symbol = str(params.get("symbol") or "")
        action = str(meta.get("action") or "")
        request_id = f"{request_file.stem}:{index}"
        if self._is_order_endpoint(endpoint):
            unsupported_reason = self._unsupported_symbol_reason(symbol)
            if unsupported_reason:
                if symbol:
                    self._invalid_symbols.add(symbol)
                return self._build_skipped_result(
                    item,
                    request_file,
                    generated_at,
                    index,
                    status="skipped_unsupported_symbol",
                    reason=unsupported_reason,
                )
        params = self._prepare_order_params(endpoint, params, meta)
        if self._is_order_endpoint(endpoint):
            params = self._prepare_close_order_against_position(symbol, action, params)
        if params.get("_skip_no_position"):
            logger.info("close skipped (no exchange position): %s action=%s", symbol, action)
            self._append_wal({
                "request_id": request_id,
                "stage": "skipped_no_position",
                "symbol": symbol,
                "endpoint": endpoint,
                "generated_at": generated_at,
            })
            return {
                "id": request_id,
                "symbol": symbol,
                "endpoint": endpoint,
                "method": method,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "request_mode": self._result_request_mode(endpoint),
                "meta": meta,
                "status": "submitted",
                "response": {"info": "no_exchange_position_to_close"},
                "follow_up_orders": [],
                "generated_at": generated_at,
            }
        params.setdefault("timestamp", str(int(datetime.now(timezone.utc).timestamp() * 1000)))
        params.setdefault("recvWindow", str(self.config.binance_recv_window_ms))
        self._append_wal({
            "request_id": request_id,
            "stage": "prepared",
            "symbol": symbol,
            "endpoint": endpoint,
            "method": method,
            "params": params,
            "meta": meta,
            "generated_at": generated_at,
        })
        if self._is_order_endpoint(endpoint):
            params.setdefault("newClientOrderId", self._client_order_id(request_id, symbol, action, generated_at))
        if meta.get("leverage") and action in {"open_long", "open_short"}:
            self._set_leverage(symbol, int(meta["leverage"]))
        params["timestamp"] = str(int(datetime.now(timezone.utc).timestamp() * 1000))
        follow_up_orders: list[dict[str, Any]] = []
        status = "submitted"
        try:
            response_payload = self._signed_request(method, endpoint, params)
        except BinanceExecutionError as exc:
            if self._is_binance_status_unknown(exc) and self._is_order_endpoint(endpoint):
                response_payload = self._reconcile_status_unknown_order(symbol, params, exc)
                status = str(response_payload.get("execution_status") or "status_unknown")
                self._append_wal({
                    "request_id": request_id,
                    "stage": status,
                    "symbol": symbol,
                    "endpoint": endpoint,
                    "status": status,
                    "error": str(exc),
                    "response": response_payload,
                    "generated_at": generated_at,
                })
            else:
                raise
        else:
            self._append_wal({
                "request_id": request_id,
                "stage": "submitted",
                "symbol": symbol,
                "endpoint": endpoint,
                "status": status,
                "generated_at": generated_at,
            })
        should_submit_protective_stop = (
            status == "submitted"
            and endpoint.endswith("/order")
            and meta.get("stop_loss_price")
            and action in {"open_long", "open_short", "take_profit_partial"}
        )
        if should_submit_protective_stop:
            protective_action = action if action in {"open_long", "open_short"} else "open_long"
            try:
                follow_up_orders.append(self._submit_protective_stop(symbol, protective_action, float(meta["stop_loss_price"])))
                self._append_wal({
                    "request_id": request_id,
                    "stage": "protective_stop_submitted",
                    "symbol": symbol,
                    "stop_loss_price": float(meta["stop_loss_price"]),
                    "generated_at": generated_at,
                })
            except BinanceExecutionError as exc:
                status = "submitted_unprotected"
                follow_up_orders.append({
                    "status": "failed",
                    "type": "protective_stop",
                    "symbol": symbol,
                    "error": str(exc),
                })
                self._append_wal({
                    "request_id": request_id,
                    "stage": "protective_stop_failed",
                    "symbol": symbol,
                    "stop_loss_price": float(meta["stop_loss_price"]),
                    "error": str(exc),
                    "generated_at": generated_at,
                })
                if action in {"open_long", "open_short"}:
                    emergency_close = self._emergency_close_after_protective_stop_failed(
                        symbol=symbol,
                        action=action,
                        request_id=request_id,
                        generated_at=generated_at,
                        error_text=str(exc),
                    )
                    follow_up_orders.append(emergency_close)
                    if emergency_close.get("status") == "submitted":
                        status = "closed_after_protective_stop_failed"
        result = {
            "id": request_id,
            "symbol": symbol,
            "endpoint": endpoint,
            "method": method,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "request_mode": self._result_request_mode(endpoint),
            "meta": meta,
            "status": status,
            "response": response_payload,
            "follow_up_orders": follow_up_orders,
            "generated_at": generated_at,
        }
        self._append_wal({
            "request_id": request_id,
            "stage": "committed",
            "symbol": symbol,
            "status": status,
            "generated_at": generated_at,
        })
        return result

    def _validate_request_payload(self, payload: dict[str, Any], request_mode: str) -> None:
        requests_payload = list(payload.get("binance_requests") or [])
        order_requests = [item for item in requests_payload if self._is_order_request(item)]
        if not order_requests:
            return
        risk_controls = dict(payload.get("risk_controls") or {})
        if risk_controls.get("status") == "blocked":
            violations = "; ".join(str(item) for item in risk_controls.get("violations", []))
            raise BinanceExecutionError(f"{request_mode} execution blocked by risk controls: {violations}")
        if request_mode == "live_order" and self.config.live_execution_requires_confirmation:
            if self.config.live_execution_confirmation != self.config.live_execution_confirmation_phrase:
                raise BinanceExecutionError(
                    "live execution requires TS_LIVE_EXECUTION_CONFIRMATION="
                    f"{self.config.live_execution_confirmation_phrase}"
                )
        max_intents = int(self.config.live_max_intents_per_request or 0)
        if max_intents > 0 and len(order_requests) > max_intents:
            raise BinanceExecutionError(
                f"{request_mode} execution request has {len(order_requests)} order intents; max is {max_intents}"
            )
        max_notional = float(self.config.live_max_order_notional_usdt or 0.0)
        if max_notional > 0:
            for item in order_requests:
                meta = dict(item.get("meta") or {})
                params = dict(item.get("params") or {})
                reduce_only = str(params.get("reduceOnly") or "").lower() == "true"
                # Check both meta hint AND actual params quantity to prevent bypass
                notional = float(meta.get("quantity_hint_usdt") or 0.0)
                qty = float(params.get("quantity") or 0.0)
                if notional <= 0 and qty > 0:
                    # Fallback: compute from params quantity × mark price
                    mark_price = float(meta.get("mark_price") or params.get("price") or 0.0)
                    notional = qty * mark_price
                # If we still can't determine notional but quantity is present, reject
                if not reduce_only and notional <= 0 and qty > 0:
                    symbol = str(params.get("symbol") or "")
                    raise BinanceExecutionError(
                        f"{request_mode} execution notional for {symbol} is indeterminate "
                        f"(qty={qty}, mark_price unavailable); rejected as safety measure"
                    )
                if not reduce_only and notional > max_notional:
                    symbol = str(params.get("symbol") or "")
                    raise BinanceExecutionError(
                        f"{request_mode} execution notional for {symbol} is {notional}; max is {max_notional}"
                    )
        loss_limit = float(self.config.live_max_request_loss_usdt or 0.0)
        risk_loss = float(risk_controls.get("worst_case_stop_loss_usdt") or 0.0)
        if loss_limit > 0 and risk_loss > loss_limit:
            raise BinanceExecutionError(
                f"{request_mode} execution worst-case stop loss is {risk_loss}; max is {loss_limit}"
            )

    @staticmethod
    def _is_order_request(item: dict[str, Any]) -> bool:
        endpoint = str(item.get("endpoint") or "")
        return BinanceExecutionRunner._is_order_endpoint(endpoint)

    @staticmethod
    def _is_order_endpoint(endpoint: str) -> bool:
        return endpoint in {"/fapi/v1/order", "/fapi/v1/order/test"}

    def _config_request_mode(self) -> str:
        raw = str(self.config.execution_order_mode or "auto").strip().lower()
        if raw == "auto":
            return "test_order" if self.config.execution_test_order else "live_order"
        return self._canonical_request_mode(raw)

    def _payload_request_mode(self, payload: dict[str, Any]) -> str:
        return self._canonical_request_mode(str(payload.get("request_mode") or self._config_request_mode()))

    @staticmethod
    def _canonical_request_mode(raw: str) -> str:
        aliases = {
            "test": "test_order",
            "test_order": "test_order",
            "dry_run": "test_order",
            "testnet": "testnet_order",
            "demo": "testnet_order",
            "testnet_order": "testnet_order",
            "live": "live_order",
            "live_order": "live_order",
        }
        return aliases.get(str(raw or "").strip().lower(), "test_order")

    def _result_request_mode(self, endpoint: str) -> str:
        if endpoint.endswith("/test"):
            return "test_order"
        if self._active_request_mode in {"testnet_order", "live_order"}:
            return self._active_request_mode
        return "live_order"

    def _base_url(self) -> str:
        if self._active_request_mode == "testnet_order":
            return self.config.binance_testnet_base_url
        return self.config.base_url

    def _api_key(self) -> str:
        if self._active_request_mode == "testnet_order":
            return self.config.binance_testnet_api_key
        return self.config.binance_api_key

    def _api_secret(self) -> str:
        if self._active_request_mode == "testnet_order":
            return self.config.binance_testnet_api_secret
        return self.config.binance_api_secret

    def _require_credentials(self, request_mode: str) -> None:
        self._active_request_mode = request_mode
        if request_mode == "testnet_order":
            if not self.config.binance_testnet_api_key:
                raise BinanceExecutionError("BINANCE_TESTNET_API_KEY is missing")
            if not self.config.binance_testnet_api_secret:
                raise BinanceExecutionError("BINANCE_TESTNET_API_SECRET is missing")
            return
        if not self.config.binance_api_key:
            raise BinanceExecutionError("BINANCE_API_KEY is missing")
        if not self.config.binance_api_secret:
            raise BinanceExecutionError("BINANCE_API_SECRET is missing")

    def _prepare_order_params(self, endpoint: str, params: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(params)
        symbol = str(prepared.get("symbol") or "")
        if endpoint.endswith("/order") or endpoint.endswith("/order/test"):
            position_mode = self._current_position_mode()
            if position_mode == "hedge":
                if symbol and "positionSide" not in prepared:
                    prepared["positionSide"] = self._position_side_from_action(str(meta.get("action") or ""), str(prepared.get("side") or ""))
                # Binance USD-M Futures does not accept reduceOnly in Hedge Mode.
                prepared.pop("reduceOnly", None)
            else:
                prepared.pop("positionSide", None)
            if symbol and "quantity" not in prepared:
                quantity = self._resolve_quantity(symbol, meta)
                if quantity is not None:
                    prepared["quantity"] = quantity
        return prepared

    def _ensure_symbol_supported(self, symbol: str) -> None:
        reason = self._unsupported_symbol_reason(symbol)
        if reason:
            raise BinanceExecutionError(reason)

    def _unsupported_symbol_reason(self, symbol: str) -> str:
        if not symbol:
            return "order symbol is missing"
        if symbol in self._invalid_symbols:
            return f"symbol previously rejected by Binance API: {symbol}"
        symbols = self._get_exchange_symbols()
        if symbol not in symbols:
            return f"symbol not available on Binance {self._active_request_mode} endpoint: {symbol}"
        return ""

    def _prepare_close_order_against_position(self, symbol: str, action: str, params: dict[str, Any]) -> dict[str, Any]:
        if action not in {"close_long", "close_short", "exit", "take_profit_partial"}:
            return params
        if self._active_request_mode == "test_order":
            return params
        available_qty = self._get_position_quantity_for_close(symbol, action)
        if available_qty <= 0:
            return {"_skip_no_position": True, "symbol": symbol, "action": action}
        prepared = dict(params)
        if action == "take_profit_partial":
            requested_qty = float(prepared.get("quantity") or 0.0)
            close_qty = available_qty if requested_qty <= 0 else min(requested_qty, available_qty)
        else:
            close_qty = available_qty
        step_size, min_qty = self._get_symbol_quantity_rules(symbol)
        close_qty = self._round_down_to_step(close_qty, step_size)
        if close_qty <= 0 or close_qty < min_qty:
            return {"_skip_no_position": True, "symbol": symbol, "action": action}
        prepared["quantity"] = self._format_decimal(close_qty)
        return prepared

    def _build_failed_result(
        self,
        item: dict[str, Any],
        request_file: Path,
        generated_at: str,
        index: int,
        exc: BinanceExecutionError,
    ) -> dict[str, Any]:
        params = dict(item.get("params") or {})
        meta = dict(item.get("meta") or {})
        endpoint = str(item.get("endpoint") or "")
        method = str(item.get("method") or "POST").upper()
        request_id = f"{request_file.stem}:{index}"
        self._append_wal({
            "request_id": request_id,
            "stage": "failed",
            "symbol": str(params.get("symbol") or ""),
            "endpoint": endpoint,
            "status": "failed",
            "error": str(exc),
            "generated_at": generated_at,
        })
        return {
            "id": request_id,
            "symbol": str(params.get("symbol") or ""),
            "endpoint": endpoint,
            "method": method,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "request_mode": self._result_request_mode(endpoint),
            "meta": meta,
            "request": {
                "endpoint": endpoint,
                "method": method,
                "params": params,
                "meta": meta,
            },
            "status": "failed",
            "error": str(exc),
            "response": {},
            "follow_up_orders": [],
            "generated_at": generated_at,
        }

    def _build_skipped_result(
        self,
        item: dict[str, Any],
        request_file: Path,
        generated_at: str,
        index: int,
        *,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        params = dict(item.get("params") or {})
        meta = dict(item.get("meta") or {})
        endpoint = str(item.get("endpoint") or "")
        method = str(item.get("method") or "POST").upper()
        request_id = f"{request_file.stem}:{index}"
        self._append_wal({
            "request_id": request_id,
            "stage": status,
            "symbol": str(params.get("symbol") or ""),
            "endpoint": endpoint,
            "status": status,
            "reason": reason,
            "generated_at": generated_at,
        })
        return {
            "id": request_id,
            "symbol": str(params.get("symbol") or ""),
            "endpoint": endpoint,
            "method": method,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "request_mode": self._result_request_mode(endpoint),
            "meta": meta,
            "request": {
                "endpoint": endpoint,
                "method": method,
                "params": params,
                "meta": meta,
            },
            "status": status,
            "reason": reason,
            "response": {"info": reason},
            "follow_up_orders": [],
            "generated_at": generated_at,
        }

    def _resolve_quantity(self, symbol: str, meta: dict[str, Any]) -> str | None:
        notional = float(meta.get("quantity_hint_usdt") or 0)
        if not symbol or notional <= 0:
            return None
        mark_price = self._get_mark_price(symbol)
        step_size, min_qty = self._get_symbol_quantity_rules(symbol)
        if mark_price <= 0 or step_size <= 0:
            return None
        raw_qty = notional / mark_price
        normalized = self._round_down_to_step(raw_qty, step_size)
        if normalized < min_qty:
            normalized = self._round_down_to_step(min_qty, step_size)
        if normalized <= 0:
            return None
        return self._format_decimal(normalized)

    def _position_side_from_action(self, action: str, side: str) -> str:
        order_side = str(side or "").upper()
        if action in {"open_short", "close_short"}:
            return "SHORT"
        if action in {"open_long", "close_long", "exit"}:
            return "LONG"
        if action == "take_profit_partial":
            return "SHORT" if order_side == "BUY" else "LONG"
        return "LONG" if order_side == "BUY" else "SHORT"

    def _current_position_mode(self) -> str:
        configured = str(self.config.binance_position_mode or "auto").strip().lower().replace("-", "_")
        aliases = {
            "oneway": "one_way",
            "one_way": "one_way",
            "oneway_mode": "one_way",
            "one_way_mode": "one_way",
            "both": "one_way",
            "hedge": "hedge",
            "hedge_mode": "hedge",
            "dual": "hedge",
            "dual_side": "hedge",
        }
        if configured in aliases:
            return aliases[configured]
        if self._position_mode_cache:
            return self._position_mode_cache
        params = {
            "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            "recvWindow": str(self.config.binance_recv_window_ms),
        }
        payload = self._signed_request("GET", "/fapi/v1/positionSide/dual", params)
        raw = payload.get("dualSidePosition")
        dual_side = raw if isinstance(raw, bool) else str(raw).strip().lower() == "true"
        self._position_mode_cache = "hedge" if dual_side else "one_way"
        logger.info("Binance futures position mode detected: %s", self._position_mode_cache)
        return self._position_mode_cache

    def _get_position_quantity_for_close(self, symbol: str, action: str) -> float:
        if not symbol:
            return 0.0
        params = {
            "symbol": symbol,
            "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            "recvWindow": str(self.config.binance_recv_window_ms),
        }
        payload = self._signed_request("GET", "/fapi/v2/positionRisk", params)
        positions = payload if isinstance(payload, list) else [payload]
        mode = self._current_position_mode()
        wants_short = action == "close_short"
        for item in positions:
            if not isinstance(item, dict) or item.get("symbol") != symbol:
                continue
            amount = float(item.get("positionAmt") or 0.0)
            if mode == "hedge":
                side = str(item.get("positionSide") or "").upper()
                if wants_short and side == "SHORT":
                    return abs(amount)
                if not wants_short and side == "LONG":
                    return max(amount, 0.0)
                continue
            if wants_short:
                return abs(min(amount, 0.0))
            return max(amount, 0.0)
        return 0.0

    def _cancel_existing_protective_orders(self, symbol: str) -> None:
        try:
            algo_params = {
                "symbol": symbol,
                "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
                "recvWindow": str(self.config.binance_recv_window_ms),
            }
            open_algos = self._signed_request("GET", "/fapi/v1/algoOrders/openOrders", algo_params)
            orders = open_algos.get("orders", []) if isinstance(open_algos, dict) else open_algos if isinstance(open_algos, list) else []
            for order in orders:
                if str(order.get("symbol", "")).upper() != symbol.upper():
                    continue
                client_id = str(order.get("clientAlgoId") or order.get("clientOrderId") or "")
                if not self._is_own_protective_client_id(client_id):
                    continue
                algo_id = order.get("algoId") or order.get("orderId")
                if algo_id:
                    cancel_params = {
                        "symbol": symbol,
                        "algoId": str(algo_id),
                        "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
                        "recvWindow": str(self.config.binance_recv_window_ms),
                    }
                    self._signed_request("DELETE", "/fapi/v1/algoOrder", cancel_params)
        except BinanceExecutionError:
            pass

    def _submit_protective_stop(self, symbol: str, action: str, stop_loss_price: float) -> dict[str, Any]:
        self._cancel_existing_protective_orders(symbol)
        stop_side = "SELL" if action == "open_long" else "BUY"
        tick_size = self._get_symbol_tick_size(symbol)
        stop_price = self._round_down_to_step(stop_loss_price, tick_size)
        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": stop_side,
            "type": "STOP_MARKET",
            "triggerPrice": self._format_decimal(stop_price),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "priceProtect": "true",
            "clientAlgoId": self._client_algo_id(symbol, action, stop_price),
            "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            "recvWindow": str(self.config.binance_recv_window_ms),
        }
        if self._current_position_mode() == "hedge":
            params["positionSide"] = "LONG" if action == "open_long" else "SHORT"
        return self._signed_request("POST", "/fapi/v1/algoOrder", params)

    def _emergency_close_after_protective_stop_failed(
        self,
        *,
        symbol: str,
        action: str,
        request_id: str,
        generated_at: str,
        error_text: str,
    ) -> dict[str, Any]:
        close_action = "close_short" if action == "open_short" else "close_long"
        try:
            quantity = self._get_position_quantity_for_close(symbol, close_action)
            step_size, min_qty = self._get_symbol_quantity_rules(symbol)
            close_qty = self._round_down_to_step(quantity, step_size)
            if close_qty <= 0 or close_qty < min_qty:
                return {
                    "status": "skipped",
                    "type": "emergency_close_after_protective_stop_failed",
                    "symbol": symbol,
                    "reason": "no_exchange_position_to_close",
                    "protective_stop_error": error_text,
                }
            side = "BUY" if close_action == "close_short" else "SELL"
            params: dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": self._format_decimal(close_qty),
                "newOrderRespType": "RESULT",
                "newClientOrderId": self._client_order_id(f"{request_id}:protective_stop_failed_close", symbol, close_action, generated_at),
                "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
                "recvWindow": str(self.config.binance_recv_window_ms),
            }
            if self._current_position_mode() == "hedge":
                params["positionSide"] = "SHORT" if close_action == "close_short" else "LONG"
            else:
                params["reduceOnly"] = "true"
            response = self._signed_request("POST", "/fapi/v1/order", params)
            self._append_wal({
                "request_id": request_id,
                "stage": "protective_stop_failed_emergency_close_submitted",
                "symbol": symbol,
                "action": close_action,
                "quantity": self._format_decimal(close_qty),
                "generated_at": generated_at,
            })
            return {
                "status": "submitted",
                "type": "emergency_close_after_protective_stop_failed",
                "symbol": symbol,
                "action": close_action,
                "side": side,
                "quantity": self._format_decimal(close_qty),
                "response": response,
                "protective_stop_error": error_text,
            }
        except BinanceExecutionError as exc:
            self._append_wal({
                "request_id": request_id,
                "stage": "protective_stop_failed_emergency_close_failed",
                "symbol": symbol,
                "action": close_action,
                "error": str(exc),
                "generated_at": generated_at,
            })
            return {
                "status": "failed",
                "type": "emergency_close_after_protective_stop_failed",
                "symbol": symbol,
                "action": close_action,
                "error": str(exc),
                "protective_stop_error": error_text,
            }

    def _set_leverage(self, symbol: str, leverage: int) -> None:
        if not symbol or leverage <= 0:
            return
        params = {
            "symbol": symbol,
            "leverage": str(leverage),
            "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            "recvWindow": str(self.config.binance_recv_window_ms),
        }
        self._signed_request("POST", "/fapi/v1/leverage", params)

    def _signed_request(self, method: str, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        method = method.upper()
        normalized = {key: str(value) for key, value in params.items() if value is not None}
        query = parse.urlencode(normalized)
        signature = hmac.new(
            self._api_secret().encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed_query = f"{query}&signature={signature}" if query else f"signature={signature}"
        url = f"{self._base_url()}{endpoint}"
        body = signed_query.encode("utf-8")
        if method == "GET":
            url = f"{url}?{signed_query}"
            body = None
        req = request.Request(
            url=url,
            data=body,
            method=method,
            headers={
                "X-MBX-APIKEY": self._api_key(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with request.urlopen(req, timeout=15, context=_https_context()) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {"status": "ok"}
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise BinanceExecutionError(
                f"Binance execution failed {exc.code}: {redact_sensitive_text(body)}"
            ) from exc
        except error.URLError as exc:
            raise BinanceExecutionError(
                f"Binance execution network failure: {redact_sensitive_text(exc)}"
            ) from exc

    def _public_request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = ""
        if params:
            query = "?" + parse.urlencode({key: str(value) for key, value in params.items() if value is not None})
        req = request.Request(url=f"{self._base_url()}{endpoint}{query}", method="GET")
        try:
            with request.urlopen(req, timeout=15, context=_https_context()) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise BinanceExecutionError(f"Binance public request failed {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise BinanceExecutionError(f"Binance public request network failure: {exc}") from exc

    def _get_mark_price(self, symbol: str) -> float:
        payload = self._public_request("/fapi/v1/premiumIndex", {"symbol": symbol})
        return float(payload.get("markPrice") or 0)

    def _get_exchange_symbols(self) -> set[str]:
        if self._exchange_symbols_cache is not None:
            return self._exchange_symbols_cache
        info = self._public_request("/fapi/v1/exchangeInfo")
        self._exchange_symbols_cache = {
            str(item.get("symbol") or "")
            for item in info.get("symbols", [])
            if str(item.get("symbol") or "")
        }
        return self._exchange_symbols_cache

    def _get_symbol_quantity_rules(self, symbol: str) -> tuple[float, float]:
        info = self._public_request("/fapi/v1/exchangeInfo")
        for item in info.get("symbols", []):
            if item.get("symbol") != symbol:
                continue
            lot_filter = next((flt for flt in item.get("filters", []) if flt.get("filterType") in {"LOT_SIZE", "MARKET_LOT_SIZE"}), None)
            if lot_filter:
                return float(lot_filter.get("stepSize") or 0), float(lot_filter.get("minQty") or 0)
        raise BinanceExecutionError(f"symbol quantity rules not found for {symbol}")

    def _get_symbol_tick_size(self, symbol: str) -> float:
        info = self._public_request("/fapi/v1/exchangeInfo")
        for item in info.get("symbols", []):
            if item.get("symbol") != symbol:
                continue
            price_filter = next((flt for flt in item.get("filters", []) if flt.get("filterType") == "PRICE_FILTER"), None)
            if price_filter:
                return float(price_filter.get("tickSize") or 0.00000001)
        return 0.00000001

    @staticmethod
    def _round_down_to_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        value_dec = Decimal(str(value))
        step_dec = Decimal(str(step))
        return float((value_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec)

    @staticmethod
    def _format_decimal(value: float) -> str:
        return format(Decimal(str(value)).normalize(), "f")

    @staticmethod
    def _rollup_status(results: list[dict[str, Any]]) -> str:
        if not results:
            return "empty"
        if any(item.get("status") != "submitted" for item in results):
            return "partial"
        return "submitted"

    @staticmethod
    def _client_order_id(request_id: str, symbol: str, action: str, generated_at: str) -> str:
        raw = f"{request_id}:{symbol}:{action}:{generated_at}"
        return "somnus" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:28]

    @staticmethod
    def _client_algo_id(symbol: str, action: str, stop_price: float) -> str:
        raw = f"{symbol}:{action}:{stop_price}"
        return "somnussl" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:27]

    @staticmethod
    def _is_own_protective_client_id(client_id: str) -> bool:
        return str(client_id or "").startswith("somnussl")

    @staticmethod
    def _is_binance_status_unknown(exc: BinanceExecutionError) -> bool:
        text = str(exc).lower()
        return (
            '"code":-1007' in text
            or '"code": -1007' in text
            or "timeout waiting for response from backend server" in text
            or "execution status unknown" in text
            or "send status unknown" in text
        )

    def _reconcile_status_unknown_order(
        self,
        symbol: str,
        order_params: dict[str, Any],
        exc: BinanceExecutionError,
    ) -> dict[str, Any]:
        client_order_id = str(order_params.get("newClientOrderId") or "")
        payload: dict[str, Any] = {
            "execution_status": "status_unknown",
            "order_found": False,
            "symbol": symbol,
            "clientOrderId": client_order_id,
            "error": str(exc),
        }
        if not symbol or not client_order_id:
            payload["lookup_error"] = "missing symbol or newClientOrderId"
            return payload
        lookup_params = {
            "symbol": symbol,
            "origClientOrderId": client_order_id,
            "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            "recvWindow": str(self.config.binance_recv_window_ms),
        }
        try:
            order = self._signed_request("GET", "/fapi/v1/order", lookup_params)
        except BinanceExecutionError as lookup_exc:
            payload["lookup_error"] = str(lookup_exc)
            return payload
        order_status = str(order.get("status") or "UNKNOWN").upper()
        payload.update({
            "execution_status": "submitted" if order_status in {"NEW", "PARTIALLY_FILLED", "FILLED"} else "status_unknown_reconciled",
            "order_found": True,
            "order_status": order_status,
            "order": order,
        })
        return payload

    def reconcile_orphan_positions(self, paper_symbols: set[str]) -> list[dict[str, Any]]:
        """Close positions on exchange that no longer exist in paper state."""
        params = {
            "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            "recvWindow": str(self.config.binance_recv_window_ms),
        }
        try:
            payload = self._signed_request("GET", "/fapi/v2/positionRisk", params)
        except BinanceExecutionError as exc:
            logger.warning("reconcile_orphan_positions: failed to query positions: %s", exc)
            return []
        positions = payload if isinstance(payload, list) else [payload]
        orphans: list[dict[str, Any]] = []
        for item in positions:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "")
            amount = float(item.get("positionAmt") or 0.0)
            if amount == 0.0 or not symbol:
                continue
            if symbol in paper_symbols:
                continue
            side = "SELL" if amount > 0 else "BUY"
            qty = abs(amount)
            step_size, min_qty = self._get_symbol_quantity_rules(symbol)
            close_qty = self._round_down_to_step(qty, step_size)
            if close_qty <= 0 or close_qty < min_qty:
                continue
            order_params: dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": self._format_decimal(close_qty),
                "reduceOnly": "true",
                "newOrderRespType": "RESULT",
                "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
                "recvWindow": str(self.config.binance_recv_window_ms),
            }
            try:
                resp = self._signed_request("POST", "/fapi/v1/order", order_params)
                status = "closed"
                logger.info("reconcile_orphan: closed %s qty=%s side=%s", symbol, close_qty, side)
            except BinanceExecutionError as exc:
                resp = {"error": str(exc)}
                status = "failed"
                logger.warning("reconcile_orphan: failed to close %s: %s", symbol, exc)
            orphans.append({
                "symbol": symbol,
                "amount": amount,
                "close_qty": close_qty,
                "side": side,
                "status": status,
                "response": resp,
            })
            self._append_wal({
                "request_id": f"orphan_reconcile_{symbol}",
                "stage": "orphan_closed" if status == "closed" else "orphan_close_failed",
                "symbol": symbol,
                "amount": amount,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
        return orphans

    def _append_wal(self, payload: dict[str, Any]) -> None:
        line = json.dumps({**payload, "logged_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False)
        append_text_line(self.config.execution_wal_file, line)
