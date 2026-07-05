from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.config import StrategyConfig
from trading_system.persistence import atomic_write_json, load_json


@dataclass(slots=True)
class ExecutionIntent:
    action: str
    symbol: str
    side: str
    order_type: str
    quantity_hint_usdt: float
    leverage: int
    reduce_only: bool
    entry_price: float | None
    stop_loss_price: float | None
    take_profit_price: float | None
    rationale: str
    reference_score: float
    smart_money_context: dict[str, Any]
    source_position_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BinanceSkillExecutionBridge:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def write_requests(self, rebalance: dict, rankings: list[dict]) -> dict:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        request_dir = self.config.execution_request_dir
        request_dir.mkdir(parents=True, exist_ok=True)

        score_map = {item["symbol"]: float(item.get("score", 0.0)) for item in rankings}
        request_mode = self._request_mode()
        signal_source = self._signal_source()
        entry_rows = self._entry_rows(rebalance, signal_source)
        exit_rows = self._exit_rows(rebalance, signal_source)
        entry_intents = [
            self._entry_intent(item, score_map.get(item["symbol"], float(item.get("score", 0.0))))
            for item in entry_rows
        ]
        exit_intents = [
            self._exit_intent(item, score_map.get(item["symbol"], float(item.get("score", 0.0))))
            for item in exit_rows
        ]
        entry_intents, suppressed_entry_symbols = self._suppress_entries_for_exit_symbols(entry_intents, exit_intents)
        ordered_intents = exit_intents + entry_intents
        risk_controls = self._build_risk_controls(ordered_intents, rebalance)
        if suppressed_entry_symbols:
            risk_controls["same_symbol_reentry_suppressed"] = suppressed_entry_symbols

        payload = {
            "skill_name": self.config.execution_skill_name,
            "request_mode": request_mode,
            "signal_source": signal_source,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "risk_controls": risk_controls,
            "intents": [item.to_dict() for item in ordered_intents],
            "binance_requests": [],
        }
        if payload["risk_controls"]["status"] != "blocked":
            payload["binance_requests"] = [
                self._to_skill_request(intent)
                for intent in ordered_intents
            ]
        request_path = request_dir / f"execution_request_{timestamp}.json"
        atomic_write_json(request_path, payload)
        return {
            "request_file": str(request_path),
            "intent_count": len(payload["binance_requests"]),
            "blocked_intent_count": len(payload["intents"]) - len(payload["binance_requests"]),
            "request_mode": payload["request_mode"],
            "signal_source": payload["signal_source"],
            "skill_name": payload["skill_name"],
            "risk_controls": payload["risk_controls"],
            "suppressed_entry_count": len(suppressed_entry_symbols),
        }

    def _entry_intent(self, item: dict, score: float) -> ExecutionIntent:
        action = str(item.get("action") or "open_long")
        side = "SELL" if action == "open_short" else "BUY"
        return ExecutionIntent(
            action=action,
            symbol=str(item["symbol"]),
            side=side,
            order_type="MARKET",
            quantity_hint_usdt=float(item.get("notional_usdt", self.config.position_notional_usdt)),
            leverage=self._entry_leverage(score),
            reduce_only=False,
            entry_price=float(item.get("entry_price")) if item.get("entry_price") is not None else None,
            stop_loss_price=float(item.get("stop_loss_price")) if item.get("stop_loss_price") is not None else None,
            take_profit_price=float(item.get("take_profit_price")) if item.get("take_profit_price") is not None else None,
            rationale=str(item.get("rationale", "")),
            reference_score=score,
            smart_money_context=dict(item.get("smart_money_context") or {}),
            source_position_key=str(item.get("shadow_position_key") or "") or None,
        )

    def _entry_leverage(self, score: float) -> int:
        base = int(self.config.default_leverage)
        if not self.config.dynamic_leverage_enabled:
            return base
        baseline = max(1.0, float(self.config.leverage_score_baseline or 80.0))
        scaled = int(round(base * max(0.5, min(1.5, float(score) / baseline))))
        return max(int(self.config.min_leverage), min(int(self.config.max_leverage), scaled))

    def _exit_intent(self, item: dict, score: float) -> ExecutionIntent:
        raw_action = str(item.get("action") or "exit").lower()
        raw_side = str(item.get("side") or "").lower()
        is_short_exit = raw_action in {"close_short", "exit_short"} or raw_side == "short"
        if raw_action == "take_profit_partial":
            action = "take_profit_partial"
        else:
            action = "close_short" if is_short_exit else "close_long"
        order_side = "BUY" if is_short_exit else "SELL"
        return ExecutionIntent(
            action=action,
            symbol=str(item["symbol"]),
            side=order_side,
            order_type="MARKET",
            quantity_hint_usdt=float(item.get("notional_usdt", self.config.position_notional_usdt)),
            leverage=self.config.default_leverage,
            reduce_only=True,
            entry_price=None,
            stop_loss_price=float(item.get("stop_loss_price")) if item.get("stop_loss_price") is not None else None,
            take_profit_price=float(item.get("take_profit_price")) if item.get("take_profit_price") is not None else None,
            rationale=str(item.get("rationale", "")),
            reference_score=score,
            smart_money_context=dict(item.get("smart_money_context") or {}),
            source_position_key=str(item.get("shadow_position_key") or "") or None,
        )

    def _to_skill_request(self, intent: ExecutionIntent) -> dict:
        request_mode = self._request_mode()
        endpoint = "/fapi/v1/order/test" if request_mode == "test_order" else "/fapi/v1/order"
        params = {
            "symbol": intent.symbol,
            "side": intent.side,
            "type": intent.order_type,
            "newOrderRespType": "RESULT",
        }
        if intent.reduce_only:
            params["reduceOnly"] = "true"
        return {
            "skill_name": self.config.execution_skill_name,
            "endpoint": endpoint,
            "method": "POST",
            "params": params,
            "meta": {
                "action": intent.action,
                "quantity_hint_usdt": intent.quantity_hint_usdt,
                "leverage": intent.leverage,
                "entry_price": intent.entry_price,
                "stop_loss_price": intent.stop_loss_price,
                "take_profit_price": intent.take_profit_price,
                "rationale": intent.rationale,
                "reference_score": intent.reference_score,
                "smart_money_context": intent.smart_money_context,
                "shadow_position_key": intent.source_position_key,
                "position_panel_monitor": self._position_panel_monitor(intent),
                "request_mode": request_mode,
            },
        }

    def _position_panel_monitor(self, intent: ExecutionIntent) -> dict[str, Any]:
        if not intent.action.startswith("open_"):
            return {}
        symbol = intent.symbol.upper()
        return {
            "symbol": symbol,
            "enabled": True,
            "panel_api": f"/api/smart-money?symbol={symbol}",
            "views": ["smart_money", "whales"],
            "chart_subjects": ["smart_money_position", "whale_position"],
            "monitor_action": intent.smart_money_context.get("monitor_action", "monitor_for_exit_on_panel_deterioration"),
            "decision_questions": ["加仓", "减仓", "平仓"],
        }

    @staticmethod
    def _suppress_entries_for_exit_symbols(
        entry_intents: list[ExecutionIntent],
        exit_intents: list[ExecutionIntent],
    ) -> tuple[list[ExecutionIntent], list[str]]:
        exit_symbols = {intent.symbol.upper() for intent in exit_intents}
        if not exit_symbols:
            return entry_intents, []
        filtered: list[ExecutionIntent] = []
        suppressed: list[str] = []
        for intent in entry_intents:
            symbol = intent.symbol.upper()
            if symbol in exit_symbols:
                suppressed.append(symbol)
                continue
            filtered.append(intent)
        return filtered, sorted(set(suppressed))

    def _build_risk_controls(self, intents: list[ExecutionIntent], rebalance: dict) -> dict[str, Any]:
        violations: list[str] = []
        if self.config.live_max_intents_per_request > 0 and len(intents) > self.config.live_max_intents_per_request:
            violations.append(
                f"intent_count {len(intents)} exceeds TS_LIVE_MAX_INTENTS_PER_REQUEST={self.config.live_max_intents_per_request}"
            )
        max_notional = float(self.config.live_max_order_notional_usdt or 0.0)
        worst_case_stop_loss_usdt = self._worst_case_stop_loss_from_intents(
            [intent for intent in intents if intent.action.startswith("open_")]
        )
        loss_limit = float(self.config.live_max_request_loss_usdt or 0.0)
        if loss_limit > 0 and worst_case_stop_loss_usdt > loss_limit:
            violations.append(
                f"worst_case_stop_loss_usdt {worst_case_stop_loss_usdt:.6f} exceeds TS_LIVE_MAX_REQUEST_LOSS_USDT={loss_limit}"
            )
        paper_risk = dict(rebalance.get("risk_controls") or {})
        if paper_risk.get("loss_fuse_triggered"):
            violations.append("paper cycle loss fuse triggered")
        request_mode = self._request_mode()
        return {
            "status": "blocked" if request_mode != "test_order" and violations else "ok",
            "mode": request_mode,
            "violations": violations,
            "worst_case_stop_loss_usdt": round(worst_case_stop_loss_usdt, 6),
            "live_max_request_loss_usdt": loss_limit,
            "live_max_order_notional_usdt": max_notional,
            "live_max_intents_per_request": self.config.live_max_intents_per_request,
            "paper_cycle": paper_risk,
        }

    def _request_mode(self) -> str:
        raw = str(self.config.execution_order_mode or "auto").strip().lower()
        if raw == "auto":
            return "test_order" if self.config.execution_test_order else "live_order"
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
        return aliases.get(raw, "test_order")

    def _signal_source(self) -> str:
        raw = str(self.config.execution_signal_source or "paper").strip().lower()
        if raw in {"paper", "shadow_paper", "both"}:
            return raw
        return "paper"

    @staticmethod
    def _entry_rows(rebalance: dict, signal_source: str) -> list[dict]:
        rows: list[dict] = []
        if signal_source in {"paper", "both"}:
            rows.extend(item for item in rebalance.get("entries", []) if isinstance(item, dict))
        if signal_source in {"shadow_paper", "both"}:
            shadow = rebalance.get("shadow_paper") or {}
            if isinstance(shadow, dict):
                rows.extend(item for item in shadow.get("entries", []) if isinstance(item, dict))
        return rows

    @staticmethod
    def _exit_rows(rebalance: dict, signal_source: str) -> list[dict]:
        rows: list[dict] = []
        if signal_source in {"paper", "both"}:
            rows.extend(item for item in rebalance.get("exits", []) if isinstance(item, dict))
        if signal_source in {"shadow_paper", "both"}:
            shadow = rebalance.get("shadow_paper") or {}
            if isinstance(shadow, dict):
                for item in shadow.get("exits", []):
                    if not isinstance(item, dict):
                        continue
                    mirror_status = item.get("binance_mirror_status")
                    if mirror_status is not None and mirror_status not in {"submitted", "submitted_unprotected"}:
                        continue
                    rows.append(item)
        return rows

    @staticmethod
    def _worst_case_stop_loss_usdt(entries: list[dict]) -> float:
        total = 0.0
        for item in entries:
            entry_price = float(item.get("entry_price") or 0.0)
            stop_loss_price = float(item.get("stop_loss_price") or 0.0)
            notional = float(item.get("notional_usdt") or 0.0)
            if entry_price <= 0 or stop_loss_price <= 0 or notional <= 0:
                continue
            total += notional * max(0.0, (entry_price - stop_loss_price) / entry_price)
        return total

    @staticmethod
    def _worst_case_stop_loss_from_intents(intents: list[ExecutionIntent]) -> float:
        total = 0.0
        for intent in intents:
            entry_price = float(intent.entry_price or 0.0)
            stop_loss_price = float(intent.stop_loss_price or 0.0)
            notional = float(intent.quantity_hint_usdt or 0.0)
            if entry_price <= 0 or stop_loss_price <= 0 or notional <= 0:
                continue
            if intent.action == "open_short":
                total += notional * max(0.0, (stop_loss_price - entry_price) / entry_price)
            else:
                total += notional * max(0.0, (entry_price - stop_loss_price) / entry_price)
        return total


def read_execution_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    results: list[dict] = []
    for file_path in sorted(path.glob("*.json")):
        payload = load_json(file_path)
        results.append({"file": str(file_path), "payload": payload})
    return results


def is_silent_execution_error(result: dict[str, Any]) -> bool:
    status = str(result.get("status") or "").lower()
    error_text = str(result.get("error") or "").lower()
    if status in {"status_unknown", "status_unknown_reconciled", "status_unknown_not_found"}:
        return True
    if not error_text:
        return False
    return (
        '"code":-1121' in error_text
        or '"code": -1121' in error_text
        or '"code":-1007' in error_text
        or '"code": -1007' in error_text
        or "timeout waiting for response from backend server" in error_text
        or "execution status unknown" in error_text
        or "send status unknown" in error_text
        or "invalid symbol" in error_text
        or "symbol not available on binance" in error_text
        or "symbol previously rejected by binance api" in error_text
    )


def summarize_execution_results(results: list[dict]) -> dict:
    statuses: dict[str, int] = {}
    actionable = 0
    latest_requests: list[dict[str, Any]] = []
    latest_errors: list[dict[str, Any]] = []
    for item in results:
        payload = item.get("payload", {})
        status = str(payload.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if status not in {"placeholder", "unknown"}:
            actionable += 1
        for result in payload.get("results", []):
            latest_requests.append(
                {
                    "symbol": str(result.get("symbol") or ""),
                    "status": str(result.get("status") or "unknown"),
                    "request_mode": str(result.get("request_mode") or payload.get("request_mode") or "unknown"),
                    "action": str((result.get("meta") or {}).get("action") or ""),
                    "side": str(((result.get("request") or {}).get("params") or {}).get("side") or ((result.get("response") or {}).get("side") or "")),
                    "quantity_hint_usdt": float((result.get("meta") or {}).get("quantity_hint_usdt") or 0),
                    "leverage": int((result.get("meta") or {}).get("leverage") or 0),
                    "rationale": str((result.get("meta") or {}).get("rationale") or ""),
                }
            )
            if result.get("status") == "failed" and not is_silent_execution_error(result):
                latest_errors.append(
                    {
                        "symbol": str(result.get("symbol") or ""),
                        "action": str((result.get("meta") or {}).get("action") or ""),
                        "error": str(result.get("error") or ""),
                        "params": dict(((result.get("request") or {}).get("params") or {})),
                    }
                )
    return {
        "result_count": len(results),
        "actionable_result_count": actionable,
        "statuses": statuses,
        "latest_requests": latest_requests[-5:],
        "latest_errors": latest_errors[-5:],
    }
