from __future__ import annotations

from typing import Any, Callable

from trading_system.risk.models import RiskAssessment


class RiskManager:
    """Behavior-preserving owner of V6 risk calculations and preflight checks."""

    def __init__(self, *, utc_now: Callable[[], str]) -> None:
        self.utc_now = utc_now

    @staticmethod
    def assess_position_capacity(*, active_position_count: int, max_positions: int) -> RiskAssessment:
        at_capacity = int(active_position_count) >= int(max_positions)
        return RiskAssessment(
            allowed=not at_capacity,
            reason="position_capacity_available" if not at_capacity else "max_positions_reached",
            metrics={
                "active_position_count": int(active_position_count),
                "max_positions": int(max_positions),
            },
        )

    @classmethod
    def at_position_capacity(cls, *, active_position_count: int, max_positions: int) -> bool:
        return not cls.assess_position_capacity(
            active_position_count=active_position_count,
            max_positions=max_positions,
        ).allowed

    @staticmethod
    def assess_scale_in_notional(
        *,
        current_notional: float,
        add_notional: float,
        max_total_notional: float,
    ) -> RiskAssessment:
        allowed = float(current_notional) + float(add_notional) <= float(max_total_notional) + 1e-9
        return RiskAssessment(
            allowed=allowed,
            reason="scale_in_notional_available" if allowed else "scale_in_max_total_notional_exceeded",
            metrics={
                "current_notional": float(current_notional),
                "add_notional": float(add_notional),
                "max_total_notional": float(max_total_notional),
            },
        )

    @classmethod
    def scale_in_notional_allowed(
        cls,
        *,
        current_notional: float,
        add_notional: float,
        max_total_notional: float,
    ) -> bool:
        return cls.assess_scale_in_notional(
            current_notional=current_notional,
            add_notional=add_notional,
            max_total_notional=max_total_notional,
        ).allowed

    def preflight_close_quantity(self, client: Any, intent: Any) -> dict[str, Any]:
        """Preserve the existing close quantity check and its exception behavior."""

        resolver = getattr(client, "resolve_close_quantity", None)
        if not callable(resolver):
            return {}
        quantity = str(resolver(intent))
        decision = (intent.risk_plan or {}).get("close_decision") or {}
        return {
            "quantity": quantity,
            "decision": dict(decision) if isinstance(decision, dict) else {},
            "checked_at": self.utc_now(),
        }

    @staticmethod
    def execution_account_metadata(config: Any) -> dict[str, Any]:
        raw_asset_mode = str(config.account_asset_mode or "SINGLE_ASSET").strip().upper().replace("-", "_")
        asset_mode = "MULTI_ASSET" if raw_asset_mode in {"MULTI", "MULTI_ASSET", "MULTI_ASSETS"} else "SINGLE_ASSET"
        margin_type = str(config.margin_type or "ISOLATED").strip().upper().replace("-", "_")
        virtual_isolation = bool(config.virtual_risk_isolation_enabled)
        return {
            "instance_name": config.instance_name,
            "account_asset_mode": asset_mode,
            "exchange_margin_type": margin_type,
            "exchange_margin_shared": margin_type == "CROSSED",
            "local_isolation_mode": "virtual_subledger" if virtual_isolation else "exchange_native",
            "virtual_risk_isolation_enabled": virtual_isolation,
            "order_mode": config.order_mode,
            "exit_control_mode": config.exit_control_mode,
        }

    @staticmethod
    def local_risk_budget_from_values(
        *,
        action: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        notional_usdt: float,
        leverage: int,
    ) -> dict[str, Any]:
        entry = float(entry_price or 0.0)
        stop = float(stop_loss_price or 0.0)
        notional = max(0.0, float(notional_usdt or 0.0))
        leverage = max(1, int(leverage or 1))
        stop_distance_pct = abs(stop - entry) / entry if entry > 0 and stop > 0 else 0.0
        return {
            "action": action,
            "side": side,
            "notional_budget_usdt": round(notional, 8),
            "initial_margin_budget_usdt": round(notional / leverage, 8),
            "leverage": leverage,
            "entry_price": round(entry, 8),
            "stop_loss_price": round(stop, 8),
            "stop_distance_pct": round(stop_distance_pct * 100.0, 8),
            "planned_loss_at_stop_usdt": round(notional * stop_distance_pct, 8),
            "calculation": "notional_budget_usdt * stop_distance_pct",
        }

    @classmethod
    def local_risk_budget_for_intent(
        cls,
        intent: Any,
        *,
        entry_price: float | None = None,
        notional_usdt: float | None = None,
    ) -> dict[str, Any]:
        return cls.local_risk_budget_from_values(
            action=intent.action,
            side="short" if intent.action in {"open_short", "close_short"} else "long",
            entry_price=float(entry_price if entry_price is not None else intent.entry_price or 0.0),
            stop_loss_price=float(intent.stop_loss_price or 0.0),
            notional_usdt=float(notional_usdt if notional_usdt is not None else intent.quantity_hint_usdt or 0.0),
            leverage=int(intent.leverage or 1),
        )

    def annotate_intent_execution_risk(self, config: Any, intent: Any) -> None:
        risk_plan = dict(intent.risk_plan or {})
        risk_plan["execution_account"] = self.execution_account_metadata(config)
        risk_plan["exit_control_mode"] = config.exit_control_mode
        if intent.action in {"open_long", "open_short"}:
            risk_plan["local_risk_budget"] = self.local_risk_budget_for_intent(intent)
        intent.risk_plan = risk_plan

    @staticmethod
    def is_reduce_only_rejected_error(error_text: str) -> bool:
        lowered = str(error_text or "").lower()
        return '"code":-2022' in lowered or "reduceonly order is rejected" in lowered

    @staticmethod
    def is_percent_price_filter_error(error_text: str) -> bool:
        lowered = str(error_text or "").lower()
        return '"code":-4131' in lowered or "percent_price filter" in lowered

    @staticmethod
    def is_status_unknown_error(error_text: str) -> bool:
        lowered = str(error_text or "").lower()
        return (
            '"code":-1007' in lowered
            or '"code": -1007' in lowered
            or "signed request failed 500" in lowered
            or "signed request failed 502" in lowered
            or "signed request failed 503" in lowered
            or "signed request failed 504" in lowered
            or "bad gateway" in lowered
            or "gateway timeout" in lowered
            or "service unavailable" in lowered
            or "internal server error" in lowered
            or "status unknown" in lowered
            or "timeout waiting for response" in lowered
        )
