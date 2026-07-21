from __future__ import annotations

from typing import Any, Callable

from trading_system.risk.models import RiskAssessment, RiskRecommendation


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

    @staticmethod
    def _first_positive_metric(payload: Any, names: tuple[str, ...]) -> float:
        if not isinstance(payload, dict):
            return 0.0
        for name in names:
            try:
                value = float(payload.get(name) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
        return 0.0

    @classmethod
    def volatility_pct_for_intent(cls, intent: Any, local_budget: dict[str, Any]) -> tuple[float, str]:
        risk_plan = intent.risk_plan if isinstance(intent.risk_plan, dict) else {}
        entry_context = risk_plan.get("entry_context") if isinstance(risk_plan.get("entry_context"), dict) else {}
        names = ("volatility_pct", "realized_volatility_pct", "atr_pct", "bb_width_pct")
        direct = cls._first_positive_metric(risk_plan, names)
        if direct > 0:
            return direct, "risk_plan"
        contextual = cls._first_positive_metric(entry_context, names)
        if contextual > 0:
            return contextual, "entry_context"
        return max(0.0, float(local_budget.get("stop_distance_pct") or 0.0)), "stop_distance_proxy"

    @staticmethod
    def assess_trade_risk(
        *,
        planned_loss_at_stop_usdt: float,
        max_risk_budget_usdt: float,
        volatility_pct: float,
        reference_volatility_pct: float,
        leverage: int,
        volatility_source: str = "unknown",
    ) -> RiskRecommendation:
        """Build an advisory score without changing allow/deny or order sizing."""

        planned_loss = max(0.0, float(planned_loss_at_stop_usdt or 0.0))
        max_budget = max(0.0, float(max_risk_budget_usdt or 0.0))
        observed_volatility = max(0.0, float(volatility_pct or 0.0))
        reference_volatility = max(0.01, float(reference_volatility_pct or 0.0))
        leverage_value = max(1, int(leverage or 1))

        budget_ratio = planned_loss / max_budget if max_budget > 0 else 0.0
        volatility_ratio = observed_volatility / reference_volatility if observed_volatility > 0 else 0.0
        budget_score = min(50.0, 30.0 * budget_ratio if budget_ratio <= 1.0 else 30.0 + 20.0 * (budget_ratio - 1.0))
        volatility_score = min(30.0, 15.0 * volatility_ratio)
        leverage_score = min(20.0, float(leverage_value))
        score = round(min(100.0, budget_score + volatility_score + leverage_score), 4)

        if score < 40.0:
            level, recommendation, score_multiplier = "low", "standard", 1.0
        elif score < 60.0:
            level, recommendation, score_multiplier = "medium", "reduced", 0.75
        elif score < 80.0:
            level, recommendation, score_multiplier = "high", "defensive", 0.50
        else:
            level, recommendation, score_multiplier = "critical", "minimum", 0.25
        volatility_adjustment = 1.0 if observed_volatility <= reference_volatility else max(
            0.25,
            reference_volatility / observed_volatility,
        )
        recommended_multiplier = round(min(score_multiplier, volatility_adjustment), 4)
        reasons: list[str] = []
        if max_budget > 0 and budget_ratio > 1.0:
            reasons.append("planned_loss_exceeds_reference_budget")
        if observed_volatility > reference_volatility:
            reasons.append("volatility_above_reference")
        if leverage_value > 8:
            reasons.append("elevated_leverage")
        if not reasons:
            reasons.append("within_reference_risk_envelope")
        return RiskRecommendation(
            risk_score=score,
            risk_level=level,
            position_recommendation=recommendation,
            position_size_multiplier=recommended_multiplier,
            volatility_adjustment=round(volatility_adjustment, 4),
            max_risk_budget_usdt=round(max_budget, 8),
            budget_utilization_pct=round(budget_ratio * 100.0, 4),
            reasons=tuple(reasons),
            metrics={
                "planned_loss_at_stop_usdt": round(planned_loss, 8),
                "volatility_pct": round(observed_volatility, 8),
                "reference_volatility_pct": round(reference_volatility, 8),
                "volatility_source": str(volatility_source or "unknown"),
                "leverage": leverage_value,
                "score_components": {
                    "budget": round(budget_score, 4),
                    "volatility": round(volatility_score, 4),
                    "leverage": round(leverage_score, 4),
                },
            },
        )

    @classmethod
    def risk_recommendation_for_intent(
        cls,
        config: Any,
        intent: Any,
        local_budget: dict[str, Any],
    ) -> RiskRecommendation:
        planned_loss = max(0.0, float(local_budget.get("planned_loss_at_stop_usdt") or 0.0))
        configured_budget = max(0.0, float(getattr(config, "risk_max_budget_usdt", 0.0) or 0.0))
        if configured_budget <= 0:
            configured_budget = max(
                0.0,
                float(getattr(config, "base_notional_usdt", 0.0) or 0.0)
                * float(getattr(config, "stop_loss_pct", 0.0) or 0.0),
            )
        if configured_budget <= 0:
            configured_budget = planned_loss
        volatility_pct, volatility_source = cls.volatility_pct_for_intent(intent, local_budget)
        reference_volatility_pct = max(
            0.01,
            float(getattr(config, "risk_reference_volatility_pct", 0.0) or 0.0)
            or float(getattr(config, "stop_loss_pct", 0.0) or 0.0) * 100.0
            or float(local_budget.get("stop_distance_pct") or 0.0)
            or 1.0,
        )
        return cls.assess_trade_risk(
            planned_loss_at_stop_usdt=planned_loss,
            max_risk_budget_usdt=configured_budget,
            volatility_pct=volatility_pct,
            reference_volatility_pct=reference_volatility_pct,
            leverage=int(intent.leverage or 1),
            volatility_source=volatility_source,
        )

    def annotate_intent_execution_risk(self, config: Any, intent: Any) -> None:
        risk_plan = dict(intent.risk_plan or {})
        risk_plan["execution_account"] = self.execution_account_metadata(config)
        risk_plan["exit_control_mode"] = config.exit_control_mode
        if intent.action in {"open_long", "open_short"}:
            local_budget = self.local_risk_budget_for_intent(intent)
            risk_plan["local_risk_budget"] = local_budget
            intent.risk_plan = risk_plan
            risk_plan["risk_engine"] = self.risk_recommendation_for_intent(config, intent, local_budget).to_dict()
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
