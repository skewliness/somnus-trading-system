from __future__ import annotations

from collections.abc import Callable, Iterable

from trading_system.exit.models import ExitRuleContext, ExitSignal


def resolve_exit_signals(signals: Iterable[ExitSignal]) -> ExitSignal:
    """Select the highest-priority triggered signal without changing its intent."""

    triggered = [signal for signal in signals if signal.triggered]
    if not triggered:
        raise ValueError("at least one triggered exit signal is required")
    return max(triggered, key=lambda signal: signal.priority)


def resolve_exit_rules(
    context: ExitRuleContext,
    rules: Iterable[Callable[[ExitRuleContext], ExitSignal]],
) -> ExitSignal | None:
    """Evaluate rules lazily in the legacy precedence order."""

    for rule in rules:
        signal = rule(context)
        if signal.triggered:
            return signal
    return None
