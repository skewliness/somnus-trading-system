from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def train_from_existing_data(
    *,
    closed_trades_path: str | Path,
    model_dir: str | Path,
    **_: Any,
) -> dict[str, Any]:
    """Lightweight fallback for exported snapshots.

    The full CoinScout trainer is private/large and is not included in this
    repository snapshot. Keep the pipeline runnable and write a clear summary
    instead of failing at import time.
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    closed_trades_path = Path(closed_trades_path)
    rows = 0
    if closed_trades_path.exists():
        rows = sum(1 for line in closed_trades_path.read_text(encoding="utf-8").splitlines() if line.strip())
    summary = {
        "status": "skipped",
        "reason": "coin_scout_trainer_full_module_not_in_snapshot",
        "closed_trades_path": str(closed_trades_path),
        "closed_trade_rows": rows,
    }
    (model_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
