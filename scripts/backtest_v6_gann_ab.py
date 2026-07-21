"""Portfolio-level V6 Gann A/B backtest / 组合层面 V6 江恩 A/B 回测.

Compares four arms on the SAME candidate stream under identical portfolio,
cost and execution rules:
四个分支使用同一份候选信号流、同一套组合/成本/执行规则：

    A0 v6_baseline              — ignore Gann entirely / 完全忽略江恩
    A1 gann_low_weight_scoring  — score' = score * (1 + 0.1 * alignment),
                                  affects same-day allocation ORDER only
                                  仅影响同日分配排序，不影响仓位
    A2 gann_position_sizing     — notional *= gann_position_multiplier(feature, side)
                                  仓位名义值乘以模块自带乘数
    A3 gann_hard_filter         — skip candidates whose Gann bias conflicts with
                                  the action side (counted in skipped_gann_filter)
                                  江恩方向冲突的候选直接跳过

Candidate input / 候选输入
--------------------------
Real V6 candidates come from the owner's production replay (NOT in this repo),
so candidates are an INPUT via ``--candidates-file path.jsonl`` — one JSON
object per line with this schema (enforced on load):
真实 V6 候选来自生产回放（不在本仓库），因此通过 ``--candidates-file`` JSONL 输入，
每行一个 JSON 对象，schema 如下（加载时强校验）：

    {
      "date": "YYYY-MM-DD",          # required. Signal day T (from T close).
      "symbol": "BTCUSDT",           # required. Binance USDS futures symbol.
      "action": "open_long"|"open_short",   # required.
      "score": 1.23,                 # required float. Higher = better; used for
                                     # same-day allocation priority.
      "entry_ref_price": 100.0,      # optional float. Fill reference price.
                                     # Default: day T+1 OPEN (next bar).
      "stop_loss_pct": 3.0,          # optional float. Default --default-stop-pct.
      "take_profit_pct": 6.0,        # optional float. Default --default-tp-pct.
      "max_hold_days": 14            # optional int. Default --max-hold-days.
    }

``--demo`` mode (no candidates file): generates a documented demo candidate set
from daily klines — rule: daily close crosses ABOVE its 20d SMA -> open_long
candidate; crosses BELOW -> open_short candidate; score = |close - sma| / sma.
Demo output is labelled "DEMO CANDIDATES — not V6 production signals" in stdout
and output CSV filenames get suffix ``_demo``.
``--demo`` 模式：无候选文件时，用日线收盘上/下穿 20 日均线生成演示候选，
score = |close-sma|/sma，输出明确标注为演示数据（非 V6 生产信号）。

Simulation rules (owner-mandated) / 模拟规则（甲方指定）
------------------------------------------------------
1. Next-bar execution: candidate dated day T fills at day T+1 OPEN (or at
   ``entry_ref_price`` if provided). Gann features for day-T signals are
   computed with ``asof_index`` = index of day T — no future data in anchors.
   次日开盘成交；江恩锚点计算传入 T 日 asof_index，绝不使用未来数据。
2. Costs: --fee-bps (default 5, taker, per side) + --slippage-bps (default 3,
   per side, adverse). Funding is NOT modeled: funding_omitted=true is printed
   and written into every output row. 资金费率未建模，输出中显式标注。
3. Portfolio: --initial-equity 10000, --risk-per-trade-pct 1.0,
   --max-concurrent 5, --max-per-symbol 1.
   Sizing formula / 仓位公式:
       risk_budget = current_realized_equity * risk_per_trade_pct / 100
       notional    = risk_budget / (stop_loss_pct / 100)        # so that a
                                                                # stop-out loss
                                                                # ≈ risk budget
       notional   *= gann_position_multiplier(feature, side)    # A2 only
       qty         = notional / fill_base_price
4. Same-day allocation: when same-day candidates exceed free slots, take the
   highest (arm-adjusted) score first; deterministic tiebreak = symbol asc.
   The rest count in skipped_full. 同日按分数（分支调整后）降序、symbol 升序分配，
   超出空位的计入 skipped_full。
5. Overlapping signals: if a position is already open for the symbol, the new
   signal is IGNORED (no adds, no resize) and counted in skipped_same_symbol.
   同 symbol 已有持仓则忽略新信号（不加仓不调仓），计入 skipped_same_symbol。
6. Exits: stop-loss / take-profit checked against daily high/low; if both are
   touched in the same bar, stop-loss fires first (conservative). Gap rule: if
   the bar OPENS beyond the stop (or target), the exit fills at that open.
   Otherwise time exit at the close of the max_hold_days-th bar held (entry
   bar counts as bar 1). Positions still open at the end of data are force
   closed at the last available close (reason "end_of_data").
   止损/止盈按日内高低价判定，同根K线双触发时保守地先止损；跳空超越止损价则按
   开盘价成交；否则持有 max_hold_days 根后收盘离场（入场当根算第 1 根）。
7. OOS split: --train-end / --validation-start. Trade stats are split by ENTRY
   date; equity/exposure/drawdown are computed on the daily equity curve
   restricted to each window's dates. No parameter is tuned inside this
   script — both windows are simply reported.
   训练/验证按入场日期切分交易统计；权益曲线类指标按窗口日期段计算；
   本脚本内部不做任何调参。

Outputs / 输出
--------------
reports/gann_eval/v6_gann_ab_portfolio_train{_demo}.csv
reports/gann_eval/v6_gann_ab_portfolio_validation{_demo}.csv
(one row per arm) + a stdout table per window.

Price data / 行情数据
---------------------
Binance USDS futures public daily klines (https://fapi.binance.com/fapi/v1/klines,
interval=1d, paginated), cached as the RAW Binance kline array JSON under
``data/gann_eval_cache/{symbol}_1d.json`` (cache path & format shared with the
Gann stratification script). ``--offline`` uses cache only; network fetch
requires an explicit ``--asof`` date so runs are deterministic (no ambient
"now"). 缓存与分层脚本共享同一路径与原始格式；--offline 只读缓存；联网抓取必须
显式给 --asof 保证可复现。

Determinism: no randomness, no ambient clock anywhere in the engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib import parse, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading_system.gann_feature import (  # noqa: E402
    GannFeature,
    compute_gann_feature,
    gann_position_multiplier,
)

BINANCE_FAPI_BASE = "https://fapi.binance.com"
DAY_MS = 86_400_000
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ACTIONS = ("open_long", "open_short")
ARM_NAMES = (
    "v6_baseline",
    "gann_low_weight_scoring",
    "gann_position_sizing",
    "gann_hard_filter",
)
DEMO_LABEL = "DEMO CANDIDATES — not V6 production signals"
FUNDING_OMITTED = True  # funding is never modeled by this script

METRIC_COLUMNS = (
    "arm",
    "window",
    "total_return_pct",
    "max_drawdown_pct",
    "profit_factor",
    "win_rate_pct",
    "avg_trade_return_pct",
    "worst_trade_pct",
    "trade_count",
    "skipped_full",
    "skipped_same_symbol",
    "skipped_gann_filter",
    "avg_exposure_pct",
    "max_simultaneous_positions",
    "funding_omitted",
)


# ---------------------------------------------------------------------------
# Data model / 数据模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    date: str
    symbol: str
    action: str  # open_long | open_short
    score: float
    entry_ref_price: float | None
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_days: int

    @property
    def side(self) -> str:
        return "long" if self.action == "open_long" else "short"


@dataclass(frozen=True)
class CostConfig:
    fee_bps: float = 5.0  # taker, per side
    slippage_bps: float = 3.0  # per side, adverse


@dataclass(frozen=True)
class PortfolioConfig:
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    max_concurrent: int = 5
    max_per_symbol: int = 1


@dataclass(frozen=True)
class ArmConfig:
    name: str
    use_gann_scoring: bool = False  # A1
    use_gann_sizing: bool = False  # A2
    use_gann_filter: bool = False  # A3

    @property
    def needs_gann(self) -> bool:
        return self.use_gann_scoring or self.use_gann_sizing or self.use_gann_filter


ARM_CONFIGS: dict[str, ArmConfig] = {
    "v6_baseline": ArmConfig("v6_baseline"),
    "gann_low_weight_scoring": ArmConfig("gann_low_weight_scoring", use_gann_scoring=True),
    "gann_position_sizing": ArmConfig("gann_position_sizing", use_gann_sizing=True),
    "gann_hard_filter": ArmConfig("gann_hard_filter", use_gann_filter=True),
}


@dataclass
class Trade:
    symbol: str
    side: str
    entry_date: str
    exit_date: str
    entry_price: float  # fill price after slippage
    exit_price: float  # fill price after slippage
    qty: float
    notional: float  # qty * fill_base_price at entry
    pnl: float  # net of fees + slippage
    return_pct: float  # pnl / notional * 100
    exit_reason: str  # stop_loss | take_profit | time_exit | end_of_data
    gann_multiplier: float = 1.0
    gann_alignment: int = 0


@dataclass
class DailyRecord:
    date: str
    equity: float  # marked-to-market at close
    exposure_pct: float  # total open notional (at close marks) / equity * 100
    open_positions: int  # max simultaneous on this day (after entries)


@dataclass
class ArmResult:
    arm: str
    initial_equity: float
    trades: list[Trade] = field(default_factory=list)
    daily: list[DailyRecord] = field(default_factory=list)
    # skip events as (fill_date, reason) so they can be windowed later
    skip_events: list[tuple[str, str]] = field(default_factory=list)
    funding_omitted: bool = FUNDING_OMITTED

    def skip_count(self, reason: str, start: str | None = None, end: str | None = None) -> int:
        return sum(
            1
            for date, why in self.skip_events
            if why == reason and _in_window(date, start, end)
        )


# ---------------------------------------------------------------------------
# Candidate loading / 候选加载
# ---------------------------------------------------------------------------


def _require(condition: bool, line_no: int, message: str) -> None:
    if not condition:
        raise ValueError(f"candidates line {line_no}: {message}")


def parse_candidate(
    obj: Any,
    line_no: int,
    *,
    default_stop_pct: float = 3.0,
    default_tp_pct: float = 6.0,
    default_max_hold_days: int = 14,
) -> Candidate:
    _require(isinstance(obj, dict), line_no, f"expected a JSON object, got {type(obj).__name__}")
    date = obj.get("date")
    _require(isinstance(date, str) and bool(DATE_RE.match(date)), line_no, f"'date' must be a 'YYYY-MM-DD' string, got {date!r}")
    symbol = obj.get("symbol")
    _require(isinstance(symbol, str) and symbol.strip() != "", line_no, f"'symbol' must be a non-empty string, got {symbol!r}")
    action = obj.get("action")
    _require(action in ACTIONS, line_no, f"'action' must be one of {ACTIONS}, got {action!r}")
    score = obj.get("score")
    _require(
        isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)),
        line_no,
        f"'score' must be a finite number, got {score!r}",
    )

    def _opt_positive(key: str, default: float | None) -> float | None:
        value = obj.get(key)
        if value is None:
            return default
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0,
            line_no,
            f"'{key}' must be a positive finite number, got {value!r}",
        )
        return float(value)

    entry_ref_price = _opt_positive("entry_ref_price", None)
    stop_loss_pct = _opt_positive("stop_loss_pct", float(default_stop_pct))
    take_profit_pct = _opt_positive("take_profit_pct", float(default_tp_pct))
    max_hold_raw = obj.get("max_hold_days")
    if max_hold_raw is None:
        max_hold_days = int(default_max_hold_days)
    else:
        _require(
            isinstance(max_hold_raw, int) and not isinstance(max_hold_raw, bool) and max_hold_raw >= 1,
            line_no,
            f"'max_hold_days' must be an integer >= 1, got {max_hold_raw!r}",
        )
        max_hold_days = int(max_hold_raw)
    return Candidate(
        date=date,
        symbol=symbol.strip().upper(),
        action=str(action),
        score=float(score),
        entry_ref_price=entry_ref_price,
        stop_loss_pct=float(stop_loss_pct),
        take_profit_pct=float(take_profit_pct),
        max_hold_days=max_hold_days,
    )


def load_candidates(
    path: Path,
    *,
    default_stop_pct: float = 3.0,
    default_tp_pct: float = 6.0,
    default_max_hold_days: int = 14,
) -> list[Candidate]:
    """Load and validate a JSONL candidates file (schema in module docstring)."""
    if not path.exists():
        raise FileNotFoundError(f"candidates file not found: {path}")
    candidates: list[Candidate] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"candidates line {line_no}: invalid JSON ({exc})") from exc
        candidates.append(
            parse_candidate(
                obj,
                line_no,
                default_stop_pct=default_stop_pct,
                default_tp_pct=default_tp_pct,
                default_max_hold_days=default_max_hold_days,
            )
        )
    if not candidates:
        raise ValueError(f"candidates file {path} contains no candidates")
    return candidates


# ---------------------------------------------------------------------------
# Klines: fetch / cache / normalize  行情抓取、缓存、标准化
# ---------------------------------------------------------------------------


def bars_from_binance_klines(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Raw Binance kline arrays -> sorted daily bar dicts (date/open/high/low/close)."""
    by_date: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            open_ms = int(row[0])
            date = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
            by_date[date] = {
                "date": date,
                "time": date,  # key consumed by gann_feature.normalize_daily_klines
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
            }
        except (TypeError, ValueError):
            continue
    return [by_date[key] for key in sorted(by_date)]


def kline_cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol.upper()}_1d.json"


def _fetch_json(url: str, retries: int = 8) -> list:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _time.sleep(min(10.0, 0.5 * (attempt + 1) * (6 if "429" in str(exc) else 1)))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def fetch_daily_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Paginated fetch of raw daily kline arrays from Binance USDS futures."""
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol.upper(), "interval": "1d", "startTime": cursor, "endTime": end_ms, "limit": 1500}
        url = f"{BINANCE_FAPI_BASE}/fapi/v1/klines?{parse.urlencode(params)}"
        batch = _fetch_json(url)
        if not batch:
            break
        for row in batch:
            if not rows or int(row[0]) > int(rows[-1][0]):
                rows.append(row)
        next_cursor = int(batch[-1][0]) + DAY_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        _time.sleep(0.12)
    return rows


def load_klines(
    symbol: str,
    cache_dir: Path,
    *,
    offline: bool,
    start_date: str,
    asof_date: str | None,
) -> list[dict[str, Any]]:
    """Load daily bars for a symbol from cache, fetching+caching when allowed.

    Cache format: RAW Binance kline array JSON at {cache_dir}/{SYMBOL}_1d.json
    (shared with the Gann stratification script).
    """
    path = kline_cache_path(cache_dir, symbol)
    raw: list = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                raw = payload
            elif isinstance(payload, dict) and isinstance(payload.get("klines"), list):
                # Wrapped cache format written by evaluate_gann_structural_features.py
                # ({"fetched_at", "symbol", "klines": [...]}).
                raw = payload["klines"]
        except json.JSONDecodeError:
            raw = []
    if offline:
        if not raw:
            raise RuntimeError(f"--offline set but no usable cache at {path}")
        return bars_from_binance_klines(raw)
    if asof_date is None:
        raise ValueError("network fetch requires an explicit --asof date (deterministic runs)")
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(
        (datetime.strptime(asof_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp() * 1000
    )
    have_ms = {int(row[0]) for row in raw if isinstance(row, (list, tuple)) and row}
    needs_fetch = True
    if have_ms:
        # fetch only when the cache does not already span [start, asof]
        needs_fetch = min(have_ms) > start_ms or max(have_ms) < end_ms - DAY_MS
    if needs_fetch:
        fetched = fetch_daily_klines(symbol, start_ms, end_ms)
        merged: dict[int, list] = {int(r[0]): list(r) for r in raw if isinstance(r, (list, tuple)) and r}
        for row in fetched:
            merged[int(row[0])] = list(row)
        raw = [merged[key] for key in sorted(merged)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw), encoding="utf-8")
    return bars_from_binance_klines(raw)


# ---------------------------------------------------------------------------
# Gann helpers / 江恩辅助
# ---------------------------------------------------------------------------


def gann_alignment(feature: GannFeature | dict[str, Any] | None, side: str) -> int:
    """+1 supportive / -1 conflicting / 0 neutral-or-unknown.

    Mirrors the usability gates of gann_position_multiplier (stale or
    confidence < 0.40 -> unusable -> 0) so all Gann arms share one notion of
    "the feature is trustworthy enough to act on".
    """
    if feature is None:
        return 0
    data = feature.to_dict() if isinstance(feature, GannFeature) else dict(feature)
    if data.get("stale"):
        return 0
    confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
    if confidence < 0.40:
        return 0
    bias = str(data.get("bias") or "")
    normalized_side = str(side or "").lower()
    if (normalized_side == "long" and bias == "support_long") or (normalized_side == "short" and bias == "support_short"):
        return 1
    if (normalized_side == "long" and bias == "support_short") or (normalized_side == "short" and bias == "support_long"):
        return -1
    return 0


def default_gann_provider(symbol: str, bars: list[dict[str, Any]], asof_index: int) -> GannFeature | None:
    """Stateless default provider (the CLI wires its own memoized one)."""
    return compute_gann_feature(symbol, bars, asof_index=asof_index)


# ---------------------------------------------------------------------------
# Portfolio simulation engine / 组合模拟引擎
# ---------------------------------------------------------------------------


@dataclass
class _OpenPosition:
    candidate: Candidate
    side_sign: int  # +1 long, -1 short
    qty: float
    base_price: float  # fill reference before slippage
    entry_fill: float  # after slippage
    entry_fee: float
    stop_price: float
    tp_price: float
    entry_date: str
    bars_held: int = 0
    gann_multiplier: float = 1.0
    gann_alignment: int = 0


def _in_window(date: str, start: str | None, end: str | None) -> bool:
    if start is not None and date < start:
        return False
    if end is not None and date > end:
        return False
    return True


def _bar_exit(pos: _OpenPosition, bar: dict[str, Any], *, is_entry_bar: bool) -> tuple[float, str] | None:
    """Return (raw_exit_price, reason) if the position exits inside this bar.

    Stop-loss is checked before take-profit (conservative same-bar rule).
    Gap rule: a bar opening beyond the stop/target fills at the open.
    On the entry bar the position fills at the bar open, so gaps at the open
    are only meaningful when entry_ref_price was supplied.
    """
    open_ = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])
    if pos.side_sign > 0:  # long
        if open_ <= pos.stop_price:
            return open_, "stop_loss"
        if low <= pos.stop_price:
            return pos.stop_price, "stop_loss"
        if not is_entry_bar and open_ >= pos.tp_price:
            return open_, "take_profit"
        if high >= pos.tp_price:
            return pos.tp_price, "take_profit"
    else:  # short
        if open_ >= pos.stop_price:
            return open_, "stop_loss"
        if high >= pos.stop_price:
            return pos.stop_price, "stop_loss"
        if not is_entry_bar and open_ <= pos.tp_price:
            return open_, "take_profit"
        if low <= pos.tp_price:
            return pos.tp_price, "take_profit"
    return None


def simulate_portfolio(
    candidates: list[Candidate],
    klines_by_symbol: dict[str, list[dict[str, Any]]],
    arm_config: ArmConfig,
    cost_config: CostConfig,
    portfolio_config: PortfolioConfig,
    gann_provider: Callable[[str, list[dict[str, Any]], int], GannFeature | None] | None = None,
) -> ArmResult:
    """Run one arm over the full candidate stream. Pure & deterministic.

    klines_by_symbol values may be bar dicts (date/open/high/low/close) or raw
    Binance kline arrays (auto-converted).
    """
    provider = gann_provider or default_gann_provider
    fee_rate = cost_config.fee_bps / 10_000.0
    slip_rate = cost_config.slippage_bps / 10_000.0

    bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
    index_by_symbol: dict[str, dict[str, int]] = {}
    for symbol, rows in klines_by_symbol.items():
        bars = rows
        if rows and not isinstance(rows[0], dict):
            bars = bars_from_binance_klines(rows)
        bars_by_symbol[symbol.upper()] = bars
        index_by_symbol[symbol.upper()] = {bar["date"]: i for i, bar in enumerate(bars)}

    result = ArmResult(arm=arm_config.name, initial_equity=portfolio_config.initial_equity)

    # Map candidates to their fill dates (T+1 bar of the candidate's symbol).
    fills_by_date: dict[str, list[tuple[Candidate, int]]] = defaultdict(list)
    for cand in candidates:
        bars = bars_by_symbol.get(cand.symbol)
        idx_map = index_by_symbol.get(cand.symbol)
        if not bars or idx_map is None or cand.date not in idx_map:
            result.skip_events.append((cand.date, "skipped_no_data"))
            continue
        idx_t = idx_map[cand.date]
        if idx_t + 1 >= len(bars):
            result.skip_events.append((cand.date, "skipped_no_next_bar"))
            continue
        fills_by_date[bars[idx_t + 1]["date"]].append((cand, idx_t))

    calendar = sorted({bar["date"] for bars in bars_by_symbol.values() for bar in bars})
    open_positions: list[_OpenPosition] = []
    equity_realized = float(portfolio_config.initial_equity)
    last_close: dict[str, float] = {}
    last_date = calendar[-1] if calendar else None

    def close_position(pos: _OpenPosition, raw_exit: float, reason: str, exit_date: str) -> None:
        nonlocal equity_realized
        # adverse slippage on the exit side (long sells lower, short buys higher)
        exit_fill = raw_exit * (1.0 - slip_rate) if pos.side_sign > 0 else raw_exit * (1.0 + slip_rate)
        exit_fee = pos.qty * exit_fill * fee_rate
        pnl = pos.side_sign * pos.qty * (exit_fill - pos.entry_fill) - pos.entry_fee - exit_fee
        notional = pos.qty * pos.base_price
        equity_realized += pnl
        result.trades.append(
            Trade(
                symbol=pos.candidate.symbol,
                side=pos.candidate.side,
                entry_date=pos.entry_date,
                exit_date=exit_date,
                entry_price=pos.entry_fill,
                exit_price=exit_fill,
                qty=pos.qty,
                notional=notional,
                pnl=pnl,
                return_pct=(pnl / notional * 100.0) if notional > 0 else 0.0,
                exit_reason=reason,
                gann_multiplier=pos.gann_multiplier,
                gann_alignment=pos.gann_alignment,
            )
        )

    for date in calendar:
        # ---- 1) entries at today's open (slots as of the open) ----
        pending = fills_by_date.get(date, [])
        ranked: list[tuple[float, str, Candidate, int, GannFeature | None, int]] = []
        for cand, idx_t in pending:
            feature: GannFeature | None = None
            alignment = 0
            if arm_config.needs_gann:
                feature = provider(cand.symbol, bars_by_symbol[cand.symbol], idx_t)
                alignment = gann_alignment(feature, cand.side)
            adj_score = cand.score
            if arm_config.use_gann_scoring:
                adj_score = cand.score * (1.0 + 0.1 * alignment)
            ranked.append((adj_score, cand.symbol, cand, idx_t, feature, alignment))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        entered_today = 0
        for adj_score, _sym, cand, idx_t, feature, alignment in ranked:
            same_symbol = sum(1 for pos in open_positions if pos.candidate.symbol == cand.symbol)
            if same_symbol >= portfolio_config.max_per_symbol:
                result.skip_events.append((date, "skipped_same_symbol"))
                continue
            if arm_config.use_gann_filter and alignment < 0:
                result.skip_events.append((date, "skipped_gann_filter"))
                continue
            if len(open_positions) >= portfolio_config.max_concurrent:
                result.skip_events.append((date, "skipped_full"))
                continue
            bars = bars_by_symbol[cand.symbol]
            fill_bar = bars[idx_t + 1]
            base_price = cand.entry_ref_price if cand.entry_ref_price is not None else float(fill_bar["open"])
            if base_price <= 0:
                result.skip_events.append((date, "skipped_bad_price"))
                continue
            multiplier = 1.0
            if arm_config.use_gann_sizing:
                multiplier = gann_position_multiplier(feature, cand.side)
            risk_budget = equity_realized * portfolio_config.risk_per_trade_pct / 100.0
            notional = risk_budget / (cand.stop_loss_pct / 100.0) * multiplier
            if notional <= 0:
                result.skip_events.append((date, "skipped_zero_notional"))
                continue
            qty = notional / base_price
            side_sign = 1 if cand.side == "long" else -1
            entry_fill = base_price * (1.0 + slip_rate) if side_sign > 0 else base_price * (1.0 - slip_rate)
            stop_price = base_price * (1.0 - side_sign * cand.stop_loss_pct / 100.0)
            tp_price = base_price * (1.0 + side_sign * cand.take_profit_pct / 100.0)
            open_positions.append(
                _OpenPosition(
                    candidate=cand,
                    side_sign=side_sign,
                    qty=qty,
                    base_price=base_price,
                    entry_fill=entry_fill,
                    entry_fee=qty * entry_fill * fee_rate,
                    stop_price=stop_price,
                    tp_price=tp_price,
                    entry_date=date,
                    gann_multiplier=multiplier,
                    gann_alignment=alignment,
                )
            )
            entered_today += 1
        max_positions_today = len(open_positions)

        # ---- 2) intraday & time exits on today's bar ----
        still_open: list[_OpenPosition] = []
        for pos in sorted(open_positions, key=lambda p: (p.candidate.symbol, p.entry_date)):
            idx_map = index_by_symbol[pos.candidate.symbol]
            bar_idx = idx_map.get(date)
            if bar_idx is None:  # symbol has no bar today (calendar gap)
                still_open.append(pos)
                continue
            bar = bars_by_symbol[pos.candidate.symbol][bar_idx]
            pos.bars_held += 1
            last_close[pos.candidate.symbol] = float(bar["close"])
            exit_hit = _bar_exit(pos, bar, is_entry_bar=pos.entry_date == date)
            if exit_hit is not None:
                close_position(pos, exit_hit[0], exit_hit[1], date)
            elif pos.bars_held >= pos.candidate.max_hold_days:
                close_position(pos, float(bar["close"]), "time_exit", date)
            elif date == last_date:
                close_position(pos, float(bar["close"]), "end_of_data", date)
            else:
                still_open.append(pos)
        open_positions = still_open

        # update marks for symbols without positions too
        for symbol, idx_map in index_by_symbol.items():
            bar_idx = idx_map.get(date)
            if bar_idx is not None:
                last_close[symbol] = float(bars_by_symbol[symbol][bar_idx]["close"])

        # ---- 3) end-of-day mark-to-market ----
        unrealized = 0.0
        gross_notional = 0.0
        for pos in open_positions:
            mark = last_close.get(pos.candidate.symbol, pos.base_price)
            unrealized += pos.side_sign * pos.qty * (mark - pos.entry_fill) - pos.entry_fee
            gross_notional += pos.qty * mark
        equity_mark = equity_realized + unrealized
        exposure_pct = (gross_notional / equity_mark * 100.0) if equity_mark > 0 else 0.0
        result.daily.append(
            DailyRecord(
                date=date,
                equity=equity_mark,
                exposure_pct=exposure_pct,
                open_positions=max_positions_today,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Metrics / 指标
# ---------------------------------------------------------------------------


def compute_metrics(
    result: ArmResult,
    window_name: str,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Metrics for one window. Trade stats split by ENTRY date; equity-curve
    stats (total return / drawdown / exposure) from the window's daily records."""
    trades = [t for t in result.trades if _in_window(t.entry_date, start, end)]
    daily = [d for d in result.daily if _in_window(d.date, start, end)]

    # window start equity = last equity mark strictly before the window
    start_equity = result.initial_equity
    if start is not None:
        prior = [d.equity for d in result.daily if d.date < start]
        if prior:
            start_equity = prior[-1]
    end_equity = daily[-1].equity if daily else start_equity
    total_return_pct = (end_equity / start_equity - 1.0) * 100.0 if start_equity > 0 else 0.0

    max_drawdown_pct = 0.0
    peak = start_equity
    for record in daily:
        peak = max(peak, record.equity)
        if peak > 0:
            max_drawdown_pct = min(max_drawdown_pct, (record.equity / peak - 1.0) * 100.0)

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    if losses:
        profit_factor = sum(wins) / abs(sum(losses))
    else:
        profit_factor = 999.0 if wins else 0.0
    win_rate_pct = len(wins) / len(trades) * 100.0 if trades else 0.0
    avg_trade_return_pct = sum(t.return_pct for t in trades) / len(trades) if trades else 0.0
    worst_trade_pct = min((t.return_pct for t in trades), default=0.0)
    avg_exposure_pct = sum(d.exposure_pct for d in daily) / len(daily) if daily else 0.0
    max_simultaneous = max((d.open_positions for d in daily), default=0)

    return {
        "arm": result.arm,
        "window": window_name,
        "total_return_pct": round(total_return_pct, 6),
        "max_drawdown_pct": round(max_drawdown_pct, 6),
        "profit_factor": round(profit_factor, 6),
        "win_rate_pct": round(win_rate_pct, 6),
        "avg_trade_return_pct": round(avg_trade_return_pct, 6),
        "worst_trade_pct": round(worst_trade_pct, 6),
        "trade_count": len(trades),
        "skipped_full": result.skip_count("skipped_full", start, end),
        "skipped_same_symbol": result.skip_count("skipped_same_symbol", start, end),
        "skipped_gann_filter": result.skip_count("skipped_gann_filter", start, end),
        "avg_exposure_pct": round(avg_exposure_pct, 6),
        "max_simultaneous_positions": max_simultaneous,
        "funding_omitted": result.funding_omitted,
    }


# ---------------------------------------------------------------------------
# Demo candidate generator / 演示候选生成器
# ---------------------------------------------------------------------------


def generate_demo_candidates(
    klines_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    sma_period: int = 20,
    default_stop_pct: float = 3.0,
    default_tp_pct: float = 6.0,
    default_max_hold_days: int = 14,
) -> list[Candidate]:
    """DEMO rule (documented, NOT V6): daily close crossing above its
    ``sma_period``-day SMA emits an open_long candidate; crossing below emits
    open_short. score = |close - sma| / sma. 演示规则：收盘上/下穿 SMA。"""
    candidates: list[Candidate] = []
    for symbol in sorted(klines_by_symbol):
        bars = klines_by_symbol[symbol]
        if bars and not isinstance(bars[0], dict):
            bars = bars_from_binance_klines(bars)
        closes = [float(bar["close"]) for bar in bars]
        for i in range(sma_period, len(closes)):
            sma_now = sum(closes[i - sma_period + 1 : i + 1]) / sma_period
            sma_prev = sum(closes[i - sma_period : i]) / sma_period
            close_now, close_prev = closes[i], closes[i - 1]
            if sma_now <= 0:
                continue
            action: str | None = None
            if close_prev <= sma_prev and close_now > sma_now:
                action = "open_long"
            elif close_prev >= sma_prev and close_now < sma_now:
                action = "open_short"
            if action is None:
                continue
            candidates.append(
                Candidate(
                    date=bars[i]["date"],
                    symbol=symbol.upper(),
                    action=action,
                    score=abs(close_now - sma_now) / sma_now,
                    entry_ref_price=None,
                    stop_loss_pct=default_stop_pct,
                    take_profit_pct=default_tp_pct,
                    max_hold_days=default_max_hold_days,
                )
            )
    return candidates


# ---------------------------------------------------------------------------
# Reporting / 输出
# ---------------------------------------------------------------------------


def write_window_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})


def format_table(rows: list[dict[str, Any]]) -> str:
    columns = list(METRIC_COLUMNS)
    rendered = [[str(row.get(col, "")) for col in columns] for row in rows]
    widths = [max(len(columns[i]), *(len(r[i]) for r in rendered)) if rendered else len(columns[i]) for i in range(len(columns))]
    lines = ["  ".join(columns[i].ljust(widths[i]) for i in range(len(columns)))]
    lines.append("  ".join("-" * widths[i] for i in range(len(columns))))
    for r in rendered:
        lines.append("  ".join(r[i].ljust(widths[i]) for i in range(len(columns))))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portfolio-level V6 Gann A/B backtest (see module docstring for schema & rules)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidates-file", type=Path, help="JSONL candidates file (schema in module docstring)")
    source.add_argument("--demo", action="store_true", help="generate DEMO candidates (20d SMA cross) — not V6 production signals")
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT",
        help="comma list; used for --demo universe (candidates mode derives symbols from the file)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "gann_eval_cache",
        help="kline cache dir (raw Binance kline array JSON, {SYMBOL}_1d.json)",
    )
    parser.add_argument("--offline", action="store_true", help="use kline cache only, no network")
    parser.add_argument("--start", default="2021-01-01", help="kline history start date (allow Gann warm-up before first candidate)")
    parser.add_argument("--asof", default=None, help="last kline date to fetch (YYYY-MM-DD); REQUIRED when fetching from network")
    parser.add_argument("--fee-bps", type=float, default=5.0, help="taker fee per side, bps")
    parser.add_argument("--slippage-bps", type=float, default=3.0, help="adverse slippage per side, bps")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=1.0)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--max-per-symbol", type=int, default=1)
    parser.add_argument("--default-stop-pct", type=float, default=3.0)
    parser.add_argument("--default-tp-pct", type=float, default=6.0)
    parser.add_argument("--max-hold-days", type=int, default=14)
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--validation-start", default="2025-01-01")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "gann_eval")
    parser.add_argument("--gann-lookback-days", type=int, default=365)
    parser.add_argument("--gann-min-cycle-days", type=int, default=14)
    parser.add_argument("--gann-min-drawdown-pct", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    demo_mode = bool(args.demo)
    if demo_mode:
        symbols = sorted({s.strip().upper() for s in str(args.symbols).split(",") if s.strip()})
        candidates: list[Candidate] | None = None
    else:
        candidates = load_candidates(
            args.candidates_file,
            default_stop_pct=args.default_stop_pct,
            default_tp_pct=args.default_tp_pct,
            default_max_hold_days=args.max_hold_days,
        )
        symbols = sorted({cand.symbol for cand in candidates})

    klines_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        klines_by_symbol[symbol] = load_klines(
            symbol,
            args.cache_dir,
            offline=args.offline,
            start_date=args.start,
            asof_date=args.asof,
        )
        if not klines_by_symbol[symbol]:
            raise RuntimeError(f"no daily klines available for {symbol}")

    if demo_mode:
        candidates = generate_demo_candidates(
            klines_by_symbol,
            default_stop_pct=args.default_stop_pct,
            default_tp_pct=args.default_tp_pct,
            default_max_hold_days=args.max_hold_days,
        )
        print(f"[{DEMO_LABEL}] generated {len(candidates)} demo candidates "
              f"(20d SMA cross) across {len(symbols)} symbols")
    assert candidates is not None

    cost_config = CostConfig(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    portfolio_config = PortfolioConfig(
        initial_equity=args.initial_equity,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_concurrent=args.max_concurrent,
        max_per_symbol=args.max_per_symbol,
    )

    gann_memo: dict[tuple[str, int], GannFeature | None] = {}

    def provider(symbol: str, bars: list[dict[str, Any]], asof_index: int) -> GannFeature | None:
        key = (symbol, asof_index)
        if key not in gann_memo:
            gann_memo[key] = compute_gann_feature(
                symbol,
                bars,
                asof_index=asof_index,
                lookback_days=args.gann_lookback_days,
                min_cycle_days=args.gann_min_cycle_days,
                min_drawdown_pct=args.gann_min_drawdown_pct,
            )
        return gann_memo[key]

    windows = [
        ("train", None, args.train_end),
        ("validation", args.validation_start, None),
    ]
    rows_by_window: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in windows}
    for arm_name in ARM_NAMES:
        result = simulate_portfolio(
            candidates,
            klines_by_symbol,
            ARM_CONFIGS[arm_name],
            cost_config,
            portfolio_config,
            gann_provider=provider,
        )
        for window_name, start, end in windows:
            rows_by_window[window_name].append(compute_metrics(result, window_name, start, end))

    suffix = "_demo" if demo_mode else ""
    print(f"\nfunding_omitted=true (funding rates are NOT modeled in this backtest)")
    print(f"candidates={len(candidates)} symbols={len(symbols)} "
          f"fee_bps={args.fee_bps} slippage_bps={args.slippage_bps} "
          f"train_end={args.train_end} validation_start={args.validation_start}")
    if demo_mode:
        print(f"NOTE: {DEMO_LABEL}")
    for window_name, _, _ in windows:
        out_path = args.output_dir / f"v6_gann_ab_portfolio_{window_name}{suffix}.csv"
        write_window_csv(out_path, rows_by_window[window_name])
        print(f"\n== {window_name} window ==")
        print(format_table(rows_by_window[window_name]))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
