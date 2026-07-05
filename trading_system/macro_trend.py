from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.binance import BinanceFuturesClient
from trading_system.persistence import atomic_write_json, load_json


@dataclass(slots=True)
class MacroTrendSignal:
    btc_price: float
    btc_ma7: float
    btc_ma20: float
    btc_trend: str
    btc_trend_strength: str
    btc_distance_pct: float
    btc_distance_ma20_pct: float
    btc_ma7_slope_pct_3d: float
    btc_ma210: float
    btc_ma672: float
    btc_distance_ma210_pct: float
    btc_distance_ma672_pct: float
    btc_ma210_slope_pct: float
    btc_ma672_slope_pct: float
    btc_ma210_above_ma672: bool
    eth_price: float
    eth_ma7: float
    eth_ma20: float
    eth_trend: str
    eth_trend_strength: str
    eth_distance_pct: float
    eth_distance_ma20_pct: float
    eth_ma7_slope_pct_3d: float
    macro_direction: str
    macro_confidence: str
    source: str
    computed_at: str
    kline_timezone: str = "UTC"
    cache_age_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def neutral(cls, *, source: str = "fallback_neutral", error: str = "") -> "MacroTrendSignal":
        return cls(
            btc_price=0.0,
            btc_ma7=0.0,
            btc_ma20=0.0,
            btc_trend="neutral",
            btc_trend_strength="flat",
            btc_distance_pct=0.0,
            btc_distance_ma20_pct=0.0,
            btc_ma7_slope_pct_3d=0.0,
            btc_ma210=0.0,
            btc_ma672=0.0,
            btc_distance_ma210_pct=0.0,
            btc_distance_ma672_pct=0.0,
            btc_ma210_slope_pct=0.0,
            btc_ma672_slope_pct=0.0,
            btc_ma210_above_ma672=False,
            eth_price=0.0,
            eth_ma7=0.0,
            eth_ma20=0.0,
            eth_trend="neutral",
            eth_trend_strength="flat",
            eth_distance_pct=0.0,
            eth_distance_ma20_pct=0.0,
            eth_ma7_slope_pct_3d=0.0,
            macro_direction="mixed",
            macro_confidence="neutral",
            source=source,
            computed_at=_utc_now_iso(),
            error=error,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MacroTrendSignal":
        neutral = cls.neutral(source=str(payload.get("source") or "cache"))
        data = neutral.to_dict()
        data.update({key: payload.get(key, value) for key, value in data.items()})
        return cls(**data)


def compute_macro_trend(
    client: BinanceFuturesClient,
    *,
    cache_file: Path,
    cache_ttl_seconds: int = 6 * 60 * 60,
    ma_period: int = 7,
    ma20_period: int = 20,
    neutral_band_pct: float = 0.75,
    slope_neutral_threshold_pct: float = 1.0,
) -> MacroTrendSignal:
    try:
        signal = _compute_live_macro_trend(
            client,
            ma_period=max(2, int(ma_period)),
            ma20_period=max(2, int(ma20_period)),
            neutral_band_pct=float(neutral_band_pct),
            slope_neutral_threshold_pct=float(slope_neutral_threshold_pct),
        )
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cache_file, signal.to_dict())
        return signal
    except Exception as exc:
        cached = _load_fresh_cache(cache_file, cache_ttl_seconds, error=str(exc))
        if cached is not None:
            return cached
        return MacroTrendSignal.neutral(error=str(exc))


def _compute_live_macro_trend(
    client: BinanceFuturesClient,
    *,
    ma_period: int,
    ma20_period: int,
    neutral_band_pct: float,
    slope_neutral_threshold_pct: float,
) -> MacroTrendSignal:
    limit = max(ma20_period + 4, ma_period + 4, 24)
    btc = _asset_signal(
        client.get_json("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1d", "limit": limit}),
        ma_period=ma_period,
        ma20_period=ma20_period,
        neutral_band_pct=neutral_band_pct,
        slope_neutral_threshold_pct=slope_neutral_threshold_pct,
    )
    eth = _asset_signal(
        client.get_json("/fapi/v1/klines", {"symbol": "ETHUSDT", "interval": "1d", "limit": limit}),
        ma_period=ma_period,
        ma20_period=ma20_period,
        neutral_band_pct=neutral_band_pct,
        slope_neutral_threshold_pct=slope_neutral_threshold_pct,
    )
    btc_hourly = _btc_hourly_ma_signal(client)
    macro_direction, macro_confidence = _btc_only_macro_direction(btc["trend"], btc_hourly)
    return MacroTrendSignal(
        btc_price=btc["price"],
        btc_ma7=btc["ma7"],
        btc_ma20=btc["ma20"],
        btc_trend=btc["trend"],
        btc_trend_strength=btc["trend_strength"],
        btc_distance_pct=btc["distance_pct"],
        btc_distance_ma20_pct=btc["distance_ma20_pct"],
        btc_ma7_slope_pct_3d=btc["ma7_slope_pct_3d"],
        btc_ma210=btc_hourly["ma210"],
        btc_ma672=btc_hourly["ma672"],
        btc_distance_ma210_pct=btc_hourly["distance_ma210_pct"],
        btc_distance_ma672_pct=btc_hourly["distance_ma672_pct"],
        btc_ma210_slope_pct=btc_hourly["ma210_slope_pct"],
        btc_ma672_slope_pct=btc_hourly["ma672_slope_pct"],
        btc_ma210_above_ma672=btc_hourly["ma210_above_ma672"],
        eth_price=eth["price"],
        eth_ma7=eth["ma7"],
        eth_ma20=eth["ma20"],
        eth_trend=eth["trend"],
        eth_trend_strength=eth["trend_strength"],
        eth_distance_pct=eth["distance_pct"],
        eth_distance_ma20_pct=eth["distance_ma20_pct"],
        eth_ma7_slope_pct_3d=eth["ma7_slope_pct_3d"],
        macro_direction=macro_direction,
        macro_confidence=macro_confidence,
        source="live",
        computed_at=_utc_now_iso(),
    )


def _asset_signal(
    klines: list[Any],
    *,
    ma_period: int,
    ma20_period: int,
    neutral_band_pct: float,
    slope_neutral_threshold_pct: float,
) -> dict[str, float | str]:
    closes = [_kline_close(item) for item in klines]
    closes = [item for item in closes if item > 0]
    required = max(ma20_period, ma_period + 3)
    if len(closes) < required:
        raise ValueError(f"not enough daily klines: got={len(closes)} required={required}")

    price = closes[-1]
    ma7 = _mean(closes[-ma_period:])
    ma20 = _mean(closes[-ma20_period:])
    ma7_3d_ago = _mean(closes[-(ma_period + 3):-3])
    distance_pct = _pct(price, ma7)
    distance_ma20_pct = _pct(price, ma20)
    slope_pct = _pct(ma7, ma7_3d_ago)
    trend = _trend_from_distance_and_slope(
        distance_pct,
        slope_pct,
        neutral_band_pct=neutral_band_pct,
        slope_neutral_threshold_pct=slope_neutral_threshold_pct,
    )
    return {
        "price": round(price, 8),
        "ma7": round(ma7, 8),
        "ma20": round(ma20, 8),
        "trend": trend,
        "trend_strength": _trend_strength(slope_pct, slope_neutral_threshold_pct),
        "distance_pct": round(distance_pct, 4),
        "distance_ma20_pct": round(distance_ma20_pct, 4),
        "ma7_slope_pct_3d": round(slope_pct, 4),
    }


def _trend_from_distance_and_slope(
    distance_pct: float,
    slope_pct: float,
    *,
    neutral_band_pct: float,
    slope_neutral_threshold_pct: float,
) -> str:
    if abs(distance_pct) < neutral_band_pct:
        return "neutral"
    if abs(slope_pct) < slope_neutral_threshold_pct:
        return "neutral"
    if distance_pct > 0 and slope_pct > 0:
        return "bullish"
    if distance_pct < 0 and slope_pct < 0:
        return "bearish"
    return "neutral"


def _trend_strength(slope_pct: float, threshold_pct: float) -> str:
    abs_slope = abs(slope_pct)
    if abs_slope < threshold_pct:
        return "flat"
    if abs_slope < threshold_pct * 2:
        return "weak"
    return "strong"


def _btc_hourly_ma_signal(client: BinanceFuturesClient) -> dict[str, float | bool]:
    """Fetch BTC 1h klines and compute MA210/MA672 indicators."""
    klines = client.get_json("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 700})
    closes = [float(k[4]) for k in klines if float(k[4]) > 0]
    price = closes[-1]
    ma210 = _mean(closes[-210:]) if len(closes) >= 210 else price
    ma672 = _mean(closes[-672:]) if len(closes) >= 672 else ma210
    ma210_24h_ago = _mean(closes[-234:-24]) if len(closes) >= 234 else ma210
    ma672_24h_ago = _mean(closes[-696:-24]) if len(closes) >= 696 else ma672
    return {
        "ma210": round(ma210, 2),
        "ma672": round(ma672, 2),
        "distance_ma210_pct": round(_pct(price, ma210), 4),
        "distance_ma672_pct": round(_pct(price, ma672), 4),
        "ma210_slope_pct": round(_pct(ma210, ma210_24h_ago), 4),
        "ma672_slope_pct": round(_pct(ma672, ma672_24h_ago), 4),
        "ma210_above_ma672": ma210 > ma672,
    }


def _btc_only_macro_direction(btc_daily_trend: str, btc_hourly: dict) -> tuple[str, str]:
    """BTC is the sole directional indicator for macro judgment."""
    bullish_count = 0
    if btc_daily_trend == "bullish":
        bullish_count += 2
    elif btc_daily_trend == "bearish":
        bullish_count -= 2
    if btc_hourly["distance_ma210_pct"] > 0:
        bullish_count += 1
    if btc_hourly["distance_ma672_pct"] > 0:
        bullish_count += 1
    if btc_hourly["ma210_slope_pct"] > 0:
        bullish_count += 1
    if btc_hourly["ma210_above_ma672"]:
        bullish_count += 1
    if bullish_count >= 4:
        return "bullish", "high"
    if bullish_count >= 2:
        return "bullish", "low"
    if bullish_count <= -1:
        return "bearish", "medium" if bullish_count <= -2 else "low"
    return "mixed", "neutral"


def _btc_primary_macro_direction(btc_trend: str, eth_trend: str) -> tuple[str, str]:
    if btc_trend not in {"bullish", "bearish"}:
        return "mixed", "neutral"
    if eth_trend == btc_trend:
        return btc_trend, "high"
    if eth_trend in {"bullish", "bearish"}:
        return btc_trend, "low"
    return btc_trend, "medium"


def _load_fresh_cache(cache_file: Path, ttl_seconds: int, *, error: str) -> MacroTrendSignal | None:
    try:
        payload = load_json(cache_file, {}, lock=False)
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload:
        return None
    signal = MacroTrendSignal.from_dict(payload)
    computed_at = _parse_iso8601(signal.computed_at)
    if computed_at is None:
        return None
    age = max(0.0, (datetime.now(timezone.utc) - computed_at).total_seconds())
    if age > ttl_seconds:
        return None
    signal.source = "cache"
    signal.cache_age_seconds = round(age, 3)
    signal.error = error
    return signal


def _kline_close(kline: Any) -> float:
    if isinstance(kline, dict):
        return float(kline.get("close") or kline.get("c") or 0.0)
    return float(kline[4] or 0.0)


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _pct(value: float, base: float) -> float:
    return ((value - base) / base * 100.0) if base else 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso8601(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
