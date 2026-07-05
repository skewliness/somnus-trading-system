from __future__ import annotations

from dataclasses import dataclass


SHARED_BB_PERIODS = (96, 105, 120, 210, 300, 672, 1440, 1920, 2880, 4707)
VISIBLE_MA_PERIODS = (7, 20, 60, 96, 105, 120)
VISIBLE_VOLUME_MA_PERIODS = (5, 10)


@dataclass(slots=True)
class BBStructureSignal:
    momo_15m_bars: int
    momo_1m_bars: int
    momo_4h_bars: int
    momo_1d_bars: int
    ma7_15m: float
    ma20_15m: float
    ma60_15m: float
    ma96_15m: float
    ma105_15m: float
    ma120_15m: float
    volume_ma5_15m: float
    volume_ma10_15m: float
    bb96_upper_15m: float
    bb96_mid_15m: float
    bb96_lower_15m: float
    bb105_upper_15m: float
    bb105_mid_15m: float
    bb105_lower_15m: float
    bb120_upper_15m: float
    bb120_mid_15m: float
    bb120_lower_15m: float
    bb210_upper_15m: float
    bb210_mid_15m: float
    bb210_lower_15m: float
    bb300_upper_15m: float
    bb300_mid_15m: float
    bb300_lower_15m: float
    bb672_upper_15m: float
    bb672_mid_15m: float
    bb672_lower_15m: float
    bb1440_upper_15m: float
    bb1440_mid_15m: float
    bb1440_lower_15m: float
    bb1920_upper_15m: float
    bb1920_mid_15m: float
    bb1920_lower_15m: float
    bb2880_upper_15m: float
    bb2880_mid_15m: float
    bb2880_lower_15m: float
    bb4404_upper_15m: float
    bb4404_mid_15m: float
    bb4404_lower_15m: float
    bb4460_upper_15m: float
    bb4460_mid_15m: float
    bb4460_lower_15m: float
    bb4707_upper_15m: float
    bb4707_mid_15m: float
    bb4707_lower_15m: float
    ma7_4h: float
    ma20_4h: float
    ma60_4h: float
    ma96_4h: float
    ma105_4h: float
    ma120_4h: float
    volume_ma5_4h: float
    volume_ma10_4h: float
    bb105_upper_4h: float
    bb105_mid_4h: float
    bb105_lower_4h: float
    bb96_upper_4h: float
    bb96_mid_4h: float
    bb96_lower_4h: float
    bb120_upper_4h: float
    bb120_mid_4h: float
    bb120_lower_4h: float
    bb210_upper_4h: float
    bb210_mid_4h: float
    bb210_lower_4h: float
    bb300_upper_4h: float
    bb300_mid_4h: float
    bb300_lower_4h: float
    bb672_upper_4h: float
    bb672_mid_4h: float
    bb672_lower_4h: float
    bb1440_upper_4h: float
    bb1440_mid_4h: float
    bb1440_lower_4h: float
    bb1920_upper_4h: float
    bb1920_mid_4h: float
    bb1920_lower_4h: float
    bb2880_upper_4h: float
    bb2880_mid_4h: float
    bb2880_lower_4h: float
    bb4404_upper_4h: float
    bb4404_mid_4h: float
    bb4404_lower_4h: float
    bb4460_upper_4h: float
    bb4460_mid_4h: float
    bb4460_lower_4h: float
    bb4707_upper_4h: float
    bb4707_mid_4h: float
    bb4707_lower_4h: float
    bb105_upper_1m: float
    bb105_mid_1m: float
    bb105_lower_1m: float
    sma210_1m: float
    bb300_upper_1m: float
    bb300_mid_1m: float
    bb300_lower_1m: float
    bb672_upper_1m: float
    bb672_mid_1m: float
    bb672_lower_1m: float
    bb1920_upper_1m: float
    bb1920_mid_1m: float
    bb1920_lower_1m: float
    ma7_1d: float
    ma20_1d: float
    ma60_1d: float
    ma96_1d: float
    ma105_1d: float
    ma120_1d: float
    volume_ma5_1d: float
    volume_ma10_1d: float
    daily_ma7: float
    daily_ma20: float
    daily_ema21: float
    daily_bb365_upper: float
    daily_bb365_mid: float
    daily_bb365_lower: float
    bb96_upper_1d: float
    bb96_mid_1d: float
    bb96_lower_1d: float
    bb105_upper_1d: float
    bb105_mid_1d: float
    bb105_lower_1d: float
    bb120_upper_1d: float
    bb120_mid_1d: float
    bb120_lower_1d: float
    bb210_upper_1d: float
    bb210_mid_1d: float
    bb210_lower_1d: float
    bb300_upper_1d: float
    bb300_mid_1d: float
    bb300_lower_1d: float
    bb672_upper_1d: float
    bb672_mid_1d: float
    bb672_lower_1d: float
    bb1440_upper_1d: float
    bb1440_mid_1d: float
    bb1440_lower_1d: float
    bb1920_upper_1d: float
    bb1920_mid_1d: float
    bb1920_lower_1d: float
    bb2880_upper_1d: float
    bb2880_mid_1d: float
    bb2880_lower_1d: float
    bb4404_upper_1d: float
    bb4404_mid_1d: float
    bb4404_lower_1d: float
    bb4460_upper_1d: float
    bb4460_mid_1d: float
    bb4460_lower_1d: float
    bb4707_upper_1d: float
    bb4707_mid_1d: float
    bb4707_lower_1d: float
    daily_price_vs_ma7_pct: float
    daily_price_vs_ma20_pct: float
    daily_price_position_365: float
    bb_price_position_96_1d: float
    bb_price_position_120_1d: float
    bb_price_position_210_1d: float
    bb_price_position_300_1d: float
    bb_price_position_672_1d: float
    bb_price_position_1440_1d: float
    bb_price_position_1920_1d: float
    bb_price_position_2880_1d: float
    bb_price_position_4404_1d: float
    bb_price_position_4460_1d: float
    bb_price_position_4707_1d: float
    bb_price_position_105_15m: float
    bb_price_position_96: float
    bb_price_position_120: float
    bb_price_position_210: float
    bb_price_position_300: float
    bb_price_position_672_15m: float
    bb_price_position_1440_15m: float
    bb_price_position_1920_15m: float
    bb_price_position_2880_15m: float
    bb_price_position_4404_15m: float
    bb_price_position_4460_15m: float
    bb_price_position_4707_15m: float
    bb_price_position_105_1d: float
    bb_price_position_105_4h: float
    bb_price_position_96_4h: float
    bb_price_position_120_4h: float
    bb_price_position_210_4h: float
    bb_price_position_300_4h: float
    bb_price_position_672_4h: float
    bb_price_position_1440_4h: float
    bb_price_position_1920_4h: float
    bb_price_position_2880_4h: float
    bb_price_position_4404_4h: float
    bb_price_position_4460_4h: float
    bb_price_position_4707_4h: float
    bb_price_position_105_1m: float
    bb_price_position_300_1m: float
    bb_price_position_672_1m: float
    bb_price_position_1920_1m: float
    williams_r_15m: float
    williams_r_15m_120: float
    williams_r_1m: float
    williams_r_15m_zone: str
    williams_r_15m_120_zone: str
    williams_r_1m_zone: str
    bb_touch_zone: str


def compute_bb(closes: list[float], period: int, std_dev: float = 2.0) -> tuple[float, float, float]:
    if len(closes) < period:
        mid = closes[-1] if closes else 0.0
        return mid, mid, mid
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    sd = variance ** 0.5
    return mid + std_dev * sd, mid, mid - std_dev * sd


def compute_bb_if_ready(closes: list[float], period: int, std_dev: float = 2.0) -> tuple[float, float, float]:
    if len(closes) < period:
        return 0.0, 0.0, 0.0
    return compute_bb(closes, period, std_dev)


def compute_sma(closes: list[float], period: int) -> float:
    if not closes:
        return 0.0
    if len(closes) < period:
        return sum(closes) / len(closes)
    return sum(closes[-period:]) / period


def compute_sma_if_ready(values: list[float], period: int) -> float:
    if len(values) < period:
        return 0.0
    return compute_sma(values, period)


def compute_ema(closes: list[float], period: int) -> float:
    if not closes:
        return 0.0
    if len(closes) < period:
        return compute_sma(closes, len(closes))
    seed = compute_sma(closes[:period], period)
    multiplier = 2.0 / (period + 1.0)
    ema = seed
    for close in closes[period:]:
        ema = (close - ema) * multiplier + ema
    return ema


def _price_position(price: float, upper: float, lower: float) -> float:
    band_width = upper - lower
    if band_width <= 0:
        return 0.5
    return (price - lower) / band_width


def _pct_diff(price: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return (price - reference) / reference * 100.0


def _klines_high_low_close(klines: list[list]) -> tuple[list[float], list[float], list[float]]:
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    for kline in klines:
        if len(kline) > 4:
            highs.append(float(kline[2]))
            lows.append(float(kline[3]))
            closes.append(float(kline[4]))
    return highs, lows, closes


def _klines_volumes(klines: list[list]) -> list[float]:
    volumes: list[float] = []
    for item in klines:
        if len(item) <= 5:
            continue
        try:
            volumes.append(float(item[5]))
        except (TypeError, ValueError):
            continue
    return volumes


def compute_williams_r(klines: list[list], period: int = 14) -> float:
    highs, lows, closes = _klines_high_low_close(klines)
    if len(closes) < period:
        return -50.0
    highest = max(highs[-period:])
    lowest = min(lows[-period:])
    if highest <= lowest:
        return -50.0
    value = -100.0 * (highest - closes[-1]) / (highest - lowest)
    return max(-100.0, min(0.0, value))


def williams_zone(value: float) -> str:
    if value >= -20.0:
        return "overbought"
    if value >= -50.0:
        return "upper_mid"
    if value >= -80.0:
        return "lower_mid"
    return "oversold"


def _classify_touch_zone(
    price: float,
    bb96_upper: float, bb96_lower: float,
    bb120_upper: float, bb120_lower: float,
    bb210_upper: float, bb210_lower: float,
    bb672_15m_upper: float = 0.0, bb672_15m_lower: float = 0.0,
    bb300_upper: float = 0.0, bb300_lower: float = 0.0,
    bb1440_upper_15m: float = 0.0, bb1440_lower_15m: float = 0.0,
    bb1920_upper_15m: float = 0.0, bb1920_lower_15m: float = 0.0,
    bb2880_upper_15m: float = 0.0, bb2880_lower_15m: float = 0.0,
    bb4404_upper_15m: float = 0.0, bb4404_lower_15m: float = 0.0,
    bb4460_upper_15m: float = 0.0, bb4460_lower_15m: float = 0.0,
    bb4707_upper_15m: float = 0.0, bb4707_lower_15m: float = 0.0,
    bb105_upper_4h: float = 0.0, bb105_lower_4h: float = 0.0,
    bb210_upper_4h: float = 0.0, bb210_lower_4h: float = 0.0,
    bb300_upper_4h: float = 0.0, bb300_lower_4h: float = 0.0,
    bb672_upper: float = 0.0, bb672_lower: float = 0.0,
    bb1440_upper_4h: float = 0.0, bb1440_lower_4h: float = 0.0,
    bb105_upper_1m: float = 0.0, bb105_lower_1m: float = 0.0,
    bb300_upper_1m: float = 0.0, bb300_lower_1m: float = 0.0,
    bb672_upper_1m: float = 0.0, bb672_lower_1m: float = 0.0,
    threshold_pct: float = 0.3,
    extra_levels: tuple[tuple[str, float], ...] = (),
) -> str:
    thr = threshold_pct / 100.0
    candidates = (
        ("bb96_upper", bb96_upper),
        ("bb96_lower", bb96_lower),
        ("bb210_upper", bb210_upper),
        ("bb210_lower", bb210_lower),
        ("bb672_15m_upper", bb672_15m_upper),
        ("bb672_15m_lower", bb672_15m_lower),
        ("bb672_upper", bb672_upper),
        ("bb672_lower", bb672_lower),
        ("bb120_upper", bb120_upper),
        ("bb120_lower", bb120_lower),
        ("bb300_upper", bb300_upper),
        ("bb300_lower", bb300_lower),
        ("bb1440_upper_15m", bb1440_upper_15m),
        ("bb1440_lower_15m", bb1440_lower_15m),
        ("bb1920_upper_15m", bb1920_upper_15m),
        ("bb1920_lower_15m", bb1920_lower_15m),
        ("bb2880_upper_15m", bb2880_upper_15m),
        ("bb2880_lower_15m", bb2880_lower_15m),
        ("bb4404_upper_15m", bb4404_upper_15m),
        ("bb4404_lower_15m", bb4404_lower_15m),
        ("bb4460_upper_15m", bb4460_upper_15m),
        ("bb4460_lower_15m", bb4460_lower_15m),
        ("bb4707_upper_15m", bb4707_upper_15m),
        ("bb4707_lower_15m", bb4707_lower_15m),
        ("bb105_upper_4h", bb105_upper_4h),
        ("bb105_lower_4h", bb105_lower_4h),
        ("bb210_upper_4h", bb210_upper_4h),
        ("bb210_lower_4h", bb210_lower_4h),
        ("bb300_upper_4h", bb300_upper_4h),
        ("bb300_lower_4h", bb300_lower_4h),
        ("bb672_upper_4h", bb672_upper),
        ("bb672_lower_4h", bb672_lower),
        ("bb1440_upper_4h", bb1440_upper_4h),
        ("bb1440_lower_4h", bb1440_lower_4h),
        ("bb105_upper_1m", bb105_upper_1m),
        ("bb105_lower_1m", bb105_lower_1m),
        ("bb300_upper_1m", bb300_upper_1m),
        ("bb300_lower_1m", bb300_lower_1m),
        ("bb672_upper_1m", bb672_upper_1m),
        ("bb672_lower_1m", bb672_lower_1m),
    )
    candidates = candidates + tuple(extra_levels or ())
    for name, reference in candidates:
        if reference > 0 and abs(price - reference) / reference <= thr:
            return name
    return "none"


def compute_bb_structure(
    klines_15m: list[list], klines_4h: list[list], mark_price: float,
    threshold_pct: float = 0.3,
    klines_1m: list[list] | None = None,
    klines_1d: list[list] | None = None,
) -> BBStructureSignal:
    closes_15m = [float(k[4]) for k in klines_15m if len(k) > 4]
    closes_4h = [float(k[4]) for k in klines_4h if len(k) > 4]
    klines_1m = klines_1m or []
    klines_1d = klines_1d or []
    closes_1m = [float(k[4]) for k in klines_1m if len(k) > 4]
    closes_1d = [float(k[4]) for k in klines_1d if len(k) > 4]
    volumes_15m = _klines_volumes(klines_15m)
    volumes_4h = _klines_volumes(klines_4h)
    volumes_1d = _klines_volumes(klines_1d)

    ma7_15m = compute_sma_if_ready(closes_15m, 7)
    ma20_15m = compute_sma_if_ready(closes_15m, 20)
    ma60_15m = compute_sma_if_ready(closes_15m, 60)
    ma96_15m = compute_sma_if_ready(closes_15m, 96)
    ma105_15m = compute_sma_if_ready(closes_15m, 105)
    ma120_15m = compute_sma_if_ready(closes_15m, 120)
    volume_ma5_15m = compute_sma_if_ready(volumes_15m, 5)
    volume_ma10_15m = compute_sma_if_ready(volumes_15m, 10)
    bb96_upper, bb96_mid, bb96_lower = compute_bb(closes_15m, 96)
    bb105_upper_15m, bb105_mid_15m, bb105_lower_15m = compute_bb_if_ready(closes_15m, 105)
    bb120_upper, bb120_mid, bb120_lower = compute_bb_if_ready(closes_15m, 120)
    bb210_upper, bb210_mid, bb210_lower = compute_bb(closes_15m, 210)
    bb300_upper, bb300_mid, bb300_lower = compute_bb_if_ready(closes_15m, 300)
    bb672_upper_15m, bb672_mid_15m, bb672_lower_15m = compute_bb_if_ready(closes_15m, 672)
    bb1440_upper_15m, bb1440_mid_15m, bb1440_lower_15m = compute_bb_if_ready(closes_15m, 1440)
    bb1920_upper_15m, bb1920_mid_15m, bb1920_lower_15m = compute_bb_if_ready(closes_15m, 1920)
    bb2880_upper_15m, bb2880_mid_15m, bb2880_lower_15m = compute_bb_if_ready(closes_15m, 2880)
    bb4404_upper_15m, bb4404_mid_15m, bb4404_lower_15m = compute_bb_if_ready(closes_15m, 4404)
    bb4460_upper_15m, bb4460_mid_15m, bb4460_lower_15m = compute_bb_if_ready(closes_15m, 4460)
    bb4707_upper_15m, bb4707_mid_15m, bb4707_lower_15m = compute_bb_if_ready(closes_15m, 4707)
    ma7_4h = compute_sma_if_ready(closes_4h, 7)
    ma20_4h = compute_sma_if_ready(closes_4h, 20)
    ma60_4h = compute_sma_if_ready(closes_4h, 60)
    ma96_4h = compute_sma_if_ready(closes_4h, 96)
    ma105_4h = compute_sma_if_ready(closes_4h, 105)
    ma120_4h = compute_sma_if_ready(closes_4h, 120)
    volume_ma5_4h = compute_sma_if_ready(volumes_4h, 5)
    volume_ma10_4h = compute_sma_if_ready(volumes_4h, 10)
    bb105_upper_4h, bb105_mid_4h, bb105_lower_4h = compute_bb_if_ready(closes_4h, 105)
    bb96_upper_4h, bb96_mid_4h, bb96_lower_4h = compute_bb_if_ready(closes_4h, 96)
    bb120_upper_4h, bb120_mid_4h, bb120_lower_4h = compute_bb_if_ready(closes_4h, 120)
    bb210_upper_4h, bb210_mid_4h, bb210_lower_4h = compute_bb_if_ready(closes_4h, 210)
    bb300_upper_4h, bb300_mid_4h, bb300_lower_4h = compute_bb_if_ready(closes_4h, 300)
    bb672_upper_4h, bb672_mid_4h, bb672_lower_4h = compute_bb_if_ready(closes_4h, 672)
    bb1440_upper_4h, bb1440_mid_4h, bb1440_lower_4h = compute_bb_if_ready(closes_4h, 1440)
    bb1920_upper_4h, bb1920_mid_4h, bb1920_lower_4h = compute_bb_if_ready(closes_4h, 1920)
    bb2880_upper_4h, bb2880_mid_4h, bb2880_lower_4h = compute_bb_if_ready(closes_4h, 2880)
    bb4404_upper_4h, bb4404_mid_4h, bb4404_lower_4h = compute_bb_if_ready(closes_4h, 4404)
    bb4460_upper_4h, bb4460_mid_4h, bb4460_lower_4h = compute_bb_if_ready(closes_4h, 4460)
    bb4707_upper_4h, bb4707_mid_4h, bb4707_lower_4h = compute_bb_if_ready(closes_4h, 4707)
    bb105_upper_1m, bb105_mid_1m, bb105_lower_1m = compute_bb_if_ready(closes_1m, 105)
    sma210_1m = compute_sma(closes_1m, 210)
    bb300_upper_1m, bb300_mid_1m, bb300_lower_1m = compute_bb_if_ready(closes_1m, 300)
    bb672_upper_1m, bb672_mid_1m, bb672_lower_1m = compute_bb_if_ready(closes_1m, 672)
    bb1920_upper_1m, bb1920_mid_1m, bb1920_lower_1m = compute_bb_if_ready(closes_1m, 1920)
    ma7_1d = compute_sma_if_ready(closes_1d, 7)
    ma20_1d = compute_sma_if_ready(closes_1d, 20)
    ma60_1d = compute_sma_if_ready(closes_1d, 60)
    ma96_1d = compute_sma_if_ready(closes_1d, 96)
    ma105_1d = compute_sma_if_ready(closes_1d, 105)
    ma120_1d = compute_sma_if_ready(closes_1d, 120)
    volume_ma5_1d = compute_sma_if_ready(volumes_1d, 5)
    volume_ma10_1d = compute_sma_if_ready(volumes_1d, 10)
    daily_ma7 = compute_sma(closes_1d, 7)
    daily_ma20 = compute_sma(closes_1d, 20)
    daily_ema21 = compute_ema(closes_1d, 21)
    daily_bb365_upper, daily_bb365_mid, daily_bb365_lower = compute_bb_if_ready(closes_1d, 365)
    bb96_upper_1d, bb96_mid_1d, bb96_lower_1d = compute_bb_if_ready(closes_1d, 96)
    bb105_upper_1d, bb105_mid_1d, bb105_lower_1d = compute_bb_if_ready(closes_1d, 105)
    bb120_upper_1d, bb120_mid_1d, bb120_lower_1d = compute_bb_if_ready(closes_1d, 120)
    bb210_upper_1d, bb210_mid_1d, bb210_lower_1d = compute_bb_if_ready(closes_1d, 210)
    bb300_upper_1d, bb300_mid_1d, bb300_lower_1d = compute_bb_if_ready(closes_1d, 300)
    bb672_upper_1d, bb672_mid_1d, bb672_lower_1d = compute_bb_if_ready(closes_1d, 672)
    bb1440_upper_1d, bb1440_mid_1d, bb1440_lower_1d = compute_bb_if_ready(closes_1d, 1440)
    bb1920_upper_1d, bb1920_mid_1d, bb1920_lower_1d = compute_bb_if_ready(closes_1d, 1920)
    bb2880_upper_1d, bb2880_mid_1d, bb2880_lower_1d = compute_bb_if_ready(closes_1d, 2880)
    bb4404_upper_1d, bb4404_mid_1d, bb4404_lower_1d = compute_bb_if_ready(closes_1d, 4404)
    bb4460_upper_1d, bb4460_mid_1d, bb4460_lower_1d = compute_bb_if_ready(closes_1d, 4460)
    bb4707_upper_1d, bb4707_mid_1d, bb4707_lower_1d = compute_bb_if_ready(closes_1d, 4707)
    williams_r_15m = compute_williams_r(klines_15m)
    williams_r_15m_120 = compute_williams_r(klines_15m, 120)
    williams_r_1m = compute_williams_r(klines_1m)

    scope = locals()
    extra_touch_levels: list[tuple[str, float]] = []
    for timeframe in ("15m", "1d"):
        for band in ("upper", "lower"):
            name = f"bb105_{band}_{timeframe}"
            extra_touch_levels.append((name, float(scope.get(name, 0.0) or 0.0)))
    for timeframe in ("4h", "1d"):
        for period in SHARED_BB_PERIODS:
            for band in ("upper", "lower"):
                name = f"bb{period}_{band}_{timeframe}"
                extra_touch_levels.append((name, float(scope.get(name, 0.0) or 0.0)))

    touch_zone = _classify_touch_zone(
        mark_price,
        bb96_upper, bb96_lower,
        bb120_upper, bb120_lower,
        bb210_upper, bb210_lower,
        bb672_upper_15m, bb672_lower_15m,
        bb300_upper, bb300_lower,
        bb1440_upper_15m, bb1440_lower_15m,
        bb1920_upper_15m, bb1920_lower_15m,
        bb2880_upper_15m, bb2880_lower_15m,
        bb4404_upper_15m, bb4404_lower_15m,
        bb4460_upper_15m, bb4460_lower_15m,
        bb4707_upper_15m, bb4707_lower_15m,
        bb105_upper_4h, bb105_lower_4h,
        bb210_upper_4h, bb210_lower_4h,
        bb300_upper_4h, bb300_lower_4h,
        bb672_upper_4h, bb672_lower_4h,
        bb1440_upper_4h, bb1440_lower_4h,
        bb105_upper_1m, bb105_lower_1m,
        bb300_upper_1m, bb300_lower_1m,
        bb672_upper_1m, bb672_lower_1m,
        threshold_pct,
        extra_levels=tuple(extra_touch_levels),
    )

    return BBStructureSignal(
        momo_15m_bars=len(closes_15m),
        momo_1m_bars=len(closes_1m),
        momo_4h_bars=len(closes_4h),
        momo_1d_bars=len(closes_1d),
        ma7_15m=ma7_15m,
        ma20_15m=ma20_15m,
        ma60_15m=ma60_15m,
        ma96_15m=ma96_15m,
        ma105_15m=ma105_15m,
        ma120_15m=ma120_15m,
        volume_ma5_15m=volume_ma5_15m,
        volume_ma10_15m=volume_ma10_15m,
        bb96_upper_15m=bb96_upper,
        bb96_mid_15m=bb96_mid,
        bb96_lower_15m=bb96_lower,
        bb105_upper_15m=bb105_upper_15m,
        bb105_mid_15m=bb105_mid_15m,
        bb105_lower_15m=bb105_lower_15m,
        bb120_upper_15m=bb120_upper,
        bb120_mid_15m=bb120_mid,
        bb120_lower_15m=bb120_lower,
        bb210_upper_15m=bb210_upper,
        bb210_mid_15m=bb210_mid,
        bb210_lower_15m=bb210_lower,
        bb300_upper_15m=bb300_upper,
        bb300_mid_15m=bb300_mid,
        bb300_lower_15m=bb300_lower,
        bb672_upper_15m=bb672_upper_15m,
        bb672_mid_15m=bb672_mid_15m,
        bb672_lower_15m=bb672_lower_15m,
        bb1440_upper_15m=bb1440_upper_15m,
        bb1440_mid_15m=bb1440_mid_15m,
        bb1440_lower_15m=bb1440_lower_15m,
        bb1920_upper_15m=bb1920_upper_15m,
        bb1920_mid_15m=bb1920_mid_15m,
        bb1920_lower_15m=bb1920_lower_15m,
        bb2880_upper_15m=bb2880_upper_15m,
        bb2880_mid_15m=bb2880_mid_15m,
        bb2880_lower_15m=bb2880_lower_15m,
        bb4404_upper_15m=bb4404_upper_15m,
        bb4404_mid_15m=bb4404_mid_15m,
        bb4404_lower_15m=bb4404_lower_15m,
        bb4460_upper_15m=bb4460_upper_15m,
        bb4460_mid_15m=bb4460_mid_15m,
        bb4460_lower_15m=bb4460_lower_15m,
        bb4707_upper_15m=bb4707_upper_15m,
        bb4707_mid_15m=bb4707_mid_15m,
        bb4707_lower_15m=bb4707_lower_15m,
        ma7_4h=ma7_4h,
        ma20_4h=ma20_4h,
        ma60_4h=ma60_4h,
        ma96_4h=ma96_4h,
        ma105_4h=ma105_4h,
        ma120_4h=ma120_4h,
        volume_ma5_4h=volume_ma5_4h,
        volume_ma10_4h=volume_ma10_4h,
        bb105_upper_4h=bb105_upper_4h,
        bb105_mid_4h=bb105_mid_4h,
        bb105_lower_4h=bb105_lower_4h,
        bb96_upper_4h=bb96_upper_4h,
        bb96_mid_4h=bb96_mid_4h,
        bb96_lower_4h=bb96_lower_4h,
        bb120_upper_4h=bb120_upper_4h,
        bb120_mid_4h=bb120_mid_4h,
        bb120_lower_4h=bb120_lower_4h,
        bb210_upper_4h=bb210_upper_4h,
        bb210_mid_4h=bb210_mid_4h,
        bb210_lower_4h=bb210_lower_4h,
        bb300_upper_4h=bb300_upper_4h,
        bb300_mid_4h=bb300_mid_4h,
        bb300_lower_4h=bb300_lower_4h,
        bb672_upper_4h=bb672_upper_4h,
        bb672_mid_4h=bb672_mid_4h,
        bb672_lower_4h=bb672_lower_4h,
        bb1440_upper_4h=bb1440_upper_4h,
        bb1440_mid_4h=bb1440_mid_4h,
        bb1440_lower_4h=bb1440_lower_4h,
        bb1920_upper_4h=bb1920_upper_4h,
        bb1920_mid_4h=bb1920_mid_4h,
        bb1920_lower_4h=bb1920_lower_4h,
        bb2880_upper_4h=bb2880_upper_4h,
        bb2880_mid_4h=bb2880_mid_4h,
        bb2880_lower_4h=bb2880_lower_4h,
        bb4404_upper_4h=bb4404_upper_4h,
        bb4404_mid_4h=bb4404_mid_4h,
        bb4404_lower_4h=bb4404_lower_4h,
        bb4460_upper_4h=bb4460_upper_4h,
        bb4460_mid_4h=bb4460_mid_4h,
        bb4460_lower_4h=bb4460_lower_4h,
        bb4707_upper_4h=bb4707_upper_4h,
        bb4707_mid_4h=bb4707_mid_4h,
        bb4707_lower_4h=bb4707_lower_4h,
        bb105_upper_1m=bb105_upper_1m,
        bb105_mid_1m=bb105_mid_1m,
        bb105_lower_1m=bb105_lower_1m,
        sma210_1m=sma210_1m,
        bb300_upper_1m=bb300_upper_1m,
        bb300_mid_1m=bb300_mid_1m,
        bb300_lower_1m=bb300_lower_1m,
        bb672_upper_1m=bb672_upper_1m,
        bb672_mid_1m=bb672_mid_1m,
        bb672_lower_1m=bb672_lower_1m,
        bb1920_upper_1m=bb1920_upper_1m,
        bb1920_mid_1m=bb1920_mid_1m,
        bb1920_lower_1m=bb1920_lower_1m,
        ma7_1d=ma7_1d,
        ma20_1d=ma20_1d,
        ma60_1d=ma60_1d,
        ma96_1d=ma96_1d,
        ma105_1d=ma105_1d,
        ma120_1d=ma120_1d,
        volume_ma5_1d=volume_ma5_1d,
        volume_ma10_1d=volume_ma10_1d,
        daily_ma7=daily_ma7,
        daily_ma20=daily_ma20,
        daily_ema21=daily_ema21,
        daily_bb365_upper=daily_bb365_upper,
        daily_bb365_mid=daily_bb365_mid,
        daily_bb365_lower=daily_bb365_lower,
        bb96_upper_1d=bb96_upper_1d,
        bb96_mid_1d=bb96_mid_1d,
        bb96_lower_1d=bb96_lower_1d,
        bb105_upper_1d=bb105_upper_1d,
        bb105_mid_1d=bb105_mid_1d,
        bb105_lower_1d=bb105_lower_1d,
        bb120_upper_1d=bb120_upper_1d,
        bb120_mid_1d=bb120_mid_1d,
        bb120_lower_1d=bb120_lower_1d,
        bb210_upper_1d=bb210_upper_1d,
        bb210_mid_1d=bb210_mid_1d,
        bb210_lower_1d=bb210_lower_1d,
        bb300_upper_1d=bb300_upper_1d,
        bb300_mid_1d=bb300_mid_1d,
        bb300_lower_1d=bb300_lower_1d,
        bb672_upper_1d=bb672_upper_1d,
        bb672_mid_1d=bb672_mid_1d,
        bb672_lower_1d=bb672_lower_1d,
        bb1440_upper_1d=bb1440_upper_1d,
        bb1440_mid_1d=bb1440_mid_1d,
        bb1440_lower_1d=bb1440_lower_1d,
        bb1920_upper_1d=bb1920_upper_1d,
        bb1920_mid_1d=bb1920_mid_1d,
        bb1920_lower_1d=bb1920_lower_1d,
        bb2880_upper_1d=bb2880_upper_1d,
        bb2880_mid_1d=bb2880_mid_1d,
        bb2880_lower_1d=bb2880_lower_1d,
        bb4404_upper_1d=bb4404_upper_1d,
        bb4404_mid_1d=bb4404_mid_1d,
        bb4404_lower_1d=bb4404_lower_1d,
        bb4460_upper_1d=bb4460_upper_1d,
        bb4460_mid_1d=bb4460_mid_1d,
        bb4460_lower_1d=bb4460_lower_1d,
        bb4707_upper_1d=bb4707_upper_1d,
        bb4707_mid_1d=bb4707_mid_1d,
        bb4707_lower_1d=bb4707_lower_1d,
        daily_price_vs_ma7_pct=_pct_diff(mark_price, daily_ma7),
        daily_price_vs_ma20_pct=_pct_diff(mark_price, daily_ma20),
        daily_price_position_365=_price_position(mark_price, daily_bb365_upper, daily_bb365_lower),
        bb_price_position_96_1d=_price_position(mark_price, bb96_upper_1d, bb96_lower_1d),
        bb_price_position_120_1d=_price_position(mark_price, bb120_upper_1d, bb120_lower_1d),
        bb_price_position_210_1d=_price_position(mark_price, bb210_upper_1d, bb210_lower_1d),
        bb_price_position_300_1d=_price_position(mark_price, bb300_upper_1d, bb300_lower_1d),
        bb_price_position_672_1d=_price_position(mark_price, bb672_upper_1d, bb672_lower_1d),
        bb_price_position_1440_1d=_price_position(mark_price, bb1440_upper_1d, bb1440_lower_1d),
        bb_price_position_1920_1d=_price_position(mark_price, bb1920_upper_1d, bb1920_lower_1d),
        bb_price_position_2880_1d=_price_position(mark_price, bb2880_upper_1d, bb2880_lower_1d),
        bb_price_position_4404_1d=_price_position(mark_price, bb4404_upper_1d, bb4404_lower_1d),
        bb_price_position_4460_1d=_price_position(mark_price, bb4460_upper_1d, bb4460_lower_1d),
        bb_price_position_4707_1d=_price_position(mark_price, bb4707_upper_1d, bb4707_lower_1d),
        bb_price_position_105_15m=_price_position(mark_price, bb105_upper_15m, bb105_lower_15m),
        bb_price_position_96=_price_position(mark_price, bb96_upper, bb96_lower),
        bb_price_position_120=_price_position(mark_price, bb120_upper, bb120_lower),
        bb_price_position_210=_price_position(mark_price, bb210_upper, bb210_lower),
        bb_price_position_300=_price_position(mark_price, bb300_upper, bb300_lower),
        bb_price_position_672_15m=_price_position(mark_price, bb672_upper_15m, bb672_lower_15m),
        bb_price_position_1440_15m=_price_position(mark_price, bb1440_upper_15m, bb1440_lower_15m),
        bb_price_position_1920_15m=_price_position(mark_price, bb1920_upper_15m, bb1920_lower_15m),
        bb_price_position_2880_15m=_price_position(mark_price, bb2880_upper_15m, bb2880_lower_15m),
        bb_price_position_4404_15m=_price_position(mark_price, bb4404_upper_15m, bb4404_lower_15m),
        bb_price_position_4460_15m=_price_position(mark_price, bb4460_upper_15m, bb4460_lower_15m),
        bb_price_position_4707_15m=_price_position(mark_price, bb4707_upper_15m, bb4707_lower_15m),
        bb_price_position_105_1d=_price_position(mark_price, bb105_upper_1d, bb105_lower_1d),
        bb_price_position_105_4h=_price_position(mark_price, bb105_upper_4h, bb105_lower_4h),
        bb_price_position_96_4h=_price_position(mark_price, bb96_upper_4h, bb96_lower_4h),
        bb_price_position_120_4h=_price_position(mark_price, bb120_upper_4h, bb120_lower_4h),
        bb_price_position_210_4h=_price_position(mark_price, bb210_upper_4h, bb210_lower_4h),
        bb_price_position_300_4h=_price_position(mark_price, bb300_upper_4h, bb300_lower_4h),
        bb_price_position_672_4h=_price_position(mark_price, bb672_upper_4h, bb672_lower_4h),
        bb_price_position_1440_4h=_price_position(mark_price, bb1440_upper_4h, bb1440_lower_4h),
        bb_price_position_1920_4h=_price_position(mark_price, bb1920_upper_4h, bb1920_lower_4h),
        bb_price_position_2880_4h=_price_position(mark_price, bb2880_upper_4h, bb2880_lower_4h),
        bb_price_position_4404_4h=_price_position(mark_price, bb4404_upper_4h, bb4404_lower_4h),
        bb_price_position_4460_4h=_price_position(mark_price, bb4460_upper_4h, bb4460_lower_4h),
        bb_price_position_4707_4h=_price_position(mark_price, bb4707_upper_4h, bb4707_lower_4h),
        bb_price_position_105_1m=_price_position(mark_price, bb105_upper_1m, bb105_lower_1m),
        bb_price_position_300_1m=_price_position(mark_price, bb300_upper_1m, bb300_lower_1m),
        bb_price_position_672_1m=_price_position(mark_price, bb672_upper_1m, bb672_lower_1m),
        bb_price_position_1920_1m=_price_position(mark_price, bb1920_upper_1m, bb1920_lower_1m),
        williams_r_15m=williams_r_15m,
        williams_r_15m_120=williams_r_15m_120,
        williams_r_1m=williams_r_1m,
        williams_r_15m_zone=williams_zone(williams_r_15m),
        williams_r_15m_120_zone=williams_zone(williams_r_15m_120),
        williams_r_1m_zone=williams_zone(williams_r_1m),
        bb_touch_zone=touch_zone,
    )
