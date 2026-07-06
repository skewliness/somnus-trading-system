from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import logging
import os
import pickle
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any
from urllib import parse

import numpy as np
import requests

from trading_system.bb_structure import SHARED_BB_PERIODS, compute_bb_structure
from trading_system.binance_public_guard import (
    BinancePublicRateLimitError,
    assert_binance_public_not_banned,
    note_binance_public_response,
    throttle_binance_public_request,
)
from trading_system.coin_scout_model import CoinFeatures
from trading_system.gate_registry import GateEventRecorder
from trading_system.gann_feature import gann_position_multiplier
from trading_system.h4_structure import compute_h4_structure
from trading_system.persistence import append_text_line, atomic_write_json, load_json, timed_file_lock
from trading_system.redaction import redact_sensitive_text
from trading_system.transition_gate_policy import TransitionGatePolicy, append_gate_decision, transition_notional_multiplier
from trading_system.v6_common.reason_utils import join_reasons


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/trading_system/model_training/ablation_v6_bitget_full/mixed_undersampled/coin_scout_model.pkl")
DEFAULT_STATE_DIR = Path("data/trading_system/v6_execution")
DEFAULT_SHARED_KLINE_CACHE_DIR = Path("data/trading_system/cache/binance_klines")
DEFAULT_EXECUTION_KLINE_CACHE_DIR = Path("data/trading_system/cache/execution_klines")
PUBLIC_FAPI_BASE_URL = "https://fapi.binance.com"
DEMO_FAPI_BASE_URL = "https://demo-fapi.binance.com"
LIVE_TRADING_CONFIRM_ENV = "TS_V6_LIVE_TRADING_CONFIRMED"
LIVE_TRADING_CONFIRM_VALUE = "YES_I_UNDERSTAND_REAL_MONEY_RISK"
KLINE_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}
V6_KLINE_LIMITS = {
    "1m": 4707,
    "15m": 4707,
    "4h": 4707,
    "1d": 4707,
}
STAGED_TAKE_PROFIT_SELECTION_STRATEGY = "bb_enum_mid_band_v1"
V6_4H_BB2880_MIN_KLINES = 4707


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_add_hours(value: str, hours: float) -> str:
    text = str(value or "").replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.astimezone(timezone.utc) + timedelta(hours=float(hours))).isoformat()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_float_list(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    values: list[float] = []
    for item in raw.replace(";", ",").split(","):
        text = item.strip()
        if not text:
            continue
        value = float(text)
        if value > 0:
            values.append(value)
    return tuple(values) if values else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default




def _local_http_proxy_available(proxy_url: str, *, timeout: float = 0.25) -> bool:
    try:
        parsed = parse.urlparse(proxy_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host or port <= 0:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_binance_proxy_url() -> str:
    configured = _first_env("TS_BINANCE_PROXY_URL", "TS_V6_BINANCE_PROXY_URL", "BINANCE_PROXY_URL")
    if configured.strip().lower() in {"0", "false", "off", "none", "direct"}:
        return ""
    if configured.strip():
        return configured.strip()
    if not _env_bool("TS_BINANCE_AUTO_LOCAL_PROXY", True):
        return ""
    local_proxy = _first_env("TS_LOCAL_PROXY_URL", "LOCAL_PROXY_URL", default="http://127.0.0.1:7897").strip()
    if local_proxy and _local_http_proxy_available(local_proxy):
        return local_proxy
    return ""

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct_diff(price: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return (price - reference) / reference * 100.0


def clipped_pct_diff(price: float, reference: float, clip_pct: float) -> float:
    return clamp(pct_diff(price, reference), -clip_pct, clip_pct)


def parse_symbols(raw: str) -> list[str]:
    symbols = []
    for item in raw.replace(";", ",").split(","):
        symbol = item.strip().upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def parse_symbol_float_map(raw: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in str(raw or "").replace(";", ",").split(","):
        text = item.strip()
        if not text:
            continue
        if ":" in text:
            symbol, value = text.split(":", 1)
        elif "=" in text:
            symbol, value = text.split("=", 1)
        else:
            continue
        symbol = symbol.strip().upper()
        if symbol and not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        try:
            threshold = float(value.strip())
        except ValueError:
            continue
        if symbol and 0.0 < threshold < 1.0:
            values[symbol] = threshold
    return values


SYMBOL_BB_ENTRY_PROFILES: dict[str, dict[str, Any]] = {
    "BTCUSDT": {
        "name": "win60_aggressive_mae_profile_20260610",
        "source": "v6_indicator_backtest_win60_filtered_15m_4h_365d; win>=60 avg>0 MAE<=3%_15m/6.5%_4h; NEAR removed",
        "allow_synthetic_15m_second_test": False,
        "allow_4h_profile_direction_override": True,
        "allow": (
            ('15m', 1440, 'upper', 'long_upper_breakout'),  # BB BB1440_upper upper_breakout long win=61.90% avg=0.2279% mae=-0.7128% events=210
            ('15m', 1440, 'lower', 'long_lower_support_rebound'),  # BB BB1440_lower lower_support_rebound long win=61.76% avg=0.0902% mae=-0.9793% events=272
            ('15m', 2880, 'upper', 'short_upper_rejection'),  # BB BB2880_upper upper_rejection short win=62.58% avg=0.0271% mae=-0.8550% events=155
            ('4h', 1440, 'lower', 'long_lower_support_rebound'),  # BB BB1440_lower lower_support_rebound long win=70.00% avg=1.4465% mae=-1.2138% events=10
            ('4h', 2880, 'lower', 'long_lower_support_rebound'),  # BB BB2880_lower lower_support_rebound long win=83.33% avg=1.0499% mae=-1.6535% events=12
            ('4h', 210, 'lower', 'long_lower_support_rebound'),  # BB BB210_lower lower_support_rebound long win=76.92% avg=0.8882% mae=-2.4288% events=13
            ('4h', 105, 'upper', 'long_upper_breakout'),  # BB BB105_upper upper_breakout long win=62.96% avg=0.3616% mae=-1.6389% events=27
            ('4h', 672, 'lower', 'short_lower_breakdown_retest'),  # BB BB672_lower lower_breakdown_retest short win=81.82% avg=1.9235% mae=-2.1038% events=11
            ('4h', 300, 'mid', 'short_mid_resistance_retest'),  # BB BB300_mid mid_resistance_retest short win=63.64% avg=1.1917% mae=-1.8989% events=22
            ('4h', 210, 'lower', 'short_lower_breakdown_retest'),  # BB BB210_lower lower_breakdown_retest short win=61.11% avg=1.2173% mae=-2.1525% events=18
            ('4h', 96, 'upper', 'short_upper_rejection'),  # BB BB96_upper upper_rejection short win=62.96% avg=0.8882% mae=-1.6653% events=27
            ('4h', 2880, 'upper', 'short_upper_rejection'),  # BB BB2880_upper upper_rejection short win=70.00% avg=1.1502% mae=-0.9652% events=10
            ('4h', 210, 'upper', 'short_upper_rejection'),  # BB BB210_upper upper_rejection short win=66.67% avg=0.6528% mae=-1.6755% events=24
            ('4h', 4707, 'upper', 'short_upper_rejection'),  # BB BB4707_upper upper_rejection short win=64.29% avg=0.5217% mae=-1.2154% events=14
        ),
    },
    "ETHUSDT": {
        "name": "win60_aggressive_mae_profile_20260610",
        "source": "v6_indicator_backtest_win60_filtered_15m_4h_365d; win>=60 avg>0 MAE<=3%_15m/6.5%_4h; NEAR removed",
        "allow_synthetic_15m_second_test": False,
        "allow_4h_profile_direction_override": True,
        "allow": (
            ('15m', 4707, 'mid', 'long_mid_support_retest'),  # BB BB4707_mid mid_support_retest long win=62.04% avg=0.2682% mae=-1.0622% events=108
            ('4h', 300, 'upper', 'long_upper_breakout'),  # BB BB300_upper upper_breakout long win=60.00% avg=0.7260% mae=-2.6906% events=10
            ('4h', 300, 'upper', 'short_upper_rejection'),  # BB BB300_upper upper_rejection short win=83.33% avg=2.7584% mae=-1.6472% events=12
            ('4h', 672, 'mid', 'short_mid_resistance_retest'),  # BB BB672_mid mid_resistance_retest short win=60.00% avg=0.3257% mae=-2.4372% events=10
        ),
    },
    "HYPEUSDT": {
        "name": "win60_aggressive_mae_profile_20260610",
        "source": "v6_indicator_backtest_win60_filtered_15m_4h_365d; win>=60 avg>0 MAE<=3%_15m/6.5%_4h; NEAR removed",
        "allow_synthetic_15m_second_test": False,
        "allow_4h_profile_direction_override": True,
        "allow": (
            ('15m', 672, 'upper', 'long_upper_breakout'),  # BB BB672_upper upper_breakout long win=62.55% avg=0.4046% mae=-1.8907% events=243
            ('15m', 1440, 'mid', 'short_mid_resistance_retest'),  # BB BB1440_mid mid_resistance_retest short win=61.74% avg=0.6854% mae=-1.5647% events=264
            ('4h', 120, 'lower', 'long_lower_support_rebound'),  # BB BB120_lower lower_support_rebound long win=91.67% avg=5.0632% mae=-4.1412% events=12
            ('4h', 210, 'lower', 'long_lower_support_rebound'),  # BB BB210_lower lower_support_rebound long win=81.82% avg=3.8500% mae=-4.7644% events=11
            ('4h', 105, 'mid', 'short_mid_resistance_retest'),  # BB BB105_mid mid_resistance_retest short win=69.57% avg=2.5818% mae=-3.2167% events=23
            ('4h', 105, 'ma', 'short_ma_resistance_retest'),  # MA MA105 ma_resistance_retest short win=69.57% avg=2.5818% mae=-3.2167% events=23
            ('4h', 96, 'upper', 'short_upper_rejection'),  # BB BB96_upper upper_rejection short win=69.23% avg=0.7943% mae=-4.6849% events=13
        ),
    },
    "BNBUSDT": {
        "name": "win60_aggressive_mae_profile_20260610",
        "source": "v6_indicator_backtest_win60_filtered_15m_4h_365d; win>=60 avg>0 MAE<=3%_15m/6.5%_4h; NEAR removed",
        "allow_synthetic_15m_second_test": False,
        "allow_4h_profile_direction_override": True,
        "allow": (
            ('15m', 1920, 'upper', 'long_upper_breakout'),  # BB BB1920_upper upper_breakout long win=61.57% avg=0.2993% mae=-0.9518% events=229
            ('4h', 672, 'upper', 'long_upper_breakout'),  # BB BB672_upper upper_breakout long win=61.11% avg=1.8873% mae=-1.7359% events=18
            ('4h', 1440, 'mid', 'long_mid_support_retest'),  # BB BB1440_mid mid_support_retest long win=82.35% avg=1.4302% mae=-1.4312% events=17
            ('4h', 120, 'upper', 'long_upper_breakout'),  # BB BB120_upper upper_breakout long win=65.62% avg=1.0486% mae=-1.7042% events=32
            ('4h', 96, 'lower', 'long_lower_support_rebound'),  # BB BB96_lower lower_support_rebound long win=66.67% avg=0.5256% mae=-3.0087% events=18
            ('4h', 105, 'lower', 'long_lower_support_rebound'),  # BB BB105_lower lower_support_rebound long win=61.90% avg=0.1347% mae=-4.5037% events=21
            ('4h', 210, 'mid', 'long_mid_support_retest'),  # BB BB210_mid mid_support_retest long win=62.16% avg=0.0551% mae=-1.8295% events=37
            ('4h', 60, 'ma', 'short_ma_resistance_retest'),  # MA MA60 ma_resistance_retest short win=61.54% avg=0.5908% mae=-1.9678% events=39
            ('4h', 60, 'ma', 'short_ma_break_cross'),  # MA MA60 ma_break_cross short win=60.00% avg=0.4017% mae=-2.4180% events=60
        ),
    },
    "ZECUSDT": {
        "name": "zec_4h_only_best_structure_20260612",
        "source": "ZEC scoped rules: 4H-only local structure; BTC/ETH daily MA7 reversal is context only",
        "force_4h_only": True,
        "disable_h4_confirmed_15m_selector": True,
        "ignore_market_daily_ma7_reversal": True,
        "allow_synthetic_15m_second_test": False,
        "allow_4h_profile_direction_override": True,
        "allow": (
            ('4h', 96, 'mid', 'long_mid_support_retest'),  # BB BB96_mid mid_support_retest long win=64.29% avg=5.6427% mae=-3.7523% events=14
            ('4h', 96, 'ma', 'long_ma_support_retest'),  # MA MA96 ma_support_retest long win=64.29% avg=5.6427% mae=-3.7523% events=14
            ('4h', 105, 'mid', 'long_mid_support_retest'),  # BB BB105_mid mid_support_retest long win=66.67% avg=4.3797% mae=-5.0722% events=15
            ('4h', 105, 'ma', 'long_ma_support_retest'),  # MA MA105 ma_support_retest long win=66.67% avg=4.3797% mae=-5.0722% events=15
            ('4h', 96, 'ma', 'long_ma_reclaim_cross'),  # MA MA96 ma_reclaim_cross long win=64.58% avg=2.4229% mae=-5.2369% events=48
            ('4h', 120, 'mid', 'long_mid_support_retest'),  # BB BB120_mid mid_support_retest long win=66.67% avg=4.2849% mae=-3.2584% events=12
            ('4h', 120, 'ma', 'long_ma_support_retest'),  # MA MA120 ma_support_retest long win=66.67% avg=4.2849% mae=-3.2584% events=12
            ('4h', 1920, 'mid', 'long_mid_support_retest'),  # BB BB1920_mid mid_support_retest long win=63.64% avg=1.1284% mae=-5.1337% events=11
            ('4h', 120, 'mid', 'short_mid_resistance_retest'),  # BB BB120_mid mid_resistance_retest short win=75.00% avg=2.9102% mae=-5.8665% events=12
            ('4h', 120, 'ma', 'short_ma_resistance_retest'),  # MA MA120 ma_resistance_retest short win=75.00% avg=2.9102% mae=-5.8665% events=12
            ('4h', 20, 'ma', 'short_ma_resistance_retest'),  # MA MA20 ma_resistance_retest short win=64.10% avg=1.8566% mae=-4.7487% events=39
        ),
    },
    "SUIUSDT": {
        "name": "win60_aggressive_mae_profile_20260610",
        "source": "v6_indicator_backtest_win60_filtered_15m_4h_365d; win>=60 avg>0 MAE<=3%_15m/6.5%_4h; NEAR removed",
        "allow_synthetic_15m_second_test": False,
        "allow_4h_profile_direction_override": True,
        "allow": (
            ('15m', 1920, 'lower', 'long_lower_support_rebound'),  # BB BB1920_lower lower_support_rebound long win=61.21% avg=0.5809% mae=-2.1424% events=165
            ('4h', 105, 'lower', 'long_lower_support_rebound'),  # BB BB105_lower lower_support_rebound long win=64.29% avg=1.7841% mae=-3.7946% events=14
            ('4h', 96, 'lower', 'long_lower_support_rebound'),  # BB BB96_lower lower_support_rebound long win=66.67% avg=0.9667% mae=-5.1099% events=15
            ('4h', 300, 'lower', 'short_lower_breakdown_retest'),  # BB BB300_lower lower_breakdown_retest short win=76.92% avg=6.1527% mae=-2.9682% events=13
            ('4h', 120, 'ma', 'short_ma_break_cross'),  # MA MA120 ma_break_cross short win=70.27% avg=3.4648% mae=-3.1613% events=37
            ('4h', 7, 'ma', 'short_ma_resistance_retest'),  # MA MA7 ma_resistance_retest short win=62.09% avg=1.8239% mae=-3.5525% events=153
            ('4h', 96, 'mid', 'short_mid_resistance_retest'),  # BB BB96_mid mid_resistance_retest short win=66.67% avg=3.1175% mae=-3.9033% events=33
            ('4h', 96, 'ma', 'short_ma_resistance_retest'),  # MA MA96 ma_resistance_retest short win=66.67% avg=3.1175% mae=-3.9033% events=33
            ('4h', 120, 'mid', 'short_mid_resistance_retest'),  # BB BB120_mid mid_resistance_retest short win=64.71% avg=1.1467% mae=-3.2103% events=17
            ('4h', 120, 'ma', 'short_ma_resistance_retest'),  # MA MA120 ma_resistance_retest short win=64.71% avg=1.1467% mae=-3.2103% events=17
        ),
    },
    "SOLUSDT": {
        "name": "win60_aggressive_mae_profile_20260610",
        "source": "v6_indicator_backtest_win60_filtered_15m_4h_365d; win>=60 avg>0 MAE<=3%_15m/6.5%_4h; NEAR removed",
        "allow_synthetic_15m_second_test": False,
        "allow_4h_profile_direction_override": True,
        "allow": (
            ('15m', 2880, 'upper', 'long_upper_breakout'),  # BB BB2880_upper upper_breakout long win=61.31% avg=0.4880% mae=-1.4034% events=137
            ('15m', 1920, 'lower', 'long_lower_support_rebound'),  # BB BB1920_lower lower_support_rebound long win=61.67% avg=0.3900% mae=-2.0066% events=120
            ('4h', 105, 'ma', 'long_ma_reclaim_cross'),  # MA MA105 ma_reclaim_cross long win=68.75% avg=1.8854% mae=-2.8033% events=48
            ('4h', 96, 'ma', 'long_ma_reclaim_cross'),  # MA MA96 ma_reclaim_cross long win=70.83% avg=1.5690% mae=-2.7310% events=48
            ('4h', 210, 'mid', 'long_mid_support_retest'),  # BB BB210_mid mid_support_retest long win=60.00% avg=2.0472% mae=-4.3139% events=15
            ('4h', 96, 'lower', 'long_lower_support_rebound'),  # BB BB96_lower lower_support_rebound long win=70.00% avg=1.9341% mae=-4.9980% events=10
            ('4h', 300, 'mid', 'long_mid_support_retest'),  # BB BB300_mid mid_support_retest long win=70.59% avg=1.2053% mae=-2.6021% events=17
            ('4h', 4707, 'lower', 'long_lower_support_rebound'),  # BB BB4707_lower lower_support_rebound long win=63.64% avg=1.2081% mae=-2.9186% events=11
            ('4h', 120, 'mid', 'long_mid_support_retest'),  # BB BB120_mid mid_support_retest long win=60.87% avg=0.8017% mae=-3.2646% events=23
            ('4h', 120, 'ma', 'long_ma_support_retest'),  # MA MA120 ma_support_retest long win=60.87% avg=0.8017% mae=-3.2646% events=23
            ('4h', 210, 'lower', 'long_lower_support_rebound'),  # BB BB210_lower lower_support_rebound long win=60.00% avg=0.1928% mae=-5.7769% events=10
            ('4h', 96, 'upper', 'short_upper_rejection'),  # BB BB96_upper upper_rejection short win=61.11% avg=1.9827% mae=-3.1207% events=18
            ('4h', 120, 'mid', 'short_mid_resistance_retest'),  # BB BB120_mid mid_resistance_retest short win=63.16% avg=0.4957% mae=-3.6233% events=19
            ('4h', 120, 'ma', 'short_ma_resistance_retest'),  # MA MA120 ma_resistance_retest short win=63.16% avg=0.4957% mae=-3.6233% events=19
        ),
    },
    "XRPUSDT": {
        "name": "win60_aggressive_mae_profile_20260610",
        "source": "v6_indicator_backtest_win60_filtered_15m_4h_365d; win>=60 avg>0 MAE<=3%_15m/6.5%_4h; NEAR removed",
        "allow_synthetic_15m_second_test": False,
        "allow_4h_profile_direction_override": True,
        "allow": (
            ('15m', 1920, 'lower', 'long_lower_support_rebound'),  # BB BB1920_lower lower_support_rebound long win=63.64% avg=0.2550% mae=-2.2131% events=121
            ('15m', 1920, 'mid', 'short_mid_resistance_retest'),  # BB BB1920_mid mid_resistance_retest short win=60.24% avg=0.4091% mae=-1.0674% events=337
            ('15m', 2880, 'mid', 'short_mid_resistance_retest'),  # BB BB2880_mid mid_resistance_retest short win=62.28% avg=0.1708% mae=-1.1740% events=228
            ('4h', 210, 'mid', 'long_mid_support_retest'),  # BB BB210_mid mid_support_retest long win=60.87% avg=1.1120% mae=-2.5796% events=23
            ('4h', 96, 'lower', 'long_lower_support_rebound'),  # BB BB96_lower lower_support_rebound long win=70.37% avg=0.8742% mae=-3.9632% events=27
            ('4h', 7, 'ma', 'short_ma_break_cross'),  # MA MA7 ma_break_cross short win=60.61% avg=1.2943% mae=-2.7708% events=231
            ('4h', 20, 'ma', 'short_ma_resistance_retest'),  # MA MA20 ma_resistance_retest short win=62.92% avg=1.6832% mae=-2.6605% events=89
            ('4h', 120, 'mid', 'short_mid_resistance_retest'),  # BB BB120_mid mid_resistance_retest short win=72.73% avg=1.7691% mae=-2.2207% events=44
            ('4h', 120, 'ma', 'short_ma_resistance_retest'),  # MA MA120 ma_resistance_retest short win=72.73% avg=1.7691% mae=-2.2207% events=44
            ('4h', 120, 'ma', 'short_ma_break_cross'),  # MA MA120 ma_break_cross short win=65.52% avg=1.6753% mae=-2.0582% events=58
            ('4h', 105, 'ma', 'short_ma_break_cross'),  # MA MA105 ma_break_cross short win=63.46% avg=1.5912% mae=-1.9937% events=52
            ('4h', 105, 'mid', 'short_mid_resistance_retest'),  # BB BB105_mid mid_resistance_retest short win=65.00% avg=1.6748% mae=-2.0302% events=40
            ('4h', 105, 'ma', 'short_ma_resistance_retest'),  # MA MA105 ma_resistance_retest short win=65.00% avg=1.6748% mae=-2.0302% events=40
        ),
    },
    "PUMPUSDT": {
        "name": "pump_dedicated_bb_ma_profile_20260628",
        "source": "PUMP local Binance futures cache backtest 2025-07-10..2026-06-28; 4H window=48h, 1D window=14d",
        "allow_synthetic_15m_second_test": False,
        "allow_4h_profile_direction_override": True,
        "allow": (
            ('4h', 120, 'upper', 'long_upper_breakout'),  # BB120 upper breakout long win=63.64% avg=+5.0683% events=11
            ('4h', 105, 'upper', 'long_upper_breakout'),  # BB105 upper breakout long win=63.64% avg=+4.2974% events=11
            ('4h', 105, 'ma', 'short_ma_break_cross'),  # MA105 break cross short win=60.42% avg=+3.1296% events=48
            ('4h', 120, 'mid', 'short_mid_resistance_retest'),  # BB120 mid resistance short win=66.67% avg=+3.5826% events=15
            ('4h', 120, 'ma', 'short_ma_resistance_retest'),  # MA120 resistance short win=66.67% avg=+3.5826% events=15
            ('4h', 300, 'ma', 'short_ma_break_cross'),  # MA300 break cross short win=60.71% avg=+2.1972% events=28
            ('1d', 20, 'ma', 'short_ma_break_cross'),  # 1D MA20 break cross short win=68.18% avg=+20.5435% events=22
            ('1d', 7, 'ma', 'short_ma_break_cross'),  # 1D MA7 break cross short win=70.59% avg=+10.8959% events=34
        ),
    },
}

def symbol_bb_entry_profile(symbol: str) -> dict[str, Any]:
    return SYMBOL_BB_ENTRY_PROFILES.get(str(symbol or "").upper(), {})


def bb_profile_rule(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("timeframe") or ""),
        int(row.get("period") or 0),
        str(row.get("band") or ""),
        str(row.get("structure_signal") or ""),
    )


def bb_entry_profile_allows(symbol: str, row: dict[str, Any]) -> bool:
    profile = symbol_bb_entry_profile(symbol)
    if not profile:
        return True
    return bb_profile_rule(row) in set(profile.get("allow") or ())


@dataclass(slots=True)
class V6ExecutionConfig:
    """Independent V6 execution settings.

    This config deliberately avoids legacy execution credentials and state paths.
    """

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    order_mode: str = "dry_run"
    public_base_url: str = PUBLIC_FAPI_BASE_URL
    demo_base_url: str = DEMO_FAPI_BASE_URL
    live_base_url: str = PUBLIC_FAPI_BASE_URL
    demo_api_key: str = ""
    demo_api_secret: str = ""
    live_api_key: str = ""
    live_api_secret: str = ""
    live_trading_confirmation: str = ""
    recv_window_ms: int = 30000
    binance_proxy_url: str = ""
    binance_request_timeout_seconds: float = 20.0
    binance_request_retries: int = 3
    position_mode: str = "auto"
    margin_type: str = "ISOLATED"
    model_path: Path = DEFAULT_MODEL_PATH
    state_dir: Path = DEFAULT_STATE_DIR
    state_file: Path = DEFAULT_STATE_DIR / "state.json"
    signals_file: Path = DEFAULT_STATE_DIR / "signals.jsonl"
    orders_file: Path = DEFAULT_STATE_DIR / "orders.jsonl"
    wal_file: Path = DEFAULT_STATE_DIR / "execution_wal.jsonl"
    latest_summary_file: Path = DEFAULT_STATE_DIR / "latest_run_summary.json"
    closed_trade_review_queue_file: Path = DEFAULT_STATE_DIR / "closed_trade_review_queue.jsonl"
    post_trade_training_samples_file: Path = DEFAULT_STATE_DIR / "post_trade_training_samples.jsonl"
    kline_cache_dir: Path = DEFAULT_EXECUTION_KLINE_CACHE_DIR
    kline_cache_enabled: bool = True
    public_cache_ttl_seconds: int = 90
    public_mark_price_ttl_seconds: int = 120
    public_ticker_24h_ttl_seconds: int = 300
    kline_refresh_interval_1m_seconds: int = 20
    kline_refresh_interval_15m_seconds: int = 120
    kline_refresh_interval_4h_seconds: int = 600
    kline_refresh_interval_1d_seconds: int = 1800
    kline_limit_1m: int = V6_KLINE_LIMITS["1m"]
    kline_limit_15m: int = V6_KLINE_LIMITS["15m"]
    kline_limit_4h: int = V6_KLINE_LIMITS["4h"]
    kline_limit_1d: int = V6_KLINE_LIMITS["1d"]
    kline_4h_supplement_enabled: bool = True
    kline_4h_supplement_min_limit: int = V6_4H_BB2880_MIN_KLINES
    base_notional_usdt: float = 300.0
    min_notional_usdt: float = 300.0
    default_leverage: int = 3
    max_positions: int = 2
    long_entry_threshold: float = 0.72
    short_entry_threshold: float = 0.72
    symbol_long_entry_thresholds: dict[str, float] = field(default_factory=dict)
    symbol_short_entry_thresholds: dict[str, float] = field(default_factory=dict)
    technical_entry_only_symbols: tuple[str, ...] = field(default_factory=tuple)
    four_h_entry_15m_selector_symbols: tuple[str, ...] = field(default_factory=tuple)
    v5_overlay_rules_enabled: bool = False
    v5_block_ma7_15m_single_entry: bool = False
    v5_block_mixed_avoid_low_quality_entry: bool = False
    v5_entry_location_guard_enabled: bool = False
    v5_entry_location_guard_near_pct: float = 1.2
    v5_entry_15m_ma20_extension_pct: float = 4.0
    v5_disaster_stop_multiplier: float = 1.0
    v5_forced_stop_loss_pct: float = 0.0
    v5_stop_loss_cooldown_minutes: float = 0.0
    v5_reentry_wait_15m_candles: int = 0
    exit_opposite_threshold: float = 0.62
    max_wait_probability_for_entry: float = 0.42
    entry_structure_filter_enabled: bool = True
    entry_structure_boundary_near_pct: float = 2.0
    entry_structure_rejection_near_pct: float = 0.8
    entry_structure_min_rejection_lines: int = 3
    fib_structure_confirmation_enabled: bool = True
    fib_structure_confirmation_near_pct: float = 1.0
    fib_structure_confirmation_lookback_4h: int = 2190
    transition_gate_policy_enabled: bool = False
    transition_gate_min_notional_usdt: float = 5.0
    gann_position_sizing_enabled: bool = False
    entry_confluence_hold_enabled: bool = True
    entry_confluence_hold_near_pct: float = 1.0
    entry_confluence_hold_break_tolerance_pct: float = 0.15
    entry_confluence_hold_min_lines: int = 3
    entry_confluence_hold_lookback_1m: int = 15
    generic_indicator_symbols: tuple[str, ...] = field(default_factory=tuple)
    generic_indicator_label: str = "\u901a\u7528\u6307\u6807"
    generic_indicator_1m_cross_lookback: int = 3
    h4_lower_chase_block_enabled: bool = True
    h4_lower_chase_near_pct: float = 1.5
    h4_lower_chase_min_lines: int = 2
    h4_lower_chase_position_max: float = 0.15
    countertrend_rebound_enabled: bool = True
    ma20_trend_guard_enabled: bool = True
    ma20_trend_guard_candles: int = 2
    ma20_trend_guard_tolerance_pct: float = 0.05
    ma20_4h_pressure_near_pct: float = 2.0
    trend_big_right_side_only: bool = True
    h4_right_side_lookback: int = 8
    h4_right_side_retest_tolerance_pct: float = 1.0
    daily_trend_big_enabled: bool = True
    daily_trend_lookback: int = 20
    daily_ma_retest_tolerance_pct: float = 0.8
    daily_ma7_short_reversal_exit_enabled: bool = True
    daily_ma7_short_reversal_symbols: tuple[str, ...] = field(default_factory=tuple)
    daily_ma7_short_reversal_market_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    daily_ma7_short_reversal_require_all_market_symbols: bool = True
    daily_ma7_short_reversal_required_1d_candles: int = 1
    daily_ma7_short_reversal_tolerance_pct: float = 0.0
    max_hold_days_1d: float = 365.0
    exit_opposite_threshold_1d_trend: float = 0.82
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    max_hold_days_15m: float = 14.0
    max_hold_days_4h: float = 180.0
    exit_opposite_threshold_4h_trend: float = 0.78
    logic_weakening_exit_enabled: bool = True
    logic_weakening_exit_required_4h_candles: int = 2
    logic_weakening_exit_tolerance_pct: float = 0.05
    mid_retest_near_pct: float = 2.0
    far_from_mid_pct: float = 8.0
    distance_clip_pct: float = 30.0
    structure_runner_max_distance_pct: float = 40.0
    staged_take_profit_enabled: bool = True
    staged_take_profit_max_targets: int = 3
    staged_take_profit_fractions: tuple[float, ...] = (0.35, 0.35, 0.30)
    roi_take_profit_enabled: bool = True
    roi_take_profit_thresholds: tuple[float, ...] = (30.0, 38.0, 45.0)
    roi_take_profit_fractions: tuple[float, ...] = (0.30, 0.20, 0.20)
    roi_take_profit_bb_resume_threshold: float = 50.0
    roi_take_profit_bb_resume_max: float = 80.0
    v5_candidate_metadata_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    v5_no_news_momentum_tp_enabled: bool = False
    v5_no_news_momentum_tp_thresholds: tuple[float, ...] = (20.0, 30.0, 50.0)
    v5_no_news_momentum_tp_fractions: tuple[float, ...] = (0.70, 0.20, 0.10)
    v5_no_news_momentum_final_giveback_pct: float = 10.0
    v5_no_news_momentum_sentiment_file: Path = Path("data/trading_system/merged_sentiment_signals.csv")
    v5_momentum_strength_tp_enabled: bool = False
    v5_momentum_strength_extra_thresholds: tuple[float, ...] = (70.0, 100.0)
    v5_momentum_guard_enabled: bool = False
    v5_momentum_guard_first_check_minutes: float = 30.0
    v5_momentum_guard_first_min_roi_pct: float = 5.0
    v5_momentum_guard_first_fraction: float = 0.50
    v5_momentum_guard_mid_check_minutes: float = 60.0
    v5_momentum_guard_mid_min_roi_pct: float = 10.0
    v5_momentum_guard_final_check_minutes: float = 120.0
    v5_momentum_guard_final_min_roi_pct: float = 18.0
    v5_momentum_guard_oi_expand_pct: float = 1.0
    v5_momentum_guard_oi_refresh_seconds: float = 60.0
    v5_momentum_guard_smart_money_ratio: float = 1.20
    v5_momentum_guard_snapshot_max_age_hours: float = 0.25
    v5_momentum_guard_smart_money_file: Path = Path("data/trading_system/skill_bridges/binance-smart-signal.json")
    v5_momentum_guard_use_hyper_panel: bool = False
    v5_momentum_guard_hyper_whale_file: Path = Path("data/trading_system/skill_bridges/hyperdash-hype-whale-distill.json")
    smart_signal_risk_guard_enabled: bool = True
    smart_signal_file: Path = Path("data/trading_system/skill_bridges/binance-smart-signal.json")
    smart_signal_snapshot_max_age_hours: float = 2.0
    smart_signal_risk_reduce_fraction: float = 0.30
    dust_position_notional_usdt: float = 1.0
    strategy_dust_position_notional_usdt: float = 5.0
    v5_profit_reentry_guard_enabled: bool = False
    v5_profit_reentry_guard_max_age_hours: float = 24.0
    v5_profit_reentry_pullback_pct: float = 2.0
    v5_profit_reentry_ma20_near_pct: float = 0.5
    v5_staged_entry_enabled: bool = False
    v5_initial_entry_notional_fraction: float = 0.30
    v5_scale_in_min_hold_minutes: float = 0.0
    v5_scale_in_require_15m_ma20_retest: bool = False
    v5_scale_in_15m_ma20_retest_near_pct: float = 0.6
    v5_scale_in_15m_ma20_max_extension_pct: float = 1.2
    v5_block_mixed_structure_initial_entry: bool = False
    v5_block_unclean_15m_ma20_initial_entry: bool = False
    profit_giveback_guard_enabled: bool = True
    profit_giveback_guard_first_giveback_pct: float = 5.0
    profit_giveback_guard_first_fraction: float = 0.70
    profit_giveback_guard_second_giveback_pct: float = 10.0
    profit_giveback_guard_second_fraction: float = 0.20
    profit_giveback_guard_entry_roi_pct: float = 5.0
    h4_lower_reversal_guard_enabled: bool = True
    h4_lower_reversal_guard_48h_hours: float = 48.0
    h4_lower_reversal_guard_5d_hours: float = 120.0
    h4_lower_reversal_guard_loss_exit_roi_pct: float = -3.0
    h4_lower_reversal_guard_no_h4_bull_fraction: float = 0.50
    h4_lower_reversal_guard_h4_bull_fraction: float = 0.50
    h4_lower_reversal_guard_no_1d_ma7_fraction: float = 0.70
    h4_lower_reversal_guard_1d_reclaim_fraction: float = 0.30
    take_profit_volume_guard_enabled: bool = True
    take_profit_volume_guard_required_candles: int = 2
    take_profit_volume_guard_min_ma5_ratio: float = 1.05
    take_profit_volume_guard_require_trades: bool = False
    take_profit_touch_confirm_window_15m: int = 2
    scale_in_enabled: bool = True
    max_scale_ins_per_position: int = 3
    scale_in_notional_fraction: float = 0.5
    scale_in_min_adverse_move_pct: float = 0.35
    scale_in_near_level_pct: float = 0.8
    scale_in_min_rejection_lines: int = 3
    scale_in_max_4h_stop_distance_pct: float = 5.0
    scale_in_max_total_notional_multiplier: float = 2.5
    commit_dry_run_state: bool = False
    percent_price_retry_enabled: bool = False
    percent_price_retry_wait_seconds: float = 60.0
    percent_price_retry_max_deviation_pct: float = 2.0
    percent_price_retry_cooldown_hours: float = 0.5
    submit_protective_stop: bool = True
    protective_stop_cancel_settle_seconds: float = 2.0
    close_retry_cooldown_minutes: int = 5
    logic_weakening_reentry_cooldown_minutes: int = 30

    @classmethod
    def from_env(cls) -> "V6ExecutionConfig":
        state_dir = Path(os.getenv("TS_V6_EXECUTION_DIR", str(DEFAULT_STATE_DIR)))
        return cls(
            symbols=parse_symbols(os.getenv("TS_V6_SYMBOLS", "BTCUSDT,ETHUSDT")),
            order_mode=canonical_order_mode(os.getenv("TS_V6_ORDER_MODE", "dry_run")),
            public_base_url=os.getenv("TS_V6_PUBLIC_BASE_URL", PUBLIC_FAPI_BASE_URL),
            demo_base_url=_first_env("BINANCE_V6_DEMO_BASE_URL", "TS_V6_BINANCE_DEMO_BASE_URL", default=DEMO_FAPI_BASE_URL),
            live_base_url=os.getenv("TS_V6_BINANCE_LIVE_BASE_URL", PUBLIC_FAPI_BASE_URL),
            demo_api_key=_first_env("BINANCE_V6_DEMO_API_KEY", "TS_V6_BINANCE_DEMO_API_KEY"),
            demo_api_secret=_first_env("BINANCE_V6_DEMO_API_SECRET", "BINANCE_V6_DEMO_SECRET_KEY", "TS_V6_BINANCE_DEMO_API_SECRET"),
            live_api_key=_first_env("BINANCE_V6_API_KEY", "TS_V6_BINANCE_API_KEY"),
            live_api_secret=_first_env("BINANCE_V6_API_SECRET", "BINANCE_V6_SECRET_KEY", "TS_V6_BINANCE_API_SECRET"),
            live_trading_confirmation=os.getenv(LIVE_TRADING_CONFIRM_ENV, ""),
            recv_window_ms=_env_int("TS_V6_BINANCE_RECV_WINDOW_MS", 30000),
            binance_proxy_url=resolve_binance_proxy_url(),
            binance_request_timeout_seconds=_env_float("TS_BINANCE_REQUEST_TIMEOUT_SECONDS", 20.0),
            binance_request_retries=_env_int("TS_BINANCE_REQUEST_RETRIES", 3),
            position_mode=os.getenv("TS_V6_BINANCE_POSITION_MODE", "auto"),
            margin_type=os.getenv("TS_V6_MARGIN_TYPE", "ISOLATED"),
            model_path=Path(os.getenv("TS_V6_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
            state_dir=state_dir,
            state_file=Path(os.getenv("TS_V6_STATE_FILE", str(state_dir / "state.json"))),
            signals_file=Path(os.getenv("TS_V6_SIGNALS_FILE", str(state_dir / "signals.jsonl"))),
            orders_file=Path(os.getenv("TS_V6_ORDERS_FILE", str(state_dir / "orders.jsonl"))),
            wal_file=Path(os.getenv("TS_V6_EXECUTION_WAL_FILE", str(state_dir / "execution_wal.jsonl"))),
            latest_summary_file=Path(os.getenv("TS_V6_LATEST_SUMMARY_FILE", str(state_dir / "latest_run_summary.json"))),
            closed_trade_review_queue_file=Path(os.getenv("TS_V6_CLOSED_TRADE_REVIEW_QUEUE_FILE", str(state_dir / "closed_trade_review_queue.jsonl"))),
            post_trade_training_samples_file=Path(os.getenv("TS_V6_POST_TRADE_TRAINING_SAMPLES_FILE", str(state_dir / "post_trade_training_samples.jsonl"))),
            kline_cache_dir=Path(
                os.getenv(
                    "TS_V6_KLINE_CACHE_DIR",
                    os.getenv("TS_EXECUTION_KLINE_CACHE_DIR", str(DEFAULT_EXECUTION_KLINE_CACHE_DIR)),
                )
            ),
            kline_cache_enabled=_env_bool("TS_V6_KLINE_CACHE_ENABLED", True),
            public_cache_ttl_seconds=_env_int("TS_V6_PUBLIC_CACHE_TTL_SECONDS", 90),
            public_mark_price_ttl_seconds=_env_int(
                "TS_V6_PUBLIC_MARK_PRICE_TTL_SECONDS",
                _env_int("TS_V6_PUBLIC_CACHE_TTL_SECONDS", 120),
            ),
            public_ticker_24h_ttl_seconds=_env_int(
                "TS_V6_PUBLIC_TICKER_24H_TTL_SECONDS",
                _env_int("TS_V6_PUBLIC_CACHE_TTL_SECONDS", 300),
            ),
            kline_refresh_interval_1m_seconds=_env_int("TS_V6_KLINE_REFRESH_1M_SECONDS", 20),
            kline_refresh_interval_15m_seconds=_env_int("TS_V6_KLINE_REFRESH_15M_SECONDS", 120),
            kline_refresh_interval_4h_seconds=_env_int("TS_V6_KLINE_REFRESH_4H_SECONDS", 600),
            kline_refresh_interval_1d_seconds=_env_int("TS_V6_KLINE_REFRESH_1D_SECONDS", 1800),
            kline_limit_1m=min(_env_int("TS_V6_KLINE_LIMIT_1M", V6_KLINE_LIMITS["1m"]), V6_KLINE_LIMITS["1m"]),
            kline_limit_15m=min(_env_int("TS_V6_KLINE_LIMIT_15M", V6_KLINE_LIMITS["15m"]), V6_KLINE_LIMITS["15m"]),
            kline_limit_4h=min(_env_int("TS_V6_KLINE_LIMIT_4H", V6_KLINE_LIMITS["4h"]), V6_KLINE_LIMITS["4h"]),
            kline_limit_1d=min(_env_int("TS_V6_KLINE_LIMIT_1D", V6_KLINE_LIMITS["1d"]), V6_KLINE_LIMITS["1d"]),
            kline_4h_supplement_enabled=_env_bool("TS_V6_KLINE_4H_SUPPLEMENT_ENABLED", True),
            kline_4h_supplement_min_limit=_env_int("TS_V6_KLINE_4H_SUPPLEMENT_MIN_LIMIT", V6_4H_BB2880_MIN_KLINES),
            base_notional_usdt=_env_float("TS_V6_BASE_NOTIONAL_USDT", 300.0),
            min_notional_usdt=_env_float("TS_V6_MIN_NOTIONAL_USDT", 300.0),
            default_leverage=_env_int("TS_V6_DEFAULT_LEVERAGE", 3),
            max_positions=_env_int("TS_V6_MAX_POSITIONS", 2),
            long_entry_threshold=_env_float("TS_V6_LONG_ENTRY_THRESHOLD", 0.72),
            short_entry_threshold=_env_float("TS_V6_SHORT_ENTRY_THRESHOLD", 0.72),
            symbol_long_entry_thresholds=parse_symbol_float_map(
                os.getenv("TS_V6_SYMBOL_LONG_ENTRY_THRESHOLDS", os.getenv("TS_V6_SYMBOL_ENTRY_THRESHOLDS", ""))
            ),
            symbol_short_entry_thresholds=parse_symbol_float_map(
                os.getenv("TS_V6_SYMBOL_SHORT_ENTRY_THRESHOLDS", os.getenv("TS_V6_SYMBOL_ENTRY_THRESHOLDS", ""))
            ),
            technical_entry_only_symbols=tuple(parse_symbols(os.getenv("TS_V6_TECHNICAL_ENTRY_ONLY_SYMBOLS", ""))),
            four_h_entry_15m_selector_symbols=tuple(
                parse_symbols(os.getenv("TS_V6_4H_ENTRY_15M_SELECTOR_SYMBOLS", "BTCUSDT,ETHUSDT"))
            ),
            exit_opposite_threshold=_env_float("TS_V6_EXIT_OPPOSITE_THRESHOLD", 0.62),
            max_wait_probability_for_entry=_env_float("TS_V6_MAX_WAIT_PROBABILITY_FOR_ENTRY", 0.42),
            entry_structure_filter_enabled=_env_bool("TS_V6_ENTRY_STRUCTURE_FILTER_ENABLED", True),
            entry_structure_boundary_near_pct=_env_float("TS_V6_ENTRY_STRUCTURE_BOUNDARY_NEAR_PCT", 2.0),
            entry_structure_rejection_near_pct=_env_float("TS_V6_ENTRY_STRUCTURE_REJECTION_NEAR_PCT", 0.8),
            entry_structure_min_rejection_lines=_env_int("TS_V6_ENTRY_STRUCTURE_MIN_REJECTION_LINES", 3),
            fib_structure_confirmation_enabled=_env_bool("TS_V6_FIB_STRUCTURE_CONFIRMATION_ENABLED", True),
            fib_structure_confirmation_near_pct=_env_float("TS_V6_FIB_STRUCTURE_CONFIRMATION_NEAR_PCT", 1.0),
            fib_structure_confirmation_lookback_4h=_env_int("TS_V6_FIB_STRUCTURE_CONFIRMATION_LOOKBACK_4H", 2190),
            transition_gate_policy_enabled=_env_bool("TS_V6_TRANSITION_GATE_POLICY_ENABLED", False),
            transition_gate_min_notional_usdt=_env_float("TS_V6_TRANSITION_GATE_MIN_NOTIONAL_USDT", 5.0),
            gann_position_sizing_enabled=_env_bool("TS_V6_GANN_POSITION_SIZING_ENABLED", False),
            entry_confluence_hold_enabled=_env_bool("TS_V6_ENTRY_CONFLUENCE_HOLD_ENABLED", True),
            entry_confluence_hold_near_pct=_env_float("TS_V6_ENTRY_CONFLUENCE_HOLD_NEAR_PCT", 1.0),
            entry_confluence_hold_break_tolerance_pct=_env_float("TS_V6_ENTRY_CONFLUENCE_HOLD_BREAK_TOLERANCE_PCT", 0.15),
            entry_confluence_hold_min_lines=_env_int("TS_V6_ENTRY_CONFLUENCE_HOLD_MIN_LINES", 3),
            entry_confluence_hold_lookback_1m=_env_int("TS_V6_ENTRY_CONFLUENCE_HOLD_LOOKBACK_1M", 15),
            generic_indicator_symbols=tuple(parse_symbols(os.getenv("TS_V6_GENERIC_INDICATOR_SYMBOLS", ""))),
            generic_indicator_label=os.getenv("TS_V6_GENERIC_INDICATOR_LABEL", "\u901a\u7528\u6307\u6807"),
            generic_indicator_1m_cross_lookback=_env_int("TS_V6_GENERIC_INDICATOR_1M_CROSS_LOOKBACK", 3),
            h4_lower_chase_block_enabled=_env_bool("TS_V6_H4_LOWER_CHASE_BLOCK_ENABLED", True),
            h4_lower_chase_near_pct=_env_float("TS_V6_H4_LOWER_CHASE_NEAR_PCT", 1.5),
            h4_lower_chase_min_lines=_env_int("TS_V6_H4_LOWER_CHASE_MIN_LINES", 2),
            h4_lower_chase_position_max=_env_float("TS_V6_H4_LOWER_CHASE_POSITION_MAX", 0.15),
            countertrend_rebound_enabled=_env_bool("TS_V6_COUNTERTREND_REBOUND_ENABLED", True),
            ma20_trend_guard_enabled=_env_bool("TS_V6_MA20_TREND_GUARD_ENABLED", True),
            ma20_trend_guard_candles=_env_int("TS_V6_MA20_TREND_GUARD_CANDLES", 2),
            ma20_trend_guard_tolerance_pct=_env_float("TS_V6_MA20_TREND_GUARD_TOLERANCE_PCT", 0.05),
            ma20_4h_pressure_near_pct=_env_float("TS_V6_4H_MA20_PRESSURE_NEAR_PCT", 2.0),
            trend_big_right_side_only=_env_bool("TS_V6_TREND_BIG_RIGHT_SIDE_ONLY", True),
            h4_right_side_lookback=_env_int("TS_V6_4H_RIGHT_SIDE_LOOKBACK", 8),
            h4_right_side_retest_tolerance_pct=_env_float("TS_V6_4H_RIGHT_SIDE_RETEST_TOLERANCE_PCT", 1.0),
            daily_trend_big_enabled=_env_bool("TS_V6_DAILY_TREND_BIG_ENABLED", True),
            daily_trend_lookback=_env_int("TS_V6_DAILY_TREND_LOOKBACK", 20),
            daily_ma_retest_tolerance_pct=_env_float("TS_V6_DAILY_MA_RETEST_TOLERANCE_PCT", 0.8),
            daily_ma7_short_reversal_exit_enabled=_env_bool("TS_V6_DAILY_MA7_SHORT_REVERSAL_EXIT_ENABLED", True),
            daily_ma7_short_reversal_symbols=tuple(
                parse_symbols(os.getenv("TS_V6_DAILY_MA7_SHORT_REVERSAL_SYMBOLS", ""))
            ),
            daily_ma7_short_reversal_market_symbols=tuple(
                parse_symbols(os.getenv("TS_V6_DAILY_MA7_SHORT_REVERSAL_MARKET_SYMBOLS", "BTCUSDT,ETHUSDT"))
            ),
            daily_ma7_short_reversal_require_all_market_symbols=_env_bool("TS_V6_DAILY_MA7_SHORT_REVERSAL_REQUIRE_ALL_MARKET_SYMBOLS", True),
            daily_ma7_short_reversal_required_1d_candles=_env_int("TS_V6_DAILY_MA7_SHORT_REVERSAL_REQUIRED_1D_CANDLES", 1),
            daily_ma7_short_reversal_tolerance_pct=_env_float("TS_V6_DAILY_MA7_SHORT_REVERSAL_TOLERANCE_PCT", 0.0),
            max_hold_days_1d=_env_float("TS_V6_1D_MAX_HOLD_DAYS", 365.0),
            exit_opposite_threshold_1d_trend=_env_float("TS_V6_1D_EXIT_OPPOSITE_THRESHOLD", 0.82),
            stop_loss_pct=_env_float("TS_V6_STOP_LOSS_PCT", 0.03),
            take_profit_pct=_env_float("TS_V6_TAKE_PROFIT_PCT", 0.06),
            max_hold_days_15m=_env_float("TS_V6_15M_MAX_HOLD_DAYS", 14.0),
            max_hold_days_4h=_env_float("TS_V6_4H_MAX_HOLD_DAYS", 180.0),
            exit_opposite_threshold_4h_trend=_env_float("TS_V6_4H_EXIT_OPPOSITE_THRESHOLD", 0.78),
            logic_weakening_exit_enabled=_env_bool("TS_V6_LOGIC_WEAKENING_EXIT_ENABLED", True),
            logic_weakening_exit_required_4h_candles=_env_int("TS_V6_LOGIC_WEAKENING_EXIT_REQUIRED_4H_CANDLES", 2),
            logic_weakening_exit_tolerance_pct=_env_float("TS_V6_LOGIC_WEAKENING_EXIT_TOLERANCE_PCT", 0.05),
            mid_retest_near_pct=_env_float("TS_V6_MID_RETEST_NEAR_PCT", 2.0),
            far_from_mid_pct=_env_float("TS_V6_FAR_FROM_MID_PCT", 8.0),
            distance_clip_pct=_env_float("TS_V6_DISTANCE_CLIP_PCT", 30.0),
            structure_runner_max_distance_pct=_env_float("TS_V6_STRUCTURE_RUNNER_MAX_DISTANCE_PCT", 40.0),
            staged_take_profit_enabled=_env_bool("TS_V6_STAGED_TAKE_PROFIT_ENABLED", True),
            staged_take_profit_max_targets=_env_int("TS_V6_STAGED_TAKE_PROFIT_MAX_TARGETS", 3),
            staged_take_profit_fractions=_env_float_list("TS_V6_STAGED_TAKE_PROFIT_FRACTIONS", (0.35, 0.35, 0.30)),
            roi_take_profit_enabled=_env_bool("TS_V6_ROI_TAKE_PROFIT_ENABLED", True),
            roi_take_profit_thresholds=_env_float_list("TS_V6_ROI_TAKE_PROFIT_THRESHOLDS", (30.0, 38.0, 45.0)),
            roi_take_profit_fractions=_env_float_list("TS_V6_ROI_TAKE_PROFIT_FRACTIONS", (0.30, 0.20, 0.20)),
            roi_take_profit_bb_resume_threshold=_env_float("TS_V6_ROI_TAKE_PROFIT_BB_RESUME_THRESHOLD", 50.0),
            roi_take_profit_bb_resume_max=_env_float("TS_V6_ROI_TAKE_PROFIT_BB_RESUME_MAX", 80.0),
            v5_no_news_momentum_tp_enabled=_env_bool("TS_V5_NO_NEWS_MOMENTUM_TP_ENABLED", False),
            v5_no_news_momentum_tp_thresholds=_env_float_list("TS_V5_NO_NEWS_MOMENTUM_TP_THRESHOLDS", (20.0, 30.0, 50.0)),
            v5_no_news_momentum_tp_fractions=_env_float_list("TS_V5_NO_NEWS_MOMENTUM_TP_FRACTIONS", (0.70, 0.20, 0.10)),
            v5_no_news_momentum_final_giveback_pct=_env_float("TS_V5_NO_NEWS_MOMENTUM_FINAL_GIVEBACK_PCT", 10.0),
            v5_no_news_momentum_sentiment_file=Path(
                os.getenv("TS_V5_NO_NEWS_MOMENTUM_SENTIMENT_FILE", "data/trading_system/merged_sentiment_signals.csv")
            ),
            v5_momentum_strength_tp_enabled=_env_bool("TS_V5_MOMENTUM_STRENGTH_TP_ENABLED", False),
            v5_momentum_strength_extra_thresholds=_env_float_list("TS_V5_MOMENTUM_STRENGTH_EXTRA_THRESHOLDS", (70.0, 100.0)),
            v5_momentum_guard_enabled=_env_bool("TS_V5_MOMENTUM_GUARD_ENABLED", False),
            v5_momentum_guard_first_check_minutes=_env_float("TS_V5_MOMENTUM_GUARD_FIRST_CHECK_MINUTES", 30.0),
            v5_momentum_guard_first_min_roi_pct=_env_float("TS_V5_MOMENTUM_GUARD_FIRST_MIN_ROI_PCT", 5.0),
            v5_momentum_guard_first_fraction=_env_float("TS_V5_MOMENTUM_GUARD_FIRST_FRACTION", 0.50),
            v5_momentum_guard_mid_check_minutes=_env_float("TS_V5_MOMENTUM_GUARD_MID_CHECK_MINUTES", 60.0),
            v5_momentum_guard_mid_min_roi_pct=_env_float("TS_V5_MOMENTUM_GUARD_MID_MIN_ROI_PCT", 10.0),
            v5_momentum_guard_final_check_minutes=_env_float("TS_V5_MOMENTUM_GUARD_FINAL_CHECK_MINUTES", 120.0),
            v5_momentum_guard_final_min_roi_pct=_env_float("TS_V5_MOMENTUM_GUARD_FINAL_MIN_ROI_PCT", 18.0),
            v5_momentum_guard_oi_expand_pct=_env_float("TS_V5_MOMENTUM_GUARD_OI_EXPAND_PCT", 1.0),
            v5_momentum_guard_oi_refresh_seconds=_env_float("TS_V5_MOMENTUM_GUARD_OI_REFRESH_SECONDS", 60.0),
            v5_momentum_guard_smart_money_ratio=_env_float("TS_V5_MOMENTUM_GUARD_SMART_MONEY_RATIO", 1.20),
            v5_momentum_guard_snapshot_max_age_hours=_env_float("TS_V5_MOMENTUM_GUARD_SNAPSHOT_MAX_AGE_HOURS", 0.25),
            v5_momentum_guard_smart_money_file=Path(
                os.getenv("TS_V5_MOMENTUM_GUARD_SMART_MONEY_FILE", "data/trading_system/skill_bridges/binance-smart-signal.json")
            ),
            v5_momentum_guard_use_hyper_panel=_env_bool("TS_V5_MOMENTUM_GUARD_USE_HYPER_PANEL", False),
            v5_momentum_guard_hyper_whale_file=Path(
                os.getenv("TS_V5_MOMENTUM_GUARD_HYPER_WHALE_FILE", "data/trading_system/skill_bridges/hyperdash-hype-whale-distill.json")
            ),
            smart_signal_risk_guard_enabled=_env_bool("TS_V6_SMART_SIGNAL_RISK_GUARD_ENABLED", True),
            smart_signal_file=Path(os.getenv("TS_V6_SMART_SIGNAL_FILE", "data/trading_system/skill_bridges/binance-smart-signal.json")),
            smart_signal_snapshot_max_age_hours=_env_float("TS_V6_SMART_SIGNAL_SNAPSHOT_MAX_AGE_HOURS", 2.0),
            smart_signal_risk_reduce_fraction=_env_float("TS_V6_SMART_SIGNAL_RISK_REDUCE_FRACTION", 0.30),
            dust_position_notional_usdt=_env_float("TS_V6_DUST_POSITION_NOTIONAL_USDT", 1.0),
            strategy_dust_position_notional_usdt=_env_float("TS_V6_STRATEGY_DUST_POSITION_NOTIONAL_USDT", 5.0),
            v5_profit_reentry_guard_enabled=_env_bool("TS_V5_PROFIT_REENTRY_GUARD_ENABLED", False),
            v5_profit_reentry_guard_max_age_hours=_env_float("TS_V5_PROFIT_REENTRY_GUARD_MAX_AGE_HOURS", 24.0),
            v5_profit_reentry_pullback_pct=_env_float("TS_V5_PROFIT_REENTRY_PULLBACK_PCT", 2.0),
            v5_profit_reentry_ma20_near_pct=_env_float("TS_V5_PROFIT_REENTRY_MA20_NEAR_PCT", 0.5),
            v5_staged_entry_enabled=_env_bool("TS_V5_STAGED_ENTRY_ENABLED", False),
            v5_initial_entry_notional_fraction=_env_float("TS_V5_INITIAL_ENTRY_NOTIONAL_FRACTION", 0.30),
            v5_scale_in_min_hold_minutes=_env_float("TS_V5_SCALE_IN_MIN_HOLD_MINUTES", 0.0),
            v5_scale_in_require_15m_ma20_retest=_env_bool("TS_V5_SCALE_IN_REQUIRE_15M_MA20_RETEST", False),
            v5_scale_in_15m_ma20_retest_near_pct=_env_float("TS_V5_SCALE_IN_15M_MA20_RETEST_NEAR_PCT", 0.6),
            v5_scale_in_15m_ma20_max_extension_pct=_env_float("TS_V5_SCALE_IN_15M_MA20_MAX_EXTENSION_PCT", 1.2),
            v5_block_mixed_structure_initial_entry=_env_bool("TS_V5_BLOCK_MIXED_STRUCTURE_INITIAL_ENTRY", False),
            v5_block_unclean_15m_ma20_initial_entry=_env_bool("TS_V5_BLOCK_UNCLEAN_15M_MA20_INITIAL_ENTRY", False),
            v5_forced_stop_loss_pct=_env_float("TS_V5_FORCED_STOP_LOSS_PCT", 0.0),
            v5_stop_loss_cooldown_minutes=_env_float("TS_V5_STOP_LOSS_COOLDOWN_MINUTES", 0.0),
            profit_giveback_guard_enabled=_env_bool("TS_V6_PROFIT_GIVEBACK_GUARD_ENABLED", True),
            profit_giveback_guard_first_giveback_pct=_env_float("TS_V6_PROFIT_GIVEBACK_GUARD_FIRST_GIVEBACK_PCT", 5.0),
            profit_giveback_guard_first_fraction=_env_float("TS_V6_PROFIT_GIVEBACK_GUARD_FIRST_FRACTION", 0.70),
            profit_giveback_guard_second_giveback_pct=_env_float("TS_V6_PROFIT_GIVEBACK_GUARD_SECOND_GIVEBACK_PCT", 10.0),
            profit_giveback_guard_second_fraction=_env_float("TS_V6_PROFIT_GIVEBACK_GUARD_SECOND_FRACTION", 0.20),
            profit_giveback_guard_entry_roi_pct=_env_float("TS_V6_PROFIT_GIVEBACK_GUARD_ENTRY_ROI_PCT", 5.0),
            h4_lower_reversal_guard_enabled=_env_bool("TS_V6_H4_LOWER_REVERSAL_GUARD_ENABLED", True),
            h4_lower_reversal_guard_48h_hours=_env_float("TS_V6_H4_LOWER_REVERSAL_GUARD_48H_HOURS", 48.0),
            h4_lower_reversal_guard_5d_hours=_env_float("TS_V6_H4_LOWER_REVERSAL_GUARD_5D_HOURS", 120.0),
            h4_lower_reversal_guard_loss_exit_roi_pct=_env_float("TS_V6_H4_LOWER_REVERSAL_GUARD_LOSS_EXIT_ROI_PCT", -3.0),
            h4_lower_reversal_guard_no_h4_bull_fraction=_env_float("TS_V6_H4_LOWER_REVERSAL_GUARD_NO_H4_BULL_FRACTION", 0.50),
            h4_lower_reversal_guard_h4_bull_fraction=_env_float("TS_V6_H4_LOWER_REVERSAL_GUARD_H4_BULL_FRACTION", 0.50),
            h4_lower_reversal_guard_no_1d_ma7_fraction=_env_float("TS_V6_H4_LOWER_REVERSAL_GUARD_NO_1D_MA7_FRACTION", 0.70),
            h4_lower_reversal_guard_1d_reclaim_fraction=_env_float("TS_V6_H4_LOWER_REVERSAL_GUARD_1D_RECLAIM_FRACTION", 0.30),
            take_profit_volume_guard_enabled=_env_bool("TS_V6_TP_VOLUME_GUARD_ENABLED", True),
            take_profit_volume_guard_required_candles=_env_int("TS_V6_TP_VOLUME_GUARD_REQUIRED_CANDLES", 2),
            take_profit_volume_guard_min_ma5_ratio=_env_float("TS_V6_TP_VOLUME_GUARD_MIN_MA5_RATIO", 1.05),
            take_profit_volume_guard_require_trades=_env_bool("TS_V6_TP_VOLUME_GUARD_REQUIRE_TRADES", False),
            take_profit_touch_confirm_window_15m=_env_int("TS_V6_TP_TOUCH_CONFIRM_WINDOW_15M", 2),
            scale_in_enabled=_env_bool("TS_V6_SCALE_IN_ENABLED", True),
            max_scale_ins_per_position=_env_int("TS_V6_MAX_SCALE_INS_PER_POSITION", 3),
            scale_in_notional_fraction=_env_float("TS_V6_SCALE_IN_NOTIONAL_FRACTION", 0.5),
            scale_in_min_adverse_move_pct=_env_float("TS_V6_SCALE_IN_MIN_ADVERSE_MOVE_PCT", 0.35),
            scale_in_near_level_pct=_env_float("TS_V6_SCALE_IN_NEAR_LEVEL_PCT", 0.8),
            scale_in_min_rejection_lines=_env_int("TS_V6_SCALE_IN_MIN_REJECTION_LINES", 3),
            scale_in_max_4h_stop_distance_pct=_env_float("TS_V6_SCALE_IN_MAX_4H_STOP_DISTANCE_PCT", 5.0),
            scale_in_max_total_notional_multiplier=_env_float("TS_V6_SCALE_IN_MAX_TOTAL_NOTIONAL_MULTIPLIER", 2.5),
            commit_dry_run_state=_env_bool("TS_V6_COMMIT_DRY_RUN_STATE", False),
            percent_price_retry_enabled=_env_bool("TS_V6_PERCENT_PRICE_RETRY_ENABLED", False),
            percent_price_retry_wait_seconds=_env_float("TS_V6_PERCENT_PRICE_RETRY_WAIT_SECONDS", 60.0),
            percent_price_retry_max_deviation_pct=_env_float("TS_V6_PERCENT_PRICE_RETRY_MAX_DEVIATION_PCT", 2.0),
            percent_price_retry_cooldown_hours=_env_float("TS_V6_PERCENT_PRICE_RETRY_COOLDOWN_HOURS", 0.5),
            submit_protective_stop=_env_bool("TS_V6_SUBMIT_PROTECTIVE_STOP", True),
            protective_stop_cancel_settle_seconds=_env_float("TS_V6_PROTECTIVE_STOP_CANCEL_SETTLE_SECONDS", 2.0),
            close_retry_cooldown_minutes=_env_int("TS_V6_CLOSE_RETRY_COOLDOWN_MINUTES", 5),
            logic_weakening_reentry_cooldown_minutes=_env_int("TS_V6_LOGIC_WEAKENING_REENTRY_COOLDOWN_MINUTES", 30),
        )

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.signals_file.parent.mkdir(parents=True, exist_ok=True)
        self.orders_file.parent.mkdir(parents=True, exist_ok=True)
        self.wal_file.parent.mkdir(parents=True, exist_ok=True)
        self.latest_summary_file.parent.mkdir(parents=True, exist_ok=True)
        self.closed_trade_review_queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.post_trade_training_samples_file.parent.mkdir(parents=True, exist_ok=True)
        self.kline_cache_dir.mkdir(parents=True, exist_ok=True)

    def kline_limit_for_interval(self, interval: str) -> int:
        limits = {
            "1m": self.kline_limit_1m,
            "15m": self.kline_limit_15m,
            "4h": self.kline_limit_4h,
            "1d": self.kline_limit_1d,
        }
        fallback = V6_KLINE_LIMITS.get(interval, max(SHARED_BB_PERIODS))
        return min(int(limits.get(interval, fallback)), int(fallback))

    def kline_refresh_interval_seconds_for_interval(self, interval: str) -> int:
        intervals = {
            "1m": self.kline_refresh_interval_1m_seconds,
            "15m": self.kline_refresh_interval_15m_seconds,
            "4h": self.kline_refresh_interval_4h_seconds,
            "1d": self.kline_refresh_interval_1d_seconds,
        }
        return max(0, int(intervals.get(str(interval or "").strip(), 120)))

    def long_entry_threshold_for(self, symbol: str) -> float:
        return float(self.symbol_long_entry_thresholds.get(str(symbol or "").upper(), self.long_entry_threshold))

    def short_entry_threshold_for(self, symbol: str) -> float:
        return float(self.symbol_short_entry_thresholds.get(str(symbol or "").upper(), self.short_entry_threshold))

    def technical_entry_only_for(self, symbol: str) -> bool:
        return str(symbol or "").upper() in {item.upper() for item in self.technical_entry_only_symbols}

    def four_h_entry_requires_15m_selector_for(self, symbol: str) -> bool:
        return str(symbol or "").upper() in {item.upper() for item in self.four_h_entry_15m_selector_symbols}

    def live_trading_confirmed(self) -> bool:
        return str(self.live_trading_confirmation or "").strip() == LIVE_TRADING_CONFIRM_VALUE

    def v5_overlay_rules_active_for(self, symbol: str) -> bool:
        if not self.v5_overlay_rules_enabled:
            return False
        technical_symbols = {item.upper() for item in self.technical_entry_only_symbols}
        if not technical_symbols:
            return True
        return str(symbol or "").upper() in technical_symbols


def canonical_order_mode(raw: str) -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "dry": "dry_run",
        "dry_run": "dry_run",
        "paper": "dry_run",
        "testnet": "testnet_order",
        "demo": "testnet_order",
        "testnet_order": "testnet_order",
        "live": "live_order",
        "live_order": "live_order",
    }
    return aliases.get(value, "dry_run")


class V6ExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class V6Signal:
    symbol: str
    computed_at: str
    price: float
    side_bias: str
    long_probability: float
    short_probability: float
    wait_probability: float
    model_quality_score: float
    reasons: list[str]
    h4: dict[str, Any]
    fifteen_m: dict[str, Any]
    one_m: dict[str, Any]
    mid_retest: dict[str, Any]
    levels: dict[str, Any]
    indicators: dict[str, Any] = field(default_factory=dict)
    market_data_freshness: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class V6OrderIntent:
    action: str
    symbol: str
    side: str
    quantity_hint_usdt: float
    leverage: int
    reduce_only: bool
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    long_probability: float
    short_probability: float
    wait_probability: float
    rationale: str
    generated_at: str
    risk_plan: dict[str, Any] = field(default_factory=dict)
    close_fraction: float = 1.0
    target_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class V6BinanceClient:
    def __init__(self, config: V6ExecutionConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.trust_env = True
        proxy_url = str(config.binance_proxy_url or "").strip()
        if proxy_url:
            self.session.proxies.update({"http": proxy_url, "https": proxy_url})
        self._exchange_info_cache: dict[str, Any] | None = None
        self._position_mode_cache: str | None = None
        self._leverage_bracket_cache: dict[str, Any] = {}
        self._market_data_freshness: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self.session.close()

    @staticmethod
    def _url_host(url: str) -> str:
        parsed = parse.urlparse(str(url or ""))
        if parsed.netloc:
            return parsed.netloc.lower()
        return str(url or "").split("/", 1)[0].lower()

    def order_endpoint_requires_live_confirmation(self) -> bool:
        if self.config.order_mode == "live_order":
            return True
        host = self._url_host(self.order_base_url())
        if "demo" in host or "testnet" in host:
            return False
        return host in {"fapi.binance.com", "dapi.binance.com"} or host.endswith(".binance.com")

    def require_live_trading_confirmation(self, action: str = "live trading request") -> None:
        if not self.order_endpoint_requires_live_confirmation():
            return
        if self.config.live_trading_confirmed():
            return
        raise V6ExecutionError(
            "V6 live trading blocked: "
            f"{action} requires {LIVE_TRADING_CONFIRM_ENV}={LIVE_TRADING_CONFIRM_VALUE}"
        )

    def reset_market_data_freshness(self, symbol: str) -> None:
        self._market_data_freshness[str(symbol or "").upper()] = {
            "symbol": str(symbol or "").upper(),
            "stale": False,
            "stale_sources": [],
            "updated_at": utc_now_iso(),
        }

    def mark_stale_market_data(self, symbol: str, source: str, reason: str) -> None:
        symbol_upper = str(symbol or "").upper()
        if not symbol_upper:
            return
        row = self._market_data_freshness.setdefault(
            symbol_upper,
            {
                "symbol": symbol_upper,
                "stale": False,
                "stale_sources": [],
                "updated_at": utc_now_iso(),
            },
        )
        sources = row.setdefault("stale_sources", [])
        item = {"source": str(source or "unknown"), "reason": str(reason or "")[:240], "at": utc_now_iso()}
        if isinstance(sources, list):
            sources.append(item)
            row["stale_sources"] = sources[-12:]
        row["stale"] = True
        row["updated_at"] = utc_now_iso()

    def market_data_freshness(self, symbol: str) -> dict[str, Any]:
        symbol_upper = str(symbol or "").upper()
        row = self._market_data_freshness.get(symbol_upper)
        if not isinstance(row, dict):
            return {"symbol": symbol_upper, "stale": False, "stale_sources": [], "updated_at": utc_now_iso()}
        return dict(row)

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        retries = max(1, int(self.config.binance_request_retries or 1))
        timeout = float(kwargs.pop("timeout", self.config.binance_request_timeout_seconds or 20.0))
        last_exc: requests.RequestException | None = None
        for attempt in range(retries):
            try:
                return self.session.request(method, url, timeout=timeout, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt + 1 < retries:
                    time.sleep(0.6 * (2 ** attempt))
        raise V6ExecutionError(
            "Binance request failed after "
            f"{retries} attempts: {type(last_exc).__name__}: "
            f"{redact_sensitive_text(last_exc)}"
        ) from last_exc

    def public_get(self, endpoint: str, params: dict[str, Any] | None = None, *, order_endpoint: bool = False) -> Any:
        base_url = self.order_base_url() if order_endpoint else self.config.public_base_url
        if not order_endpoint:
            try:
                assert_binance_public_not_banned(f"v6:{endpoint}")
            except BinancePublicRateLimitError as exc:
                raise V6ExecutionError(str(exc)) from exc
        response = None
        for attempt in range(3):
            if not order_endpoint:
                throttle_binance_public_request()
            response = self._request("GET", f"{base_url}{endpoint}", params=params or {})
            note_binance_public_response(response.status_code, response.text, f"v6:{endpoint}")
            if response.status_code == 418:
                break
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
        if response is None:
            raise V6ExecutionError("Binance public request failed before response")
        if response.status_code != 200:
            raise V6ExecutionError(
                "Binance public request failed "
                f"{response.status_code}: {redact_sensitive_text(response.text)}"
            )
        return response.json()

    def public_cache_path(self, name: str, symbol: str) -> Path:
        safe_name = "".join(ch for ch in name if ch.isalnum() or ch in {"_", "-"})
        safe_symbol = "".join(ch for ch in symbol.upper() if ch.isalnum() or ch in {"_", "-"})
        return self.config.kline_cache_dir / "public_meta" / safe_name / f"{safe_symbol}.json"

    def load_public_cache(self, name: str, symbol: str, *, max_age_seconds: int | None = None) -> dict[str, Any]:
        path = self.public_cache_path(name, symbol)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        if max_age_seconds is not None and max_age_seconds > 0:
            try:
                updated_at = float(payload.get("updated_at_epoch") or 0)
            except Exception:
                updated_at = 0.0
            if updated_at <= 0 or time.time() - updated_at > max_age_seconds:
                return {}
        data = payload.get("data")
        return data if isinstance(data, dict) else {}

    def save_public_cache(self, name: str, symbol: str, data: dict[str, Any]) -> None:
        path = self.public_cache_path(name, symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            path,
            {
                "source": "binance_futures_mainnet",
                "name": name,
                "symbol": symbol.upper(),
                "updated_at": utc_now_iso(),
                "updated_at_epoch": time.time(),
                "data": data,
            },
        )

    def signed_request(self, method: str, endpoint: str, params: dict[str, Any]) -> Any:
        method = method.upper()
        if method not in {"GET", "HEAD", "OPTIONS"}:
            self.require_live_trading_confirmation(f"{method} {endpoint}")
        self.require_credentials()
        normalized = {key: str(value) for key, value in params.items() if value is not None}
        query = parse.urlencode(normalized)
        signature = hmac.new(
            self.api_secret().encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed_query = f"{query}&signature={signature}" if query else f"signature={signature}"
        url = f"{self.order_base_url()}{endpoint}"
        data = signed_query
        if method == "GET":
            url = f"{url}?{signed_query}"
            data = None
        response = self._request(
            method,
            url,
            data=data,
            headers={
                "X-MBX-APIKEY": self.api_key(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=self.config.binance_request_timeout_seconds,
        )
        if response.status_code >= 400:
            raise V6ExecutionError(
                "Binance signed request failed "
                f"{response.status_code}: {redact_sensitive_text(response.text)}"
            )
        if not response.text:
            return {"status": "ok"}
        return response.json()

    def order_base_url(self) -> str:
        if self.config.order_mode == "testnet_order":
            return self.config.demo_base_url
        if self.config.order_mode == "live_order":
            return self.config.live_base_url
        return self.config.demo_base_url

    def api_key(self) -> str:
        if self.config.order_mode == "testnet_order":
            return self.config.demo_api_key
        if self.config.order_mode == "live_order":
            return self.config.live_api_key
        return ""

    def api_secret(self) -> str:
        if self.config.order_mode == "testnet_order":
            return self.config.demo_api_secret
        if self.config.order_mode == "live_order":
            return self.config.live_api_secret
        return ""

    def require_credentials(self) -> None:
        if self.config.order_mode == "dry_run":
            return
        if not self.api_key():
            raise V6ExecutionError(f"V6 {self.config.order_mode} API key is missing")
        if not self.api_secret():
            raise V6ExecutionError(f"V6 {self.config.order_mode} API secret is missing")

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[list[Any]]:
        limit = max(1, int(limit))
        if not self.config.kline_cache_enabled:
            return self.fetch_klines_from_api_backward(symbol, interval, limit)[0]
        return self.fetch_klines_with_cache(symbol, interval, limit)

    def fetch_klines_from_api_backward(
        self,
        symbol: str,
        interval: str,
        limit: int,
        *,
        end_time: int | None = None,
    ) -> tuple[list[list[Any]], bool]:
        remaining = max(1, int(limit))
        collected: list[list[Any]] = []
        next_end_time = end_time
        exhausted = False
        while remaining > 0:
            batch_limit = min(1500, remaining)
            params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": batch_limit}
            if next_end_time is not None:
                params["endTime"] = next_end_time
            batch = self.public_get("/fapi/v1/klines", params)
            if not isinstance(batch, list) or not batch:
                exhausted = True
                break
            collected = batch + collected
            remaining -= len(batch)
            first_open = int(batch[0][0])
            next_end = first_open - 1
            if next_end_time is not None and next_end >= next_end_time:
                exhausted = True
                break
            next_end_time = next_end
            if len(batch) < batch_limit:
                exhausted = True
                break
            time.sleep(0.08)
        return self.normalize_klines(collected)[-limit:], exhausted or len(collected) < limit

    def fetch_klines_from_api_forward(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: int,
        limit: int,
    ) -> list[list[Any]]:
        remaining = max(1, int(limit))
        collected: list[list[Any]] = []
        next_start = int(start_time)
        interval_ms = KLINE_INTERVAL_MS.get(interval, 0)
        while remaining > 0:
            batch_limit = min(1500, remaining)
            params: dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "limit": batch_limit,
                "startTime": next_start,
            }
            batch = self.public_get("/fapi/v1/klines", params)
            if not isinstance(batch, list) or not batch:
                break
            collected.extend(batch)
            remaining -= len(batch)
            last_open = int(batch[-1][0])
            next_start = last_open + interval_ms if interval_ms > 0 else last_open + 1
            if len(batch) < batch_limit:
                break
            time.sleep(0.08)
        return self.normalize_klines(collected)

    def fetch_klines_with_cache(self, symbol: str, interval: str, limit: int) -> list[list[Any]]:
        symbol = symbol.upper()
        interval = interval.strip()
        limit = max(1, int(limit))
        payload = self.load_kline_cache(symbol, interval)
        cached = self.normalize_klines(payload.get("klines") or [])
        earliest_exhausted = bool(payload.get("earliest_exhausted"))
        interval_ms = KLINE_INTERVAL_MS.get(interval, 0)
        cache_changed = False

        if cached and interval_ms > 0:
            refresh_interval = self.config.kline_refresh_interval_seconds_for_interval(interval)
            cache_age = self.kline_cache_age_seconds(payload)
            should_refresh_tail = refresh_interval <= 0 or cache_age is None or cache_age >= refresh_interval
            if should_refresh_tail:
                tail_refresh_limits = {
                    "1m": 180,
                    "15m": 128,
                    "4h": 72,
                    "1d": 30,
                }
                tail_limit = min(limit, tail_refresh_limits.get(interval, min(limit, 120)))
                try:
                    latest_tail, _ = self.fetch_klines_from_api_backward(
                        symbol,
                        interval,
                        tail_limit,
                    )
                    if latest_tail:
                        cached = self.merge_klines(cached, latest_tail)
                        cache_changed = True
                except V6ExecutionError as exc:
                    self.mark_stale_market_data(symbol, f"kline_{interval}", str(exc))
                    LOGGER.warning("V6 using stale cached %s %s klines after public request failure: %s", symbol, interval, exc)

        if not cached:
            cached, earliest_exhausted = self.fetch_klines_from_api_backward(symbol, interval, limit)
            cache_changed = True
        elif len(cached) < limit and not earliest_exhausted:
            missing = limit - len(cached)
            first_open = int(cached[0][0])
            try:
                older, exhausted = self.fetch_klines_from_api_backward(
                    symbol,
                    interval,
                    missing,
                    end_time=first_open - 1,
                )
                if older:
                    cached = self.merge_klines(older, cached)
                    cache_changed = True
                earliest_exhausted = exhausted or len(cached) < limit
                cache_changed = True
            except V6ExecutionError as exc:
                LOGGER.warning("V6 keeping partial cached %s %s klines after backfill failure: cached=%s desired=%s error=%s", symbol, interval, len(cached), limit, exc)
                earliest_exhausted = len(cached) < limit

        cached = self.normalize_klines(cached)
        if len(cached) > limit:
            cached = cached[-limit:]
            earliest_exhausted = False
            cache_changed = True
        if cache_changed:
            self.save_kline_cache(symbol, interval, cached, earliest_exhausted=earliest_exhausted)
        return cached[-limit:]

    def kline_cache_path(self, symbol: str, interval: str) -> Path:
        return self.config.kline_cache_dir / interval / f"{symbol.upper()}.json"

    def load_kline_cache(self, symbol: str, interval: str) -> dict[str, Any]:
        path = self.kline_cache_path(symbol, interval)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return {}
        if isinstance(payload, list):
            return {"klines": payload, "earliest_exhausted": False}
        if isinstance(payload, dict):
            return payload
        return {}

    @staticmethod
    def kline_cache_age_seconds(payload: dict[str, Any]) -> float | None:
        try:
            updated_at = float(payload.get("updated_at_epoch") or 0)
        except Exception:
            updated_at = 0.0
        if updated_at <= 0:
            return None
        return max(0.0, time.time() - updated_at)

    def save_kline_cache(
        self,
        symbol: str,
        interval: str,
        klines: list[list[Any]],
        *,
        earliest_exhausted: bool,
    ) -> None:
        path = self.kline_cache_path(symbol, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": "binance_futures_mainnet",
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": len(klines),
            "earliest_exhausted": bool(earliest_exhausted),
            "first_open_time": int(klines[0][0]) if klines else None,
            "last_open_time": int(klines[-1][0]) if klines else None,
            "updated_at": utc_now_iso(),
            "updated_at_epoch": time.time(),
            "klines": klines,
        }
        atomic_write_json(path, payload)

    @staticmethod
    def normalize_klines(klines: list[Any]) -> list[list[Any]]:
        rows: dict[int, list[Any]] = {}
        for row in klines:
            if not isinstance(row, list) or not row:
                continue
            try:
                open_time = int(row[0])
            except (TypeError, ValueError):
                continue
            rows[open_time] = row
        return [rows[key] for key in sorted(rows)]

    @classmethod
    def merge_klines(cls, *groups: list[list[Any]]) -> list[list[Any]]:
        merged: list[Any] = []
        for group in groups:
            merged.extend(group)
        return cls.normalize_klines(merged)

    def supplement_4h_klines_for_bb2880(self, symbol: str, klines: list[list[Any]], target_limit: int) -> list[list[Any]]:
        if not self.config.kline_4h_supplement_enabled:
            return klines
        min_limit = int(self.config.kline_4h_supplement_min_limit)
        if len(klines) >= min_limit:
            return klines
        supplemental = self.fetch_supplemental_4h_klines(symbol, min_limit)
        if len(supplemental) < min_limit:
            return klines
        merged = self.merge_klines(supplemental, klines)
        return merged[-max(target_limit, min_limit):]

    def fetch_supplemental_4h_klines(self, symbol: str, limit: int) -> list[list[Any]]:
        for source in ("bybit_linear", "bitget_usdt_futures"):
            try:
                rows = self.fetch_supplemental_4h_klines_with_cache(symbol, source, limit)
            except Exception as exc:
                LOGGER.warning("V6 supplemental 4h kline fetch failed source=%s symbol=%s: %s", source, symbol, exc)
                continue
            if len(rows) >= limit:
                return rows[-limit:]
        return []

    def fetch_supplemental_4h_klines_with_cache(self, symbol: str, source: str, limit: int) -> list[list[Any]]:
        symbol = symbol.upper()
        limit = max(1, int(limit))
        payload = self.load_supplemental_kline_cache(source, symbol, "4h")
        cached = self.normalize_klines(payload.get("klines") or [])
        if len(cached) >= limit:
            return cached[-limit:]
        if source == "bybit_linear":
            rows, exhausted = self.fetch_bybit_linear_4h_klines_backward(symbol, limit)
        elif source == "bitget_usdt_futures":
            rows, exhausted = self.fetch_bitget_usdt_futures_4h_klines_backward(symbol, limit)
        else:
            raise V6ExecutionError(f"unsupported supplemental kline source: {source}")
        if rows:
            cached = self.merge_klines(rows)
            self.save_supplemental_kline_cache(source, symbol, "4h", cached, earliest_exhausted=exhausted)
        return cached[-limit:]

    def supplemental_kline_cache_path(self, source: str, symbol: str, interval: str) -> Path:
        return self.config.kline_cache_dir / "supplemental" / source / interval / f"{symbol.upper()}.json"

    def load_supplemental_kline_cache(self, source: str, symbol: str, interval: str) -> dict[str, Any]:
        path = self.supplemental_kline_cache_path(source, symbol, interval)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def save_supplemental_kline_cache(
        self,
        source: str,
        symbol: str,
        interval: str,
        klines: list[list[Any]],
        *,
        earliest_exhausted: bool,
    ) -> None:
        path = self.supplemental_kline_cache_path(source, symbol, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": source,
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": len(klines),
            "earliest_exhausted": bool(earliest_exhausted),
            "first_open_time": int(klines[0][0]) if klines else None,
            "last_open_time": int(klines[-1][0]) if klines else None,
            "updated_at": utc_now_iso(),
            "klines": klines,
        }
        atomic_write_json(path, payload)

    def fetch_bybit_linear_4h_klines_backward(self, symbol: str, limit: int) -> tuple[list[list[Any]], bool]:
        rows: list[list[Any]] = []
        end_time: int | None = None
        exhausted = False
        remaining = max(1, int(limit))
        while remaining > 0:
            batch_limit = min(1000, remaining)
            params = {
                "category": "linear",
                "symbol": symbol.upper(),
                "interval": "240",
                "limit": str(batch_limit),
            }
            if end_time is not None:
                params["end"] = str(end_time)
            response = self.session.get("https://api.bybit.com/v5/market/kline", params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if response.status_code != 200:
                raise V6ExecutionError(f"Bybit kline request failed {response.status_code}: {response.text}")
            payload = response.json()
            batch = (payload.get("result") or {}).get("list") or []
            if payload.get("retCode") != 0 or not batch:
                exhausted = True
                break
            converted = [self.bybit_kline_to_binance_shape(item) for item in batch]
            rows.extend(converted)
            remaining -= len(converted)
            oldest = min(int(item[0]) for item in converted)
            next_end = oldest - 1
            if end_time == next_end:
                exhausted = True
                break
            end_time = next_end
            if len(batch) < batch_limit:
                exhausted = True
                break
            time.sleep(0.08)
        rows = self.normalize_klines(rows)
        return rows[-limit:], exhausted or len(rows) < limit

    def fetch_bitget_usdt_futures_4h_klines_backward(self, symbol: str, limit: int) -> tuple[list[list[Any]], bool]:
        rows: list[list[Any]] = []
        end_time: int | None = None
        exhausted = False
        remaining = max(1, int(limit))
        while remaining > 0:
            batch_limit = min(200, remaining)
            params = {
                "symbol": symbol.upper(),
                "granularity": "4H",
                "limit": str(batch_limit),
                "productType": "usdt-futures",
            }
            if end_time is not None:
                params["endTime"] = str(end_time)
            response = self.session.get("https://api.bitget.com/api/v2/mix/market/history-candles", params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if response.status_code != 200:
                raise V6ExecutionError(f"Bitget kline request failed {response.status_code}: {response.text}")
            payload = response.json()
            batch = payload.get("data") or []
            if payload.get("code") != "00000" or not batch:
                exhausted = True
                break
            converted = [self.bitget_kline_to_binance_shape(item) for item in batch]
            rows.extend(converted)
            remaining -= len(converted)
            oldest = min(int(item[0]) for item in converted)
            next_end = oldest - 1
            if end_time == next_end:
                exhausted = True
                break
            end_time = next_end
            if len(batch) < batch_limit:
                exhausted = True
                break
            time.sleep(0.08)
        rows = self.normalize_klines(rows)
        return rows[-limit:], exhausted or len(rows) < limit

    @staticmethod
    def bybit_kline_to_binance_shape(row: list[Any]) -> list[Any]:
        return [int(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5]) if len(row) > 5 else "0"]

    @staticmethod
    def bitget_kline_to_binance_shape(row: list[Any]) -> list[Any]:
        return [int(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5]) if len(row) > 5 else "0"]

    def mark_price(self, symbol: str) -> float:
        cached = self.load_public_cache("premiumIndex", symbol, max_age_seconds=self.config.public_mark_price_ttl_seconds)
        if cached:
            return float(cached.get("markPrice") or 0.0)
        try:
            payload = self.public_get("/fapi/v1/premiumIndex", {"symbol": symbol})
        except V6ExecutionError as exc:
            stale = self.load_public_cache("premiumIndex", symbol, max_age_seconds=None)
            if stale:
                self.mark_stale_market_data(symbol, "mark_price", str(exc))
                LOGGER.warning("V6 using stale cached mark price for %s after public request failure: %s", symbol, exc)
                return float(stale.get("markPrice") or 0.0)
            raise
        if isinstance(payload, dict):
            self.save_public_cache("premiumIndex", symbol, payload)
        return float(payload.get("markPrice") or 0.0)

    def order_endpoint_reference_price(self, symbol: str) -> float:
        try:
            payload = self.public_get("/fapi/v1/premiumIndex", {"symbol": symbol}, order_endpoint=True)
            if isinstance(payload, dict):
                value = float(payload.get("markPrice") or payload.get("indexPrice") or 0.0)
                if value > 0:
                    return value
        except Exception:
            pass
        try:
            payload = self.public_get("/fapi/v1/ticker/24hr", {"symbol": symbol}, order_endpoint=True)
            if isinstance(payload, dict):
                value = float(payload.get("lastPrice") or payload.get("weightedAvgPrice") or 0.0)
                if value > 0:
                    return value
        except Exception:
            pass
        return 0.0

    def order_endpoint_book_ticker(self, symbol: str) -> dict[str, Any]:
        payload = self.public_get("/fapi/v1/ticker/bookTicker", {"symbol": symbol}, order_endpoint=True)
        return payload if isinstance(payload, dict) else {}

    def order_endpoint_ticker_24h(self, symbol: str) -> dict[str, Any]:
        payload = self.public_get("/fapi/v1/ticker/24hr", {"symbol": symbol}, order_endpoint=True)
        return payload if isinstance(payload, dict) else {}

    def percent_price_filter(self, symbol: str) -> dict[str, Any]:
        for item in self.exchange_info().get("symbols", []):
            if item.get("symbol") != symbol:
                continue
            return next(
                (flt for flt in item.get("filters", []) if flt.get("filterType") in {"PERCENT_PRICE", "PERCENT_PRICE_BY_SIDE"}),
                {},
            )
        return {}

    def order_endpoint_percent_price_context(self, symbol: str, action: str) -> dict[str, Any]:
        book = self.order_endpoint_book_ticker(symbol)
        ticker = self.order_endpoint_ticker_24h(symbol)
        price_filter = self.percent_price_filter(symbol)

        bid = float(book.get("bidPrice") or 0.0)
        ask = float(book.get("askPrice") or 0.0)
        weighted_avg = float(ticker.get("weightedAvgPrice") or 0.0)
        mark = 0.0
        try:
            mark = float(self.public_get("/fapi/v1/premiumIndex", {"symbol": symbol}, order_endpoint=True).get("markPrice") or 0.0)
        except Exception:
            mark = 0.0

        filter_type = str(price_filter.get("filterType") or "")
        if filter_type == "PERCENT_PRICE_BY_SIDE":
            buy_up = float(price_filter.get("bidMultiplierUp") or 0.0)
            buy_down = float(price_filter.get("bidMultiplierDown") or 0.0)
            sell_up = float(price_filter.get("askMultiplierUp") or 0.0)
            sell_down = float(price_filter.get("askMultiplierDown") or 0.0)
        else:
            buy_up = sell_up = float(price_filter.get("multiplierUp") or 0.0)
            buy_down = sell_down = float(price_filter.get("multiplierDown") or 0.0)

        is_buy = action in {"open_long", "close_short"}
        counterparty = ask if is_buy else bid
        upper_multiplier = buy_up if is_buy else sell_up
        lower_multiplier = buy_down if is_buy else sell_down
        upper_limit = weighted_avg * upper_multiplier if weighted_avg > 0 and upper_multiplier > 0 else 0.0
        lower_limit = weighted_avg * lower_multiplier if weighted_avg > 0 and lower_multiplier > 0 else 0.0

        valid = True
        reason = "ok"
        if counterparty <= 0 or weighted_avg <= 0 or not price_filter:
            valid = True
            reason = "insufficient_filter_context"
        elif is_buy and upper_limit > 0 and counterparty > upper_limit:
            valid = False
            reason = "buy_counterparty_above_percent_price_upper"
        elif not is_buy and lower_limit > 0 and counterparty < lower_limit:
            valid = False
            reason = "sell_counterparty_below_percent_price_lower"

        return {
            "symbol": symbol,
            "action": action,
            "side": "buy" if is_buy else "sell",
            "valid": bool(valid),
            "reason": reason,
            "bid": round(bid, 10),
            "ask": round(ask, 10),
            "counterparty_price": round(counterparty, 10),
            "weighted_avg_price": round(weighted_avg, 10),
            "mark_price": round(mark, 10),
            "filter_type": filter_type,
            "upper_limit": round(upper_limit, 10),
            "lower_limit": round(lower_limit, 10),
            "upper_multiplier": upper_multiplier,
            "lower_multiplier": lower_multiplier,
        }

    def ticker_24h(self, symbol: str) -> dict[str, Any]:
        cached = self.load_public_cache("ticker_24hr", symbol, max_age_seconds=self.config.public_ticker_24h_ttl_seconds)
        if cached:
            return cached
        try:
            payload = self.public_get("/fapi/v1/ticker/24hr", {"symbol": symbol})
        except V6ExecutionError as exc:
            stale = self.load_public_cache("ticker_24hr", symbol, max_age_seconds=None)
            if stale:
                self.mark_stale_market_data(symbol, "ticker_24h", str(exc))
                LOGGER.warning("V6 using stale cached 24h ticker for %s after public request failure: %s", symbol, exc)
                return stale
            raise
        if not isinstance(payload, dict):
            return {}
        self.save_public_cache("ticker_24hr", symbol, payload)
        return payload

    def exchange_info(self) -> dict[str, Any]:
        if self._exchange_info_cache is None:
            payload = self.public_get("/fapi/v1/exchangeInfo", order_endpoint=self.config.order_mode != "dry_run")
            if not isinstance(payload, dict):
                raise V6ExecutionError("exchangeInfo returned non-dict payload")
            self._exchange_info_cache = payload
        return self._exchange_info_cache

    def quantity_rules(self, symbol: str) -> tuple[float, float]:
        for item in self.exchange_info().get("symbols", []):
            if item.get("symbol") != symbol:
                continue
            lot_filter = next(
                (flt for flt in item.get("filters", []) if flt.get("filterType") in {"LOT_SIZE", "MARKET_LOT_SIZE"}),
                None,
            )
            if lot_filter:
                return float(lot_filter.get("stepSize") or 0.0), float(lot_filter.get("minQty") or 0.0)
        raise V6ExecutionError(f"quantity rules not found for {symbol}")

    def price_tick_size(self, symbol: str) -> float:
        for item in self.exchange_info().get("symbols", []):
            if item.get("symbol") != symbol:
                continue
            price_filter = next(
                (flt for flt in item.get("filters", []) if flt.get("filterType") == "PRICE_FILTER"),
                None,
            )
            if price_filter:
                return float(price_filter.get("tickSize") or 0.0)
        raise V6ExecutionError(f"price rules not found for {symbol}")

    def resolve_price(self, symbol: str, price: float) -> str:
        if price <= 0:
            raise V6ExecutionError(f"cannot resolve price for {symbol}: price={price}")
        tick_size = self.price_tick_size(symbol)
        rounded = self.round_down_to_step(price, tick_size)
        if rounded <= 0:
            raise V6ExecutionError(f"resolved zero price for {symbol}")
        return self.format_decimal(rounded)

    def resolve_quantity(self, symbol: str, notional_usdt: float, price: float) -> str:
        if price <= 0 or notional_usdt <= 0:
            raise V6ExecutionError(f"cannot resolve quantity for {symbol}: price={price} notional={notional_usdt}")
        step, min_qty = self.quantity_rules(symbol)
        raw_qty = notional_usdt / price
        qty = self.round_down_to_step(raw_qty, step)
        if qty < min_qty:
            qty = self.round_down_to_step(min_qty, step)
        if qty <= 0:
            raise V6ExecutionError(f"resolved zero quantity for {symbol}")
        return self.format_decimal(qty)

    def resolve_close_quantity(self, intent: V6OrderIntent) -> str:
        actual = self.actual_position_for_intent(intent)
        if actual is None:
            raise V6ExecutionError(f"no exchange position found for {intent.symbol} {intent.action}")
        amount = float(actual.get("positionAmt") or 0.0)
        expected_side = "short" if intent.action == "close_short" else "long"
        actual_side = self.position_side_from_risk_row(actual)
        if actual_side != expected_side:
            raise V6ExecutionError(
                f"exchange position side mismatch for {intent.symbol}: expected={expected_side} actual={actual_side} amount={amount}"
            )
        quantity = abs(amount)
        if quantity <= 0:
            raise V6ExecutionError(f"zero exchange position quantity for {intent.symbol} {intent.action}")
        step, _ = self.quantity_rules(intent.symbol)
        close_fraction = clamp(float(intent.close_fraction or 1.0), 0.0, 1.0)
        if 0.0 < close_fraction < 0.999:
            quantity = self.round_down_to_step(quantity * close_fraction, step)
            if quantity <= 0:
                raise V6ExecutionError(f"resolved zero partial close quantity for {intent.symbol} {intent.action}")
        else:
            quantity = self.round_down_to_step(quantity, step)
        return self.format_decimal(quantity)

    def actual_position_for_intent(self, intent: V6OrderIntent) -> dict[str, Any] | None:
        expected_side = "short" if intent.action in {"open_short", "close_short"} else "long"
        for row in self.actual_position_rows(symbol=intent.symbol):
            if self.position_side_from_risk_row(row) == expected_side:
                return row
        return None

    def actual_position_rows(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = self.signed_params()
        if symbol:
            params["symbol"] = symbol
        payload = self.signed_request("GET", "/fapi/v2/positionRisk", params)
        rows = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
        return [row for row in rows if isinstance(row, dict) and abs(float(row.get("positionAmt") or 0.0)) > 0]

    @staticmethod
    def exchange_position_snapshot(row: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(row, dict) or not row:
            return {}

        def as_float(key: str) -> float:
            try:
                return float(row.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return {
            "symbol": str(row.get("symbol") or "").upper(),
            "position_side": str(row.get("positionSide") or ""),
            "side": V6BinanceClient.position_side_from_risk_row(row),
            "position_amt": as_float("positionAmt"),
            "abs_position_qty": abs(as_float("positionAmt")),
            "entry_price": as_float("entryPrice"),
            "break_even_price": as_float("breakEvenPrice"),
            "mark_price": as_float("markPrice"),
            "notional": as_float("notional"),
            "unrealized_profit": as_float("unRealizedProfit"),
            "leverage": int(as_float("leverage")) if as_float("leverage") > 0 else 0,
            "margin_type": str(row.get("marginType") or ""),
            "isolated": str(row.get("isolated") or "").lower() == "true" or row.get("isolated") is True,
            "update_time": row.get("updateTime"),
        }

    def exchange_snapshot_notional(self, snapshot: dict[str, Any]) -> float:
        try:
            notional = abs(float(snapshot.get("notional") or 0.0))
        except (TypeError, ValueError):
            notional = 0.0
        if notional > 0:
            return notional
        try:
            qty = abs(float(snapshot.get("position_amt") or snapshot.get("abs_position_qty") or 0.0))
            mark = float(snapshot.get("mark_price") or snapshot.get("entry_price") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return abs(qty * mark)

    def is_dust_exchange_snapshot(self, snapshot: dict[str, Any]) -> bool:
        if not snapshot:
            return False
        threshold = max(
            0.0,
            float(self.config.dust_position_notional_usdt or 0.0),
            float(getattr(self.config, "strategy_dust_position_notional_usdt", 0.0) or 0.0),
        )
        if threshold <= 0:
            return False
        try:
            qty = abs(float(snapshot.get("position_amt") or snapshot.get("abs_position_qty") or 0.0))
        except (TypeError, ValueError):
            qty = 0.0
        return qty > 0 and self.exchange_snapshot_notional(snapshot) < threshold

    def position_reconcile_payload(
        self,
        intent: V6OrderIntent,
        before_row: dict[str, Any] | None,
        after_row: dict[str, Any] | None,
        *,
        before_error: str = "",
        after_error: str = "",
    ) -> dict[str, Any]:
        before = self.exchange_position_snapshot(before_row)
        after = self.exchange_position_snapshot(after_row)
        before_qty = abs(float(before.get("position_amt") or 0.0))
        after_qty = abs(float(after.get("position_amt") or 0.0))
        if intent.action in {"open_long", "open_short"}:
            delta_qty = max(0.0, after_qty - before_qty)
        elif intent.action in {"close_long", "close_short"}:
            delta_qty = max(0.0, before_qty - after_qty)
        else:
            delta_qty = 0.0
        return {
            "status": "ok" if not before_error and not after_error else "partial",
            "symbol": intent.symbol,
            "side": "short" if intent.action in {"open_short", "close_short"} else "long",
            "action": intent.action,
            "before": before,
            "after": after,
            "delta_qty_abs": round(delta_qty, 12),
            "before_error": before_error,
            "after_error": after_error,
            "synced_at": utc_now_iso(),
        }

    @staticmethod
    def submitted_order_has_no_confirmed_position_delta(result: dict[str, Any]) -> bool:
        status = str(result.get("status") or "")
        if status not in {"submitted", "submitted_unprotected", "submitted_no_position_delta"}:
            return False
        reconcile = result.get("position_reconcile")
        if not isinstance(reconcile, dict):
            return False
        try:
            delta_qty = abs(float(reconcile.get("delta_qty_abs") or 0.0))
        except Exception:
            delta_qty = 0.0
        return delta_qty <= 0.0

    @staticmethod
    def position_side_from_risk_row(row: dict[str, Any]) -> str:
        position_side = str(row.get("positionSide") or "").upper()
        amount = float(row.get("positionAmt") or 0.0)
        if position_side == "LONG":
            return "long"
        if position_side == "SHORT":
            return "short"
        if amount > 0:
            return "long"
        if amount < 0:
            return "short"
        return ""

    def current_position_mode(self) -> str:
        configured = self.config.position_mode.strip().lower().replace("-", "_")
        aliases = {
            "oneway": "one_way",
            "one_way": "one_way",
            "both": "one_way",
            "hedge": "hedge",
            "dual": "hedge",
        }
        if configured in aliases:
            return aliases[configured]
        if self.config.order_mode == "dry_run":
            return "one_way"
        if self._position_mode_cache:
            return self._position_mode_cache
        payload = self.signed_request("GET", "/fapi/v1/positionSide/dual", self.signed_params())
        raw = payload.get("dualSidePosition")
        dual = raw if isinstance(raw, bool) else str(raw).lower() == "true"
        self._position_mode_cache = "hedge" if dual else "one_way"
        return self._position_mode_cache

    def signed_params(self) -> dict[str, Any]:
        return {
            "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            "recvWindow": str(self.config.recv_window_ms),
        }

    def desired_margin_type(self) -> str:
        raw = str(self.config.margin_type or "").strip().upper().replace("-", "_")
        aliases = {
            "ISOLATED": "ISOLATED",
            "ISOLATE": "ISOLATED",
            "ISOLATED_MARGIN": "ISOLATED",
            "CROSS": "CROSSED",
            "CROSSED": "CROSSED",
            "CROSS_MARGIN": "CROSSED",
        }
        return aliases.get(raw, "")

    def set_margin_type(self, symbol: str, margin_type: str) -> dict[str, Any]:
        desired = str(margin_type or "").strip().upper()
        if desired not in {"ISOLATED", "CROSSED"}:
            return {
                "status": "skipped",
                "reason": f"unsupported margin type: {margin_type}",
                "submitted_at": utc_now_iso(),
            }
        params = {
            "symbol": symbol,
            "marginType": desired,
            **self.signed_params(),
        }
        try:
            payload = self.signed_request("POST", "/fapi/v1/marginType", params)
            return {
                "status": "submitted",
                "request": {k: v for k, v in params.items() if k not in {"timestamp", "signature"}},
                "response": payload,
                "submitted_at": utc_now_iso(),
            }
        except V6ExecutionError as exc:
            message = str(exc)
            if "-4046" in message or "No need to change margin type" in message:
                return {
                    "status": "already_set",
                    "request": {k: v for k, v in params.items() if k not in {"timestamp", "signature"}},
                    "response": {"code": -4046, "msg": "No need to change margin type"},
                    "submitted_at": utc_now_iso(),
                }
            if "-4047" in message or "exists open orders" in message:
                return {
                    "status": "skipped_open_orders_exist",
                    "request": {k: v for k, v in params.items() if k not in {"timestamp", "signature"}},
                    "response": {"code": -4047, "msg": "Margin type cannot be changed if there exists open orders"},
                    "submitted_at": utc_now_iso(),
                }
            raise

    def ensure_margin_type_for_open(self, symbol: str) -> dict[str, Any] | None:
        desired = self.desired_margin_type()
        if not desired or self.config.order_mode == "dry_run":
            return None
        return self.set_margin_type(symbol, desired)

    def max_initial_leverage(self, symbol: str) -> int:
        symbol = symbol.upper()
        cached = self._leverage_bracket_cache.get(symbol)
        if cached is None:
            cached = self.signed_request("GET", "/fapi/v1/leverageBracket", {
                "symbol": symbol,
                **self.signed_params(),
            })
            self._leverage_bracket_cache[symbol] = cached
        rows = cached if isinstance(cached, list) else [cached] if isinstance(cached, dict) else []
        values: list[int] = []
        for row in rows:
            brackets = row.get("brackets") if isinstance(row, dict) else None
            for bracket in brackets or []:
                if not isinstance(bracket, dict):
                    continue
                try:
                    value = int(float(bracket.get("initialLeverage") or 0))
                except (TypeError, ValueError):
                    value = 0
                if value > 0:
                    values.append(value)
        return max(values or [0])

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        requested_leverage = int(leverage)
        if requested_leverage <= 0:
            raise V6ExecutionError(f"Invalid leverage: {requested_leverage}")
        max_leverage = 0
        resolved_leverage = requested_leverage
        try:
            max_leverage = self.max_initial_leverage(symbol)
        except Exception as exc:
            LOGGER.warning("Failed to load leverage bracket for %s: %s", symbol, exc)
        if max_leverage > 0:
            resolved_leverage = min(requested_leverage, max_leverage)
        params = {
            "symbol": symbol,
            "leverage": str(resolved_leverage),
            **self.signed_params(),
        }
        payload = self.signed_request("POST", "/fapi/v1/leverage", params)
        return {
            "status": "submitted",
            "requested_leverage": requested_leverage,
            "resolved_leverage": resolved_leverage,
            "max_initial_leverage": max_leverage,
            "leverage_adjusted": resolved_leverage != requested_leverage,
            "request": {k: v for k, v in params.items() if k not in {"timestamp", "signature"}},
            "response": payload,
            "submitted_at": utc_now_iso(),
        }

    def submit_order(self, intent: V6OrderIntent) -> dict[str, Any]:
        if self.config.order_mode == "dry_run":
            return {
                "status": "dry_run",
                "order_mode": "dry_run",
                "intent": intent.to_dict(),
                "submitted_at": utc_now_iso(),
            }
        self.require_live_trading_confirmation("submit_order")
        if self.config.submit_protective_stop and intent.action in {"open_long", "open_short"}:
            protective_precheck = self.protective_stop_precheck(intent)
            if protective_precheck.get("would_immediately_trigger"):
                return {
                    "status": "skipped_protective_stop_immediate_trigger",
                    "order_mode": self.config.order_mode,
                    "intent": intent.to_dict(),
                    "protective_stop_precheck": protective_precheck,
                    "error": "",
                    "telegram_silent": True,
                    "maintenance_only": True,
                    "submitted_at": utc_now_iso(),
                    "note": "Open skipped because the protective stop would immediately trigger at current mark price.",
                }
        before_position = None
        before_position_error = ""
        try:
            before_position = self.actual_position_for_intent(intent)
        except Exception as exc:
            before_position_error = f"{type(exc).__name__}: {exc}"
        margin_type_result = None
        leverage_result = None
        if intent.action in {"open_long", "open_short"}:
            margin_type_result = self.ensure_margin_type_for_open(intent.symbol)
            leverage_result = self.set_leverage(intent.symbol, intent.leverage)
            try:
                intent.leverage = int(leverage_result.get("resolved_leverage") or intent.leverage)
            except (TypeError, ValueError):
                pass
        quantity = self.resolve_close_quantity(intent) if intent.action in {"close_long", "close_short"} else self.resolve_quantity(intent.symbol, intent.quantity_hint_usdt, intent.entry_price)
        params: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": intent.side,
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "RESULT",
            **self.signed_params(),
        }
        position_mode = self.current_position_mode()
        if position_mode == "hedge":
            params["positionSide"] = position_side_for_action(intent.action, intent.side)
        elif intent.reduce_only:
            params["reduceOnly"] = "true"
        payload = self.signed_request("POST", "/fapi/v1/order", params)
        payload = self.enrich_order_fill_payload(intent.symbol, payload)
        result = {
            "status": "submitted",
            "order_mode": self.config.order_mode,
            "intent": intent.to_dict(),
            "request": {k: v for k, v in params.items() if k not in {"timestamp", "signature"}},
            "response": payload,
            "submitted_at": utc_now_iso(),
        }
        if margin_type_result is not None:
            result["margin_type"] = margin_type_result
        if leverage_result is not None:
            result["leverage"] = leverage_result
        if self.config.submit_protective_stop and intent.action in {"open_long", "open_short"}:
            adjustment = self.adjust_open_stop_after_fill(intent, payload)
            if adjustment:
                result["protective_stop_adjustment"] = adjustment
                result["intent"] = intent.to_dict()
            try:
                result["protective_stop"] = self.submit_protective_stop(intent, quantity=quantity)
            except Exception as exc:
                result["protective_stop"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "submitted_at": utc_now_iso(),
                }
                try:
                    result["emergency_close"] = self.submit_reduce_only_close_for_unprotected_open(
                        intent,
                        quantity=self.filled_quantity_from_response(payload, quantity),
                    )
                    result["status"] = "closed_after_protective_stop_failed"
                except Exception as close_exc:
                    result["emergency_close"] = {
                        "status": "failed",
                        "error": f"{type(close_exc).__name__}: {close_exc}",
                        "submitted_at": utc_now_iso(),
                    }
                    result["status"] = "submitted_unprotected"
        after_position = None
        after_position_error = ""
        try:
            after_position = self.actual_position_for_intent(intent)
        except Exception as exc:
            after_position_error = f"{type(exc).__name__}: {exc}"
        result["position_reconcile"] = self.position_reconcile_payload(
            intent,
            before_position,
            after_position,
            before_error=before_position_error,
            after_error=after_position_error,
        )
        if self.submitted_order_has_no_confirmed_position_delta(result):
            result["status"] = "submitted_no_position_delta"
            result["reconcile_warning"] = "exchange position quantity did not change after submitted order"
        return result

    def enrich_order_fill_payload(self, symbol: str, payload: Any) -> Any:
        if self.config.order_mode == "dry_run" or not isinstance(payload, dict):
            return payload
        if self.filled_price_from_response(payload, 0.0) > 0:
            return payload
        order_id = payload.get("orderId")
        if not order_id:
            return payload
        enriched = dict(payload)
        errors: list[str] = []
        try:
            order = self.signed_request("GET", "/fapi/v1/order", {
                "symbol": symbol,
                "orderId": str(order_id),
                **self.signed_params(),
            })
            if isinstance(order, dict):
                for key in ("avgPrice", "cumQuote", "executedQty", "origQty"):
                    value = order.get(key)
                    try:
                        parsed = float(value or 0.0)
                    except (TypeError, ValueError):
                        parsed = 0.0
                    if parsed > 0:
                        enriched[key] = value
        except Exception as exc:
            errors.append(f"order:{type(exc).__name__}: {exc}")

        if self.filled_price_from_response(enriched, 0.0) <= 0:
            try:
                trades = self.signed_request("GET", "/fapi/v1/userTrades", {
                    "symbol": symbol,
                    "orderId": str(order_id),
                    **self.signed_params(),
                })
                if isinstance(trades, list):
                    executed_qty = 0.0
                    cum_quote = 0.0
                    for trade in trades:
                        if not isinstance(trade, dict):
                            continue
                        try:
                            qty = float(trade.get("qty") or 0.0)
                            quote = float(trade.get("quoteQty") or 0.0)
                        except (TypeError, ValueError):
                            qty = 0.0
                            quote = 0.0
                        executed_qty += qty
                        cum_quote += quote
                    if executed_qty > 0 and cum_quote > 0:
                        enriched["executedQty"] = self.format_decimal(executed_qty)
                        enriched["cumQuote"] = self.format_decimal(cum_quote)
                        enriched["avgPrice"] = self.format_decimal(cum_quote / executed_qty)
                        enriched["fillTradeCount"] = len(trades)
            except Exception as exc:
                errors.append(f"trades:{type(exc).__name__}: {exc}")
        if errors:
            enriched["fillDetailErrors"] = errors[-3:]
        return enriched

    def submit_protective_stop(self, intent: V6OrderIntent, quantity: str | None = None) -> dict[str, Any]:
        stop_side = "SELL" if intent.action == "open_long" else "BUY"
        position_mode = self.current_position_mode()
        params: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": stop_side,
            "positionSide": "LONG" if position_mode == "hedge" and intent.action == "open_long" else "SHORT" if position_mode == "hedge" else "BOTH",
            "algoType": "CONDITIONAL",
            "type": "STOP_MARKET",
            "triggerPrice": self.resolve_price(intent.symbol, intent.stop_loss_price),
            "quantity": quantity or self.resolve_quantity(intent.symbol, intent.quantity_hint_usdt, intent.entry_price),
            "workingType": "MARK_PRICE",
            **self.signed_params(),
        }
        if position_mode != "hedge":
            params["reduceOnly"] = "true"
        payload = self.signed_request("POST", "/fapi/v1/algoOrder", params)
        return {
            "status": "submitted",
            "request": {k: v for k, v in params.items() if k not in {"timestamp", "signature"}},
            "response": payload,
            "submitted_at": utc_now_iso(),
        }

    def protective_stop_precheck(self, intent: V6OrderIntent) -> dict[str, Any]:
        if intent.action not in {"open_long", "open_short"}:
            return {"checked": False, "reason": "not_open_action"}
        try:
            mark = float(self.mark_price(intent.symbol) or 0.0)
            stop_price = float(self.resolve_price(intent.symbol, intent.stop_loss_price))
        except Exception as exc:
            return {
                "checked": False,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if mark <= 0 or stop_price <= 0:
            return {
                "checked": False,
                "status": "invalid_price",
                "mark_price": mark,
                "stop_loss_price": stop_price,
            }
        would_trigger = (
            mark <= stop_price if intent.action == "open_long" else mark >= stop_price
        )
        return {
            "checked": True,
            "symbol": intent.symbol,
            "action": intent.action,
            "mark_price": mark,
            "stop_loss_price": stop_price,
            "would_immediately_trigger": bool(would_trigger),
        }

    def cancel_order(self, symbol: str, order_id: int | str) -> dict[str, Any]:
        payload = self.signed_request("DELETE", "/fapi/v1/algoOrder", {
            "algoId": str(order_id),
            **self.signed_params(),
        })
        return {"status": "submitted", "response": payload, "submitted_at": utc_now_iso()}

    def submit_reduce_only_close_for_unprotected_open(self, intent: V6OrderIntent, quantity: str) -> dict[str, Any]:
        close_action = "close_short" if intent.action == "open_short" else "close_long"
        close_side = "BUY" if intent.action == "open_short" else "SELL"
        params: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "RESULT",
            **self.signed_params(),
        }
        position_mode = self.current_position_mode()
        if position_mode == "hedge":
            params["positionSide"] = position_side_for_action(close_action, close_side)
        else:
            params["reduceOnly"] = "true"
        payload = self.signed_request("POST", "/fapi/v1/order", params)
        return {
            "status": "submitted",
            "request": {k: v for k, v in params.items() if k not in {"timestamp", "signature"}},
            "response": payload,
            "submitted_at": utc_now_iso(),
        }

    def adjust_open_stop_after_fill(self, intent: V6OrderIntent, payload: Any) -> dict[str, Any] | None:
        if intent.action not in {"open_long", "open_short"}:
            return None
        fill_price = self.filled_price_from_response(payload, intent.entry_price)
        if fill_price <= 0:
            return None
        try:
            mark = float(self.mark_price(intent.symbol) or 0.0)
        except Exception:
            mark = 0.0
        original_stop = float(intent.stop_loss_price or 0.0)
        if original_stop <= 0:
            return None
        if intent.action == "open_long":
            fallback_stop = fill_price * (1.0 - abs(float(self.config.stop_loss_pct)))
            adjusted_stop = min(original_stop, fallback_stop)
            if mark > 0:
                adjusted_stop = min(adjusted_stop, mark * 0.9975)
            needs_adjustment = original_stop >= fill_price or (mark > 0 and original_stop >= mark)
        else:
            fallback_stop = fill_price * (1.0 + abs(float(self.config.stop_loss_pct)))
            adjusted_stop = max(original_stop, fallback_stop)
            if mark > 0:
                adjusted_stop = max(adjusted_stop, mark * 1.0025)
            needs_adjustment = original_stop <= fill_price or (mark > 0 and original_stop <= mark)
        if not needs_adjustment or adjusted_stop <= 0:
            return None
        rounded_stop = round(float(adjusted_stop), 8)
        intent.stop_loss_price = rounded_stop
        if isinstance(intent.risk_plan, dict):
            intent.risk_plan["stop_loss_price"] = rounded_stop
            intent.risk_plan["protective_stop_adjusted_after_fill"] = {
                "original_stop_loss_price": round(original_stop, 8),
                "adjusted_stop_loss_price": rounded_stop,
                "fill_price": round(fill_price, 8),
                "mark_price": round(mark, 8) if mark > 0 else 0.0,
                "reason": "post_fill_stop_would_immediately_trigger_or_cross_fill",
            }
        return {
            "original_stop_loss_price": round(original_stop, 8),
            "adjusted_stop_loss_price": rounded_stop,
            "fill_price": round(fill_price, 8),
            "mark_price": round(mark, 8) if mark > 0 else 0.0,
            "reason": "post_fill_stop_would_immediately_trigger_or_cross_fill",
        }

    @staticmethod
    def filled_quantity_from_response(payload: Any, fallback: str) -> str:
        if isinstance(payload, dict):
            for key in ("executedQty", "origQty"):
                value = payload.get(key)
                if value is not None:
                    try:
                        if float(value) > 0:
                            return str(value)
                    except (TypeError, ValueError):
                        pass
        return fallback

    @staticmethod
    def filled_price_from_response(payload: Any, fallback: float) -> float:
        if isinstance(payload, dict):
            for key in ("avgPrice", "price"):
                value = payload.get(key)
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    parsed = 0.0
                if parsed > 0:
                    return parsed
            try:
                executed_qty = float(payload.get("executedQty") or 0.0)
                cum_quote = float(payload.get("cumQuote") or 0.0)
            except (TypeError, ValueError):
                executed_qty = 0.0
                cum_quote = 0.0
            if executed_qty > 0 and cum_quote > 0:
                return cum_quote / executed_qty
        try:
            return float(fallback or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def round_down_to_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        value_dec = Decimal(str(value))
        step_dec = Decimal(str(step))
        return float((value_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec)

    @staticmethod
    def format_decimal(value: float) -> str:
        return format(Decimal(str(value)).normalize(), "f")


def local_side_for_action(action: str) -> str:
    return "short" if action in {"open_short", "close_short"} else "long"


def position_side_for_action(action: str, side: str) -> str:
    if action in {"open_short", "close_short"}:
        return "SHORT"
    if action in {"open_long", "close_long"}:
        return "LONG"
    return "LONG" if side.upper() == "BUY" else "SHORT"


class V6ModelScorer:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = None
        if model_path.exists():
            with model_path.open("rb") as fh:
                self.model = pickle.load(fh)

    def quality_score(self, features: CoinFeatures) -> float:
        if self.model is None:
            return 0.5
        vector = np.array([features.to_feature_vector()])
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vector)[0]
            if len(probs) > 1:
                return clamp(float(probs[1]), 0.0, 1.0)
        raw = self.model.predict(vector)
        return 0.65 if int(raw[0]) == 1 else 0.35


class V6ExecutionEngine:
    def __init__(self, config: V6ExecutionConfig | None = None) -> None:
        self.config = config or V6ExecutionConfig.from_env()
        self.config.ensure_dirs()
        self.client = V6BinanceClient(self.config)
        self.scorer = V6ModelScorer(self.config.model_path)
        self._market_daily_ma7_reversal_cache: dict[str, Any] | None = None
        self._v5_sentiment_metadata_cache: dict[str, dict[str, Any]] | None = None
        self._v5_momentum_panel_cache: dict[str, dict[str, dict[str, Any]]] | None = None
        self._smart_signal_cache: dict[str, dict[str, Any]] | None = None
        self.gate_recorder = GateEventRecorder.from_env()
        self.transition_gate_policy = TransitionGatePolicy.from_env()
        if self.config.transition_gate_policy_enabled and not self.transition_gate_policy.enabled:
            self.transition_gate_policy = TransitionGatePolicy(enabled=True)

    def close(self) -> None:
        self.client.close()

    def execution_lock_target(self) -> Path:
        return Path(os.getenv("TS_V6_EXECUTION_LOCK_TARGET", str(self.config.state_dir / "v6_execution")))

    def execution_lock(self):
        return timed_file_lock(
            self.execution_lock_target(),
            timeout_seconds=_env_float("TS_V6_EXECUTION_LOCK_TIMEOUT_SECONDS", 55.0),
            poll_seconds=_env_float("TS_V6_EXECUTION_LOCK_POLL_SECONDS", 0.25),
        )

    def run_once(self, *, submit: bool = False, commit_dry_run_state: bool | None = None) -> dict[str, Any]:
        if _env_bool("TS_V6_EXECUTION_LOCK_HELD", False):
            return self._run_once_locked(submit=submit, commit_dry_run_state=commit_dry_run_state)
        with self.execution_lock():
            return self._run_once_locked(submit=submit, commit_dry_run_state=commit_dry_run_state)

    def _run_once_locked(self, *, submit: bool = False, commit_dry_run_state: bool | None = None) -> dict[str, Any]:
        self._market_daily_ma7_reversal_cache = None
        self._smart_signal_cache = None
        if submit:
            self.client.require_live_trading_confirmation("run_once submit")
        state = self.load_state()
        maintenance: list[dict[str, Any]] = []
        if _env_bool("TS_V6_RECONCILE_EVERY_RUN", False):
            reconcile_result = self.reconcile_state_with_exchange(state)
            if reconcile_result:
                maintenance.append(reconcile_result)
        signals: list[V6Signal] = []
        intents: list[V6OrderIntent] = []
        results: list[dict[str, Any]] = []
        for symbol in self.config.symbols:
            try:
                signal = self.build_signal(symbol)
                signals.append(signal)
                intent = self.intent_for_signal(signal, state)
                if intent:
                    intents.append(intent)
            except Exception as exc:
                LOGGER.exception("V6 signal build failed for %s", symbol)
                results.append({
                    "symbol": symbol,
                    "status": "signal_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "logged_at": utc_now_iso(),
                })
        def compact_signal_for_jsonl(signal_data: dict[str, Any]) -> dict[str, Any]:
            h4 = signal_data.get("h4") if isinstance(signal_data.get("h4"), dict) else {}
            fifteen_m = signal_data.get("fifteen_m") if isinstance(signal_data.get("fifteen_m"), dict) else {}
            return {
                "symbol": signal_data.get("symbol"),
                "computed_at": signal_data.get("computed_at"),
                "price": signal_data.get("price"),
                "side_bias": signal_data.get("side_bias"),
                "long_probability": signal_data.get("long_probability"),
                "short_probability": signal_data.get("short_probability"),
                "wait_probability": signal_data.get("wait_probability"),
                "model_quality_score": signal_data.get("model_quality_score"),
                "reasons": list(signal_data.get("reasons") or [])[:8],
                "h4": {
                    "structure_state": h4.get("structure_state"),
                    "price_location": h4.get("price_location"),
                    "above_ma20": h4.get("above_ma20"),
                    "ma20_slope_pct": h4.get("ma20_slope_pct"),
                    "right_side_state": (h4.get("right_side") or {}).get("state") if isinstance(h4.get("right_side"), dict) else None,
                },
                "fifteen_m": {
                    "structure_state": fifteen_m.get("structure_state"),
                    "price_location": fifteen_m.get("price_location"),
                    "above_ma20": fifteen_m.get("above_ma20"),
                    "ma20_slope_pct": fifteen_m.get("ma20_slope_pct"),
                },
            }

        signal_dicts = [signal.to_dict() for signal in signals]
        compact_signal_dicts = [compact_signal_for_jsonl(signal_data) for signal_data in signal_dicts]
        for signal_data in compact_signal_dicts:
            append_text_line(self.config.signals_file, json.dumps(signal_data, ensure_ascii=False))
        should_commit_dry = self.config.commit_dry_run_state if commit_dry_run_state is None else commit_dry_run_state
        for intent in intents:
            result: dict[str, Any]
            existing_position: dict[str, Any] | None = None
            pre_submit_cancellation: dict[str, Any] | None = None
            try:
                existing_position = self.find_position(state, intent.symbol, side=local_side_for_action(intent.action)) if intent.action in {"close_long", "close_short"} or self.is_scale_in_intent(intent) else None
                if submit and existing_position and intent.action in {"close_long", "close_short"}:
                    pre_submit_cancellation = self.cancel_protective_stop_if_present(existing_position)
                    if pre_submit_cancellation:
                        cancel_wait = self.wait_for_protective_stop_cancel(
                            intent.symbol,
                            existing_position.get("protective_stop_order_id"),
                        )
                        pre_submit_cancellation["clear_wait"] = cancel_wait
                        if cancel_wait.get("status") in {"still_open", "failed"}:
                            retry_wait = None
                            if self.is_staged_take_profit_intent(intent):
                                retry_wait = self.mark_partial_take_profit_retry_wait(
                                    existing_position,
                                    intent,
                                    "protective stop cancel is still pending before reduce-only close",
                                    reason="protective_stop_cancel_pending",
                                    minutes=2,
                                )
                                if retry_wait:
                                    maintenance.append(retry_wait)
                            result = {
                                "status": "close_waiting_for_protective_stop_cancel",
                                "order_mode": self.config.order_mode,
                                "intent": intent.to_dict(),
                                "pre_submit_cancelled_protective_stop": pre_submit_cancellation,
                                "retry_wait": retry_wait or {},
                                "error": "",
                                "telegram_silent": True,
                                "maintenance_only": True,
                                "submitted_at": utc_now_iso(),
                                "note": "Protective stop cancellation had not cleared yet; close was deferred to avoid Binance reduce-only rejection.",
                            }
                            results.append(result)
                            append_text_line(self.config.orders_file, json.dumps(result, ensure_ascii=False))
                            continue
                        settle_seconds = max(0.0, float(self.config.protective_stop_cancel_settle_seconds))
                        if settle_seconds > 0:
                            time.sleep(settle_seconds)
                            pre_submit_cancellation["settle_wait_seconds"] = settle_seconds
                if submit:
                    result = self.client.submit_order(intent)
                    if pre_submit_cancellation:
                        result["pre_submit_cancelled_protective_stop"] = pre_submit_cancellation
                else:
                    result = {
                        "status": "planned",
                        "order_mode": self.config.order_mode,
                        "intent": intent.to_dict(),
                        "submitted_at": utc_now_iso(),
                    }
                    if pre_submit_cancellation:
                        result["pre_submit_cancelled_protective_stop"] = pre_submit_cancellation
                if (
                    result.get("status") == "submitted"
                    and existing_position
                    and not self.is_partial_take_profit_intent(intent)
                    and not pre_submit_cancellation
                ):
                    cancellation = self.cancel_protective_stop_if_present(existing_position)
                    if cancellation:
                        result["cancelled_protective_stop"] = cancellation
                results.append(result)
                append_text_line(self.config.orders_file, json.dumps(result, ensure_ascii=False))
                if result.get("status") == "skipped_protective_stop_immediate_trigger" and intent.action in {"open_long", "open_short"}:
                    self.set_entry_cooldown(
                        state,
                        intent.symbol,
                        hours=0.5,
                        reason="protective_stop_would_immediately_trigger",
                        side="short" if intent.action == "open_short" else "long",
                    )
                    journal = list(state.get("journal") or [])
                    journal.append({
                        "timestamp": utc_now_iso(),
                        "intent": intent.to_dict(),
                        "result_status": result.get("status"),
                        "order_mode": result.get("order_mode"),
                        "state_mutation": "skipped_protective_stop_immediate_trigger",
                        "protective_stop_precheck": result.get("protective_stop_precheck"),
                    })
                    state["journal"] = journal[-1000:]
                if result.get("status") in {"submitted", "submitted_unprotected", "submitted_no_position_delta"} or (result.get("status") == "dry_run" and should_commit_dry):
                    self.apply_intent_to_state(state, intent, result)
                    if result.get("status") == "submitted" and existing_position and self.is_partial_take_profit_intent(intent):
                        updated_position = self.find_position(state, intent.symbol, side=local_side_for_action(intent.action))
                        if updated_position:
                            sync = self.refresh_protective_stop_after_partial_take_profit(updated_position)
                            if sync:
                                result["partial_take_profit_protective_stop_sync"] = sync
                                maintenance.append(sync)
                                append_text_line(self.config.orders_file, json.dumps({"event": "partial_take_profit_protective_stop_sync", "intent": intent.to_dict(), "sync": sync}, ensure_ascii=False))
                    if result.get("status") == "submitted" and existing_position and self.is_scale_in_intent(intent):
                        updated_position = self.find_position(state, intent.symbol, side=local_side_for_action(intent.action))
                        if updated_position:
                            sync = self.refresh_protective_stop_after_partial_take_profit(updated_position)
                            if sync:
                                result["scale_in_protective_stop_sync"] = sync
                                maintenance.append(sync)
                                append_text_line(self.config.orders_file, json.dumps({"event": "scale_in_protective_stop_sync", "intent": intent.to_dict(), "sync": sync}, ensure_ascii=False))
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                failed = {
                    "status": "failed",
                    "intent": intent.to_dict(),
                    "error": error_text,
                    "submitted_at": utc_now_iso(),
                }
                if pre_submit_cancellation:
                    failed["pre_submit_cancelled_protective_stop"] = pre_submit_cancellation
                if (
                    submit
                    and intent.action in {"open_long", "open_short"}
                    and self.is_percent_price_filter_error(error_text)
                ):
                    retry_result = self.retry_percent_price_order_after_spike(intent, error_text)
                    if retry_result is not None:
                        if retry_result.get("status") == "submitted":
                            results.append(retry_result)
                            append_text_line(self.config.orders_file, json.dumps(retry_result, ensure_ascii=False))
                            self.apply_intent_to_state(state, intent, retry_result)
                            continue
                        expected_side = "short" if intent.action == "open_short" else "long"
                        cooldown = self.set_entry_cooldown(
                            state,
                            intent.symbol,
                            hours=max(0.0, float(self.config.percent_price_retry_cooldown_hours)),
                            reason=str(retry_result.get("status") or "percent_price_retry_skipped"),
                            side=expected_side,
                        )
                        retry_result["entry_cooldown"] = cooldown
                        results.append(retry_result)
                        append_text_line(self.config.orders_file, json.dumps(retry_result, ensure_ascii=False))
                        continue
                if (
                    submit
                    and intent.action in {"close_long", "close_short"}
                    and self.is_percent_price_filter_error(error_text)
                ):
                    retry_result = self.retry_percent_price_order_after_spike(intent, error_text)
                    if retry_result is not None:
                        if retry_result.get("status") == "submitted":
                            results.append(retry_result)
                            append_text_line(self.config.orders_file, json.dumps(retry_result, ensure_ascii=False))
                            self.apply_intent_to_state(state, intent, retry_result)
                            continue
                        try:
                            reconcile_result = self.reconcile_state_with_exchange(state)
                        except Exception as reconcile_exc:
                            retry_result["reconcile"] = {
                                "event": "exchange_position_reconcile_after_percent_price_close_rejected",
                                "status": "failed",
                                "error": f"{type(reconcile_exc).__name__}: {reconcile_exc}",
                                "submitted_at": utc_now_iso(),
                            }
                        else:
                            if reconcile_result:
                                maintenance.append(reconcile_result)
                            retry_result["reconcile"] = reconcile_result or {}
                        retry_position = self.find_position(state, intent.symbol, side=local_side_for_action(intent.action))
                        if retry_position:
                            sync = self.refresh_protective_stop_after_partial_take_profit(retry_position)
                            if sync:
                                retry_result["protective_stop_resync_after_failed_close"] = sync
                                maintenance.append(sync)
                                append_text_line(self.config.orders_file, json.dumps({"event": "protective_stop_resync_after_failed_percent_price_close", "intent": intent.to_dict(), "sync": sync}, ensure_ascii=False))
                        results.append(retry_result)
                        append_text_line(self.config.orders_file, json.dumps(retry_result, ensure_ascii=False))
                        continue
                if (
                    submit
                    and intent.action in {"close_long", "close_short"}
                    and self.is_status_unknown_error(error_text)
                ):
                    try:
                        reconcile_result = self.reconcile_state_with_exchange(state)
                    except Exception as reconcile_exc:
                        failed["reconcile"] = {
                            "event": "exchange_position_reconcile_after_close_status_unknown",
                            "status": "failed",
                            "error": f"{type(reconcile_exc).__name__}: {reconcile_exc}",
                            "submitted_at": utc_now_iso(),
                        }
                    else:
                        if reconcile_result:
                            maintenance.append(reconcile_result)
                        retry_position = self.find_position(state, intent.symbol, side=local_side_for_action(intent.action))
                        retry_block_until = None
                        if retry_position:
                            cooldown_minutes = max(1, int(self.config.close_retry_cooldown_minutes or 5))
                            retry_block_until = (datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)).isoformat()
                            retry_position["close_retry_block_until"] = retry_block_until
                            retry_position["close_retry_block_reason"] = "binance_status_unknown_position_still_open"
                            retry_position["close_retry_last_error"] = error_text
                            retry_position["close_retry_last_at"] = utc_now_iso()
                        failed.update({
                            "status": "status_unknown_position_still_open" if retry_position else "status_unknown_closed_reconciled",
                            "error": "",
                            "original_error": error_text,
                            "reconcile": reconcile_result or {},
                            "retry_block_until": retry_block_until,
                            "telegram_silent": True,
                            "maintenance_only": True,
                            "note": "Binance returned close status unknown; exchange ledger was reconciled before any retry.",
                        })
                elif (
                    intent.action in {"close_long", "close_short"}
                    and "no exchange position found" in str(exc)
                ):
                    try:
                        reconcile_result = self.reconcile_state_with_exchange(state)
                    except Exception as reconcile_exc:
                        failed["reconcile"] = {
                            "event": "exchange_position_reconcile_after_missing_close",
                            "status": "failed",
                            "error": f"{type(reconcile_exc).__name__}: {reconcile_exc}",
                            "submitted_at": utc_now_iso(),
                        }
                    else:
                        if reconcile_result:
                            maintenance.append(reconcile_result)
                        failed.update({
                            "status": "reconciled_missing_exchange_position",
                            "error": "",
                            "original_error": error_text,
                            "reconcile": reconcile_result or {},
                            "telegram_silent": True,
                            "maintenance_only": True,
                            "note": "Exchange position was already absent; local V6 ledger was reconciled instead of retrying the close.",
                        })
                elif (
                    intent.action in {"close_long", "close_short"}
                    and "resolved zero partial close quantity" in str(exc)
                ):
                    retry_wait = None
                    try:
                        reconcile_result = self.reconcile_state_with_exchange(state)
                    except Exception as reconcile_exc:
                        failed["reconcile"] = {
                            "event": "exchange_position_reconcile_after_zero_partial_close",
                            "status": "failed",
                            "error": f"{type(reconcile_exc).__name__}: {reconcile_exc}",
                            "submitted_at": utc_now_iso(),
                        }
                    else:
                        if reconcile_result:
                            maintenance.append(reconcile_result)
                        retry_position = self.find_position(state, intent.symbol, side=local_side_for_action(intent.action))
                        if retry_position and self.is_staged_take_profit_intent(intent):
                            retry_wait = self.mark_partial_take_profit_retry_wait(
                                retry_position,
                                intent,
                                error_text,
                                reason="zero_partial_close_quantity_after_exchange_precision",
                                minutes=60,
                            )
                            if retry_wait:
                                maintenance.append(retry_wait)
                        failed.update({
                            "status": "skipped_zero_partial_close_quantity",
                            "error": "",
                            "original_error": error_text,
                            "reconcile": reconcile_result or {},
                            "retry_wait": retry_wait or {},
                            "telegram_silent": True,
                            "maintenance_only": True,
                            "note": "Partial close quantity rounded to zero after exchange precision rules; exchange ledger was reconciled and the staged close was cooled down or dust-cleared.",
                        })
                elif (
                    submit
                    and intent.action in {"close_long", "close_short"}
                    and self.is_reduce_only_rejected_error(error_text)
                ):
                    retry_wait = None
                    try:
                        reconcile_result = self.reconcile_state_with_exchange(state)
                    except Exception as reconcile_exc:
                        failed["reconcile"] = {
                            "event": "exchange_position_reconcile_after_reduce_only_rejected",
                            "status": "failed",
                            "error": f"{type(reconcile_exc).__name__}: {reconcile_exc}",
                            "submitted_at": utc_now_iso(),
                        }
                    else:
                        if reconcile_result:
                            maintenance.append(reconcile_result)
                        retry_position = self.find_position(state, intent.symbol, side=local_side_for_action(intent.action))
                        close_retry: dict[str, Any] | None = None
                        close_retry_error = ""
                        close_retry_cancellation: dict[str, Any] | None = None
                        if retry_position:
                            close_retry_cancellation = self.cancel_all_protective_stops_for_position(retry_position)
                            try:
                                close_retry = self.client.submit_order(intent)
                            except Exception as retry_exc:
                                close_retry_error = f"{type(retry_exc).__name__}: {retry_exc}"
                            else:
                                close_retry["reduce_only_rejected_retry"] = {
                                    "enabled": True,
                                    "original_error": error_text,
                                    "pre_retry_cancellation": close_retry_cancellation or {},
                                    "retried_at": utc_now_iso(),
                                }
                                results.append(close_retry)
                                append_text_line(self.config.orders_file, json.dumps(close_retry, ensure_ascii=False))
                                if close_retry.get("status") in {"submitted", "submitted_unprotected", "submitted_no_position_delta"}:
                                    self.apply_intent_to_state(state, intent, close_retry)
                                    if self.is_partial_take_profit_intent(intent):
                                        updated_position = self.find_position(state, intent.symbol, side=local_side_for_action(intent.action))
                                        if updated_position:
                                            sync = self.refresh_protective_stop_after_partial_take_profit(updated_position)
                                            if sync:
                                                close_retry["partial_take_profit_protective_stop_sync"] = sync
                                                maintenance.append(sync)
                                                append_text_line(self.config.orders_file, json.dumps({"event": "partial_take_profit_protective_stop_sync_after_reduce_only_retry", "intent": intent.to_dict(), "sync": sync}, ensure_ascii=False))
                                continue
                        if retry_position and self.is_staged_take_profit_intent(intent):
                            retry_wait = self.mark_partial_take_profit_retry_wait(
                                retry_position,
                                intent,
                                error_text,
                                reason="reduce_only_rejected_after_protective_stop_conflict",
                                minutes=30,
                            )
                            if retry_wait:
                                maintenance.append(retry_wait)
                        close_retry_wait = None
                        if retry_position:
                            cooldown_minutes = max(1, int(self.config.close_retry_cooldown_minutes or 5))
                            close_retry_wait = (datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)).isoformat()
                            retry_position["close_retry_block_until"] = close_retry_wait
                            retry_position["close_retry_block_reason"] = "reduce_only_rejected"
                            retry_position["close_retry_last_error"] = close_retry_error or error_text
                            retry_position["close_retry_last_at"] = utc_now_iso()
                        failed.update({
                            "status": "reduce_only_rejected_reconciled",
                            "error": "",
                            "original_error": error_text,
                            "reconcile": reconcile_result or {},
                            "close_retry": {
                                "pre_retry_cancellation": close_retry_cancellation or {},
                                "error": close_retry_error,
                                "retry_block_until": close_retry_wait,
                            } if retry_position else {},
                            "retry_wait": retry_wait or {},
                            "telegram_silent": True,
                            "maintenance_only": True,
                            "note": "Reduce-only close was rejected by Binance; exchange state was reconciled. A one-shot close retry after cancelling same-side protective stops also failed or was not applicable.",
                        })
                elif submit and intent.action in {"open_long", "open_short"}:
                    try:
                        reconcile_result = self.reconcile_state_with_exchange(state)
                    except Exception as reconcile_exc:
                        failed["reconcile"] = {
                            "event": "exchange_position_reconcile_after_open_exception",
                            "status": "failed",
                            "error": f"{type(reconcile_exc).__name__}: {reconcile_exc}",
                            "submitted_at": utc_now_iso(),
                        }
                    else:
                        if reconcile_result:
                            maintenance.append(reconcile_result)
                        expected_side = "short" if intent.action == "open_short" else "long"
                        expected_position = f"{intent.symbol.upper()}:{expected_side}"
                        actual_positions = {
                            str(item).upper()
                            for item in (reconcile_result or {}).get("actual_positions") or []
                        }
                        status_unknown_error = self.is_status_unknown_error(error_text)
                        if status_unknown_error and expected_position.upper() not in actual_positions:
                            self.set_entry_cooldown(
                                state,
                                intent.symbol,
                                hours=0.5,
                                reason="binance_status_unknown_no_position",
                                side=expected_side,
                            )
                            failed.update({
                                "status": "status_unknown_no_position",
                                "reconcile": reconcile_result or {},
                                "original_error": error_text,
                                "error": "",
                                "telegram_silent": True,
                                "maintenance_only": True,
                                "note": "Binance returned status unknown, but exchange reconcile found no matching position; entry cooldown applied.",
                            })
                        else:
                            failed.update({
                                "status": "failed_open_reconciled",
                                "reconcile": reconcile_result or {},
                                "original_error": error_text,
                                "error": error_text,
                                "note": "Open action raised after/around submission; exchange position reconcile was run to prevent orphaned or duplicate local positions.",
                            })
                results.append(failed)
                append_text_line(self.config.orders_file, json.dumps(failed, ensure_ascii=False))
        state["updated_at"] = utc_now_iso()
        atomic_write_json(self.config.state_file, state)
        post_action_reconcile = self.run_ledger_reconcile_after_action(results)
        if post_action_reconcile:
            maintenance.append(post_action_reconcile)
            if post_action_reconcile.get("status") in {"ok", "repaired"}:
                state = self.load_state()
        summary = {
            "generated_at": utc_now_iso(),
            "order_mode": self.config.order_mode,
            "submit": submit,
            "state_file": str(self.config.state_file),
            "signals_file": str(self.config.signals_file),
            "orders_file": str(self.config.orders_file),
            "symbols": self.config.symbols,
            "entry_thresholds": {
                symbol: {
                    "long": self.config.long_entry_threshold_for(symbol),
                    "short": self.config.short_entry_threshold_for(symbol),
                }
                for symbol in self.config.symbols
            },
            "technical_entry_only_symbols": list(self.config.technical_entry_only_symbols),
            "four_h_entry_15m_selector_symbols": list(self.config.four_h_entry_15m_selector_symbols),
            "entry_mode_by_symbol": {
                symbol: (
                    "technical_indicators_only"
                    if self.config.technical_entry_only_for(symbol)
                    else "probability_threshold"
                )
                for symbol in self.config.symbols
            },
            "signals": signal_dicts,
            "intents": [intent.to_dict() for intent in intents],
            "results": results,
            "maintenance": maintenance,
            "positions": state.get("positions", []),
        }
        atomic_write_json(self.config.latest_summary_file, summary)
        compact_wal = {
            "event": "run_once",
            "generated_at": summary["generated_at"],
            "order_mode": summary["order_mode"],
            "submit": summary["submit"],
            "symbols": summary["symbols"],
            "signal_count": len(summary["signals"]),
            "intent_count": len(summary["intents"]),
            "result_count": len(summary["results"]),
            "signals": compact_signal_dicts,
            "intents": [
                {
                    "symbol": item.get("symbol"),
                    "action": item.get("action"),
                    "side": item.get("side"),
                    "quantity": item.get("quantity"),
                    "entry_price": item.get("entry_price"),
                    "stop_loss": item.get("stop_loss"),
                    "take_profit": item.get("take_profit"),
                    "reason": str(item.get("reason") or "")[:500],
                }
                for item in summary["intents"]
                if isinstance(item, dict)
            ],
            "results": [
                {
                    "symbol": item.get("symbol") or (item.get("intent") or {}).get("symbol") if isinstance(item.get("intent"), dict) else item.get("symbol"),
                    "action": (item.get("intent") or {}).get("action") if isinstance(item.get("intent"), dict) else item.get("action"),
                    "status": item.get("status"),
                    "order_id": item.get("order_id") or item.get("client_order_id"),
                    "error": str(item.get("error") or "")[:500],
                }
                for item in summary["results"]
                if isinstance(item, dict)
            ],
            "maintenance": [
                {
                    "event": item.get("event"),
                    "status": item.get("status"),
                }
                for item in summary["maintenance"]
                if isinstance(item, dict)
            ],
            "position_count": len(summary["positions"]),
            "positions": [
                {
                    "symbol": item.get("symbol"),
                    "side": item.get("side"),
                    "position_qty": item.get("position_qty") or item.get("remaining_quantity"),
                    "entry_price": item.get("entry_price"),
                    "current_price": item.get("current_price"),
                    "horizon": item.get("horizon"),
                    "entry_timeframe": item.get("entry_timeframe"),
                }
                for item in summary["positions"]
                if isinstance(item, dict)
            ],
        }
        append_text_line(self.config.wal_file, json.dumps(compact_wal, ensure_ascii=False))
        return summary

    @staticmethod
    def result_should_trigger_ledger_reconcile(result: dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        status = str(result.get("status") or "")
        if status not in {"submitted", "submitted_unprotected", "submitted_no_position_delta", "closed_after_protective_stop_failed"}:
            return False
        intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
        action = str(intent.get("action") or "")
        return action in {"open_long", "open_short", "close_long", "close_short"}

    def run_ledger_reconcile_after_action(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.config.order_mode == "dry_run" or not _env_bool("TS_V6_RECONCILE_AFTER_ACTION", True):
            return None
        triggers: list[dict[str, Any]] = []
        for result in results:
            if not self.result_should_trigger_ledger_reconcile(result):
                continue
            intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
            triggers.append({
                "symbol": intent.get("symbol"),
                "action": intent.get("action"),
                "status": result.get("status"),
                "order_id": self.order_id_from_result(result),
            })
        if not triggers:
            return None

        command = [
            os.getenv("TS_V6_LEDGER_RECONCILE_PYTHON", sys.executable or "python3"),
            os.getenv("TS_V6_LEDGER_RECONCILE_SCRIPT", "scripts/run_v6_ledger_reconcile.py"),
        ]
        timeout_seconds = _env_int("TS_V6_LEDGER_RECONCILE_AFTER_ACTION_TIMEOUT_SECONDS", 45)
        started_at = utc_now_iso()
        child_env = os.environ.copy()
        child_env["TS_V6_EXECUTION_LOCK_HELD"] = "1"
        try:
            completed = subprocess.run(
                command,
                cwd=Path.cwd(),
                env=child_env,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(5, timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "event": "post_action_ledger_reconcile",
                "status": "timeout",
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "timeout_seconds": timeout_seconds,
                "triggers": triggers,
                "error": f"TimeoutExpired: {exc}",
            }
        except Exception as exc:
            return {
                "event": "post_action_ledger_reconcile",
                "status": "failed",
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "triggers": triggers,
                "error": f"{type(exc).__name__}: {exc}",
            }

        status = "ok" if completed.returncode == 0 else "failed"
        return {
            "event": "post_action_ledger_reconcile",
            "status": status,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "returncode": completed.returncode,
            "command": command,
            "triggers": triggers,
            "stdout_tail": (completed.stdout or "")[-2000:],
            "stderr_tail": (completed.stderr or "")[-2000:],
        }

    def build_signal(self, symbol: str) -> V6Signal:
        self.client.reset_market_data_freshness(symbol)
        klines_4h = self.client.fetch_klines(symbol, "4h", self.config.kline_limit_for_interval("4h"))
        klines_4h = self.client.supplement_4h_klines_for_bb2880(symbol, klines_4h, self.config.kline_limit_for_interval("4h"))
        klines_15m = self.client.fetch_klines(symbol, "15m", self.config.kline_limit_for_interval("15m"))
        klines_1m = self.client.fetch_klines(symbol, "1m", self.config.kline_limit_for_interval("1m"))
        klines_1d = self.client.fetch_klines(symbol, "1d", self.config.kline_limit_for_interval("1d"))
        if len(klines_4h) < 120 or len(klines_15m) < 96:
            raise V6ExecutionError(f"insufficient kline history for {symbol}")
        mark_price = self.client.mark_price(symbol)
        if mark_price <= 0:
            mark_price = float(klines_4h[-1][4])
        ticker = self.client.ticker_24h(symbol)
        bb = compute_bb_structure(
            klines_15m,
            klines_4h,
            mark_price,
            threshold_pct=0.3,
            klines_1m=klines_1m,
            klines_1d=klines_1d,
        )
        h4 = compute_h4_structure(symbol, klines_4h, mark_price)
        features = self.features_from_context(symbol, mark_price, ticker, bb, h4, klines_4h, klines_15m)
        quality = self.scorer.quality_score(features)
        fifteen_m = self.fifteen_m_state(mark_price, bb, klines_15m)
        mid_retest = self.mid_retest_state(mark_price, bb)
        one_m = self.one_m_state(mark_price, bb)
        levels = self.levels(mark_price, bb, h4, klines_15m=klines_15m, klines_4h=klines_4h, klines_1d=klines_1d)
        levels["generic_indicator_1m"] = self.generic_indicator_1m_state(mark_price, klines_1m)
        indicators = self.indicator_snapshot(mark_price, bb, klines_15m=klines_15m)
        indicators["recent_extremes"] = {
            "1m": self.recent_kline_extremes(klines_1m, lookback=3),
            "15m_closed": self.recent_kline_extremes(self.closed_kline_rows(klines_15m), lookback=1),
            "1m_confluence": self.recent_kline_extremes(
                klines_1m,
                lookback=max(3, int(self.config.entry_confluence_hold_lookback_1m)),
            ),
        }
        long_prob, short_prob, wait_prob, reasons = self.direction_probabilities(
            mark_price=mark_price,
            quality=quality,
            h4=h4,
            fifteen_m=fifteen_m,
            mid_retest=mid_retest,
            one_m=one_m,
            entry_scan=levels.get("structure_entry_scan"),
            ma20_4h_pressure=levels.get("ma20_4h_pressure"),
            h4_right_side=levels.get("h4_right_side"),
            daily_ma7_trend=levels.get("daily_ma7_trend"),
            bb96_210_second_test_15m=levels.get("bb96_210_second_test_15m"),
        )
        reasons = self.apply_symbol_entry_profile_reasons(symbol, reasons, levels)
        side_bias = "neutral"
        if long_prob >= max(short_prob, wait_prob):
            side_bias = "long"
        if short_prob >= max(long_prob, wait_prob):
            side_bias = "short"
        return V6Signal(
            symbol=symbol,
            computed_at=utc_now_iso(),
            price=round(mark_price, 8),
            side_bias=side_bias,
            long_probability=round(long_prob, 4),
            short_probability=round(short_prob, 4),
            wait_probability=round(wait_prob, 4),
            model_quality_score=round(quality, 4),
            reasons=reasons,
            h4={
                "structure_state": h4.h4_structure_state,
                "price_location": h4.h4_price_location,
                "pattern_type": h4.h4_pattern_type,
                "above_ma20": h4.h4_close_above_ma20,
                "ma20_slope_pct": h4.h4_ma20_slope_pct,
                "price_vs_ma20_pct": h4.h4_price_vs_ma20_pct,
                "close_position_20bars": h4.h4_close_position_20bars,
                "range_low": h4.h4_range_low,
                "recent_swing_low": h4.h4_recent_swing_low,
                "right_side": levels.get("h4_right_side", {}),
            },
            fifteen_m=fifteen_m,
            one_m=one_m,
            mid_retest=mid_retest,
            levels=levels,
            indicators=indicators,
            market_data_freshness=self.client.market_data_freshness(symbol),
        )

    @staticmethod
    def recent_kline_extremes(klines: list[list[Any]], *, lookback: int) -> dict[str, Any]:
        rows = [row for row in klines[-max(1, lookback):] if isinstance(row, list) and len(row) >= 5]
        if not rows:
            return {}
        lows = [float(row[3]) for row in rows]
        highs = [float(row[2]) for row in rows]
        return {
            "lookback": len(rows),
            "low": min(lows),
            "high": max(highs),
            "last_open_time": int(rows[-1][0]),
            "last_close": float(rows[-1][4]),
        }

    def ma20_trend_state(self, price: float, bb: Any, klines_15m: list[list[Any]] | None) -> dict[str, Any]:
        required = max(1, int(self.config.ma20_trend_guard_candles))
        tolerance_pct = max(0.0, float(self.config.ma20_trend_guard_tolerance_pct))
        tolerance = tolerance_pct / 100.0
        current_ma20 = float(getattr(bb, "ma20_15m", getattr(bb, "bb20_mid_15m", 0.0)) or 0.0)
        rows = [row for row in (klines_15m or []) if isinstance(row, list) and len(row) >= 5]
        if current_ma20 <= 0 or len(rows) < 20 + required:
            return {
                "ready": False,
                "timeframe": "15m",
                "period": 20,
                "required_candles": required,
                "checked_candles": 0,
                "ma20": round(current_ma20, 8),
                "reason": "insufficient 15M MA20 history",
            }
        closes = [float(row[4]) for row in rows]
        end_index = len(rows) - 2 if len(rows) >= 20 + required + 1 else len(rows) - 1
        start_index = end_index - required + 1
        if start_index < 19:
            return {
                "ready": False,
                "timeframe": "15m",
                "period": 20,
                "required_candles": required,
                "checked_candles": 0,
                "ma20": round(current_ma20, 8),
                "reason": "insufficient closed 15M candles for MA20 guard",
            }

        checked: list[dict[str, Any]] = []
        for index in range(start_index, end_index + 1):
            row = rows[index]
            ma20 = sum(closes[index - 19:index + 1]) / 20.0
            low = float(row[3])
            high = float(row[2])
            close = float(row[4])
            low_held = low >= ma20 * (1.0 - tolerance)
            high_rejected = high <= ma20 * (1.0 + tolerance)
            close_above = close >= ma20 * (1.0 - tolerance)
            close_below = close <= ma20 * (1.0 + tolerance)
            checked.append({
                "open_time": int(row[0]),
                "ma20": round(ma20, 8),
                "low": round(low, 8),
                "high": round(high, 8),
                "close": round(close, 8),
                "low_held_above_ma20": low_held,
                "high_rejected_below_ma20": high_rejected,
                "close_above_ma20": close_above,
                "close_below_ma20": close_below,
                "low_distance_pct": round(pct_diff(low, ma20), 6),
                "high_distance_pct": round(pct_diff(high, ma20), 6),
                "close_distance_pct": round(pct_diff(close, ma20), 6),
            })

        bullish_hold_count = sum(1 for item in checked if item["low_held_above_ma20"] and item["close_above_ma20"])
        bearish_reject_count = sum(1 for item in checked if item["high_rejected_below_ma20"] and item["close_below_ma20"])
        low_break_count = sum(1 for item in checked if not item["low_held_above_ma20"])
        high_break_count = sum(1 for item in checked if not item["high_rejected_below_ma20"])
        price_vs_ma20 = pct_diff(price, current_ma20)
        current_above = price >= current_ma20 * (1.0 - tolerance)
        current_below = price <= current_ma20 * (1.0 + tolerance)
        bullish_alive = bullish_hold_count >= required and current_above
        bearish_alive = bearish_reject_count >= required and current_below
        bullish_broken = low_break_count > 0 or price < current_ma20 * (1.0 - tolerance)
        bearish_broken = high_break_count > 0 or price > current_ma20 * (1.0 + tolerance)
        if bullish_alive and not bearish_alive:
            state = "bullish_trend_alive"
        elif bearish_alive and not bullish_alive:
            state = "bearish_trend_alive"
        elif bullish_broken or bearish_broken:
            state = "broken_or_mixed"
        else:
            state = "neutral"
        return {
            "ready": True,
            "timeframe": "15m",
            "period": 20,
            "required_candles": required,
            "checked_candles": len(checked),
            "uses_closed_candles": True,
            "tolerance_pct": tolerance_pct,
            "ma20": round(current_ma20, 8),
            "price": round(price, 8),
            "price_distance_pct": round(price_vs_ma20, 6),
            "bullish_hold_count": bullish_hold_count,
            "bearish_reject_count": bearish_reject_count,
            "low_break_count": low_break_count,
            "high_break_count": high_break_count,
            "bullish_trend_alive": bullish_alive,
            "bearish_trend_alive": bearish_alive,
            "bullish_trend_broken": bullish_broken,
            "bearish_trend_broken": bearish_broken,
            "state": state,
            "candles": checked,
        }

    def tick_volume_expansion_state(self, klines_15m: list[list[Any]] | None) -> dict[str, Any]:
        required = max(2, int(self.config.take_profit_volume_guard_required_candles))
        rows = self.closed_kline_rows([
            row for row in (klines_15m or [])
            if isinstance(row, list) and len(row) >= 6
        ])
        if len(rows) < max(5, required + 1):
            return {
                "ready": False,
                "timeframe": "15m",
                "required_candles": required,
                "reason": "insufficient closed 15M volume candles",
            }
        recent = rows[-required:]
        compare_window = rows[-(required + 1):]
        volumes = [float(row[5] or 0.0) for row in compare_window]
        quote_volumes = [float(row[7] or 0.0) if len(row) > 7 else 0.0 for row in compare_window]
        trade_counts = [float(row[8] or 0.0) if len(row) > 8 else 0.0 for row in compare_window]
        volume_ma5 = sum(float(row[5] or 0.0) for row in rows[-5:]) / 5.0
        quote_ma5_values = [float(row[7] or 0.0) if len(row) > 7 else 0.0 for row in rows[-5:]]
        quote_ma5 = sum(quote_ma5_values) / len(quote_ma5_values) if quote_ma5_values else 0.0
        trade_ma5_values = [float(row[8] or 0.0) if len(row) > 8 else 0.0 for row in rows[-5:]]
        trade_ma5 = sum(trade_ma5_values) / len(trade_ma5_values) if trade_ma5_values else 0.0

        def strictly_increasing(values: list[float]) -> bool:
            if len(values) < required + 1:
                return False
            tail = values[-(required + 1):]
            return all(tail[index] > tail[index - 1] for index in range(1, len(tail)))

        volume_increasing = strictly_increasing(volumes)
        quote_increasing = strictly_increasing(quote_volumes) if any(value > 0 for value in quote_volumes) else False
        trades_ready = any(value > 0 for value in trade_counts)
        trades_increasing = strictly_increasing(trade_counts) if trades_ready else False
        latest_volume = volumes[-1]
        latest_quote = quote_volumes[-1] if quote_volumes else 0.0
        latest_trades = trade_counts[-1] if trade_counts else 0.0
        volume_ratio = latest_volume / volume_ma5 if volume_ma5 > 0 else 0.0
        quote_ratio = latest_quote / quote_ma5 if quote_ma5 > 0 else 0.0
        trade_ratio = latest_trades / trade_ma5 if trade_ma5 > 0 else 0.0
        min_ratio = max(0.0, float(self.config.take_profit_volume_guard_min_ma5_ratio))
        amount_expanding = volume_increasing or quote_increasing
        trades_ok = trades_increasing if self.config.take_profit_volume_guard_require_trades else True
        ma_ratio_ok = max(volume_ratio, quote_ratio) >= min_ratio
        single_candle_spike = bool(ma_ratio_ok)
        active = bool((amount_expanding or single_candle_spike) and trades_ok and ma_ratio_ok)
        return {
            "ready": True,
            "timeframe": "15m",
            "required_candles": required,
            "active": active,
            "amount_expanding": amount_expanding,
            "single_candle_spike": single_candle_spike,
            "volume_increasing": volume_increasing,
            "quote_volume_increasing": quote_increasing,
            "trades_ready": trades_ready,
            "trades_increasing": trades_increasing,
            "trades_required": bool(self.config.take_profit_volume_guard_require_trades),
            "latest_volume": round(latest_volume, 8),
            "latest_quote_volume": round(latest_quote, 8),
            "latest_trade_count": round(latest_trades, 8),
            "volume_ma5": round(volume_ma5, 8),
            "quote_volume_ma5": round(quote_ma5, 8),
            "trade_count_ma5": round(trade_ma5, 8),
            "volume_vs_ma5_ratio": round(volume_ratio, 6),
            "quote_volume_vs_ma5_ratio": round(quote_ratio, 6),
            "trade_count_vs_ma5_ratio": round(trade_ratio, 6),
            "min_ma5_ratio": min_ratio,
            "recent": [
                {
                    "open_time": int(row[0]),
                    "volume": round(float(row[5] or 0.0), 8),
                    "quote_volume": round(float(row[7] or 0.0), 8) if len(row) > 7 else 0.0,
                    "trade_count": round(float(row[8] or 0.0), 8) if len(row) > 8 else 0.0,
                    "close": round(float(row[4] or 0.0), 8),
                }
                for row in recent
            ],
        }

    def indicator_snapshot(self, price: float, bb: Any, *, klines_15m: list[list[Any]] | None = None) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for timeframe in ("15m", "4h", "1d"):
            ma_rows: dict[str, Any] = {}
            for period in (7, 20, 60, 96, 105, 120):
                value = float(getattr(bb, f"ma{period}_{timeframe}", 0.0) or 0.0)
                ma_rows[f"ma{period}"] = {
                    "value": round(value, 8),
                    "distance_pct": round(clipped_pct_diff(price, value, self.config.distance_clip_pct), 6) if value > 0 else 0.0,
                    "ready": value > 0,
                }
            upper = float(getattr(bb, f"bb105_upper_{timeframe}", 0.0) or 0.0)
            mid = float(getattr(bb, f"bb105_mid_{timeframe}", 0.0) or 0.0)
            lower = float(getattr(bb, f"bb105_lower_{timeframe}", 0.0) or 0.0)
            width = upper - lower
            rows[timeframe] = {
                "kline_count": int(getattr(bb, f"momo_{timeframe}_bars", 0) or 0),
                "ma": ma_rows,
                "volume_ma5": round(float(getattr(bb, f"volume_ma5_{timeframe}", 0.0) or 0.0), 8),
                "volume_ma10": round(float(getattr(bb, f"volume_ma10_{timeframe}", 0.0) or 0.0), 8),
                "bb105": {
                    "upper": round(upper, 8),
                    "mid": round(mid, 8),
                    "lower": round(lower, 8),
                    "position": round((price - lower) / width, 6) if width > 0 else 0.5,
                    "distance_mid_pct": round(clipped_pct_diff(price, mid, self.config.distance_clip_pct), 6) if mid > 0 else 0.0,
                    "ready": upper > 0 and mid > 0 and lower > 0,
                },
            }
        rows["ma20_trend"] = self.ma20_trend_state(price, bb, klines_15m or [])
        rows["tick_volume_15m"] = self.tick_volume_expansion_state(klines_15m or [])
        return rows

    def features_from_context(
        self,
        symbol: str,
        mark_price: float,
        ticker: dict[str, Any],
        bb: Any,
        h4: Any,
        klines_4h: list[list[Any]],
        klines_15m: list[list[Any]],
    ) -> CoinFeatures:
        closes_4h = [float(k[4]) for k in klines_4h]
        closes_15m = [float(k[4]) for k in klines_15m]
        features = CoinFeatures(symbol=symbol, timestamp=utc_now_iso())
        features.quote_volume_usdt_24h = float(ticker.get("quoteVolume") or 0.0)
        features.price_change_pct_24h = float(ticker.get("priceChangePercent") or 0.0)
        features.price_change_pct_4h = pct_diff(mark_price, closes_4h[-2]) if len(closes_4h) >= 2 else 0.0
        features.price_change_pct_1h = pct_diff(mark_price, closes_15m[-5]) if len(closes_15m) >= 5 else 0.0
        features.price_change_pct_15m = pct_diff(mark_price, closes_15m[-2]) if len(closes_15m) >= 2 else 0.0
        features.price_position_7d = price_position_pct(closes_4h, mark_price, 42)
        features.price_position_14d = price_position_pct(closes_4h, mark_price, 84)
        features.price_position_20d = price_position_pct(closes_4h, mark_price, 120)
        low_7d = min(closes_4h[-42:]) if len(closes_4h) >= 42 else min(closes_4h)
        features.breakout_distance_7d = (mark_price - low_7d) / low_7d if low_7d > 0 else 0.0
        features.compression_score = compression_score(closes_4h, 20)
        features.momentum_score = features.price_change_pct_4h
        features.structure_quality = clamp(float(h4.h4_close_position_20bars) * 100.0, 0.0, 100.0)
        volumes_15m = [float(k[5]) for k in klines_15m if len(k) > 5]
        latest_volume_15m = volumes_15m[-1] if volumes_15m else 0.0
        features.kline_ma20_distance_pct = clipped_pct_diff(
            mark_price,
            float(getattr(bb, "ma20_15m", getattr(bb, "bb20_mid_15m", 0.0)) or 0.0),
            self.config.distance_clip_pct,
        )
        features.kline_volume_vs_ma5 = (
            (latest_volume_15m / float(getattr(bb, "volume_ma5_15m", 0.0) or 0.0)) - 1.0
            if latest_volume_15m > 0 and float(getattr(bb, "volume_ma5_15m", 0.0) or 0.0) > 0
            else 0.0
        )
        features.kline_volume_trend = (
            pct_diff(float(getattr(bb, "volume_ma5_15m", 0.0) or 0.0), float(getattr(bb, "volume_ma10_15m", 0.0) or 0.0))
            if float(getattr(bb, "volume_ma5_15m", 0.0) or 0.0) > 0 and float(getattr(bb, "volume_ma10_15m", 0.0) or 0.0) > 0
            else 0.0
        )
        now = datetime.now(timezone.utc)
        features.hour_of_day = float(now.hour)
        features.day_of_week = float(now.weekday())
        features.is_asian_session = 1.0 if 0 <= now.hour < 8 else 0.0
        features.is_us_session = 1.0 if 13 <= now.hour < 21 else 0.0
        features.bb_price_position_96 = clamp(float(bb.bb_price_position_96), 0.0, 1.0)
        features.bb_price_position_105_4h = clamp(float(bb.bb_price_position_105_4h), 0.0, 1.0)
        features.bb_price_position_672_4h = clamp(float(bb.bb_price_position_672_4h), 0.0, 1.0)
        features.bb_price_position_1440_4h = clamp(float(bb.bb_price_position_1440_4h), 0.0, 1.0)
        features.bb_price_position_105_1m = clamp(float(bb.bb_price_position_105_1m), 0.0, 1.0)
        features.bb_price_position_300_1m = clamp(float(bb.bb_price_position_300_1m), 0.0, 1.0)
        features.bb105_mid_4h_distance_pct = clipped_pct_diff(mark_price, float(bb.bb105_mid_4h), self.config.distance_clip_pct)
        features.bb672_mid_4h_distance_pct = clipped_pct_diff(mark_price, float(bb.bb672_mid_4h), self.config.distance_clip_pct)
        features.bb1440_mid_4h_distance_pct = clipped_pct_diff(mark_price, float(bb.bb1440_mid_4h), self.config.distance_clip_pct)
        features.bb105_upper_1m_distance_pct = clipped_pct_diff(mark_price, float(bb.bb105_upper_1m), self.config.distance_clip_pct)
        features.bb300_upper_1m_distance_pct = clipped_pct_diff(mark_price, float(bb.bb300_upper_1m), self.config.distance_clip_pct)
        features.bb96_upper_15m_distance_pct = clipped_pct_diff(mark_price, float(bb.bb96_upper_15m), self.config.distance_clip_pct)
        features.short_4h_mid_retest_distance_min_pct = min(
            abs(features.bb105_mid_4h_distance_pct),
            abs(features.bb672_mid_4h_distance_pct),
            abs(features.bb1440_mid_4h_distance_pct),
        )
        features.short_1m_upper_pressure_max = max(features.bb_price_position_105_1m, features.bb_price_position_300_1m)
        features.short_15m_bb96_position = features.bb_price_position_96
        return features

    def fifteen_m_state(self, price: float, bb: Any, klines_15m: list[list[Any]] | None = None) -> dict[str, Any]:
        periods: dict[str, tuple[float, float, float]] = {}
        for period in SHARED_BB_PERIODS:
            position_attr = f"bb_price_position_{period}" if period in {96, 120, 210, 300} else f"bb_price_position_{period}_15m"
            periods[f"bb{period}"] = (
                float(getattr(bb, f"bb{period}_mid_15m", 0.0) or 0.0),
                float(getattr(bb, f"bb{period}_lower_15m", 0.0) or 0.0),
                float(getattr(bb, position_attr, 0.5) or 0.5),
            )
        long_cycle_names = {"bb1440", "bb1920", "bb2880", "bb4707"}
        below_mid: list[str] = []
        below_lower: list[str] = []
        near_lower: list[str] = []
        distances: dict[str, float] = {}
        positions: dict[str, float] = {}
        for name, (mid, lower, position) in periods.items():
            if mid > 0:
                distances[f"{name}_mid_pct"] = round(clipped_pct_diff(price, mid, self.config.distance_clip_pct), 6)
                if price < mid:
                    below_mid.append(name)
            if lower > 0:
                lower_distance = pct_diff(price, lower)
                distances[f"{name}_lower_pct"] = round(clipped_pct_diff(price, lower, self.config.distance_clip_pct), 6)
                if price < lower:
                    below_lower.append(name)
                if abs(lower_distance) <= self.config.mid_retest_near_pct:
                    near_lower.append(name)
            positions[name] = round(clamp(float(position), 0.0, 1.0), 6)
        long_cycle_below_mid = [name for name in below_mid if name in long_cycle_names]
        long_cycle_below_lower = [name for name in below_lower if name in long_cycle_names]
        failed = len(long_cycle_below_mid) >= 3 or len(long_cycle_below_lower) >= 1
        return {
            "long_cycle_failed": failed,
            "below_mid_count": len(below_mid),
            "below_lower_count": len(below_lower),
            "long_cycle_below_mid_count": len(long_cycle_below_mid),
            "long_cycle_below_lower_count": len(long_cycle_below_lower),
            "below_mid": below_mid,
            "below_lower": below_lower,
            "near_lower": near_lower,
            "positions": positions,
            "clipped_mid_distance_pct": distances,
            "bb672_lower_active": "bb672" in near_lower,
            "bb672_lower_distance_pct": distances.get("bb672_lower_pct"),
            "ma20_trend": self.ma20_trend_state(price, bb, klines_15m or []),
        }

    def mid_retest_state(self, price: float, bb: Any) -> dict[str, Any]:
        periods = {
            f"bb{period}_4h": getattr(bb, f"bb{period}_mid_4h", 0.0)
            for period in SHARED_BB_PERIODS
        }
        rows: dict[str, dict[str, Any]] = {}
        nearest_name = ""
        nearest_abs = 10**9
        retest_candidates: list[str] = []
        far_below_count = 0
        for name, mid in periods.items():
            mid = float(mid or 0.0)
            if mid <= 0:
                continue
            raw = pct_diff(price, mid)
            clipped = clipped_pct_diff(price, mid, self.config.distance_clip_pct)
            near = abs(raw) <= self.config.mid_retest_near_pct
            below = raw < 0
            far_below = raw < -self.config.far_from_mid_pct
            if far_below:
                far_below_count += 1
            if below and near:
                retest_candidates.append(name)
            if abs(raw) < nearest_abs:
                nearest_abs = abs(raw)
                nearest_name = name
            rows[name] = {
                "mid": round(mid, 8),
                "raw_distance_pct": round(raw, 6),
                "clipped_distance_pct": round(clipped, 6),
                "below_mid": below,
                "above_mid": raw > 0,
                "near_mid": near,
                "far_below_mid": far_below,
            }
        return {
            "nearest_mid": nearest_name,
            "nearest_abs_distance_pct": round(nearest_abs, 6),
            "retest_candidates": retest_candidates,
            "far_below_count": far_below_count,
            "periods": rows,
        }

    def one_m_state(self, price: float, bb: Any) -> dict[str, Any]:
        p105 = clamp(float(bb.bb_price_position_105_1m), 0.0, 1.0)
        p300 = clamp(float(bb.bb_price_position_300_1m), 0.0, 1.0)
        upper_pressure = max(p105, p300)
        return {
            "bb105_position": round(p105, 6),
            "bb300_position": round(p300, 6),
            "upper_pressure": round(upper_pressure, 6),
            "upper_pressure_active": upper_pressure >= 0.85,
        }

    def generic_indicator_enabled_for(self, symbol: str) -> bool:
        return str(symbol or "").upper() in set(self.config.generic_indicator_symbols or ())

    def generic_indicator_1m_state(self, price: float, klines_1m: list[list[Any]] | None) -> dict[str, Any]:
        rows = self.closed_kline_rows(klines_1m or [])
        periods = (105, 300)
        required = max(periods) + 2
        if price <= 0 or len(rows) < required:
            return {
                "ready": False,
                "label": self.config.generic_indicator_label,
                "timeframe": "1m",
                "periods": list(periods),
                "reason": "insufficient 1M candles for MA105/MA300 generic indicator",
            }
        closes = [float(row[4]) for row in rows]
        lookback = max(1, int(self.config.generic_indicator_1m_cross_lookback))
        current_index = len(rows) - 1
        trigger_start = max(max(periods), len(rows) - lookback)
        ma_rows: dict[str, dict[str, Any]] = {}
        long_triggers: list[dict[str, Any]] = []
        short_triggers: list[dict[str, Any]] = []
        for period in periods:
            current_ma = self.sma_at(closes, current_index, period)
            previous_ma = self.sma_at(closes, current_index - 1, period)
            if current_ma <= 0 or previous_ma <= 0:
                continue
            row_state = {
                "period": period,
                "ma": round(current_ma, 8),
                "price_distance_pct": round(pct_diff(price, current_ma), 6),
                "price_above": price > current_ma,
                "price_below": price < current_ma,
            }
            for index in range(trigger_start, len(rows)):
                ma_now = self.sma_at(closes, index, period)
                ma_prev = self.sma_at(closes, index - 1, period)
                if ma_now <= 0 or ma_prev <= 0:
                    continue
                prev_close = float(rows[index - 1][4])
                close = float(rows[index][4])
                if prev_close <= ma_prev and close > ma_now and price >= current_ma:
                    long_triggers.append({
                        "name": f"ma{period}_1m_reclaim",
                        "period": period,
                        "timeframe": "1m",
                        "ma": round(ma_now, 8),
                        "close": round(close, 8),
                        "open_time": int(rows[index][0]),
                        "distance_pct": round(pct_diff(close, ma_now), 6),
                    })
                if prev_close >= ma_prev and close < ma_now and price <= current_ma:
                    short_triggers.append({
                        "name": f"ma{period}_1m_break",
                        "period": period,
                        "timeframe": "1m",
                        "ma": round(ma_now, 8),
                        "close": round(close, 8),
                        "open_time": int(rows[index][0]),
                        "distance_pct": round(pct_diff(close, ma_now), 6),
                    })
            ma_rows[str(period)] = row_state
        long_confirmed = bool(long_triggers)
        short_confirmed = bool(short_triggers)
        return {
            "ready": True,
            "label": self.config.generic_indicator_label,
            "timeframe": "1m",
            "periods": list(periods),
            "cross_lookback": lookback,
            "price": round(price, 8),
            "ma": ma_rows,
            "long_confirmed": long_confirmed,
            "short_confirmed": short_confirmed,
            "confirmed_side": "long" if long_confirmed and not short_confirmed else "short" if short_confirmed and not long_confirmed else "conflict" if long_confirmed and short_confirmed else "",
            "long_triggers": long_triggers[:4],
            "short_triggers": short_triggers[:4],
        }

    def generic_indicator_entry_context(
        self,
        action: str,
        levels: dict[str, Any],
        *,
        symbol: str,
        selected: dict[str, Any],
    ) -> dict[str, Any]:
        enabled = self.generic_indicator_enabled_for(symbol)
        label = str(self.config.generic_indicator_label or "\u901a\u7528\u6307\u6807")
        state = levels.get("generic_indicator_1m") if isinstance(levels.get("generic_indicator_1m"), dict) else {}
        timeframe = str(selected.get("timeframe") or "")
        applicable = bool(enabled and action in {"open_short", "open_long"} and timeframe == "15m")
        if not enabled:
            return {"enabled": False, "label": label, "symbol": str(symbol or "").upper()}
        if not applicable:
            return {
                "enabled": True,
                "applicable": False,
                "label": label,
                "symbol": str(symbol or "").upper(),
                "reason": "generic indicator only confirms 15M entries",
                "state": state,
            }
        side = "short" if action == "open_short" else "long"
        confirmed = bool(state.get("short_confirmed") if side == "short" else state.get("long_confirmed"))
        trigger_key = "short_triggers" if side == "short" else "long_triggers"
        return {
            "enabled": True,
            "applicable": True,
            "label": label,
            "symbol": str(symbol or "").upper(),
            "side": side,
            "required_trigger": "1M MA105/MA300 break" if side == "short" else "1M MA105/MA300 reclaim",
            "confirmed": confirmed,
            "trigger": dict((state.get(trigger_key) or [{}])[0]) if state.get(trigger_key) else {},
            "state": state,
        }

    def apply_symbol_entry_profile_reasons(self, symbol: str, reasons: list[str], levels: dict[str, Any]) -> list[str]:
        profile = symbol_bb_entry_profile(symbol)
        if not profile:
            return reasons
        filtered = list(reasons)
        if profile.get("suppress_unprofiled_structure_reasons"):
            filtered = [
                reason for reason in filtered
                if not str(reason).startswith("BB structure long entry candidates near price:")
                and not str(reason).startswith("BB structure short entry candidates near price:")
            ]
        if profile.get("suppress_15m_profile_reasons"):
            filtered = [
                reason for reason in filtered
                if not str(reason).startswith("15M MA20")
                and not str(reason).startswith("15M BB96/BB210")
                and not str(reason).startswith("15M BB672")
                and not str(reason).startswith("15M long-cycle")
            ]
        selected_rows: list[str] = []
        blocked_15m = False
        for action in ("open_short", "open_long"):
            context = self.entry_context_for_action(action, levels, symbol=symbol)
            selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
            if selected:
                selected_rows.append(
                    f"{action}:{selected.get('name') or '--'} {context.get('horizon') or '--'}"
                )
            profile_filter = context.get("symbol_bb_entry_profile") if isinstance(context.get("symbol_bb_entry_profile"), dict) else {}
            blocked_candidates = profile_filter.get("blocked_candidates") if isinstance(profile_filter.get("blocked_candidates"), list) else []
            if any(str(item.get("timeframe") or "") == "15m" for item in blocked_candidates if isinstance(item, dict)):
                blocked_15m = True
            if profile_filter.get("synthetic_second_test_candidate_blocked"):
                blocked_15m = True
        profile_name = str(profile.get("name") or "")
        if selected_rows:
            message = f"symbol entry profile active: {profile_name}; selected {'; '.join(selected_rows)}"
        elif blocked_15m:
            message = f"symbol entry profile active: {profile_name}; 15M scalp candidates ignored, waiting for allowed 4H setup"
        else:
            message = f"symbol entry profile active: {profile_name}; waiting for allowed 4H setup"
        if message not in filtered:
            filtered.append(message)
        return filtered

    def direction_probabilities(
        self,
        *,
        mark_price: float,
        quality: float,
        h4: Any,
        fifteen_m: dict[str, Any],
        mid_retest: dict[str, Any],
        one_m: dict[str, Any],
        entry_scan: dict[str, Any] | None = None,
        ma20_4h_pressure: dict[str, Any] | None = None,
        h4_right_side: dict[str, Any] | None = None,
        daily_ma7_trend: dict[str, Any] | None = None,
        bb96_210_second_test_15m: dict[str, Any] | None = None,
    ) -> tuple[float, float, float, list[str]]:
        long_score = 0.25
        short_score = 0.25
        wait_score = 0.25
        reasons: list[str] = []
        if h4.h4_close_above_ma20 and h4.h4_ma20_slope_pct > 0:
            long_score += 0.24
            reasons.append("4H above MA20 with positive MA20 slope")
        if (not h4.h4_close_above_ma20) and h4.h4_ma20_slope_pct < 0:
            short_score += 0.28
            reasons.append("4H below MA20 with negative MA20 slope")
        if str(h4.h4_structure_state) == "broken":
            short_score += 0.14
            reasons.append("4H structure is broken")
        if float(h4.h4_close_position_20bars) <= 0.25:
            wait_score += 0.10
            reasons.append("price is already near the lower side of the 4H 20-bar range")
        ma20_trend = fifteen_m.get("ma20_trend") if isinstance(fifteen_m.get("ma20_trend"), dict) else {}
        if ma20_trend.get("bullish_trend_alive"):
            long_score += 0.08
            short_score -= 0.06
            reasons.append("15M MA20 held for 2 candles; short-term uptrend remains active")
        elif ma20_trend.get("bearish_trend_alive"):
            short_score += 0.08
            long_score -= 0.06
            reasons.append("15M MA20 rejected for 2 candles; short-term downtrend remains active")
        elif ma20_trend.get("bullish_trend_broken") or ma20_trend.get("bearish_trend_broken"):
            wait_score += 0.03
            reasons.append("15M MA20 trend guard is broken or mixed")
        ma20_4h_pressure = ma20_4h_pressure or {}
        h4_right_side = h4_right_side or {}
        daily_ma7_trend = daily_ma7_trend or {}
        if daily_ma7_trend.get("bearish_stack"):
            short_score += 0.10
            long_score -= 0.04
            reasons.append("1D MA20 is above MA7; daily trend remains bearish until MA7 is reclaimed")
        if daily_ma7_trend.get("ma7_reclaimed"):
            wait_score += 0.04
            reasons.append("1D MA7 is reclaimed; first daily bearish checkpoint is under test")
        if h4_right_side.get("long_confirmed"):
            long_score += 0.12
            reasons.append("4H right-side long confirmation is active")
        if h4_right_side.get("short_confirmed"):
            short_score += 0.12
            reasons.append("4H right-side short confirmation is active")
        if daily_ma7_trend.get("long_confirmed"):
            long_score += 0.16
            reasons.append("1D MA7 breakout-retest-rebreak trend long is confirmed")
        if daily_ma7_trend.get("short_confirmed"):
            short_score += 0.16
            reasons.append("1D MA7 breakdown-retest-rebreak trend short is confirmed")
        bb96_210_second_test_15m = bb96_210_second_test_15m or {}
        upper_second_test = bb96_210_second_test_15m.get("upper") if isinstance(bb96_210_second_test_15m.get("upper"), dict) else {}
        lower_second_test = bb96_210_second_test_15m.get("lower") if isinstance(bb96_210_second_test_15m.get("lower"), dict) else {}
        if upper_second_test.get("confirmed"):
            short_score += 0.12
            reasons.append("15M BB96/BB210 upper overlap second-test rejection is active")
        if lower_second_test.get("confirmed"):
            long_score += 0.12
            reasons.append("15M BB96/BB210 lower overlap second-test rebound is active")
        if upper_second_test.get("confirmed") and lower_second_test.get("confirmed"):
            wait_score += 0.08
            reasons.append("15M BB96/BB210 second-test zones conflict; keep wait weight")
        if ma20_4h_pressure.get("pressure_active"):
            short_score += 0.08
            reasons.append("4H MA20 pressure is near price")
        elif ma20_4h_pressure.get("support_active"):
            long_score += 0.05
            reasons.append("4H MA20 support is near price")
        if fifteen_m.get("long_cycle_failed"):
            short_score += 0.14
            long_score -= 0.05
            reasons.append("15M long-cycle structure is failed; use 4H as primary frame")
        if fifteen_m.get("bb672_lower_active") and str(h4.h4_structure_state) == "broken":
            short_score += 0.10
            wait_score -= 0.02
            distance = fifteen_m.get("bb672_lower_distance_pct")
            if distance is None:
                reasons.append("15M BB672 lower band is active near price")
            else:
                reasons.append(f"15M BB672 lower band is active near price ({distance:+.2f}%)")
        if mid_retest.get("retest_candidates"):
            short_score += 0.18
            wait_score -= 0.04
            reasons.append(f"4H mid retest candidate: {','.join(mid_retest['retest_candidates'])}")
        if int(mid_retest.get("far_below_count") or 0) >= 2:
            wait_score += 0.18
            short_score -= 0.06
            reasons.append("price is far below multiple 4H mids; avoid chasing")
        if one_m.get("upper_pressure_active"):
            short_score += 0.10
            reasons.append("1M upper pressure is active")
        entry_scan = entry_scan or {}
        long_candidates = list(entry_scan.get("long_candidates") or [])
        short_candidates = list(entry_scan.get("short_candidates") or [])
        long_candidate_count = int(entry_scan.get("long_candidate_count") or len(long_candidates))
        short_candidate_count = int(entry_scan.get("short_candidate_count") or len(short_candidates))
        if long_candidate_count:
            long_boost = min(0.12, 0.04 + 0.02 * min(long_candidate_count, 4))
            long_score += long_boost
            names = ",".join(str(item.get("name")) for item in long_candidates[:3] if item.get("name"))
            reasons.append(f"BB structure long entry candidates near price: {names or long_candidate_count}")
        if short_candidate_count:
            short_boost = min(0.12, 0.04 + 0.02 * min(short_candidate_count, 4))
            short_score += short_boost
            names = ",".join(str(item.get("name")) for item in short_candidates[:3] if item.get("name"))
            reasons.append(f"BB structure short entry candidates near price: {names or short_candidate_count}")
        if long_candidate_count and short_candidate_count:
            wait_score += 0.06
            reasons.append("BB structure entry scan is mixed; keep conflict wait weight")
        if quality < 0.35:
            wait_score += 0.12
            reasons.append("V6 model quality score is low; increase wait weight")
        elif quality > 0.65:
            if short_score > long_score:
                short_score += 0.08
            elif long_score > short_score:
                long_score += 0.08
            reasons.append("V6 model quality score supports following the structure")
        long_score = max(0.0, long_score)
        short_score = max(0.0, short_score)
        wait_score = max(0.05, wait_score)
        total = long_score + short_score + wait_score
        return long_score / total, short_score / total, wait_score / total, reasons

    def shared_bb_level_rows(
        self,
        bb: Any,
        *,
        timeframes: tuple[str, ...] = ("15m", "4h", "1d"),
        bands: tuple[str, ...] = ("lower", "mid", "upper"),
    ) -> list[tuple[str, float]]:
        rows: list[tuple[str, float]] = []
        for timeframe in timeframes:
            for period in SHARED_BB_PERIODS:
                for band in bands:
                    name = f"bb{period}_{band}_{timeframe}"
                    value = float(getattr(bb, name, 0.0) or 0.0)
                    if value > 0:
                        rows.append((name, value))
        return rows

    def shared_ma_level_rows(self, bb: Any) -> list[tuple[str, float]]:
        rows: list[tuple[str, float]] = []
        for timeframe, periods in (
            ("15m", (7, 20, 60, 96, 105, 120)),
            ("4h", (7, 20, 60, 96, 105, 120)),
            ("1d", (7, 20)),
        ):
            for period in periods:
                name = f"ma{period}_{timeframe}"
                value = float(getattr(bb, name, 0.0) or 0.0)
                if value > 0:
                    rows.append((name, value))
        return rows

    def ma20_4h_pressure_state(self, price: float, bb: Any) -> dict[str, Any]:
        value = float(getattr(bb, "ma20_4h", 0.0) or 0.0)
        if price <= 0 or value <= 0:
            return {"ready": False, "name": "ma20_4h", "value": round(value, 8)}
        distance = pct_diff(price, value)
        near_pct = float(self.config.ma20_4h_pressure_near_pct)
        return {
            "ready": True,
            "name": "ma20_4h",
            "timeframe": "4h",
            "period": 20,
            "value": round(value, 8),
            "distance_pct": round(distance, 6),
            "abs_distance_pct": round(abs(distance), 6),
            "near_pct": near_pct,
            "pressure_active": value > price and abs(distance) <= near_pct,
            "support_active": value < price and abs(distance) <= near_pct,
            "role": "pressure" if value > price else "support" if value < price else "at_price",
        }

    @staticmethod
    def sma_at(closes: list[float], index: int, period: int) -> float:
        if period <= 0 or index < period - 1:
            return 0.0
        window = closes[index - period + 1:index + 1]
        if len(window) < period:
            return 0.0
        return sum(window) / period

    @staticmethod
    def closed_kline_rows(klines: list[list[Any]]) -> list[list[Any]]:
        rows = [row for row in klines if isinstance(row, list) and len(row) >= 5]
        if len(rows) > 1:
            return rows[:-1]
        return rows

    def h4_right_side_state(self, price: float, bb: Any, klines_4h: list[list[Any]]) -> dict[str, Any]:
        rows = self.closed_kline_rows(klines_4h)
        lookback = max(3, int(self.config.h4_right_side_lookback))
        tolerance_pct = max(0.0, float(self.config.h4_right_side_retest_tolerance_pct))
        if len(rows) < 24:
            return {"ready": False, "timeframe": "4h", "reason": "insufficient 4H candles"}
        closes = [float(row[4]) for row in rows]
        ma20 = float(self.sma_at(closes, len(closes) - 1, 20) or getattr(bb, "ma20_4h", 0.0) or 0.0)
        if ma20 <= 0:
            return {"ready": False, "timeframe": "4h", "reason": "missing 4H MA20"}
        tolerance = tolerance_pct / 100.0
        recent = rows[-lookback:]
        cross_up = False
        cross_down = False
        retest_hold = None
        retest_reject = None
        for i in range(max(20, len(rows) - lookback - 8), len(rows)):
            ma_now = self.sma_at(closes, i, 20)
            ma_prev = self.sma_at(closes, i - 1, 20)
            if ma_now <= 0 or ma_prev <= 0:
                continue
            close_now = float(rows[i][4])
            close_prev = float(rows[i - 1][4])
            if close_prev <= ma_prev and close_now > ma_now:
                cross_up = True
            if close_prev >= ma_prev and close_now < ma_now:
                cross_down = True
        recent_start = len(rows) - len(recent)
        for offset, row in enumerate(recent):
            idx = recent_start + offset
            ma = self.sma_at(closes, idx, 20)
            if ma <= 0:
                continue
            high = float(row[2])
            low = float(row[3])
            close = float(row[4])
            if low <= ma * (1.0 + tolerance) and close >= ma * (1.0 - tolerance):
                retest_hold = {
                    "open_time": int(row[0]),
                    "ma20": round(ma, 8),
                    "low": round(low, 8),
                    "close": round(close, 8),
                    "low_distance_pct": round(pct_diff(low, ma), 6),
                }
            if high >= ma * (1.0 - tolerance) and close <= ma * (1.0 + tolerance):
                retest_reject = {
                    "open_time": int(row[0]),
                    "ma20": round(ma, 8),
                    "high": round(high, 8),
                    "close": round(close, 8),
                    "high_distance_pct": round(pct_diff(high, ma), 6),
                }
        last_close = float(rows[-1][4])
        long_confirmed = bool(cross_up and retest_hold and price > ma20 and last_close >= ma20 * (1.0 - tolerance))
        short_confirmed = bool(cross_down and retest_reject and price < ma20 and last_close <= ma20 * (1.0 + tolerance))
        weak_required = max(1, int(self.config.logic_weakening_exit_required_4h_candles))
        weak_tolerance_pct = max(0.0, float(self.config.logic_weakening_exit_tolerance_pct))
        weak_tolerance = weak_tolerance_pct / 100.0
        weak_recent: list[dict[str, Any]] = []
        for idx in range(max(19, len(rows) - max(weak_required, 2)), len(rows)):
            ma = self.sma_at(closes, idx, 20)
            if ma <= 0:
                continue
            row = rows[idx]
            close = float(row[4])
            weak_recent.append({
                "open_time": int(row[0]),
                "ma20": round(ma, 8),
                "close": round(close, 8),
                "close_distance_pct": round(pct_diff(close, ma), 6),
                "long_logic_weakened": close < ma * (1.0 - weak_tolerance),
                "short_logic_weakened": close > ma * (1.0 + weak_tolerance),
            })
        weak_tail = weak_recent[-weak_required:] if len(weak_recent) >= weak_required else []
        long_logic_weakened = bool(len(weak_tail) == weak_required and all(item["long_logic_weakened"] for item in weak_tail))
        short_logic_weakened = bool(len(weak_tail) == weak_required and all(item["short_logic_weakened"] for item in weak_tail))
        return {
            "ready": True,
            "timeframe": "4h",
            "period": 20,
            "lookback": lookback,
            "ma20": round(ma20, 8),
            "price": round(price, 8),
            "price_distance_pct": round(pct_diff(price, ma20), 6),
            "cross_up_seen": cross_up,
            "cross_down_seen": cross_down,
            "retest_hold": retest_hold,
            "retest_reject": retest_reject,
            "long_confirmed": long_confirmed,
            "short_confirmed": short_confirmed,
            "state": "long_right_side" if long_confirmed else "short_right_side" if short_confirmed else "not_confirmed",
            "right_side_only": bool(self.config.trend_big_right_side_only),
            "logic_weakening_exit": {
                "enabled": bool(self.config.logic_weakening_exit_enabled),
                "timeframe": "4h",
                "period": 20,
                "required_closed_candles": weak_required,
                "checked_closed_candles": len(weak_tail),
                "tolerance_pct": weak_tolerance_pct,
                "recent_closed_candles": weak_recent,
                "long_logic_weakened": long_logic_weakened,
                "short_logic_weakened": short_logic_weakened,
                "long_weakened_count": sum(1 for item in weak_tail if item["long_logic_weakened"]),
                "short_weakened_count": sum(1 for item in weak_tail if item["short_logic_weakened"]),
            },
        }

    def daily_ma7_trend_state(self, price: float, bb: Any, klines_1d: list[list[Any]]) -> dict[str, Any]:
        if not self.config.daily_trend_big_enabled:
            return {"ready": False, "timeframe": "1d", "enabled": False}
        rows = self.closed_kline_rows(klines_1d)
        lookback = max(5, int(self.config.daily_trend_lookback))
        tolerance_pct = max(0.0, float(self.config.daily_ma_retest_tolerance_pct))
        if len(rows) < 24:
            return {"ready": False, "timeframe": "1d", "reason": "insufficient 1D candles"}
        closes = [float(row[4]) for row in rows]
        current_ma7 = float(self.sma_at(closes, len(closes) - 1, 7) or getattr(bb, "ma7_1d", 0.0) or 0.0)
        current_ma20 = float(self.sma_at(closes, len(closes) - 1, 20) or getattr(bb, "ma20_1d", 0.0) or 0.0)
        if current_ma7 <= 0 or current_ma20 <= 0:
            return {"ready": False, "timeframe": "1d", "reason": "missing daily MA7/MA20"}
        bearish_stack = current_ma20 > current_ma7
        bullish_stack = current_ma7 >= current_ma20
        ma7_reclaimed = price > current_ma7
        ma20_reclaimed = price > current_ma20
        tolerance = tolerance_pct / 100.0
        start = max(20, len(rows) - lookback)
        long_breakout = None
        short_breakdown = None
        long_retest = None
        short_retest = None
        long_breakout_index = -1
        short_breakdown_index = -1
        for i in range(start, len(rows)):
            ma7 = self.sma_at(closes, i, 7)
            ma20 = self.sma_at(closes, i, 20)
            ma7_prev = self.sma_at(closes, i - 1, 7)
            ma20_prev = self.sma_at(closes, i - 1, 20)
            if ma7 <= 0 or ma20 <= 0 or ma7_prev <= 0 or ma20_prev <= 0:
                continue
            close = float(rows[i][4])
            prev_close = float(rows[i - 1][4])
            high = float(rows[i][2])
            low = float(rows[i][3])
            if prev_close <= ma7_prev and close > ma7 and ma7 >= ma20:
                long_breakout = {"open_time": int(rows[i][0]), "ma7": round(ma7, 8), "ma20": round(ma20, 8), "close": round(close, 8)}
                long_breakout_index = i
            if prev_close >= ma7_prev and close < ma7 and ma7 <= ma20:
                short_breakdown = {"open_time": int(rows[i][0]), "ma7": round(ma7, 8), "ma20": round(ma20, 8), "close": round(close, 8)}
                short_breakdown_index = i
            if long_breakout and i > long_breakout_index:
                zone_high = max(ma7, ma20)
                zone_low = min(ma7, ma20)
                if low <= zone_high * (1.0 + tolerance) and low >= zone_low * (1.0 - tolerance) and close >= zone_low * (1.0 - tolerance):
                    long_retest = {"open_time": int(rows[i][0]), "ma7": round(ma7, 8), "ma20": round(ma20, 8), "low": round(low, 8), "close": round(close, 8)}
            if short_breakdown and i > short_breakdown_index:
                zone_high = max(ma7, ma20)
                zone_low = min(ma7, ma20)
                if high >= zone_low * (1.0 - tolerance) and high <= zone_high * (1.0 + tolerance) and close <= zone_high * (1.0 + tolerance):
                    short_retest = {"open_time": int(rows[i][0]), "ma7": round(ma7, 8), "ma20": round(ma20, 8), "high": round(high, 8), "close": round(close, 8)}
        long_confirmed = bool(long_breakout and long_retest and price > current_ma7 and current_ma7 >= current_ma20)
        short_confirmed = bool(short_breakdown and short_retest and price < current_ma7 and current_ma7 <= current_ma20)
        if long_confirmed:
            entry_level = {"name": "ma7_1d", "price": round(current_ma7, 8), "distance_pct": round(pct_diff(price, current_ma7), 6)}
        elif short_confirmed:
            entry_level = {"name": "ma7_1d", "price": round(current_ma7, 8), "distance_pct": round(pct_diff(price, current_ma7), 6)}
        else:
            entry_level = {}
        reclaim_required = max(1, int(self.config.daily_ma7_short_reversal_required_1d_candles))
        reclaim_tolerance_pct = max(0.0, float(self.config.daily_ma7_short_reversal_tolerance_pct))
        reclaim_tolerance = reclaim_tolerance_pct / 100.0
        reclaim_recent: list[dict[str, Any]] = []
        for idx in range(max(6, len(rows) - reclaim_required), len(rows)):
            ma7 = self.sma_at(closes, idx, 7)
            if ma7 <= 0:
                continue
            row = rows[idx]
            close = float(row[4])
            reclaim_recent.append({
                "open_time": int(row[0]),
                "ma7": round(ma7, 8),
                "close": round(close, 8),
                "close_distance_pct": round(pct_diff(close, ma7), 6),
                "close_above_ma7": close >= ma7 * (1.0 + reclaim_tolerance),
            })
        reclaim_tail = reclaim_recent[-reclaim_required:] if len(reclaim_recent) >= reclaim_required else []
        ma7_reclaim_held = bool(
            len(reclaim_tail) == reclaim_required
            and all(item["close_above_ma7"] for item in reclaim_tail)
            and price >= current_ma7 * (1.0 + reclaim_tolerance)
        )
        ma20_reclaim_recent: list[dict[str, Any]] = []
        ma7_loss_recent: list[dict[str, Any]] = []
        ma20_loss_recent: list[dict[str, Any]] = []
        for idx in range(max(19, len(rows) - reclaim_required), len(rows)):
            ma7 = self.sma_at(closes, idx, 7)
            ma20 = self.sma_at(closes, idx, 20)
            if ma7 <= 0 or ma20 <= 0:
                continue
            row = rows[idx]
            close = float(row[4])
            ma20_reclaim_recent.append({
                "open_time": int(row[0]),
                "ma20": round(ma20, 8),
                "close": round(close, 8),
                "close_distance_pct": round(pct_diff(close, ma20), 6),
                "close_above_ma20": close >= ma20 * (1.0 + reclaim_tolerance),
            })
            ma7_loss_recent.append({
                "open_time": int(row[0]),
                "ma7": round(ma7, 8),
                "close": round(close, 8),
                "close_distance_pct": round(pct_diff(close, ma7), 6),
                "close_below_ma7": close <= ma7 * (1.0 - reclaim_tolerance),
            })
            ma20_loss_recent.append({
                "open_time": int(row[0]),
                "ma20": round(ma20, 8),
                "close": round(close, 8),
                "close_distance_pct": round(pct_diff(close, ma20), 6),
                "close_below_ma20": close <= ma20 * (1.0 - reclaim_tolerance),
            })
        ma20_reclaim_tail = ma20_reclaim_recent[-reclaim_required:] if len(ma20_reclaim_recent) >= reclaim_required else []
        ma7_loss_tail = ma7_loss_recent[-reclaim_required:] if len(ma7_loss_recent) >= reclaim_required else []
        ma20_loss_tail = ma20_loss_recent[-reclaim_required:] if len(ma20_loss_recent) >= reclaim_required else []
        ma20_reclaim_held = bool(
            len(ma20_reclaim_tail) == reclaim_required
            and all(item["close_above_ma20"] for item in ma20_reclaim_tail)
            and price >= current_ma20 * (1.0 + reclaim_tolerance)
        )
        ma7_loss_held = bool(
            len(ma7_loss_tail) == reclaim_required
            and all(item["close_below_ma7"] for item in ma7_loss_tail)
            and price <= current_ma7 * (1.0 - reclaim_tolerance)
        )
        ma20_loss_held = bool(
            len(ma20_loss_tail) == reclaim_required
            and all(item["close_below_ma20"] for item in ma20_loss_tail)
            and price <= current_ma20 * (1.0 - reclaim_tolerance)
        )
        return {
            "ready": True,
            "timeframe": "1d",
            "ma7": round(current_ma7, 8),
            "ma20": round(current_ma20, 8),
            "bearish_stack": bearish_stack,
            "bullish_stack": bullish_stack,
            "first_checkpoint": "ma7_1d" if bearish_stack else "",
            "ma7_reclaimed": ma7_reclaimed,
            "ma7_reclaim_held": ma7_reclaim_held,
            "ma20_reclaimed": ma20_reclaimed,
            "ma20_reclaim_held": ma20_reclaim_held,
            "ma7_lost": price < current_ma7,
            "ma7_loss_held": ma7_loss_held,
            "ma20_lost": price < current_ma20,
            "ma20_loss_held": ma20_loss_held,
            "short_reversal_on_ma7_reclaim": bool(bearish_stack and ma7_reclaim_held),
            "short_reversal_on_ma20_reclaim": bool(bearish_stack and ma7_reclaim_held and ma20_reclaim_held),
            "long_reversal_on_ma7_loss": bool(bullish_stack and ma7_loss_held),
            "long_reversal_on_ma20_loss": bool(bullish_stack and ma7_loss_held and ma20_loss_held),
            "ma7_reclaim_hold": {
                "enabled": bool(self.config.daily_ma7_short_reversal_exit_enabled),
                "timeframe": "1d",
                "period": 7,
                "required_closed_candles": reclaim_required,
                "checked_closed_candles": len(reclaim_tail),
                "tolerance_pct": reclaim_tolerance_pct,
                "held": ma7_reclaim_held,
                "held_count": sum(1 for item in reclaim_tail if item["close_above_ma7"]),
                "current_price_above_ma7": price >= current_ma7 * (1.0 + reclaim_tolerance),
                "recent_closed_candles": reclaim_recent,
            },
            "ma20_reclaim_hold": {
                "enabled": bool(self.config.daily_ma7_short_reversal_exit_enabled),
                "timeframe": "1d",
                "period": 20,
                "required_closed_candles": reclaim_required,
                "checked_closed_candles": len(ma20_reclaim_tail),
                "tolerance_pct": reclaim_tolerance_pct,
                "held": ma20_reclaim_held,
                "held_count": sum(1 for item in ma20_reclaim_tail if item["close_above_ma20"]),
                "current_price_above_ma20": price >= current_ma20 * (1.0 + reclaim_tolerance),
                "recent_closed_candles": ma20_reclaim_recent,
            },
            "ma7_loss_hold": {
                "enabled": bool(self.config.daily_ma7_short_reversal_exit_enabled),
                "timeframe": "1d",
                "period": 7,
                "required_closed_candles": reclaim_required,
                "checked_closed_candles": len(ma7_loss_tail),
                "tolerance_pct": reclaim_tolerance_pct,
                "held": ma7_loss_held,
                "held_count": sum(1 for item in ma7_loss_tail if item["close_below_ma7"]),
                "current_price_below_ma7": price <= current_ma7 * (1.0 - reclaim_tolerance),
                "recent_closed_candles": ma7_loss_recent,
            },
            "ma20_loss_hold": {
                "enabled": bool(self.config.daily_ma7_short_reversal_exit_enabled),
                "timeframe": "1d",
                "period": 20,
                "required_closed_candles": reclaim_required,
                "checked_closed_candles": len(ma20_loss_tail),
                "tolerance_pct": reclaim_tolerance_pct,
                "held": ma20_loss_held,
                "held_count": sum(1 for item in ma20_loss_tail if item["close_below_ma20"]),
                "current_price_below_ma20": price <= current_ma20 * (1.0 - reclaim_tolerance),
                "recent_closed_candles": ma20_loss_recent,
            },
            "daily_background": "bearish" if bearish_stack else "bullish_or_neutral",
            "price": round(price, 8),
            "price_vs_ma7_pct": round(pct_diff(price, current_ma7), 6),
            "price_vs_ma20_pct": round(pct_diff(price, current_ma20), 6),
            "lookback": lookback,
            "tolerance_pct": tolerance_pct,
            "long_breakout": long_breakout,
            "long_retest": long_retest,
            "long_confirmed": long_confirmed,
            "short_breakdown": short_breakdown,
            "short_retest": short_retest,
            "short_confirmed": short_confirmed,
            "entry_level": entry_level,
            "state": "long_ma7_rebreak" if long_confirmed else "short_ma7_rebreak" if short_confirmed else "bearish_stack_ma20_above_ma7" if bearish_stack else "not_confirmed",
            "scale_in_levels": [
                {"name": "ma7_1d", "price": round(current_ma7, 8)},
                {"name": "ma20_1d", "price": round(current_ma20, 8)},
            ],
        }

    def market_daily_ma7_reversal_state(self) -> dict[str, Any]:
        if self._market_daily_ma7_reversal_cache is not None:
            return self._market_daily_ma7_reversal_cache
        if not bool(self.config.daily_ma7_short_reversal_exit_enabled):
            self._market_daily_ma7_reversal_cache = {"enabled": False, "ready": False}
            return self._market_daily_ma7_reversal_cache

        market_symbols = [symbol.upper() for symbol in self.config.daily_ma7_short_reversal_market_symbols if str(symbol or "").strip()]
        require_all = bool(self.config.daily_ma7_short_reversal_require_all_market_symbols)
        if not market_symbols:
            self._market_daily_ma7_reversal_cache = {
                "enabled": True,
                "ready": False,
                "reason": "missing market symbols for daily MA7 reversal",
                "market_symbols": [],
                "require_all": require_all,
            }
            return self._market_daily_ma7_reversal_cache

        drivers: list[dict[str, Any]] = []
        for symbol in market_symbols:
            try:
                limit = max(60, min(int(self.config.kline_limit_for_interval("1d")), 180))
                klines = self.client.fetch_klines(symbol, "1d", limit)
                mark_price = self.client.mark_price(symbol)
                if mark_price <= 0 and klines:
                    mark_price = float(klines[-1][4])
                daily = self.daily_ma7_trend_state(mark_price, object(), klines)
                hold = daily.get("ma7_reclaim_hold") if isinstance(daily.get("ma7_reclaim_hold"), dict) else {}
                drivers.append({
                    "symbol": symbol,
                    "ready": bool(daily.get("ready")),
                    "price": daily.get("price"),
                    "ma7": daily.get("ma7"),
                    "ma20": daily.get("ma20"),
                    "ma7_reclaimed": bool(daily.get("ma7_reclaimed")),
                    "ma7_reclaim_held": bool(daily.get("ma7_reclaim_held")),
                    "ma20_reclaimed": bool(daily.get("ma20_reclaimed")),
                    "ma20_reclaim_held": bool(daily.get("ma20_reclaim_held")),
                    "held_count": int(hold.get("held_count") or 0),
                    "required_closed_candles": int(hold.get("required_closed_candles") or self.config.daily_ma7_short_reversal_required_1d_candles),
                    "current_price_above_ma7": bool(hold.get("current_price_above_ma7")),
                    "daily_background": daily.get("daily_background"),
                    "daily_ma7_trend": daily,
                })
            except Exception as exc:
                drivers.append({
                    "symbol": symbol,
                    "ready": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        ready_count = sum(1 for item in drivers if item.get("ready"))
        ma7_held_count = sum(1 for item in drivers if item.get("ma7_reclaim_held"))
        ma20_held_count = sum(1 for item in drivers if item.get("ma20_reclaim_held"))
        reclaimed_count = sum(1 for item in drivers if item.get("ma7_reclaimed") or item.get("held_count"))
        ma20_reclaimed_count = sum(1 for item in drivers if item.get("ma20_reclaimed") or item.get("ma20_reclaim_held"))
        if require_all:
            reversal_confirmed = bool(
                ready_count == len(market_symbols)
                and ma7_held_count == len(market_symbols)
                and ma20_held_count == len(market_symbols)
            )
        else:
            reversal_confirmed = ma7_held_count > 0 and ma20_held_count > 0
        state = "reversal_confirmed" if reversal_confirmed else "ma20_watch" if ma20_reclaimed_count > 0 else "ma7_watch" if reclaimed_count > 0 else "not_confirmed"
        self._market_daily_ma7_reversal_cache = {
            "enabled": True,
            "ready": ready_count == len(market_symbols) if require_all else ready_count > 0,
            "timeframe": "1d",
            "basis": "BTC/ETH daily MA7 first checkpoint + MA20 second checkpoint",
            "market_symbols": market_symbols,
            "require_all": require_all,
            "ready_count": ready_count,
            "held_count": ma20_held_count,
            "ma7_held_count": ma7_held_count,
            "ma20_held_count": ma20_held_count,
            "reclaimed_count": reclaimed_count,
            "ma20_reclaimed_count": ma20_reclaimed_count,
            "ma7_reclaim_confirmed": bool(ma7_held_count == len(market_symbols) if require_all else ma7_held_count > 0),
            "ma20_reclaim_confirmed": bool(ma20_held_count == len(market_symbols) if require_all else ma20_held_count > 0),
            "reversal_confirmed": reversal_confirmed,
            "state": state,
            "checked_at": utc_now_iso(),
            "drivers": drivers,
        }
        return self._market_daily_ma7_reversal_cache

    def bb96_210_second_test_15m_state(self, price: float, bb: Any, klines_15m: list[list[Any]] | None) -> dict[str, Any]:
        rows = self.closed_kline_rows(klines_15m or [])
        if price <= 0 or len(rows) < 32:
            return {"ready": False, "timeframe": "15m", "reason": "insufficient 15M candles for BB96/BB210 second-test"}

        overlap_max_pct = min(1.0, max(0.2, float(self.config.entry_structure_rejection_near_pct)))
        touch_near_pct = min(1.0, max(0.2, float(self.config.entry_structure_rejection_near_pct)))
        previous_near_pct = min(1.2, max(overlap_max_pct, touch_near_pct))
        current_near_pct = previous_near_pct
        recent_lookback = 6
        prior_lookback = 96
        recent = rows[-recent_lookback:]
        prior = rows[-(prior_lookback + recent_lookback):-recent_lookback]
        if len(prior) < 12:
            prior = rows[:-recent_lookback]

        def band_value(period: int, band: str) -> float:
            return float(getattr(bb, f"bb{period}_{band}_15m", 0.0) or 0.0)

        def overlap_zone(first: float, second: float) -> dict[str, Any]:
            if first <= 0 or second <= 0:
                return {"ready": False}
            low = min(first, second)
            high = max(first, second)
            mid = (low + high) / 2.0
            width_pct = pct_diff(high, low)
            return {
                "ready": True,
                "low": round(low, 8),
                "high": round(high, 8),
                "mid": round(mid, 8),
                "width_pct": round(width_pct, 6),
                "max_width_pct": round(overlap_max_pct, 6),
                "overlapped": width_pct <= overlap_max_pct,
            }

        def upper_state() -> dict[str, Any]:
            zone = overlap_zone(band_value(96, "upper"), band_value(210, "upper"))
            if not zone.get("ready"):
                return {"ready": False, "side": "short", "reason": "missing BB96/BB210 upper"}
            recent_high = max(float(row[2]) for row in recent)
            recent_close = float(recent[-1][4])
            prior_high = max(float(row[2]) for row in prior) if prior else 0.0
            zone_low = float(zone["low"])
            zone_high = float(zone["high"])
            zone_mid = float(zone["mid"])
            recent_touched = recent_high >= zone_low * (1.0 - touch_near_pct / 100.0)
            close_rejected = recent_close <= zone_high * (1.0 + touch_near_pct / 100.0) and price <= zone_high * (1.0 + touch_near_pct / 100.0)
            near_previous = prior_high > 0 and abs(pct_diff(recent_high, prior_high)) <= previous_near_pct
            current_near_zone = abs(pct_diff(price, zone_mid)) <= current_near_pct
            confirmed = bool(zone.get("overlapped") and recent_touched and close_rejected and near_previous and current_near_zone)
            return {
                **zone,
                "side": "short",
                "structure_signal": "short_bb96_210_upper_second_test_rejection",
                "name": "bb96_210_upper_second_test_15m",
                "price": round(zone_mid, 8),
                "recent_high": round(recent_high, 8),
                "recent_close": round(recent_close, 8),
                "prior_high": round(prior_high, 8),
                "recent_high_vs_zone_mid_pct": round(pct_diff(recent_high, zone_mid), 6),
                "recent_high_vs_prior_high_pct": round(pct_diff(recent_high, prior_high), 6) if prior_high > 0 else 0.0,
                "touch_near_pct": round(touch_near_pct, 6),
                "previous_near_pct": round(previous_near_pct, 6),
                "current_near_pct": round(current_near_pct, 6),
                "recent_touched": recent_touched,
                "close_rejected": close_rejected,
                "near_previous_high": near_previous,
                "current_near_zone": current_near_zone,
                "confirmed": confirmed,
            }

        def lower_state() -> dict[str, Any]:
            zone = overlap_zone(band_value(96, "lower"), band_value(210, "lower"))
            if not zone.get("ready"):
                return {"ready": False, "side": "long", "reason": "missing BB96/BB210 lower"}
            recent_low = min(float(row[3]) for row in recent)
            recent_close = float(recent[-1][4])
            prior_low = min(float(row[3]) for row in prior) if prior else 0.0
            zone_low = float(zone["low"])
            zone_high = float(zone["high"])
            zone_mid = float(zone["mid"])
            recent_touched = recent_low <= zone_high * (1.0 + touch_near_pct / 100.0)
            close_rebounded = recent_close >= zone_low * (1.0 - touch_near_pct / 100.0) and price >= zone_low * (1.0 - touch_near_pct / 100.0)
            near_previous = prior_low > 0 and abs(pct_diff(recent_low, prior_low)) <= previous_near_pct
            current_near_zone = abs(pct_diff(price, zone_mid)) <= current_near_pct
            confirmed = bool(zone.get("overlapped") and recent_touched and close_rebounded and near_previous and current_near_zone)
            return {
                **zone,
                "side": "long",
                "structure_signal": "long_bb96_210_lower_second_test_rebound",
                "name": "bb96_210_lower_second_test_15m",
                "price": round(zone_mid, 8),
                "recent_low": round(recent_low, 8),
                "recent_close": round(recent_close, 8),
                "prior_low": round(prior_low, 8),
                "recent_low_vs_zone_mid_pct": round(pct_diff(recent_low, zone_mid), 6),
                "recent_low_vs_prior_low_pct": round(pct_diff(recent_low, prior_low), 6) if prior_low > 0 else 0.0,
                "touch_near_pct": round(touch_near_pct, 6),
                "previous_near_pct": round(previous_near_pct, 6),
                "current_near_pct": round(current_near_pct, 6),
                "recent_touched": recent_touched,
                "close_rebounded": close_rebounded,
                "near_previous_low": near_previous,
                "current_near_zone": current_near_zone,
                "confirmed": confirmed,
            }

        upper = upper_state()
        lower = lower_state()
        return {
            "ready": True,
            "timeframe": "15m",
            "periods": [96, 210],
            "meaning": "BB96/BB210 overlap area plus second test near previous high/low",
            "recent_lookback": recent_lookback,
            "prior_lookback": len(prior),
            "upper": upper,
            "lower": lower,
            "active_side": "short" if upper.get("confirmed") and not lower.get("confirmed") else "long" if lower.get("confirmed") and not upper.get("confirmed") else "conflict" if upper.get("confirmed") and lower.get("confirmed") else "",
        }

    def fib_structure_state(self, price: float, klines_4h: list[list[Any]]) -> dict[str, Any]:
        near_pct = max(0.1, float(self.config.fib_structure_confirmation_near_pct))
        rows = self.closed_kline_rows(klines_4h or [])
        min_rows = 120
        if price <= 0 or len(rows) < min_rows:
            return {
                "ready": False,
                "timeframe": "4h",
                "near_pct": near_pct,
                "reason": "insufficient 4H candles for fib structure",
                "kline_count": len(rows),
                "required_kline_count": min_rows,
            }

        configured_lookback = max(min_rows, int(self.config.fib_structure_confirmation_lookback_4h or 0))
        lookback = min(configured_lookback, len(rows))
        window = rows[-lookback:]

        def safe_float(row: list[Any], index: int) -> float:
            try:
                return float(row[index])
            except (TypeError, ValueError, IndexError):
                return 0.0

        def row_time(row: list[Any]) -> str:
            try:
                return datetime.fromtimestamp(float(row[0]) / 1000.0, tz=timezone.utc).isoformat()
            except (TypeError, ValueError, IndexError, OSError):
                return ""

        high_row = max(window, key=lambda item: safe_float(item, 2))
        low_row = min(window, key=lambda item: safe_float(item, 3))
        swing_high = safe_float(high_row, 2)
        swing_low = safe_float(low_row, 3)
        if swing_high <= 0 or swing_low <= 0 or swing_high <= swing_low:
            return {
                "ready": False,
                "timeframe": "4h",
                "near_pct": near_pct,
                "reason": "invalid 4H swing range for fib structure",
                "kline_count": len(rows),
                "lookback": lookback,
                "swing_high": round(swing_high, 8),
                "swing_low": round(swing_low, 8),
            }

        span = swing_high - swing_low
        ratios = (0.236, 0.382, 0.5, 0.618, 0.786, 0.886)
        levels: list[dict[str, Any]] = []
        for ratio in ratios:
            level = swing_high - span * ratio
            if level <= 0:
                continue
            distance = pct_diff(price, level)
            levels.append({
                "name": f"fib{int(round(ratio * 1000)):03d}_4h",
                "ratio": ratio,
                "price": round(level, 8),
                "distance_pct": round(distance, 6),
                "abs_distance_pct": round(abs(distance), 6),
                "role": "support" if level <= price else "resistance",
            })
        levels.sort(key=lambda item: float(item.get("abs_distance_pct") or 999.0))
        supports = [item for item in levels if item.get("role") == "support"]
        resistances = [item for item in levels if item.get("role") == "resistance"]
        near_levels = [item for item in levels if float(item.get("abs_distance_pct") or 999.0) <= near_pct]
        return {
            "ready": True,
            "timeframe": "4h",
            "near_pct": near_pct,
            "lookback": lookback,
            "kline_count": len(rows),
            "swing_high": round(swing_high, 8),
            "swing_high_time": row_time(high_row),
            "swing_low": round(swing_low, 8),
            "swing_low_time": row_time(low_row),
            "levels": levels,
            "nearest_support": supports[0] if supports else {},
            "nearest_resistance": resistances[0] if resistances else {},
            "near_levels": near_levels,
        }

    def levels(self, price: float, bb: Any, h4: Any, *, klines_15m: list[list[Any]] | None = None, klines_4h: list[list[Any]] | None = None, klines_1d: list[list[Any]] | None = None) -> dict[str, Any]:
        shared_levels = self.shared_bb_level_rows(bb)
        ma_levels = self.shared_ma_level_rows(bb)
        ma20_4h_pressure = self.ma20_4h_pressure_state(price, bb)
        h4_right_side = self.h4_right_side_state(price, bb, klines_4h or [])
        daily_ma7_trend = self.daily_ma7_trend_state(price, bb, klines_1d or [])
        market_daily_ma7_reversal = self.market_daily_ma7_reversal_state()
        bb96_210_second_test_15m = self.bb96_210_second_test_15m_state(price, bb, klines_15m or [])
        fib_structure_4h = self.fib_structure_state(price, klines_4h or [])
        supports = [
            ("h4_recent_swing_low", float(h4.h4_recent_swing_low or 0.0)),
            ("h4_range_low", float(h4.h4_range_low or 0.0)),
            *[(name, value) for name, value in ma_levels if value < price],
            *[(name, value) for name, value in shared_levels if value < price],
        ]
        resistances = [
            *[(name, value) for name, value in ma_levels if value > price],
            *[(name, value) for name, value in shared_levels if value > price],
        ]
        support_rows = [(name, round(value, 8)) for name, value in supports if 0 < value < price]
        resistance_rows = [(name, round(value, 8)) for name, value in resistances if value > price]
        support_rows.sort(key=lambda item: price - item[1])
        resistance_rows.sort(key=lambda item: item[1] - price)
        coverage = self.bb_coverage(shared_levels)
        structure_scan = self.structure_entry_scan(
            price,
            shared_levels,
            ma_levels,
            {"15m": klines_15m or [], "4h": klines_4h or [], "1d": klines_1d or []},
        )
        return {
            "price": round(price, 8),
            "nearest_supports": support_rows[:8],
            "nearest_resistances": resistance_rows[:8],
            "all_supports": support_rows,
            "all_resistances": resistance_rows,
            "support_count": len(support_rows),
            "resistance_count": len(resistance_rows),
            "bb_coverage": coverage,
            "structure_entry_scan": structure_scan,
            "shared_bb_timeframes": ["15m", "4h", "1d"],
            "shared_bb_periods": list(SHARED_BB_PERIODS),
            "shared_ma_levels": [(name, round(value, 8)) for name, value in ma_levels],
            "ma20_4h_pressure": ma20_4h_pressure,
            "h4_right_side": h4_right_side,
            "daily_ma7_trend": daily_ma7_trend,
            "market_daily_ma7_reversal": market_daily_ma7_reversal,
            "bb96_210_second_test_15m": bb96_210_second_test_15m,
            "fib_structure_4h": fib_structure_4h,
            "short_entry_retest_zone": self.short_retest_zone(price, bb),
        }

    def bb_coverage(self, shared_levels: list[tuple[str, float]]) -> dict[str, Any]:
        by_timeframe: dict[str, int] = {"15m": 0, "4h": 0, "1d": 0}
        by_band: dict[str, int] = {"lower": 0, "mid": 0, "upper": 0}
        by_period: dict[str, int] = {str(period): 0 for period in SHARED_BB_PERIODS}
        for name, _ in shared_levels:
            parsed = self.parse_shared_bb_name(name)
            if not parsed:
                continue
            timeframe = str(parsed["timeframe"])
            band = str(parsed["band"])
            period = str(parsed["period"])
            by_timeframe[timeframe] = by_timeframe.get(timeframe, 0) + 1
            by_band[band] = by_band.get(band, 0) + 1
            by_period[period] = by_period.get(period, 0) + 1
        configured_total = len(SHARED_BB_PERIODS) * 3 * 3
        active_total = len(shared_levels)
        return {
            "configured_level_count": configured_total,
            "active_level_count": active_total,
            "missing_level_count": max(0, configured_total - active_total),
            "timeframes": ["15m", "4h", "1d"],
            "periods": list(SHARED_BB_PERIODS),
            "bands": ["lower", "mid", "upper"],
            "by_timeframe": by_timeframe,
            "by_band": by_band,
            "by_period": by_period,
        }

    @staticmethod
    def parse_shared_bb_name(name: str) -> dict[str, Any] | None:
        parts = str(name or "").split("_")
        if len(parts) != 3 or not parts[0].startswith("bb"):
            return None
        try:
            period = int(parts[0][2:])
        except ValueError:
            return None
        band = parts[1]
        timeframe = parts[2]
        if band not in {"lower", "mid", "upper"} or timeframe not in {"15m", "4h", "1d"}:
            return None
        return {"period": period, "band": band, "timeframe": timeframe}

    @staticmethod
    def parse_shared_ma_name(name: str) -> dict[str, Any] | None:
        parts = str(name or "").split("_")
        if len(parts) != 2 or not parts[0].startswith("ma"):
            return None
        try:
            period = int(parts[0][2:])
        except ValueError:
            return None
        timeframe = parts[1]
        if timeframe not in {"15m", "4h", "1d"}:
            return None
        return {"period": period, "band": "ma", "timeframe": timeframe}

    def structure_entry_scan(
        self,
        price: float,
        shared_levels: list[tuple[str, float]],
        ma_levels: list[tuple[str, float]] | None = None,
        klines_by_timeframe: dict[str, list[list[Any]]] | None = None,
    ) -> dict[str, Any]:
        near_levels: list[dict[str, Any]] = []
        long_candidates: list[dict[str, Any]] = []
        short_candidates: list[dict[str, Any]] = []
        near_pct = float(self.config.mid_retest_near_pct)

        def append_candidate(row: dict[str, Any]) -> None:
            signal = str(row.get("structure_signal") or "")
            if signal.startswith("long_"):
                long_candidates.append(row)
            elif signal.startswith("short_"):
                short_candidates.append(row)
            near_levels.append(row)

        for name, value in shared_levels:
            parsed = self.parse_shared_bb_name(name)
            if not parsed or value <= 0:
                continue
            distance = pct_diff(price, value)
            abs_distance = abs(distance)
            if abs_distance > near_pct:
                continue
            band = str(parsed["band"])
            row = {
                "name": name,
                "price": round(value, 8),
                "distance_pct": round(distance, 6),
                "abs_distance_pct": round(abs_distance, 6),
                "period": parsed["period"],
                "band": band,
                "timeframe": parsed["timeframe"],
            }
            if band == "lower":
                if distance >= 0:
                    row["structure_signal"] = "long_lower_support_rebound"
                    long_candidates.append(row)
                else:
                    row["structure_signal"] = "short_lower_breakdown_retest"
                    short_candidates.append(row)
            elif band == "mid":
                if distance >= 0:
                    row["structure_signal"] = "long_mid_support_retest"
                    long_candidates.append(row)
                else:
                    row["structure_signal"] = "short_mid_resistance_retest"
                    short_candidates.append(row)
            elif band == "upper":
                if distance >= 0:
                    row["structure_signal"] = "long_upper_breakout"
                    long_candidates.append(row)
                else:
                    row["structure_signal"] = "short_upper_rejection"
                    short_candidates.append(row)
            near_levels.append(row)

        for name, value in ma_levels or []:
            parsed = self.parse_shared_ma_name(name)
            if not parsed or value <= 0 or parsed["timeframe"] not in {"15m", "4h"}:
                continue
            distance = pct_diff(price, value)
            abs_distance = abs(distance)
            if abs_distance > near_pct:
                continue
            row = {
                "name": name,
                "price": round(value, 8),
                "distance_pct": round(distance, 6),
                "abs_distance_pct": round(abs_distance, 6),
                "period": parsed["period"],
                "band": "ma",
                "timeframe": parsed["timeframe"],
                "indicator_family": "ma",
            }
            if distance >= 0:
                row["structure_signal"] = "long_ma_support_retest"
            else:
                row["structure_signal"] = "short_ma_resistance_retest"
            append_candidate(row)

        klines_by_timeframe = klines_by_timeframe or {}
        ma_cross_periods_by_timeframe = {
            "15m": (7, 20, 60, 96, 105, 120),
            "4h": (7, 20, 60, 96, 105, 120, 300),
            "1d": (7, 20),
        }
        for timeframe, rows in klines_by_timeframe.items():
            periods = ma_cross_periods_by_timeframe.get(str(timeframe))
            if not periods:
                continue
            closed = self.closed_kline_rows(rows or [])
            if len(closed) < 3:
                continue
            closes = [float(row[4]) for row in closed]
            current_index = len(closed) - 1
            previous_index = current_index - 1
            for period in periods:
                if len(closes) < period + 1:
                    continue
                ma_now = self.sma_at(closes, current_index, period)
                ma_prev = self.sma_at(closes, previous_index, period)
                if ma_now <= 0 or ma_prev <= 0:
                    continue
                prev_close = float(closed[previous_index][4])
                close_now = float(closed[current_index][4])
                distance = pct_diff(price, ma_now)
                if abs(distance) > near_pct:
                    continue
                base = {
                    "name": f"ma{period}_{timeframe}",
                    "price": round(ma_now, 8),
                    "distance_pct": round(distance, 6),
                    "abs_distance_pct": round(abs(distance), 6),
                    "period": period,
                    "band": "ma",
                    "timeframe": timeframe,
                    "indicator_family": "ma",
                    "source": "ma_cross",
                    "previous_close": round(prev_close, 8),
                    "current_close": round(close_now, 8),
                }
                if prev_close <= ma_prev and close_now > ma_now:
                    row = dict(base)
                    row["structure_signal"] = "long_ma_reclaim_cross"
                    append_candidate(row)
                if prev_close >= ma_prev and close_now < ma_now:
                    row = dict(base)
                    row["structure_signal"] = "short_ma_break_cross"
                    append_candidate(row)
        near_levels.sort(key=lambda item: (item["abs_distance_pct"], item["timeframe"], int(item["period"])))
        long_candidates.sort(key=lambda item: (item["abs_distance_pct"], item["timeframe"], int(item["period"])))
        short_candidates.sort(key=lambda item: (item["abs_distance_pct"], item["timeframe"], int(item["period"])))
        return {
            "near_pct": near_pct,
            "near_level_count": len(near_levels),
            "long_candidate_count": len(long_candidates),
            "short_candidate_count": len(short_candidates),
            "near_levels": near_levels[:12],
            "long_candidates": long_candidates[:8],
            "short_candidates": short_candidates[:8],
        }

    def short_retest_zone(self, price: float, bb: Any) -> dict[str, Any]:
        mids = [
            (f"bb{period}_mid_4h", float(getattr(bb, f"bb{period}_mid_4h", 0.0) or 0.0))
            for period in SHARED_BB_PERIODS
        ]
        candidates = [(name, mid, abs(pct_diff(price, mid))) for name, mid in mids if mid > price > 0]
        if not candidates:
            return {}
        name, mid, _ = min(candidates, key=lambda item: item[2])
        return {
            "target_mid": name,
            "mid_price": round(mid, 8),
            "zone_low": round(mid * 0.98, 8),
            "zone_high": round(mid * 0.995, 8),
            "distance_pct": round(pct_diff(price, mid), 6),
        }

    def staged_take_profit_targets(self, target_rows: list[dict[str, Any]], action: str = "", horizon: str = "") -> list[dict[str, Any]]:
        if not self.config.staged_take_profit_enabled or not target_rows:
            return []
        max_targets = max(1, int(self.config.staged_take_profit_max_targets))
        primary_rows: list[dict[str, Any]] = []
        preferred_timeframes = {"4h", "1d"} if str(horizon) == "4h_trend" else {"15m", "4h"}
        for row in target_rows:
            parsed = self.parse_shared_bb_name(str(row.get("name") or ""))
            if parsed and parsed["timeframe"] in preferred_timeframes:
                primary_rows.append(row)
        candidates = primary_rows or target_rows
        selected = self.select_staged_take_profit_rows(candidates, max_targets, action=action, horizon=horizon)
        fractions = self.staged_take_profit_fractions_for_rows(selected)
        stages: list[dict[str, Any]] = []
        for index, row in enumerate(selected):
            parsed = self.parse_shared_bb_name(str(row.get("name") or "")) or {}
            stages.append({
                "index": index + 1,
                "name": str(row.get("name") or ""),
                "price": round(float(row.get("price") or 0.0), 8),
                "distance_pct": round(float(row.get("distance_pct") or 0.0), 6),
                "fraction_of_initial": fractions[index],
                "timeframe": parsed.get("timeframe", ""),
                "period": parsed.get("period", 0),
                "band": parsed.get("band", ""),
                "status": "pending",
                "selection_strategy": STAGED_TAKE_PROFIT_SELECTION_STRATEGY,
            })
        return stages

    def select_staged_take_profit_rows(self, rows: list[dict[str, Any]], max_targets: int, action: str = "", horizon: str = "") -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            name = str(row.get("name") or "")
            price = float(row.get("price") or 0.0)
            if not name or price <= 0 or name in seen:
                continue
            seen.add(name)
            unique.append(row)
        if max_targets == 1:
            return [unique[0]]

        parsed_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in unique:
            parsed = self.parse_shared_bb_name(str(row.get("name") or ""))
            if parsed and parsed["timeframe"] in {"15m", "4h"}:
                parsed_rows.append((row, parsed))
        if not parsed_rows:
            return unique[:max_targets]

        action = str(action or "").lower()
        if action == "open_long":
            first_bands = ("mid",)
            runner_bands = ("upper",)
        else:
            first_bands = ("mid",)
            runner_bands = ("lower",)

        prefer_15m_default = str(horizon) != "4h_trend"

        def band_rows(bands: tuple[str, ...], *, prefer_15m: bool) -> list[dict[str, Any]]:
            candidates = [
                (row, parsed) for row, parsed in parsed_rows
                if parsed["band"] in bands
            ]
            candidates.sort(key=lambda item: (
                0 if prefer_15m and item[1]["timeframe"] == "15m" else 1 if prefer_15m else 0,
                abs(float(item[0].get("distance_pct") or 0.0)),
                int(item[1]["period"]),
            ))
            return [row for row, _ in candidates]

        selected: list[dict[str, Any]] = []
        selected_names: set[str] = set()

        def add(row: dict[str, Any] | None) -> None:
            if not row or len(selected) >= max_targets:
                return
            name = str(row.get("name") or "")
            if not name or name in selected_names:
                return
            selected_names.add(name)
            selected.append(row)

        for row in band_rows(first_bands, prefer_15m=prefer_15m_default):
            add(row)
            if len(selected) >= min(2, max_targets):
                break
        if selected:
            selected_prices = [float(row.get("price") or 0.0) for row in selected]
            runner_boundary = min(selected_prices) if action != "open_long" else max(selected_prices)
        else:
            runner_boundary = 0.0
        for row in band_rows(runner_bands, prefer_15m=False):
            price = float(row.get("price") or 0.0)
            if selected and action == "open_long" and price <= runner_boundary:
                continue
            if selected and action != "open_long" and price >= runner_boundary:
                continue
            add(row)
            break
        if not selected:
            for row in unique:
                add(row)
                if len(selected) >= max_targets:
                    break
        return selected[:max_targets]

    def staged_take_profit_fractions_for_rows(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        first = self.parse_shared_bb_name(str(rows[0].get("name") or "")) or {}
        if first.get("band") == "mid":
            if len(rows) == 1:
                return [1.0]
            remaining = 0.5
            tail = len(rows) - 1
            fractions = [0.5] + [round(remaining / tail, 6) for _ in range(tail)]
            fractions[-1] = round(fractions[-1] + (1.0 - sum(fractions)), 6)
            return fractions
        return self.staged_take_profit_fractions(len(rows))

    def staged_take_profit_fractions(self, stage_count: int) -> list[float]:
        if stage_count <= 0:
            return []
        configured = [value for value in self.config.staged_take_profit_fractions if value > 0]
        fractions: list[float] = []
        remaining = 1.0
        for index in range(stage_count):
            if index == stage_count - 1:
                fraction = remaining
            elif index < len(configured):
                fraction = min(configured[index], remaining)
            else:
                fraction = remaining / (stage_count - index)
            fraction = max(0.0, min(fraction, remaining))
            fractions.append(round(fraction, 6))
            remaining = max(0.0, remaining - fraction)
        if fractions and sum(fractions) < 0.999:
            fractions[-1] = round(fractions[-1] + (1.0 - sum(fractions)), 6)
        return fractions

    def roi_take_profit_stage_templates(self, symbol: str = "") -> list[dict[str, Any]]:
        if not self.config.roi_take_profit_enabled:
            return []
        symbol_upper = str(symbol or "").upper()
        if symbol_upper == "ZECUSDT":
            thresholds = [25.0, 30.0, 38.0]
            fractions = [0.30, 0.20, 0.20]
            strategy = "roi_profit_protection_zec_v1"
        elif symbol_upper == "BNBUSDT":
            thresholds = [8.0, 10.0]
            fractions = [0.20, 0.10]
            strategy = "roi_profit_protection_bnb_v1"
        else:
            thresholds = [float(value) for value in self.config.roi_take_profit_thresholds if float(value) > 0]
            fractions = [float(value) for value in self.config.roi_take_profit_fractions if float(value) > 0]
            strategy = "roi_profit_protection_v1"
        templates: list[dict[str, Any]] = []
        for index, threshold in enumerate(thresholds):
            if index >= len(fractions):
                break
            fraction = clamp(fractions[index], 0.0, 1.0)
            if fraction <= 0:
                continue
            templates.append({
                "index": index + 1,
                "name": f"roi_tp_{int(threshold) if threshold.is_integer() else threshold:g}",
                "roi_threshold_pct": round(threshold, 6),
                "fraction_of_initial": round(fraction, 6),
                "status": "pending",
                "selection_strategy": strategy,
            })
        templates.sort(key=lambda item: float(item.get("roi_threshold_pct") or 0.0))
        return templates

    def ensure_position_roi_take_profit_stages(self, position: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.config.roi_take_profit_enabled:
            return []
        templates = self.roi_take_profit_stage_templates(str(position.get("symbol") or ""))
        if not templates:
            return []
        existing = [stage for stage in position.get("roi_take_profit_stages") or [] if isinstance(stage, dict)]
        existing_by_name = {str(stage.get("name") or ""): stage for stage in existing}
        stages: list[dict[str, Any]] = []
        for template in templates:
            name = str(template.get("name") or "")
            old = existing_by_name.get(name)
            if old:
                merged = dict(template)
                merged.update(old)
                stages.append(merged)
            else:
                stages.append(dict(template))
        position["roi_take_profit_stages"] = stages
        position.setdefault("remaining_fraction", 1.0)
        return stages

    @staticmethod
    def _metadata_tokens(value: Any) -> set[str]:
        if isinstance(value, str):
            raw = value.replace("|", ",").replace(";", ",")
            return {item.strip().lower() for item in raw.split(",") if item.strip()}
        if isinstance(value, (list, tuple, set)):
            return {str(item).strip().lower() for item in value if str(item).strip()}
        return set()

    @staticmethod
    def candidate_has_news_confirmation(metadata: dict[str, Any]) -> bool:
        if not isinstance(metadata, dict):
            return False
        if bool(metadata.get("has_news_confirmation")):
            return True
        sources = V6ExecutionEngine._metadata_tokens(metadata.get("sentiment_source") or metadata.get("source"))
        if sources & {"opennews_news", "opennews_listing", "opennews_onchain", "opentwitter_official"}:
            return True
        rationale = metadata.get("rationale") or metadata.get("reasons") or []
        if isinstance(rationale, str):
            rationale = [rationale]
        return any(str(item).startswith("news_confirm=") for item in rationale if str(item or ""))

    @staticmethod
    def candidate_is_price_momentum_only(metadata: dict[str, Any]) -> bool:
        if not isinstance(metadata, dict) or not metadata:
            return False
        if V6ExecutionEngine.candidate_has_news_confirmation(metadata):
            return False
        if bool(metadata.get("price_momentum_only")):
            return True
        sources = V6ExecutionEngine._metadata_tokens(metadata.get("sentiment_source") or metadata.get("source"))
        if sources & {"binance_top_movers", "coingecko_trending", "crypto-market-rank", "meme-rush"}:
            return True
        notes = str(metadata.get("notes") or metadata.get("rationale") or "").lower()
        return any(token in notes for token in ("24h_rank", "move=", "top_movers", "hotness="))

    def v5_sentiment_metadata_cache(self) -> dict[str, dict[str, Any]]:
        if self._v5_sentiment_metadata_cache is not None:
            return self._v5_sentiment_metadata_cache
        rows: dict[str, dict[str, Any]] = {}
        try:
            path = self.config.v5_no_news_momentum_sentiment_file
            if path.exists():
                with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
                    for row in csv.DictReader(handle):
                        symbol = str(row.get("symbol") or "").upper().strip()
                        if symbol:
                            rows[symbol] = dict(row)
        except Exception as exc:
            LOGGER.warning("V5 no-news momentum metadata load failed: %s", exc)
        self._v5_sentiment_metadata_cache = rows
        return rows

    def v5_candidate_metadata_for_symbol(self, symbol: str) -> dict[str, Any]:
        symbol_upper = str(symbol or "").upper()
        configured = self.config.v5_candidate_metadata_by_symbol.get(symbol_upper)
        if isinstance(configured, dict) and configured:
            return dict(configured)
        row = self.v5_sentiment_metadata_cache().get(symbol_upper)
        return dict(row) if isinstance(row, dict) else {}

    def apply_candidate_metadata_to_risk_plan(self, symbol: str, risk_plan: dict[str, Any]) -> None:
        metadata = self.v5_candidate_metadata_for_symbol(symbol)
        if not metadata:
            return
        risk_plan["v5_candidate_metadata"] = {
            "symbol": str(metadata.get("symbol") or symbol).upper(),
            "score": metadata.get("score"),
            "sentiment_score": metadata.get("sentiment_score") or metadata.get("hotness_score"),
            "sentiment_mentions": metadata.get("sentiment_mentions") or metadata.get("mention_count"),
            "sentiment_source": metadata.get("sentiment_source") or metadata.get("source") or "",
            "rationale": list(metadata.get("rationale") or metadata.get("reasons") or [])[:10]
            if isinstance(metadata.get("rationale") or metadata.get("reasons"), list)
            else [],
            "notes": str(metadata.get("notes") or "")[:300],
            "has_news_confirmation": self.candidate_has_news_confirmation(metadata),
            "price_momentum_only": self.candidate_is_price_momentum_only(metadata),
        }

    def v5_no_news_momentum_position_metadata(self, position: dict[str, Any]) -> dict[str, Any]:
        risk_plan = position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {}
        metadata = risk_plan.get("v5_candidate_metadata") if isinstance(risk_plan.get("v5_candidate_metadata"), dict) else {}
        if metadata:
            return dict(metadata)
        return self.v5_candidate_metadata_for_symbol(str(position.get("symbol") or ""))

    def is_v5_no_news_momentum_position(self, position: dict[str, Any]) -> bool:
        if not self.config.v5_no_news_momentum_tp_enabled:
            return False
        return self.candidate_is_price_momentum_only(self.v5_no_news_momentum_position_metadata(position))

    def v5_profit_reentry_guard_block(self, signal: V6Signal, state: dict[str, Any], action: str) -> dict[str, Any] | None:
        if not self.config.v5_profit_reentry_guard_enabled or action not in {"open_long", "open_short"}:
            return None
        guards = state.get("v5_profit_reentry_guards") if isinstance(state.get("v5_profit_reentry_guards"), dict) else {}
        guard = guards.get(signal.symbol.upper()) if isinstance(guards, dict) else None
        if not isinstance(guard, dict):
            return None
        expected_side = "long" if action == "open_long" else "short"
        if str(guard.get("side") or "").lower() != expected_side:
            return None
        try:
            created_at = datetime.fromisoformat(str(guard.get("created_at") or "").replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - created_at).total_seconds() / 3600.0 > float(self.config.v5_profit_reentry_guard_max_age_hours):
                return None
        except Exception:
            pass
        exit_price = float(guard.get("exit_price") or 0.0)
        price = float(signal.price)
        if exit_price <= 0 or price <= 0:
            return None
        pullback_pct = max(0.0, float(self.config.v5_profit_reentry_pullback_pct))
        ma20_near_pct = max(0.0, float(self.config.v5_profit_reentry_ma20_near_pct))
        fifteen_m = signal.fifteen_m if isinstance(signal.fifteen_m, dict) else {}
        ma20_trend = fifteen_m.get("ma20_trend") if isinstance(fifteen_m.get("ma20_trend"), dict) else {}
        ma20 = float(ma20_trend.get("ma20") or 0.0)
        if expected_side == "long":
            pullback_ok = price <= exit_price * (1.0 - pullback_pct / 100.0)
            ma20_retest_ok = ma20 > 0 and price <= ma20 * (1.0 + ma20_near_pct / 100.0)
        else:
            pullback_ok = price >= exit_price * (1.0 + pullback_pct / 100.0)
            ma20_retest_ok = ma20 > 0 and price >= ma20 * (1.0 - ma20_near_pct / 100.0)
        if pullback_ok or ma20_retest_ok:
            return None
        return {
            "enabled": True,
            "blocked": True,
            "reason": "v5_profit_reentry_no_pullback",
            "message": "entry blocked: V5 symbol has recent profitable exit; waiting for pullback/retest before re-entry",
            "symbol": signal.symbol,
            "action": action,
            "side": expected_side,
            "price": round(price, 8),
            "last_profit_exit_price": round(exit_price, 8),
            "last_profit_roi_pct": guard.get("roi_pct"),
            "required_pullback_pct": pullback_pct,
            "ma20_15m": round(ma20, 8) if ma20 > 0 else 0.0,
            "ma20_near_pct": ma20_near_pct,
            "created_at": guard.get("created_at"),
        }

    def record_v5_profit_reentry_guard(
        self,
        state: dict[str, Any],
        position: dict[str, Any],
        intent: V6OrderIntent,
        result: dict[str, Any],
        *,
        event: str,
    ) -> None:
        if not self.config.v5_profit_reentry_guard_enabled:
            return
        if str(result.get("status") or "") not in {"submitted", "submitted_unprotected", "submitted_no_position_delta"}:
            return
        if self.submitted_order_has_no_confirmed_position_delta(result):
            return
        side = str(position.get("side") or local_side_for_action(intent.action) or "").lower()
        if side not in {"long", "short"}:
            return
        entry = float(position.get("entry_price") or 0.0)
        close_price = float(intent.entry_price or 0.0)
        leverage = float(position.get("leverage") or intent.leverage or self.config.default_leverage or 1.0)
        roi_pct = self.position_roi_pct(side, entry, close_price, leverage)
        if roi_pct <= 0:
            return
        guards = state.get("v5_profit_reentry_guards") if isinstance(state.get("v5_profit_reentry_guards"), dict) else {}
        guards[intent.symbol.upper()] = {
            "symbol": intent.symbol.upper(),
            "side": side,
            "exit_price": round(close_price, 8),
            "entry_price": round(entry, 8),
            "roi_pct": round(roi_pct, 6),
            "event": event,
            "order_status": result.get("status"),
            "created_at": result.get("submitted_at") or utc_now_iso(),
            "pullback_pct": float(self.config.v5_profit_reentry_pullback_pct),
            "ma20_near_pct": float(self.config.v5_profit_reentry_ma20_near_pct),
            "source": "v5_profit_reentry_guard",
        }
        state["v5_profit_reentry_guards"] = guards

    @staticmethod
    def position_roi_pct(side: str, entry_price: float, touch_price: float, leverage: float) -> float:
        if entry_price <= 0 or touch_price <= 0 or leverage <= 0:
            return 0.0
        if side == "short":
            return ((entry_price - touch_price) / entry_price) * leverage * 100.0
        if side == "long":
            return ((touch_price - entry_price) / entry_price) * leverage * 100.0
        return 0.0

    def entry_context_for_action(self, action: str, levels: dict[str, Any], *, symbol: str = "") -> dict[str, Any]:
        side_key = "short_candidates" if action == "open_short" else "long_candidates"
        raw_candidates = [dict(item) for item in ((levels.get("structure_entry_scan") or {}).get(side_key) or []) if isinstance(item, dict)]
        candidates = list(raw_candidates)
        second_test = levels.get("bb96_210_second_test_15m") if isinstance(levels.get("bb96_210_second_test_15m"), dict) else {}
        second_test_candidate: dict[str, Any] = {}
        if action == "open_short":
            upper = second_test.get("upper") if isinstance(second_test.get("upper"), dict) else {}
            if upper.get("confirmed"):
                level_price = float(upper.get("price") or 0.0)
                second_test_candidate = {
                    "name": str(upper.get("name") or "bb96_210_upper_second_test_15m"),
                    "price": round(level_price, 8),
                    "distance_pct": round(pct_diff(float(levels.get("price") or 0.0), level_price), 6) if float(levels.get("price") or 0.0) > 0 and level_price > 0 else round(float(upper.get("recent_high_vs_zone_mid_pct") or 0.0), 6),
                    "abs_distance_pct": round(abs(float(upper.get("recent_high_vs_zone_mid_pct") or 0.0)), 6),
                    "period": 0,
                    "band": "cluster",
                    "timeframe": "15m",
                    "structure_signal": "short_bb96_210_upper_second_test_rejection",
                    "line_count": 2,
                    "source": "bb96_210_second_test_15m",
                    "second_test_zone": upper,
                }
        elif action == "open_long":
            lower = second_test.get("lower") if isinstance(second_test.get("lower"), dict) else {}
            if lower.get("confirmed"):
                level_price = float(lower.get("price") or 0.0)
                second_test_candidate = {
                    "name": str(lower.get("name") or "bb96_210_lower_second_test_15m"),
                    "price": round(level_price, 8),
                    "distance_pct": round(pct_diff(float(levels.get("price") or 0.0), level_price), 6) if float(levels.get("price") or 0.0) > 0 and level_price > 0 else round(float(lower.get("recent_low_vs_zone_mid_pct") or 0.0), 6),
                    "abs_distance_pct": round(abs(float(lower.get("recent_low_vs_zone_mid_pct") or 0.0)), 6),
                    "period": 0,
                    "band": "cluster",
                    "timeframe": "15m",
                    "structure_signal": "long_bb96_210_lower_second_test_rebound",
                    "line_count": 2,
                    "source": "bb96_210_second_test_15m",
                    "second_test_zone": lower,
                }
        profile = symbol_bb_entry_profile(symbol)
        profile_filter: dict[str, Any] = {}
        if profile:
            candidates = [item for item in raw_candidates if bb_entry_profile_allows(symbol, item)]
            blocked = [item for item in raw_candidates if not bb_entry_profile_allows(symbol, item)]
            profile_filter = {
                "enabled": True,
                "symbol": str(symbol or "").upper(),
                "profile": str(profile.get("name") or ""),
                "source": str(profile.get("source") or ""),
                "allowed_rules": [
                    {
                        "timeframe": timeframe,
                        "period": period,
                        "band": band,
                        "structure_signal": structure_signal,
                    }
                    for timeframe, period, band, structure_signal in profile.get("allow") or ()
                ],
                "input_candidate_count": len(raw_candidates),
                "allowed_candidate_count": len(candidates),
                "blocked_candidate_count": len(blocked),
                "blocked_candidates": blocked[:8],
            }
        allow_synthetic_second_test = not profile or bool(profile.get("allow_synthetic_15m_second_test", True))
        if second_test_candidate and allow_synthetic_second_test:
            candidates.insert(0, second_test_candidate)
            if profile_filter:
                profile_filter["synthetic_second_test_candidate_added"] = dict(second_test_candidate)
        elif second_test_candidate and profile_filter:
            profile_filter["synthetic_second_test_candidate_blocked"] = dict(second_test_candidate)
        profile_force_4h_only = bool(profile and profile.get("force_4h_only"))
        profile_disables_h4_15m_selector = bool(profile and profile.get("disable_h4_confirmed_15m_selector"))
        ma20_pressure = levels.get("ma20_4h_pressure") if isinstance(levels.get("ma20_4h_pressure"), dict) else {}
        h4_right_side = levels.get("h4_right_side") if isinstance(levels.get("h4_right_side"), dict) else {}
        daily_trend = levels.get("daily_ma7_trend") if isinstance(levels.get("daily_ma7_trend"), dict) else {}
        side = "short" if action == "open_short" else "long"
        h4_same_side_confirmed = bool(
            h4_right_side.get("short_confirmed") if side == "short" else h4_right_side.get("long_confirmed")
        )
        ma20_candidate: dict[str, Any] = {}
        daily_candidate: dict[str, Any] = {}
        ma20_value = float(ma20_pressure.get("value") or 0.0)
        profile_allows_4h_ma20 = bool(profile and profile.get("allow_4h_ma20_pressure"))
        if (not profile or profile_allows_4h_ma20) and action == "open_short" and ma20_pressure.get("pressure_active") and h4_right_side.get("short_confirmed") and ma20_value > 0:
            ma20_candidate = {
                "name": "ma20_4h",
                "price": round(ma20_value, 8),
                "distance_pct": round(float(ma20_pressure.get("distance_pct") or 0.0), 6),
                "abs_distance_pct": round(abs(float(ma20_pressure.get("distance_pct") or 0.0)), 6),
                "period": 20,
                "band": "ma",
                "timeframe": "4h",
                "structure_signal": "short_4h_ma20_pressure_retest",
            }
        elif (not profile or profile_allows_4h_ma20) and action == "open_long" and ma20_pressure.get("support_active") and h4_right_side.get("long_confirmed") and ma20_value > 0:
            ma20_candidate = {
                "name": "ma20_4h",
                "price": round(ma20_value, 8),
                "distance_pct": round(float(ma20_pressure.get("distance_pct") or 0.0), 6),
                "abs_distance_pct": round(abs(float(ma20_pressure.get("distance_pct") or 0.0)), 6),
                "period": 20,
                "band": "ma",
                "timeframe": "4h",
                "structure_signal": "long_4h_ma20_support_retest",
            }
        if not profile and action == "open_long" and daily_trend.get("long_confirmed"):
            daily_candidate = dict(daily_trend.get("entry_level") or {})
            daily_candidate.setdefault("name", "ma7_1d")
            daily_candidate.setdefault("timeframe", "1d")
            daily_candidate.setdefault("period", 7)
            daily_candidate.setdefault("band", "ma")
            daily_candidate.setdefault("structure_signal", "long_1d_ma7_rebreak")
        elif not profile and action == "open_short" and daily_trend.get("short_confirmed"):
            daily_candidate = dict(daily_trend.get("entry_level") or {})
            daily_candidate.setdefault("name", "ma7_1d")
            daily_candidate.setdefault("timeframe", "1d")
            daily_candidate.setdefault("period", 7)
            daily_candidate.setdefault("band", "ma")
            daily_candidate.setdefault("structure_signal", "short_1d_ma7_rebreak")
        one_d = [item for item in candidates if str(item.get("timeframe") or "") == "1d"] if profile else []
        four_h_raw = [item for item in candidates if str(item.get("timeframe") or "") == "4h"]
        if action == "open_long":
            four_h = four_h_raw if h4_right_side.get("long_confirmed") else []
        else:
            four_h = four_h_raw if h4_right_side.get("short_confirmed") else []
        fifteen_m = [item for item in candidates if str(item.get("timeframe") or "") == "15m"]
        h4_selector_fifteen_m = list(fifteen_m)
        if h4_same_side_confirmed and not profile_disables_h4_15m_selector:
            seen_15m = {str(item.get("name") or "") for item in h4_selector_fifteen_m}
            profile_bypass: list[dict[str, Any]] = []
            for item in raw_candidates:
                if str(item.get("timeframe") or "") != "15m":
                    continue
                name = str(item.get("name") or "")
                if name in seen_15m:
                    continue
                seen_15m.add(name)
                profile_bypass.append(item)
                h4_selector_fifteen_m.append(item)
            if second_test_candidate and str(second_test_candidate.get("timeframe") or "") == "15m":
                name = str(second_test_candidate.get("name") or "")
                if name and name not in seen_15m:
                    h4_selector_fifteen_m.insert(0, second_test_candidate)
                    profile_bypass.insert(0, second_test_candidate)
            if profile_filter and profile_bypass:
                profile_filter["h4_confirmed_15m_selector_bypass"] = profile_bypass[:8]
        elif h4_same_side_confirmed and profile_filter and profile_disables_h4_15m_selector:
            profile_filter["h4_confirmed_15m_selector_bypass_blocked"] = True
        non_trend_fallback = [] if profile_force_4h_only else [
            item for item in candidates
            if str(item.get("timeframe") or "") not in {"4h", "1d"}
        ]
        selected = daily_candidate if daily_candidate else one_d[0] if one_d else four_h[0] if four_h else ma20_candidate if ma20_candidate else h4_selector_fifteen_m[0] if h4_selector_fifteen_m else non_trend_fallback[0] if non_trend_fallback else {}
        if profile_force_4h_only and selected and str(selected.get("timeframe") or "") != "4h":
            selected = {}
            if profile_filter:
                profile_filter["force_4h_only_rejected_selected"] = True
        timeframe = str(selected.get("timeframe") or "15m") if selected else "15m"
        h4_confirmed_15m_selector = bool(h4_same_side_confirmed and selected and timeframe == "15m")
        countertrend_rebound = bool(
            self.config.countertrend_rebound_enabled
            and selected
            and timeframe in {"15m", "4h"}
            and (
                (action == "open_long" and daily_trend.get("short_confirmed"))
                or (action == "open_short" and daily_trend.get("long_confirmed"))
            )
        )
        if timeframe == "1d":
            horizon = "1d_trend"
        elif timeframe == "4h" or h4_confirmed_15m_selector:
            horizon = "4h_trend"
        else:
            horizon = "15m_scalp"
        if countertrend_rebound:
            horizon = "15m_rebound"
        if horizon == "1d_trend":
            max_hold_days = float(self.config.max_hold_days_1d)
        elif horizon == "4h_trend":
            max_hold_days = float(self.config.max_hold_days_4h)
        else:
            max_hold_days = float(self.config.max_hold_days_15m)
        generic_indicator = self.generic_indicator_entry_context(action, levels, symbol=symbol, selected=selected)
        strategy_labels: list[str] = []
        if generic_indicator.get("confirmed"):
            label = str(generic_indicator.get("label") or "")
            if label:
                strategy_labels.append(label)
            selected = dict(selected)
            selected["generic_indicator"] = {
                "label": label,
                "trigger": dict(generic_indicator.get("trigger") or {}),
                "required_trigger": generic_indicator.get("required_trigger") or "",
            }
            selected["strategy_labels"] = list(strategy_labels)
        return {
            "horizon": horizon,
            "entry_timeframe": timeframe,
            "max_hold_days": max_hold_days,
            "selected_entry_level": selected,
            "strategy_labels": strategy_labels,
            "candidate_count": len(candidates) + (1 if ma20_candidate else 0) + (1 if daily_candidate else 0),
            "one_d_candidate_count": len(one_d) + (1 if daily_candidate else 0),
            "four_h_candidate_count": len(four_h) + (1 if ma20_candidate else 0),
            "fifteen_m_candidate_count": len(fifteen_m),
            "h4_confirmed_15m_selector": h4_confirmed_15m_selector,
            "h4_confirmed_15m_selector_candidate_count": len(h4_selector_fifteen_m),
            "ma20_4h_candidate": ma20_candidate,
            "daily_ma7_candidate": daily_candidate,
            "h4_right_side": h4_right_side,
            "daily_ma7_trend": daily_trend,
            "bb96_210_second_test_15m": second_test,
            "bb96_210_second_test_candidate": second_test_candidate,
            "generic_indicator": generic_indicator,
            "symbol_bb_entry_profile": profile_filter,
            "countertrend_rebound": countertrend_rebound,
            "macro_trend_context": (
                "daily_short_rebound_long" if countertrend_rebound and action == "open_long"
                else "daily_long_rebound_short" if countertrend_rebound and action == "open_short"
                else ""
            ),
        }

    def structure_risk_plan(self, price: float, action: str, levels: dict[str, Any], *, symbol: str = "") -> dict[str, Any]:
        supports = list(levels.get("all_supports") or levels.get("nearest_supports") or [])
        resistances = list(levels.get("all_resistances") or levels.get("nearest_resistances") or [])
        max_distance = float(self.config.structure_runner_max_distance_pct)
        entry_context = self.entry_context_for_action(action, levels, symbol=symbol)
        horizon = str(entry_context.get("horizon") or "15m_scalp")
        v5_forced_stop_loss_pct = (
            max(0.0, float(self.config.v5_forced_stop_loss_pct or 0.0))
            if self.config.v5_overlay_rules_active_for(symbol) and action in {"open_long", "open_short"}
            else 0.0
        )
        stop_loss_pct = v5_forced_stop_loss_pct if v5_forced_stop_loss_pct > 0 else float(self.config.stop_loss_pct)
        default_stop = price * (1.0 + stop_loss_pct) if action == "open_short" else price * (1.0 - stop_loss_pct)
        default_take_profit = price * (1.0 - self.config.take_profit_pct) if action == "open_short" else price * (1.0 + self.config.take_profit_pct)
        if action == "open_short":
            stop_anchor = resistances[0] if resistances else None
            stop = default_stop
            if stop_anchor and v5_forced_stop_loss_pct <= 0:
                stop = max(stop, float(stop_anchor[1]) * 1.005)
            targets = [item for item in supports if str(item[0]).startswith("bb")]
            target_rows = [
                {"name": str(name), "price": float(value), "distance_pct": round(pct_diff(price, float(value)), 6)}
                for name, value in targets
                if 0 < float(value) < price and abs(pct_diff(price, float(value))) <= max_distance
            ]
            target_rows.sort(key=lambda item: abs(float(item["distance_pct"])))
            tp1 = target_rows[0] if target_rows else {"name": "default_pct", "price": default_take_profit, "distance_pct": round(pct_diff(price, default_take_profit), 6)}
            tp2 = target_rows[min(1, len(target_rows) - 1)] if target_rows else tp1
            runner = target_rows[-1] if target_rows else tp1
            selected_tp = min(default_take_profit, runner["price"])
            staged_targets = self.staged_take_profit_targets(target_rows, action=action, horizon=horizon)
            if staged_targets:
                tp1 = staged_targets[0]
                tp2 = staged_targets[min(1, len(staged_targets) - 1)]
                runner = staged_targets[-1]
                selected_tp = float(runner["price"])
        else:
            stop_anchor = supports[0] if supports else None
            stop = default_stop
            if stop_anchor and v5_forced_stop_loss_pct <= 0:
                stop = min(stop, float(stop_anchor[1]) * 0.995)
            targets = [item for item in resistances if str(item[0]).startswith("bb")]
            target_rows = [
                {"name": str(name), "price": float(value), "distance_pct": round(pct_diff(price, float(value)), 6)}
                for name, value in targets
                if float(value) > price and abs(pct_diff(price, float(value))) <= max_distance
            ]
            target_rows.sort(key=lambda item: abs(float(item["distance_pct"])))
            tp1 = target_rows[0] if target_rows else {"name": "default_pct", "price": default_take_profit, "distance_pct": round(pct_diff(price, default_take_profit), 6)}
            tp2 = target_rows[min(1, len(target_rows) - 1)] if target_rows else tp1
            runner = target_rows[-1] if target_rows else tp1
            selected_tp = max(default_take_profit, runner["price"])
            staged_targets = self.staged_take_profit_targets(target_rows, action=action, horizon=horizon)
            if staged_targets:
                tp1 = staged_targets[0]
                tp2 = staged_targets[min(1, len(staged_targets) - 1)]
                runner = staged_targets[-1]
                selected_tp = float(runner["price"])
        stop_anchor_payload = None
        if stop_anchor:
            stop_anchor_payload = {"name": str(stop_anchor[0]), "price": float(stop_anchor[1]), "distance_pct": round(pct_diff(price, float(stop_anchor[1])), 6)}
        original_structure_stop = stop
        v5_stop_overlay: dict[str, Any] = {}
        if v5_forced_stop_loss_pct > 0:
            v5_stop_overlay = {
                "enabled": False,
                "mode": "forced_hard_stop",
                "forced_stop_loss_pct": round(v5_forced_stop_loss_pct * 100.0, 6),
                "hard_stop_distance_pct": round(abs(pct_diff(price, stop)), 6),
                "note": "V5 forced hard stop; structure anchor and disaster-stop widening are bypassed.",
            }
        elif self.config.v5_overlay_rules_active_for(symbol) and action in {"open_long", "open_short"}:
            stop_multiplier = max(1.0, float(self.config.v5_disaster_stop_multiplier or 1.0))
            stop_distance = abs(price - stop)
            if stop_multiplier > 1.0 and stop_distance > 0:
                stop = price + stop_distance * stop_multiplier if action == "open_short" else price - stop_distance * stop_multiplier
                v5_stop_overlay = {
                    "enabled": True,
                    "mode": "disaster_hard_stop",
                    "disaster_stop_multiplier": stop_multiplier,
                    "original_structure_stop_price": round(original_structure_stop, 8),
                    "original_structure_stop_distance_pct": round(abs(pct_diff(price, original_structure_stop)), 6),
                    "hard_stop_distance_pct": round(abs(pct_diff(price, stop)), 6),
                    "soft_stop_note": "V5 overlay keeps the exchange stop as a wider disaster stop; close confirmation is handled by V5 position management.",
                }
        strategy_labels = [
            str(item)
            for item in (entry_context.get("strategy_labels") or [])
            if str(item or "")
        ]
        return {
            "mode": "bb_structure_runner",
            "action": action,
            "strategy_labels": strategy_labels,
            "strategy_label": strategy_labels[0] if strategy_labels else "",
            "horizon": horizon,
            "entry_timeframe": entry_context.get("entry_timeframe"),
            "max_hold_days": float(entry_context.get("max_hold_days") or 0.0),
            "entry_context": entry_context,
            "fib_structure_confirmation": levels.get("fib_structure_confirmation") or entry_context.get("fib_structure_confirmation") or {},
            "entry_price": round(price, 8),
            "stop_loss_price": round(stop, 8),
            "take_profit_price": round(selected_tp, 8),
            "default_stop_price": round(default_stop, 8),
            "default_take_profit_price": round(default_take_profit, 8),
            "v5_original_structure_stop_price": round(original_structure_stop, 8),
            "v5_forced_stop_loss_pct": round(v5_forced_stop_loss_pct, 8),
            "v5_stop_overlay": v5_stop_overlay,
            "stop_anchor": stop_anchor_payload,
            "tp1": tp1,
            "tp2": tp2,
            "runner": runner,
            "target_ladder": target_rows[:12],
            "staged_take_profit_targets": staged_targets,
            "runner_max_distance_pct": max_distance,
            "structure_entry_scan": levels.get("structure_entry_scan") or {},
        }

    def intent_for_signal(self, signal: V6Signal, state: dict[str, Any]) -> V6OrderIntent | None:
        positions = self.find_positions(state, signal.symbol)
        for position in positions:
            exit_intent = self.exit_intent_if_needed(signal, position)
            if exit_intent:
                return exit_intent
        if self.entry_cooldown_active(state, signal.symbol):
            return None
        if len(self.active_positions(state)) >= self.config.max_positions:
            return None

        if self.config.technical_entry_only_for(signal.symbol):
            return self.intent_for_technical_entry_signal(signal, state, positions)

        if signal.wait_probability > self.config.max_wait_probability_for_entry:
            return None

        if signal.short_probability >= self.config.short_entry_threshold_for(signal.symbol):
            same_side = self.find_position(state, signal.symbol, side="short")
            if same_side:
                return self.scale_in_intent_if_needed(signal, same_side)
            if positions and not self.can_open_opposite_symbol_position():
                return None
            chase_block = self.v5_profit_reentry_guard_block(signal, state, "open_short")
            if chase_block:
                self.apply_entry_structure_block(signal, chase_block)
                return None
            block = self.entry_structure_block(signal, "open_short")
            if block:
                self.apply_entry_structure_block(signal, block)
                return None
            return self.open_intent(signal, "open_short")
        if signal.long_probability >= self.config.long_entry_threshold_for(signal.symbol):
            same_side = self.find_position(state, signal.symbol, side="long")
            if same_side:
                return self.scale_in_intent_if_needed(signal, same_side)
            if positions and not self.can_open_opposite_symbol_position():
                return None
            chase_block = self.v5_profit_reentry_guard_block(signal, state, "open_long")
            if chase_block:
                self.apply_entry_structure_block(signal, chase_block)
                return None
            block = self.entry_structure_block(signal, "open_long")
            if block:
                self.apply_entry_structure_block(signal, block)
                return None
            return self.open_intent(signal, "open_long")
        return None

    def intent_for_technical_entry_signal(
        self,
        signal: V6Signal,
        state: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> V6OrderIntent | None:
        candidates: list[dict[str, Any]] = []
        last_block: dict[str, Any] | None = None
        for action in ("open_short", "open_long"):
            context = self.entry_context_for_action(action, signal.levels or {}, symbol=signal.symbol)
            macro_block = self.macro_direction_entry_block(signal, action, context)
            macro_block = self.maybe_downgrade_entry_block(signal, macro_block)
            if macro_block:
                last_block = macro_block
                continue
            side = "short" if action == "open_short" else "long"
            same_side = self.find_position(state, signal.symbol, side=side)
            if same_side:
                scale_intent = self.scale_in_intent_if_needed(signal, same_side, require_probability=False)
                if scale_intent:
                    candidates.append({
                        "action": action,
                        "intent": scale_intent,
                        "context": (scale_intent.risk_plan or {}).get("entry_context") or {},
                        "rank": (250.0, 0.0, 0.0),
                })
                continue
            if positions and not self.can_open_opposite_symbol_position():
                continue
            chase_block = self.v5_profit_reentry_guard_block(signal, state, action)
            if chase_block:
                last_block = chase_block
                continue
            ready, block, context = self.technical_entry_ready(signal, action, context)
            if ready:
                candidates.append({
                    "action": action,
                    "context": context,
                    "rank": self.technical_entry_rank(signal, action, context),
                })
            elif block:
                last_block = block
        if not candidates:
            if last_block:
                self.apply_entry_structure_block(signal, last_block)
            return None
        candidates.sort(key=lambda item: item["rank"], reverse=True)
        selected = candidates[0]
        if isinstance(selected.get("intent"), V6OrderIntent):
            return selected["intent"]
        context = selected.get("context") if isinstance(selected.get("context"), dict) else {}
        level = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
        reason = (
            f"technical-only entry selected: {selected['action']} "
            f"horizon={context.get('horizon')} level={level.get('name') or '--'}"
        )
        if reason not in signal.reasons:
            signal.reasons.append(reason)
        return self.open_intent(signal, str(selected["action"]))

    def macro_direction_entry_block(self, signal: V6Signal, action: str, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        side_bias = str(signal.side_bias or "neutral").lower()
        context = context if isinstance(context, dict) else self.entry_context_for_action(action, signal.levels or {}, symbol=signal.symbol)
        profile = symbol_bb_entry_profile(signal.symbol)
        if profile.get("allow_4h_profile_direction_override") and str(context.get("horizon") or "") == "4h_trend":
            selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
            h4_right_side = context.get("h4_right_side") if isinstance(context.get("h4_right_side"), dict) else {}
            selected_timeframe = str(selected.get("timeframe") or "")
            if selected_timeframe == "4h" or context.get("h4_confirmed_15m_selector"):
                if action == "open_short" and h4_right_side.get("short_confirmed"):
                    return None
                if action == "open_long" and h4_right_side.get("long_confirmed"):
                    return None
        daily_trend = context.get("daily_ma7_trend") if isinstance(context.get("daily_ma7_trend"), dict) else {}
        macro_source = "probability"
        effective_bias = side_bias
        if effective_bias not in {"short", "long"}:
            if daily_trend.get("short_confirmed"):
                effective_bias = "short"
                macro_source = "daily_ma7_trend_fallback"
            elif daily_trend.get("long_confirmed"):
                effective_bias = "long"
                macro_source = "daily_ma7_trend_fallback"
        expected_action = "open_short" if effective_bias == "short" else "open_long" if effective_bias == "long" else ""
        if not expected_action:
            return {
                "enabled": True,
                "blocked": True,
                "action": action,
                "reason": "macro_direction_neutral",
                "message": "entry blocked: macro direction is neutral; probability and daily MA7 trend do not provide a clear direction",
                "side_bias": side_bias,
                "effective_bias": effective_bias,
                "macro_source": macro_source,
                "long_probability": signal.long_probability,
                "short_probability": signal.short_probability,
                "entry_context": context,
            }
        if action == expected_action:
            return None
        if self.config.countertrend_rebound_enabled and context.get("countertrend_rebound"):
            return None
        return {
            "enabled": True,
            "blocked": True,
            "action": action,
            "reason": "macro_direction_mismatch",
            "message": "entry blocked: technical setup direction conflicts with macro direction and is not a 15M/4H countertrend rebound",
            "side_bias": side_bias,
            "effective_bias": effective_bias,
            "macro_source": macro_source,
            "long_probability": signal.long_probability,
            "short_probability": signal.short_probability,
            "entry_context": context,
        }

    def v5_overlay_entry_block(
        self,
        signal: V6Signal,
        action: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.config.v5_overlay_rules_active_for(signal.symbol):
            return None
        side = "short" if action == "open_short" else "long"
        selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
        selected_name = str(selected.get("name") or "").lower()
        selected_timeframe = str(selected.get("timeframe") or "").lower()
        selected_family = str(selected.get("indicator_family") or selected.get("family") or "").lower()
        selected_period = str(selected.get("period") or "")
        selected_band = str(selected.get("band") or "").lower()
        structure_signal = str(selected.get("structure_signal") or "")
        reason_text = " | ".join(str(item or "") for item in signal.reasons).lower()
        if self.config.v5_block_unclean_15m_ma20_initial_entry and "15m ma20 trend guard is broken or mixed" in reason_text:
            return {
                "enabled": True,
                "blocked": True,
                "watchable": True,
                "action": action,
                "side": side,
                "reason": "v5_wait_clean_15m_ma20_pullback",
                "message": "entry waiting: V5 15M MA20 trend guard is broken/mixed; wait for a clean 15M pullback/retest before opening",
                "entry_context": context,
                "selected_entry_level": selected,
            }
        if self.config.v5_block_mixed_structure_initial_entry and "bb structure entry scan is mixed" in reason_text:
            return {
                "enabled": True,
                "blocked": True,
                "watchable": True,
                "action": action,
                "side": side,
                "reason": "v5_wait_mixed_structure_resolution",
                "message": "entry waiting: V5 BB structure scan is mixed; keep symbol on watchlist until 15M direction resolves",
                "entry_context": context,
                "selected_entry_level": selected,
            }
        if self.config.v5_entry_location_guard_enabled:
            if action == "open_long" and (selected_band == "upper" or structure_signal == "long_upper_breakout"):
                return {
                    "enabled": True,
                    "blocked": True,
                    "watchable": True,
                    "action": action,
                    "side": side,
                    "reason": "v5_long_upper_band_breakout_wait_retest",
                    "message": "entry waiting: V5 long selected an upper-band breakout; wait for pullback/retest instead of chasing the upper band",
                    "entry_context": context,
                    "selected_entry_level": selected,
                }
            if action == "open_short" and (selected_band == "lower" or structure_signal == "short_lower_breakdown_retest"):
                return {
                    "enabled": True,
                    "blocked": True,
                    "watchable": True,
                    "action": action,
                    "side": side,
                    "reason": "v5_short_lower_band_breakdown_wait_retest",
                    "message": "entry waiting: V5 short selected a lower-band breakdown; wait for rebound/retest instead of chasing the lower band",
                    "entry_context": context,
                    "selected_entry_level": selected,
                }
            ma20_15m = None
            for source_key in ("all_supports", "all_resistances"):
                for row in (signal.levels or {}).get(source_key) or []:
                    if isinstance(row, (list, tuple)) and len(row) >= 2 and str(row[0]).lower() in {"ma20_15m", "sma20_15m"}:
                        try:
                            ma20_15m = float(row[1] or 0.0)
                        except (TypeError, ValueError):
                            ma20_15m = None
                        break
                if ma20_15m:
                    break
            if ma20_15m and ma20_15m > 0 and signal.price and float(signal.price) > 0:
                distance_from_ma20 = pct_diff(float(signal.price), ma20_15m)
                max_extension = float(self.config.v5_entry_15m_ma20_extension_pct)
                if action == "open_long" and distance_from_ma20 >= max_extension:
                    return {
                        "enabled": True,
                        "blocked": True,
                        "watchable": True,
                        "action": action,
                        "side": side,
                        "reason": "v5_long_wait_15m_ma20_pullback",
                        "message": "entry waiting: V5 long is extended too far above 15M MA20; wait for pullback before opening",
                        "entry_context": context,
                        "selected_entry_level": selected,
                        "ma20_15m": round(ma20_15m, 8),
                        "distance_pct": round(distance_from_ma20, 6),
                        "max_extension_pct": max_extension,
                    }
                if action == "open_short" and distance_from_ma20 <= -max_extension:
                    return {
                        "enabled": True,
                        "blocked": True,
                        "watchable": True,
                        "action": action,
                        "side": side,
                        "reason": "v5_short_wait_15m_ma20_retest",
                        "message": "entry waiting: V5 short is extended too far below 15M MA20; wait for retest before opening",
                        "entry_context": context,
                        "selected_entry_level": selected,
                        "ma20_15m": round(ma20_15m, 8),
                        "distance_pct": round(distance_from_ma20, 6),
                        "max_extension_pct": max_extension,
                    }
            boundary = self.entry_boundary_chase_candidate(signal, side)
            if boundary and float(boundary.get("abs_distance_pct") or 999.0) <= float(self.config.v5_entry_location_guard_near_pct):
                if action == "open_long":
                    allowed_structures = {
                        "long_lower_support_rebound",
                        "long_bb96_210_lower_second_test_rebound",
                    }
                    reason = "v5_long_wait_below_15m_resistance"
                    message = "entry waiting: V5 long signal is too close to 15M mid/upper resistance; track price until pullback/support entry is available"
                else:
                    allowed_structures = {
                        "short_upper_rejection",
                        "short_bb96_210_upper_second_test_rejection",
                    }
                    reason = "v5_short_wait_above_15m_support"
                    message = "entry waiting: V5 short signal is too close to 15M mid/lower support; track price until retest/pressure entry is available"
                if structure_signal not in allowed_structures:
                    return {
                        "enabled": True,
                        "blocked": True,
                        "watchable": True,
                        "action": action,
                        "side": side,
                        "reason": reason,
                        "message": message,
                        "entry_context": context,
                        "selected_entry_level": selected,
                        "boundary": boundary,
                        "near_pct": float(self.config.v5_entry_location_guard_near_pct),
                    }
        if self.config.v5_block_ma7_15m_single_entry:
            is_ma7_15m = (
                selected_name in {"ma7_15m", "sma7_15m"}
                or (selected_family in {"ma", "sma"} and selected_period == "7" and selected_timeframe == "15m")
            )
            if is_ma7_15m:
                return {
                    "enabled": True,
                    "blocked": True,
                    "watchable": True,
                    "action": action,
                    "side": side,
                    "reason": "v5_ma7_15m_single_entry_blocked",
                    "message": "entry blocked: V5 small-coin overlay no longer allows MA7 15M as the selected entry trigger",
                    "entry_context": context,
                    "selected_entry_level": selected,
                }
        if self.config.v5_block_mixed_avoid_low_quality_entry:
            mixed_scan = "bb structure entry scan is mixed" in reason_text
            avoid_chasing = "avoid chasing" in reason_text
            low_quality = "model quality score is low" in reason_text
            if mixed_scan and (avoid_chasing or low_quality):
                return {
                    "enabled": True,
                    "blocked": True,
                    "watchable": True,
                    "action": action,
                    "side": "short" if action == "open_short" else "long",
                    "reason": "v5_mixed_chase_low_quality_blocked",
                    "message": "entry blocked: V5 small-coin overlay detected mixed BB structure with chase/low-quality risk",
                    "entry_context": context,
                    "mixed_scan": mixed_scan,
                    "avoid_chasing": avoid_chasing,
                    "low_quality": low_quality,
                }
        return None

    def technical_entry_ready(
        self,
        signal: V6Signal,
        action: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any]]:
        context = context if isinstance(context, dict) else self.entry_context_for_action(action, signal.levels or {}, symbol=signal.symbol)
        selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
        if not selected:
            return False, {
                "enabled": True,
                "blocked": True,
                "action": action,
                "reason": "technical_entry_only_no_indicator_level",
                "message": "entry blocked: technical-only symbol requires BB/MA/right-side indicator entry level",
                "entry_context": context,
            }, context
        v5_block = self.v5_overlay_entry_block(signal, action, context)
        if v5_block:
            downgraded = self.maybe_downgrade_entry_block(signal, v5_block)
            if downgraded is None and self.transition_gate_policy.should_downgrade(v5_block):
                return True, None, context
            return False, downgraded, context
        block = self.entry_structure_block(signal, action, context)
        if block:
            return False, block, context
        if context.get("countertrend_rebound"):
            message = (
                f"countertrend rebound entry selected: {action} "
                f"context={context.get('macro_trend_context')} level={selected.get('name') or '--'}"
            )
            if message not in signal.reasons:
                signal.reasons.append(message)
        if context.get("horizon") in {"4h_trend", "1d_trend"}:
            return True, None, context
        if self.directional_entry_level_touched(signal, action):
            return True, None, context
        side = "short" if action == "open_short" else "long"
        if self.scale_in_trigger_candidate(
            signal,
            side,
            near_pct=float(self.config.entry_structure_rejection_near_pct),
            min_lines=max(1, int(self.config.entry_structure_min_rejection_lines)),
        ):
            return True, None, context
        unconfirmed_block = {
            "enabled": True,
            "blocked": True,
            "action": action,
            "reason": "technical_entry_only_unconfirmed_15m_trigger",
            "message": "entry blocked: technical-only 15M setup requires touched BB level or 3-line confirmation",
            "entry_context": context,
        }
        downgraded = self.maybe_downgrade_entry_block(signal, unconfirmed_block)
        if downgraded is None and self.transition_gate_policy.should_downgrade(unconfirmed_block):
            return True, None, context
        return False, downgraded, context

    def technical_entry_rank(self, signal: V6Signal, action: str, context: dict[str, Any]) -> tuple[float, float, float]:
        horizon_score = {
            "1d_trend": 300.0,
            "4h_trend": 200.0,
            "15m_rebound": 120.0,
            "15m_scalp": 100.0,
        }.get(str(context.get("horizon") or ""), 0.0)
        selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
        distance_score = -abs(float(selected.get("distance_pct") or 999.0))
        return (horizon_score, distance_score, 0.0)

    def can_open_opposite_symbol_position(self) -> bool:
        try:
            return self.client.current_position_mode() == "hedge"
        except Exception:
            return False

    def ma20_trend_entry_block(self, signal: V6Signal, action: str) -> dict[str, Any] | None:
        if not self.config.ma20_trend_guard_enabled:
            return None
        if action not in {"open_short", "open_long"}:
            return None
        fifteen_m = signal.fifteen_m if isinstance(signal.fifteen_m, dict) else {}
        trend = fifteen_m.get("ma20_trend") if isinstance(fifteen_m.get("ma20_trend"), dict) else {}
        if not trend.get("ready"):
            return None
        if action == "open_short" and trend.get("bullish_trend_alive"):
            reason = "short_blocked_15m_ma20_trend_alive"
            message = "entry blocked: 15M MA20 held for 2 candles; short-term uptrend remains active"
        elif action == "open_long" and trend.get("bearish_trend_alive"):
            reason = "long_blocked_15m_ma20_downtrend_alive"
            message = "entry blocked: 15M MA20 rejected for 2 candles; short-term downtrend remains active"
        else:
            return None
        return {
            "enabled": True,
            "blocked": True,
            "action": action,
            "side": "short" if action == "open_short" else "long",
            "reason": reason,
            "message": message,
            "ma20_trend": trend,
        }

    def trend_big_entry_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any]) -> dict[str, Any] | None:
        if not self.config.trend_big_right_side_only:
            return None
        horizon = str(entry_context.get("horizon") or "")
        side = "short" if action == "open_short" else "long"
        if horizon == "4h_trend":
            state = entry_context.get("h4_right_side") if isinstance(entry_context.get("h4_right_side"), dict) else {}
            key = "short_confirmed" if side == "short" else "long_confirmed"
            if not state.get(key):
                return {
                    "enabled": True,
                    "blocked": True,
                    "action": action,
                    "side": side,
                    "reason": "trend_big_blocked_without_4h_right_side",
                    "message": "entry blocked: 4H trend big order requires right-side confirmation",
                    "h4_right_side": state,
                }
        if horizon == "1d_trend":
            state = entry_context.get("daily_ma7_trend") if isinstance(entry_context.get("daily_ma7_trend"), dict) else {}
            key = "short_confirmed" if side == "short" else "long_confirmed"
            if not state.get(key):
                return {
                    "enabled": True,
                    "blocked": True,
                    "action": action,
                    "side": side,
                    "reason": "trend_big_blocked_without_1d_ma7_rebreak",
                    "message": "entry blocked: 1D trend big order requires MA7 breakout-retest-rebreak confirmation",
                    "daily_ma7_trend": state,
                }
        return None

    def h4_lower_chase_short_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any]) -> dict[str, Any] | None:
        if not self.config.h4_lower_chase_block_enabled or action != "open_short":
            return None
        price = float(signal.price or 0.0)
        if price <= 0:
            return None
        h4_right_side = entry_context.get("h4_right_side") if isinstance(entry_context.get("h4_right_side"), dict) else {}
        symbol_upper = str(signal.symbol or "").upper()
        strict_confirmed_chase_block = symbol_upper == "ZECUSDT"
        if h4_right_side.get("short_confirmed") and strict_confirmed_chase_block:
            h4 = signal.h4 if isinstance(signal.h4, dict) else {}
            mid_retest = signal.mid_retest if isinstance(signal.mid_retest, dict) else {}
            ma20_distance = float(h4.get("price_vs_ma20_pct") or 0.0)
            close_position = h4.get("close_position_20bars")
            close_position_value = float(close_position) if close_position is not None else 1.0
            far_below_count = int(mid_retest.get("far_below_count") or 0)
            triggers: list[str] = []
            if close_position_value <= 0.30:
                triggers.append("near_4h_range_low")
            if ma20_distance <= -1.5:
                triggers.append("below_4h_ma20_by_1_5pct")
            if far_below_count >= 3:
                triggers.append("far_below_multiple_4h_mids")
            if len(triggers) >= 2:
                return {
                    "enabled": True,
                    "blocked": True,
                    "action": action,
                    "side": "short",
                    "reason": "zec_short_blocked_confirmed_4h_low_chase",
                    "message": "entry blocked: ZEC 4H short is confirmed, but price is already low/away from 4H MA20; wait for pressure retest instead of chasing short",
                    "symbol": symbol_upper,
                    "triggers": triggers,
                    "required_trigger_count": 2,
                    "close_position_20bars": round(close_position_value, 6),
                    "price_vs_4h_ma20_pct": round(ma20_distance, 6),
                    "far_below_4h_mid_count": far_below_count,
                    "h4_right_side": h4_right_side,
                }
        if h4_right_side.get("short_confirmed"):
            return None

        rows: dict[int, dict[str, float]] = {}
        for name, value in list((signal.levels or {}).get("all_supports") or []) + list((signal.levels or {}).get("all_resistances") or []):
            parsed = self.parse_shared_bb_name(str(name))
            if not parsed or parsed.get("timeframe") != "4h":
                continue
            band = str(parsed.get("band") or "")
            if band not in {"upper", "mid", "lower"}:
                continue
            period = int(parsed.get("period") or 0)
            if period <= 0:
                continue
            rows.setdefault(period, {})[band] = float(value or 0.0)

        near_pct = max(0.0, float(self.config.h4_lower_chase_near_pct))
        position_max = float(self.config.h4_lower_chase_position_max)
        min_lines = max(1, int(self.config.h4_lower_chase_min_lines))
        candidates: list[dict[str, Any]] = []
        complete_rows: list[dict[str, Any]] = []
        for period, values in rows.items():
            upper = float(values.get("upper") or 0.0)
            lower = float(values.get("lower") or 0.0)
            mid = float(values.get("mid") or 0.0)
            if upper <= 0 or lower <= 0 or upper <= lower:
                continue
            position = (price - lower) / (upper - lower)
            lower_distance = pct_diff(price, lower)
            row = {
                "name": f"bb{period}_lower_4h",
                "period": period,
                "lower": round(lower, 8),
                "mid": round(mid, 8),
                "upper": round(upper, 8),
                "position": round(position, 6),
                "distance_to_lower_pct": round(lower_distance, 6),
                "abs_distance_to_lower_pct": round(abs(lower_distance), 6),
            }
            complete_rows.append(row)
            if abs(lower_distance) <= near_pct or (-0.20 <= position <= position_max):
                candidates.append(row)

        if not candidates:
            return None
        candidates.sort(key=lambda item: (float(item.get("abs_distance_to_lower_pct") or 999.0), int(item.get("period") or 0)))
        if len(candidates) < min_lines and float(candidates[0].get("abs_distance_to_lower_pct") or 999.0) > near_pct:
            return None
        return {
            "enabled": True,
            "blocked": True,
            "action": action,
            "side": "short",
            "reason": "short_blocked_4h_lower_band_chase",
            "message": "entry blocked: price is standing near 4H BB lower support; do not chase short without 4H short right-side confirmation",
            "near_pct": near_pct,
            "position_max": position_max,
            "required_lines": min_lines,
            "matched_lines": candidates[:8],
            "complete_4h_bb_count": len(complete_rows),
            "h4_right_side": h4_right_side,
        }

    def h4_short_pressure_confirmation(self, signal: V6Signal) -> dict[str, Any]:
        price = float(signal.price or 0.0)
        if price <= 0:
            return {"confirmed": False, "matched_lines": []}
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        near_pct = max(
            float(self.config.mid_retest_near_pct),
            float(self.config.ma20_4h_pressure_near_pct),
            float(self.config.entry_structure_rejection_near_pct),
        )
        matched: list[dict[str, Any]] = []
        ma20 = levels.get("ma20_4h_pressure") if isinstance(levels.get("ma20_4h_pressure"), dict) else {}
        if ma20.get("pressure_active"):
            matched.append({
                "name": ma20.get("name") or "ma20_4h",
                "price": round(float(ma20.get("value") or 0.0), 8),
                "distance_pct": round(float(ma20.get("distance_pct") or 0.0), 6),
                "timeframe": "4h",
                "period": 20,
                "band": "ma",
                "source": "4h_ma20_pressure",
            })
        for row in levels.get("all_resistances") or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            name = str(row[0])
            value = float(row[1] or 0.0)
            parsed = self.parse_shared_bb_name(name)
            if not parsed or parsed.get("timeframe") != "4h" or parsed.get("band") not in {"mid", "upper"}:
                continue
            if value <= 0:
                continue
            distance = pct_diff(price, value)
            if abs(distance) > near_pct:
                continue
            matched.append({
                "name": name,
                "price": round(value, 8),
                "distance_pct": round(distance, 6),
                "abs_distance_pct": round(abs(distance), 6),
                "timeframe": "4h",
                "period": parsed["period"],
                "band": parsed["band"],
                "source": "4h_bb_mid_upper_pressure",
            })
        matched.sort(key=lambda item: (float(item.get("abs_distance_pct") or abs(float(item.get("distance_pct") or 999.0))), int(item.get("period") or 0)))
        return {
            "confirmed": bool(matched),
            "near_pct": near_pct,
            "matched_lines": matched[:8],
        }

    def short_15m_entry_quality_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any]) -> dict[str, Any] | None:
        if action != "open_short":
            return None
        selected = entry_context.get("selected_entry_level") if isinstance(entry_context.get("selected_entry_level"), dict) else {}
        if str(selected.get("timeframe") or "") != "15m":
            return None

        h4_right_side = entry_context.get("h4_right_side") if isinstance(entry_context.get("h4_right_side"), dict) else {}
        if h4_right_side.get("long_confirmed"):
            return {
                "enabled": True,
                "blocked": True,
                "action": action,
                "side": "short",
                "reason": "short_15m_blocked_by_4h_right_side_long",
                "message": "entry blocked: 4H right-side long confirmation is active; ordinary 15M short is forbidden",
                "selected_entry_level": selected,
                "h4_right_side": h4_right_side,
            }

        profile_filter = entry_context.get("symbol_bb_entry_profile") if isinstance(entry_context.get("symbol_bb_entry_profile"), dict) else {}
        if profile_filter.get("enabled") and selected:
            return None

        second_test = (signal.levels or {}).get("bb96_210_second_test_15m") if isinstance(signal.levels, dict) else {}
        upper_second_test = second_test.get("upper") if isinstance(second_test, dict) and isinstance(second_test.get("upper"), dict) else {}
        if upper_second_test.get("confirmed"):
            return None

        required_lines = max(3, int(self.config.entry_structure_min_rejection_lines))
        rejection = self.scale_in_trigger_candidate(
            signal,
            "short",
            near_pct=float(self.config.entry_structure_rejection_near_pct),
            min_lines=required_lines,
        )
        h4_pressure = self.h4_short_pressure_confirmation(signal)
        if rejection or h4_pressure.get("confirmed") or h4_right_side.get("short_confirmed"):
            return None

        parsed = self.parse_shared_bb_name(str(selected.get("name") or ""))
        symbol = str(signal.symbol or "").upper()
        fragile_single_line = (
            symbol == "HYPEUSDT"
            and parsed
            and parsed.get("timeframe") == "15m"
            and parsed.get("band") == "upper"
            and int(parsed.get("period") or 0) in {96, 120}
        )
        return {
            "enabled": True,
            "blocked": True,
            "action": action,
            "side": "short",
            "reason": "short_15m_requires_three_line_or_4h_pressure",
            "message": (
                "entry blocked: 15M short requires 3-line 15M pressure confluence or 4H MA20/BB mid-upper pressure; "
                "single BB96/120 upper rejection is not enough for HYPE"
                if fragile_single_line
                else "entry blocked: 15M short requires 3-line 15M pressure confluence or 4H MA20/BB mid-upper pressure"
            ),
            "selected_entry_level": selected,
            "required_15m_pressure_lines": required_lines,
            "three_line_rejection": rejection,
            "h4_pressure_confirmation": h4_pressure,
            "h4_right_side": h4_right_side,
            "fragile_single_line_symbol_rule": fragile_single_line,
        }

    def symbol_bb_entry_profile_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any]) -> dict[str, Any] | None:
        profile = entry_context.get("symbol_bb_entry_profile") if isinstance(entry_context.get("symbol_bb_entry_profile"), dict) else {}
        if not profile.get("enabled"):
            return None
        selected = entry_context.get("selected_entry_level") if isinstance(entry_context.get("selected_entry_level"), dict) else {}
        symbol = str(signal.symbol or "").upper()

        def block(reason: str, message: str) -> dict[str, Any]:
            return {
                "enabled": True,
                "blocked": True,
                "action": action,
                "side": "short" if action == "open_short" else "long",
                "reason": reason,
                "message": message,
                "symbol": symbol,
                "selected_entry_level": selected,
                "symbol_bb_entry_profile": profile,
                "entry_context": entry_context,
            }

        if not selected:
            return block(
                "symbol_bb_entry_profile_no_allowed_level",
                "entry blocked: this symbol only trades its backtested high-winrate BB/MA profile; no allowed indicator level is active",
            )

        selected_rule = bb_profile_rule(selected)
        blocked_candidates = profile.get("blocked_candidates") if isinstance(profile.get("blocked_candidates"), list) else []
        blocked_rules = {bb_profile_rule(item) for item in blocked_candidates if isinstance(item, dict)}
        if selected_rule in blocked_rules:
            return block(
                "symbol_bb_entry_profile_selected_level_blocked",
                "entry blocked: selected indicator level is explicitly blocked by this symbol's BB/MA profile",
            )

        allowed_rules = profile.get("allowed_rules") if isinstance(profile.get("allowed_rules"), list) else []
        allowed_rule_set = {bb_profile_rule(item) for item in allowed_rules if isinstance(item, dict)}
        if selected_rule in allowed_rule_set:
            return None

        synthetic = profile.get("synthetic_second_test_candidate_added") if isinstance(profile.get("synthetic_second_test_candidate_added"), dict) else {}
        if synthetic and bb_profile_rule(synthetic) == selected_rule:
            return None

        for candidate_key in ("ma20_4h_candidate", "daily_ma7_candidate", "bb96_210_second_test_candidate"):
            candidate = entry_context.get(candidate_key) if isinstance(entry_context.get(candidate_key), dict) else {}
            if candidate and bb_profile_rule(candidate) == selected_rule:
                return None

        if allowed_rule_set:
            return block(
                "symbol_bb_entry_profile_selected_level_not_allowed",
                "entry blocked: selected indicator level is not in this symbol's backtested high-winrate BB/MA profile",
            )

        return None

    def bb96_210_second_test_opposite_block(self, signal: V6Signal, action: str) -> dict[str, Any] | None:
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        state = levels.get("bb96_210_second_test_15m") if isinstance(levels.get("bb96_210_second_test_15m"), dict) else {}
        upper = state.get("upper") if isinstance(state.get("upper"), dict) else {}
        lower = state.get("lower") if isinstance(state.get("lower"), dict) else {}
        if action == "open_short" and lower.get("confirmed"):
            return {
                "enabled": True,
                "blocked": True,
                "action": action,
                "side": "short",
                "reason": "short_blocked_by_15m_bb96_210_lower_second_test_support",
                "message": "entry blocked: 15M BB96/BB210 lower overlap second-test support is confirmed; do not short into confirmed support",
                "bb96_210_second_test_15m": state,
            }
        if action == "open_long" and upper.get("confirmed"):
            return {
                "enabled": True,
                "blocked": True,
                "action": action,
                "side": "long",
                "reason": "long_blocked_by_15m_bb96_210_upper_second_test_pressure",
                "message": "entry blocked: 15M BB96/BB210 upper overlap second-test pressure is confirmed; do not chase long into confirmed pressure",
                "bb96_210_second_test_15m": state,
            }
        return None

    def generic_indicator_entry_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any]) -> dict[str, Any] | None:
        generic = entry_context.get("generic_indicator") if isinstance(entry_context.get("generic_indicator"), dict) else {}
        if not generic.get("enabled") or not generic.get("applicable"):
            return None
        if generic.get("confirmed"):
            return None
        return {
            "enabled": True,
            "blocked": True,
            "action": action,
            "side": "short" if action == "open_short" else "long",
            "reason": "generic_indicator_1m_ma105_ma300_not_confirmed",
            "message": "entry blocked: generic indicator test requires 1M MA105/MA300 trigger to confirm the 15M setup",
            "strategy_label": generic.get("label") or self.config.generic_indicator_label,
            "generic_indicator": generic,
            "entry_context": entry_context,
        }

    def four_h_entry_15m_selector_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any]) -> dict[str, Any] | None:
        if str(entry_context.get("horizon") or "") != "4h_trend":
            return None
        if not self.config.four_h_entry_requires_15m_selector_for(signal.symbol):
            return None

        side = "short" if action == "open_short" else "long"
        fifteen_m = signal.fifteen_m if isinstance(signal.fifteen_m, dict) else {}
        trend = fifteen_m.get("ma20_trend") if isinstance(fifteen_m.get("ma20_trend"), dict) else {}
        ma20_confirmed = bool(
            trend.get("bearish_trend_alive") if side == "short" else trend.get("bullish_trend_alive")
        )

        levels = signal.levels if isinstance(signal.levels, dict) else {}
        scan = levels.get("structure_entry_scan") if isinstance(levels.get("structure_entry_scan"), dict) else {}
        candidate_key = "short_candidates" if side == "short" else "long_candidates"
        side_candidates = [
            dict(item)
            for item in (scan.get(candidate_key) or [])
            if isinstance(item, dict) and str(item.get("timeframe") or "") == "15m"
        ]

        second_test = levels.get("bb96_210_second_test_15m") if isinstance(levels.get("bb96_210_second_test_15m"), dict) else {}
        zone_key = "upper" if side == "short" else "lower"
        second_zone = second_test.get(zone_key) if isinstance(second_test.get(zone_key), dict) else {}
        second_test_confirmed = bool(second_zone.get("confirmed"))

        if ma20_confirmed or side_candidates or second_test_confirmed:
            return None

        return {
            "enabled": True,
            "blocked": True,
            "action": action,
            "side": side,
            "reason": "four_h_trend_waiting_for_15m_selector",
            "message": "entry blocked: 4H signal defines the profit zone; 15M MA20/BB selector has not confirmed direction yet",
            "horizon": "4h_trend",
            "ma20_trend": trend,
            "fifteen_m_candidate_count": len(side_candidates),
            "bb96_210_second_test_15m": second_test,
            "entry_context": entry_context,
        }

    def fib_structure_confirmation_block(
        self,
        signal: V6Signal,
        action: str,
        entry_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.config.fib_structure_confirmation_enabled:
            return None
        if action not in {"open_short", "open_long"}:
            return None
        if str(entry_context.get("horizon") or "") == "1d_trend":
            return None
        selected = entry_context.get("selected_entry_level") if isinstance(entry_context.get("selected_entry_level"), dict) else {}
        if not selected:
            return None
        structure_signal = str(selected.get("structure_signal") or "")
        point_structure_signals = {
            "long_lower_support_rebound",
            "long_mid_support_retest",
            "long_ma_support_retest",
            "long_4h_ma20_support_retest",
            "long_bb96_210_lower_second_test_rebound",
            "short_mid_resistance_retest",
            "short_upper_rejection",
            "short_ma_resistance_retest",
            "short_4h_ma20_pressure_retest",
            "short_bb96_210_upper_second_test_rejection",
        }
        if structure_signal not in point_structure_signals:
            return None

        price = float(signal.price or 0.0)
        selected_price = float(selected.get("price") or 0.0)
        if price <= 0 or selected_price <= 0:
            return None
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        fib_state = levels.get("fib_structure_4h") if isinstance(levels.get("fib_structure_4h"), dict) else {}
        if not fib_state.get("ready"):
            return None

        near_pct = max(0.1, float(self.config.fib_structure_confirmation_near_pct))
        side = "short" if action == "open_short" else "long"
        fib_levels = [dict(item) for item in (fib_state.get("levels") or []) if isinstance(item, dict)]
        if side == "long":
            side_levels = [
                item for item in fib_levels
                if float(item.get("price") or 0.0) <= price * (1.0 + near_pct / 100.0)
            ]
        else:
            side_levels = [
                item for item in fib_levels
                if float(item.get("price") or 0.0) >= price * (1.0 - near_pct / 100.0)
            ]
        matched: list[dict[str, Any]] = []
        for item in side_levels:
            fib_price = float(item.get("price") or 0.0)
            if fib_price <= 0:
                continue
            distance_to_selected = pct_diff(selected_price, fib_price)
            row = dict(item)
            row["selected_level_distance_pct"] = round(distance_to_selected, 6)
            row["selected_level_abs_distance_pct"] = round(abs(distance_to_selected), 6)
            if abs(distance_to_selected) <= near_pct:
                matched.append(row)
        matched.sort(key=lambda item: float(item.get("selected_level_abs_distance_pct") or 999.0))
        if matched:
            confirmation = {
                "enabled": True,
                "confirmed": True,
                "near_pct": near_pct,
                "side": side,
                "structure_signal": structure_signal,
                "selected_entry_level": {
                    "name": selected.get("name") or "",
                    "price": round(selected_price, 8),
                    "timeframe": selected.get("timeframe") or "",
                    "band": selected.get("band") or "",
                    "period": selected.get("period") or 0,
                },
                "matched_fib_levels": matched[:3],
            }
            selected = dict(selected)
            selected["fib_structure_confirmation"] = confirmation
            entry_context["selected_entry_level"] = selected
            entry_context["fib_structure_confirmation"] = confirmation
            levels["fib_structure_confirmation"] = confirmation
            signal.levels = levels
            best_match = matched[0]
            message = (
                f"4H Fibonacci confluence confirmed for {structure_signal}: "
                f"{selected.get('name') or '--'} near {best_match.get('name') or '--'}"
            )
            if message not in signal.reasons:
                signal.reasons.append(message)
            return None

        side_levels.sort(key=lambda item: abs(pct_diff(selected_price, float(item.get("price") or 0.0))) if float(item.get("price") or 0.0) > 0 else 999.0)
        return {
            "enabled": True,
            "blocked": True,
            "watchable": True,
            "action": action,
            "side": side,
            "reason": "fib_structure_confirmation_missing",
            "message": "entry blocked: point-based BB/MA structure entry requires nearby 4H Fibonacci confluence",
            "near_pct": near_pct,
            "structure_signal": structure_signal,
            "selected_entry_level": selected,
            "nearest_fib_candidates": side_levels[:5],
            "fib_structure_4h": {
                "ready": True,
                "timeframe": "4h",
                "lookback": fib_state.get("lookback"),
                "swing_high": fib_state.get("swing_high"),
                "swing_low": fib_state.get("swing_low"),
                "nearest_support": fib_state.get("nearest_support") or {},
                "nearest_resistance": fib_state.get("nearest_resistance") or {},
            },
            "entry_context": entry_context,
        }

    def entry_structure_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.config.entry_structure_filter_enabled:
            return None
        if action not in {"open_short", "open_long"}:
            return None
        side = "short" if action == "open_short" else "long"
        profile = symbol_bb_entry_profile(signal.symbol)
        ignore_15m_entry_blocks = bool(profile.get("ignore_15m_entry_blocks"))
        entry_context = entry_context if isinstance(entry_context, dict) else self.entry_context_for_action(action, signal.levels or {}, symbol=signal.symbol)
        freshness_block = self.market_data_freshness_entry_block(signal, action)
        if freshness_block:
            return self.maybe_downgrade_entry_block(signal, freshness_block)
        smart_signal_block = self.smart_signal_entry_block(signal, action, entry_context)
        if smart_signal_block:
            return self.maybe_downgrade_entry_block(signal, smart_signal_block)
        ma20_block = None if ignore_15m_entry_blocks else self.ma20_trend_entry_block(signal, action)
        if ma20_block:
            return self.maybe_downgrade_entry_block(signal, ma20_block)
        daily_ma7_short_reversal_block = self.daily_ma7_short_reversal_entry_block(signal, action, entry_context)
        if daily_ma7_short_reversal_block:
            return self.maybe_downgrade_entry_block(signal, daily_ma7_short_reversal_block)
        daily_ma20_long_reversal_block = self.daily_ma20_long_reversal_entry_block(signal, action, entry_context)
        if daily_ma20_long_reversal_block:
            return self.maybe_downgrade_entry_block(signal, daily_ma20_long_reversal_block)
        profile_block = self.symbol_bb_entry_profile_block(signal, action, entry_context)
        if profile_block:
            return self.maybe_downgrade_entry_block(signal, profile_block)
        trend_block = self.trend_big_entry_block(signal, action, entry_context)
        if trend_block:
            return self.maybe_downgrade_entry_block(signal, trend_block)
        h4_lower_block = self.h4_lower_chase_short_block(signal, action, entry_context)
        if h4_lower_block:
            return self.maybe_downgrade_entry_block(signal, h4_lower_block)
        second_test_opposite_block = None if ignore_15m_entry_blocks else self.bb96_210_second_test_opposite_block(signal, action)
        if second_test_opposite_block:
            return self.maybe_downgrade_entry_block(signal, second_test_opposite_block)
        generic_indicator_block = self.generic_indicator_entry_block(signal, action, entry_context)
        if generic_indicator_block:
            return self.maybe_downgrade_entry_block(signal, generic_indicator_block)
        four_h_selector_block = self.four_h_entry_15m_selector_block(signal, action, entry_context)
        if four_h_selector_block:
            return self.maybe_downgrade_entry_block(signal, four_h_selector_block)
        short_15m_quality_block = self.short_15m_entry_quality_block(signal, action, entry_context)
        if short_15m_quality_block:
            return self.maybe_downgrade_entry_block(signal, short_15m_quality_block)
        fib_structure_block = self.fib_structure_confirmation_block(signal, action, entry_context)
        if fib_structure_block:
            return self.maybe_downgrade_entry_block(signal, fib_structure_block)
        if entry_context.get("horizon") in {"4h_trend", "1d_trend"}:
            confluence_hold = self.indicator_confluence_hold(signal, side)
            return self.maybe_downgrade_entry_block(signal, confluence_hold)
        confluence_hold = self.indicator_confluence_hold(signal, side)
        if confluence_hold:
            return self.maybe_downgrade_entry_block(signal, confluence_hold)
        if self.directional_entry_level_touched(signal, action):
            return None
        boundary = self.entry_boundary_chase_candidate(signal, side)
        if not boundary:
            return None
        rejection = self.scale_in_trigger_candidate(
            signal,
            side,
            near_pct=float(self.config.entry_structure_rejection_near_pct),
            min_lines=max(1, int(self.config.entry_structure_min_rejection_lines)),
        )
        if rejection:
            return None
        if side == "short":
            reason = "short_low_chase_without_15m_rejection"
            message = "entry blocked: 15M mid/lower support is near, but no 3-line 15M mid/upper rejection; avoid low-side chase short"
        else:
            reason = "long_high_chase_without_15m_reclaim"
            message = "entry blocked: 15M mid/upper resistance is near, but no 3-line 15M mid/lower reclaim; avoid high-side chase long"
        return self.maybe_downgrade_entry_block(signal, {
            "enabled": True,
            "blocked": True,
            "action": action,
            "side": side,
            "reason": reason,
            "message": message,
            "boundary": boundary,
            "required_rejection_lines": max(1, int(self.config.entry_structure_min_rejection_lines)),
            "rejection_near_pct": float(self.config.entry_structure_rejection_near_pct),
            "boundary_near_pct": float(self.config.entry_structure_boundary_near_pct),
        })

    def directional_entry_level_touched(self, signal: V6Signal, action: str) -> dict[str, Any] | None:
        context = self.entry_context_for_action(action, signal.levels or {}, symbol=signal.symbol)
        selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
        if not selected:
            return None
        structure_signal = str(selected.get("structure_signal") or "")
        level_price = float(selected.get("price") or 0.0)
        if level_price <= 0:
            return None
        if structure_signal in {"long_ma_reclaim_cross", "short_ma_break_cross"} and selected.get("source") == "ma_cross":
            row = dict(selected)
            row["touch_price"] = round(float(selected.get("current_close") or signal.price or level_price), 8)
            row["entry_context"] = context
            return row
        recent = (
            (signal.indicators or {})
            .get("recent_extremes", {})
            .get("1m", {})
        )
        if not isinstance(recent, dict):
            return None
        current_price = float(signal.price or 0.0)
        tolerance = max(0.0, float(self.config.entry_structure_rejection_near_pct)) / 100.0
        if structure_signal in {"short_bb96_210_upper_second_test_rejection", "long_bb96_210_lower_second_test_rebound"}:
            second_test_zone = selected.get("second_test_zone") if isinstance(selected.get("second_test_zone"), dict) else {}
            if second_test_zone.get("confirmed"):
                row = dict(selected)
                row["touch_price"] = round(
                    float(second_test_zone.get("recent_high") or second_test_zone.get("recent_low") or current_price),
                    8,
                )
                row["entry_context"] = context
                return row
        if action == "open_long" and structure_signal in {"long_lower_support_rebound", "long_mid_support_retest", "long_upper_breakout", "long_4h_ma20_support_retest", "long_bb96_210_lower_second_test_rebound", "long_ma_support_retest", "long_ma_reclaim_cross"}:
            touch_price = float(recent.get("low") or current_price)
            if touch_price <= level_price and current_price >= level_price * (1.0 - tolerance):
                row = dict(selected)
                row["touch_price"] = round(touch_price, 8)
                row["entry_context"] = context
                return row
        if action == "open_short" and structure_signal in {"short_mid_resistance_retest", "short_upper_rejection", "short_lower_breakdown_retest", "short_4h_ma20_pressure_retest", "short_bb96_210_upper_second_test_rejection", "short_ma_resistance_retest", "short_ma_break_cross"}:
            touch_price = float(recent.get("high") or current_price)
            if touch_price >= level_price and current_price <= level_price * (1.0 + tolerance):
                row = dict(selected)
                row["touch_price"] = round(touch_price, 8)
                row["entry_context"] = context
                return row
        return None

    def indicator_confluence_hold(self, signal: V6Signal, side: str) -> dict[str, Any] | None:
        if not self.config.entry_confluence_hold_enabled:
            return None
        price = float(signal.price or 0.0)
        if price <= 0 or side not in {"short", "long"}:
            return None
        recent = (
            (signal.indicators or {})
            .get("recent_extremes", {})
            .get("1m_confluence", {})
        )
        if not isinstance(recent, dict):
            return None
        recent_low = float(recent.get("low") or 0.0)
        recent_high = float(recent.get("high") or 0.0)
        recent_close = float(recent.get("last_close") or price)
        if recent_low <= 0 or recent_high <= 0:
            return None
        if side == "short":
            source_rows = (signal.levels or {}).get("all_supports") or []
            allowed_bands = {"mid", "lower"}
            reason = "short_support_confluence_hold"
            message = "entry blocked: multi-BB support confluence is holding; do not short while price is standing on 3+ lines"
        else:
            source_rows = (signal.levels or {}).get("all_resistances") or []
            allowed_bands = {"mid", "upper"}
            reason = "long_resistance_confluence_hold"
            message = "entry blocked: multi-BB resistance confluence is holding; do not chase long under 3+ lines"
        near_pct = float(self.config.entry_confluence_hold_near_pct)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in source_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            name = str(row[0])
            if name in seen:
                continue
            seen.add(name)
            value = float(row[1] or 0.0)
            parsed = self.parse_shared_bb_name(name)
            if not parsed or parsed.get("band") not in allowed_bands:
                continue
            if parsed.get("timeframe") not in {"15m", "4h", "1d"}:
                continue
            if value <= 0 or abs(pct_diff(price, value)) > near_pct:
                continue
            candidates.append({
                "name": name,
                "price": round(value, 8),
                "distance_pct": round(pct_diff(price, value), 6),
                "abs_distance_pct": round(abs(pct_diff(price, value)), 6),
                "period": parsed["period"],
                "band": parsed["band"],
                "timeframe": parsed["timeframe"],
            })
        min_lines = max(1, int(self.config.entry_confluence_hold_min_lines))
        if len(candidates) < min_lines:
            return None
        candidates.sort(key=lambda item: (float(item.get("abs_distance_pct") or 0.0), str(item.get("timeframe") or ""), int(item.get("period") or 0)))
        selected = candidates[:min_lines]
        prices = [float(item.get("price") or 0.0) for item in selected]
        tolerance = max(0.0, float(self.config.entry_confluence_hold_break_tolerance_pct)) / 100.0
        cluster_avg = sum(prices) / len(prices)
        if side == "short":
            cluster_floor = min(prices)
            if recent_low < cluster_floor * (1.0 - tolerance):
                return None
            if min(price, recent_close) < cluster_avg * (1.0 - tolerance):
                return None
        else:
            cluster_ceiling = max(prices)
            if recent_high > cluster_ceiling * (1.0 + tolerance):
                return None
            if max(price, recent_close) > cluster_avg * (1.0 + tolerance):
                return None
        return {
            "enabled": True,
            "blocked": True,
            "side": side,
            "action": "open_short" if side == "short" else "open_long",
            "reason": reason,
            "message": message,
            "confluence": {
                "line_count": len(candidates),
                "required_line_count": min_lines,
                "selected_lines": selected,
                "cluster_avg": round(cluster_avg, 8),
                "recent_low": round(recent_low, 8),
                "recent_high": round(recent_high, 8),
                "recent_close": round(recent_close, 8),
                "lookback": recent.get("lookback"),
                "near_pct": near_pct,
                "break_tolerance_pct": float(self.config.entry_confluence_hold_break_tolerance_pct),
            },
        }

    def entry_boundary_chase_candidate(self, signal: V6Signal, side: str) -> dict[str, Any] | None:
        price = float(signal.price or 0.0)
        if price <= 0:
            return None
        if side == "short":
            rows = (signal.levels or {}).get("all_supports") or []
            expected_bands = {"mid", "lower"}
        elif side == "long":
            rows = (signal.levels or {}).get("all_resistances") or []
            expected_bands = {"mid", "upper"}
        else:
            return None
        near_pct = float(self.config.entry_structure_boundary_near_pct)
        candidates: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            name = str(row[0])
            value = float(row[1] or 0.0)
            parsed = self.parse_shared_bb_name(name)
            if not parsed or parsed.get("timeframe") != "15m" or parsed.get("band") not in expected_bands:
                continue
            if value <= 0:
                continue
            distance = pct_diff(price, value)
            if abs(distance) > near_pct:
                continue
            candidates.append({
                "name": name,
                "price": round(value, 8),
                "distance_pct": round(distance, 6),
                "abs_distance_pct": round(abs(distance), 6),
                "period": parsed["period"],
                "band": parsed["band"],
                "timeframe": parsed["timeframe"],
            })
        if not candidates:
            return None
        candidates.sort(key=lambda item: (float(item.get("abs_distance_pct") or 0.0), int(item.get("period") or 0)))
        return candidates[0]

    def maybe_downgrade_entry_block(self, signal: V6Signal, block: dict[str, Any] | None) -> dict[str, Any] | None:
        if not block:
            return None
        policy = getattr(self, "transition_gate_policy", TransitionGatePolicy(enabled=False))
        decision = policy.decision_for(block)
        if not policy.enabled or decision is None:
            return block
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        append_gate_decision(levels, block, decision)
        signal.levels = levels
        message = (
            f"transition gate downgraded {decision.reason}: "
            f"{decision.classification}, notional x{decision.notional_multiplier:.2f}"
        )
        if message not in signal.reasons:
            signal.reasons.append(message)
        recorder = getattr(self, "gate_recorder", None)
        if recorder is not None:
            recorder.record_entry_block(
                signal,
                {
                    **dict(block),
                    "downgraded_by_transition_gate_policy": True,
                    "transition_gate_decision": decision.to_dict(),
                },
            )
        return None

    def apply_entry_structure_block(self, signal: V6Signal, block: dict[str, Any]) -> None:
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        levels["entry_structure_filter"] = dict(block)
        signal.levels = levels
        message = str(block.get("message") or block.get("reason") or "entry blocked by structure filter")
        if message and message not in signal.reasons:
            signal.reasons.append(message)
        self.gate_recorder.record_entry_block(signal, block)

    def open_intent(self, signal: V6Signal, action: str) -> V6OrderIntent:
        side = "SELL" if action == "open_short" else "BUY"
        stop, take_profit, risk_plan = self.stop_take_profit(signal, action)
        risk_plan["entry_indicators"] = dict(getattr(signal, "indicators", {}) or {})
        local_side = "short" if action == "open_short" else "long"
        smart_signal_context = self.v6_smart_signal_context(signal, local_side)
        if smart_signal_context.get("has_signal"):
            risk_plan["smart_signal_risk_guard"] = smart_signal_context
        if smart_signal_context.get("supportive_to_side"):
            if local_side == "long":
                message = "Binance smart-signal confirms crowded shorts and 15M reclaim; squeeze risk supports long bias"
            else:
                message = "Binance smart-signal confirms crowded longs and 15M loss; crowded flush risk supports short bias"
            if message not in signal.reasons:
                signal.reasons.append(message)
        self.apply_candidate_metadata_to_risk_plan(signal.symbol, risk_plan)
        base_notional = max(float(self.config.base_notional_usdt), float(self.config.min_notional_usdt))
        notional = base_notional
        transition_multiplier = transition_notional_multiplier(signal.levels if isinstance(signal.levels, dict) else {})
        gann_multiplier = 1.0
        if self.config.gann_position_sizing_enabled:
            gann_state = (signal.levels or {}).get("gann_feature") if isinstance(signal.levels, dict) else None
            gann_multiplier = gann_position_multiplier(gann_state, local_side)
        combined_multiplier = max(0.05, min(1.25, transition_multiplier * gann_multiplier))
        if abs(combined_multiplier - 1.0) > 1e-9:
            risk_plan["position_sizing_overlays"] = {
                "transition_gate_multiplier": round(transition_multiplier, 6),
                "gann_multiplier": round(gann_multiplier, 6),
                "combined_multiplier": round(combined_multiplier, 6),
            }
            if combined_multiplier < 1.0:
                notional_floor = max(5.0, float(self.config.transition_gate_min_notional_usdt))
            else:
                notional_floor = float(self.config.min_notional_usdt)
            notional = max(notional_floor, base_notional * combined_multiplier)
        if (
            self.config.v5_staged_entry_enabled
            and self.config.v5_overlay_rules_active_for(signal.symbol)
            and action in {"open_long", "open_short"}
        ):
            first_fraction = clamp(float(self.config.v5_initial_entry_notional_fraction), 0.05, 1.0)
            notional = max(float(self.config.min_notional_usdt), base_notional * first_fraction)
            risk_plan["v5_staged_entry"] = {
                "enabled": True,
                "mode": "light_first_entry_then_15m_pullback_scale_in",
                "base_notional_usdt": round(base_notional, 6),
                "initial_fraction": round(first_fraction, 6),
                "initial_notional_usdt": round(notional, 6),
                "scale_in_fraction": round(float(self.config.scale_in_notional_fraction), 6),
                "max_scale_ins": int(self.config.max_scale_ins_per_position),
                "scale_in_min_hold_minutes": float(self.config.v5_scale_in_min_hold_minutes),
                "scale_in_min_adverse_move_pct": float(self.config.scale_in_min_adverse_move_pct),
            }
        return V6OrderIntent(
            action=action,
            symbol=signal.symbol,
            side=side,
            quantity_hint_usdt=round(notional, 6),
            leverage=int(self.config.default_leverage),
            reduce_only=False,
            entry_price=signal.price,
            stop_loss_price=stop,
            take_profit_price=take_profit,
            long_probability=signal.long_probability,
            short_probability=signal.short_probability,
            wait_probability=signal.wait_probability,
            rationale=join_reasons(signal.reasons, separator="; "),
            generated_at=utc_now_iso(),
            risk_plan=risk_plan,
        )

    @staticmethod
    def is_staged_take_profit_intent(intent: V6OrderIntent) -> bool:
        return bool(intent.target_name) and bool((intent.risk_plan or {}).get("partial_take_profit"))

    @staticmethod
    def is_partial_take_profit_intent(intent: V6OrderIntent) -> bool:
        return V6ExecutionEngine.is_staged_take_profit_intent(intent) and float(intent.close_fraction or 1.0) < 0.999

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

    def retry_percent_price_order_after_spike(
        self,
        intent: V6OrderIntent,
        original_error: str,
    ) -> dict[str, Any] | None:
        if not self.config.percent_price_retry_enabled:
            return None
        if intent.action not in {"open_long", "open_short", "close_long", "close_short"}:
            return None
        is_close = intent.action in {"close_long", "close_short"}
        wait_seconds = max(0.0, float(self.config.percent_price_retry_wait_seconds))
        max_deviation = max(0.0, float(self.config.percent_price_retry_max_deviation_pct))
        retry_started_at = utc_now_iso()
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        reference_price = 0.0
        try:
            reference_price = float(self.client.order_endpoint_reference_price(intent.symbol) or 0.0)
        except Exception:
            reference_price = 0.0
        entry_price = float(intent.entry_price or 0.0)
        if is_close and entry_price <= 0 and reference_price > 0:
            entry_price = reference_price
        deviation_pct = abs(pct_diff(reference_price, entry_price)) if reference_price > 0 and entry_price > 0 else 999.0
        retry_meta = {
            "enabled": True,
            "reason": "binance_percent_price_filter_after_possible_spike",
            "action_type": "close" if is_close else "open",
            "wait_seconds": wait_seconds,
            "max_deviation_pct": max_deviation,
            "entry_price": round(entry_price, 8),
            "reference_price": round(reference_price, 8),
            "deviation_pct": round(deviation_pct, 6),
            "started_at": retry_started_at,
            "checked_at": utc_now_iso(),
        }
        if reference_price <= 0 or entry_price <= 0 or deviation_pct > max_deviation:
            return {
                "status": "deferred_percent_price_close_deviation" if is_close else "skipped_percent_price_retry_deviation",
                "order_mode": self.config.order_mode,
                "intent": intent.to_dict(),
                "error": "",
                "original_error": original_error,
                "telegram_silent": True,
                "maintenance_only": True,
                "percent_price_retry": retry_meta,
                "submitted_at": utc_now_iso(),
                "note": (
                    "Close deferred after Binance percent-price rejection because the order-endpoint reference price moved too far from the original close reference."
                    if is_close
                    else "Open skipped after Binance percent-price rejection because the order-endpoint reference price moved too far from the original entry."
                ),
            }
        counterparty_context: dict[str, Any] = {}
        try:
            counterparty_context = self.client.order_endpoint_percent_price_context(intent.symbol, intent.action)
            retry_meta["counterparty_filter"] = counterparty_context
        except Exception as exc:
            retry_meta["counterparty_filter_error"] = f"{type(exc).__name__}: {exc}"
        if counterparty_context and counterparty_context.get("valid") is False:
            return {
                "status": "deferred_percent_price_close_counterparty_filter" if is_close else "skipped_percent_price_retry_counterparty_filter",
                "order_mode": self.config.order_mode,
                "intent": intent.to_dict(),
                "error": "",
                "original_error": original_error,
                "telegram_silent": True,
                "maintenance_only": True,
                "percent_price_retry": retry_meta,
                "submitted_at": utc_now_iso(),
                "note": (
                    "Close deferred after Binance percent-price rejection because the order-endpoint counterparty price is still outside the allowed percent-price range."
                    if is_close
                    else "Open skipped after Binance percent-price rejection because the order-endpoint counterparty price is still outside the allowed percent-price range."
                ),
            }
        try:
            result = self.client.submit_order(intent)
        except Exception as retry_exc:
            retry_error = f"{type(retry_exc).__name__}: {retry_exc}"
            retry_meta["retry_error"] = retry_error
            if self.is_percent_price_filter_error(retry_error):
                return {
                    "status": "deferred_percent_price_close_failed" if is_close else "skipped_percent_price_retry_failed",
                    "order_mode": self.config.order_mode,
                    "intent": intent.to_dict(),
                    "error": "",
                    "original_error": original_error,
                    "retry_error": retry_error,
                    "telegram_silent": True,
                    "maintenance_only": True,
                    "percent_price_retry": retry_meta,
                    "submitted_at": utc_now_iso(),
                    "note": (
                        "Close deferred after retry because Binance percent-price protection still rejected the reduce-only market order."
                        if is_close
                        else "Open skipped after retry because Binance percent-price protection still rejected the market order."
                    ),
                }
            raise
        result["percent_price_retry"] = retry_meta | {"status": "retried_submitted", "retried_at": utc_now_iso()}
        result["original_error"] = original_error
        return result

    def mark_partial_take_profit_retry_wait(
        self,
        position: dict[str, Any],
        intent: V6OrderIntent,
        error_text: str,
        *,
        reason: str,
        minutes: int = 30,
    ) -> dict[str, Any] | None:
        partial = (intent.risk_plan or {}).get("partial_take_profit") or {}
        if not partial:
            return None
        partial_type = str(partial.get("type") or "bb_take_profit")
        if partial_type == "roi_take_profit":
            stage_key = "roi_take_profit_stages"
            stages = self.ensure_position_roi_take_profit_stages(position)
        elif partial_type in {
            "profit_giveback_guard",
            "h4_lower_reversal_guard",
            "v5_no_news_momentum_tp",
            "v5_no_news_momentum_final_giveback_exit",
        }:
            stage_key = "profit_guard_stages"
            stages = [stage for stage in position.get(stage_key) or [] if isinstance(stage, dict)]
        else:
            stage_key = "take_profit_stages"
            stages = self.ensure_position_take_profit_stages(position)

        target_names = [name for name in str(intent.target_name or "").split(",") if name]
        if not target_names:
            return None
        matched = [
            stage for stage in stages
            if str(stage.get("name") or "") in target_names
            and str(stage.get("status") or "pending") in {"pending", "retry_wait"}
        ]
        if not matched:
            return None

        retry_after = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
        now = utc_now_iso()
        for stage in matched:
            stage["status"] = "retry_wait"
            stage["retry_after"] = retry_after
            stage["retry_reason"] = reason
            stage["last_failed_at"] = now
            stage["last_failed_error"] = error_text
        position[stage_key] = stages

        history = list(position.get("partial_take_profit_skipped_history") or [])
        history.append({
            "type": partial_type,
            "target_name": ",".join(target_names),
            "submitted_at": now,
            "reason": reason,
            "retry_after": retry_after,
            "error": error_text,
        })
        position["partial_take_profit_skipped_history"] = history[-20:]
        return {
            "event": "partial_take_profit_retry_wait",
            "symbol": intent.symbol,
            "side": local_side_for_action(intent.action),
            "target_name": ",".join(target_names),
            "reason": reason,
            "retry_after": retry_after,
            "stage_key": stage_key,
        }

    def ensure_position_take_profit_stages(self, position: dict[str, Any], signal: V6Signal | None = None) -> list[dict[str, Any]]:
        if not self.config.staged_take_profit_enabled:
            return []
        stages = [stage for stage in position.get("take_profit_stages") or [] if isinstance(stage, dict)]
        risk_plan = position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {}
        if stages:
            if (
                risk_plan.get("target_ladder")
                and any(str(stage.get("selection_strategy") or "") != STAGED_TAKE_PROFIT_SELECTION_STRATEGY for stage in stages)
            ):
                action = str(risk_plan.get("action") or "")
                stages = self.migrate_take_profit_stages(position, risk_plan, stages, action)
                if stages:
                    position["take_profit_stages"] = stages
            return stages
        if signal is not None and not risk_plan.get("target_ladder"):
            side = str(position.get("side") or "").lower()
            action = "open_short" if side == "short" else "open_long"
            risk_plan = self.structure_risk_plan(float(signal.price), action, signal.levels, symbol=str(position.get("symbol") or signal.symbol))
            position["risk_plan"] = risk_plan
        stages = [
            dict(stage) for stage in risk_plan.get("staged_take_profit_targets") or []
            if isinstance(stage, dict)
        ]
        if not stages:
            stages = self.staged_take_profit_targets([
                dict(row) for row in risk_plan.get("target_ladder") or []
                if isinstance(row, dict)
            ], action=str(risk_plan.get("action") or ""), horizon=str(risk_plan.get("horizon") or ""))
        if stages:
            position["take_profit_stages"] = stages
            position.setdefault("original_notional_usdt", float(position.get("notional_usdt") or self.config.base_notional_usdt))
            position.setdefault("remaining_fraction", 1.0)
        return stages

    def migrate_take_profit_stages(
        self,
        position: dict[str, Any],
        risk_plan: dict[str, Any],
        old_stages: list[dict[str, Any]],
        action: str,
    ) -> list[dict[str, Any]]:
        new_stages = self.staged_take_profit_targets([
            dict(row) for row in risk_plan.get("target_ladder") or []
            if isinstance(row, dict)
        ], action=action, horizon=str(risk_plan.get("horizon") or ""))
        if not new_stages:
            return old_stages
        old_by_name = {str(stage.get("name") or ""): stage for stage in old_stages}
        migrated: list[dict[str, Any]] = []
        for stage in new_stages:
            name = str(stage.get("name") or "")
            old = old_by_name.get(name)
            if old and str(old.get("status") or "") == "executed":
                merged = dict(stage)
                merged.update({
                    key: value for key, value in old.items()
                    if key not in {"selection_strategy"}
                })
                merged["selection_strategy"] = STAGED_TAKE_PROFIT_SELECTION_STRATEGY
                migrated.append(merged)
            else:
                migrated.append(dict(stage))

        executed_fraction = sum(
            clamp(float(stage.get("fraction_of_initial") or 0.0), 0.0, 1.0)
            for stage in migrated
            if str(stage.get("status") or "pending") == "executed"
        )
        remaining_fraction = clamp(float(position.get("remaining_fraction") or max(0.0, 1.0 - executed_fraction)), 0.0, 1.0)
        pending = [stage for stage in migrated if str(stage.get("status") or "pending") == "pending"]
        if pending and remaining_fraction > 0:
            share = round(remaining_fraction / len(pending), 6)
            for stage in pending:
                stage["fraction_of_initial"] = share
            pending[-1]["fraction_of_initial"] = round(pending[-1]["fraction_of_initial"] + (remaining_fraction - share * len(pending)), 6)
        if migrated:
            final_stage = migrated[-1]
            final_target = {
                "name": str(final_stage.get("name") or ""),
                "price": float(final_stage.get("price") or 0.0),
                "distance_pct": float(final_stage.get("distance_pct") or 0.0),
            }
            risk_plan["staged_take_profit_targets"] = [dict(stage) for stage in migrated]
            risk_plan["tp1"] = migrated[0]
            risk_plan["tp2"] = migrated[min(1, len(migrated) - 1)]
            risk_plan["runner"] = final_target
            if final_target["price"] > 0:
                risk_plan["take_profit_price"] = final_target["price"]
                position["take_profit_price"] = final_target["price"]
        return migrated

    @staticmethod
    def staged_take_profit_touched(side: str, price: float, target_price: float) -> bool:
        if target_price <= 0:
            return False
        if side == "short":
            return price <= target_price
        return price >= target_price

    @staticmethod
    def staged_take_profit_touch_price(signal: V6Signal, side: str) -> float:
        price = float(signal.price)
        recent = (
            (signal.indicators or {})
            .get("recent_extremes", {})
            .get("1m", {})
        )
        if not isinstance(recent, dict):
            return price
        if side == "short":
            low = float(recent.get("low") or price)
            return min(price, low)
        if side == "long":
            high = float(recent.get("high") or price)
            return max(price, high)
        return price

    @staticmethod
    def parse_utc_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def take_profit_touch_confirm_deadline(self, touched_at: datetime) -> datetime:
        window = max(0, int(self.config.take_profit_touch_confirm_window_15m))
        return touched_at + timedelta(minutes=15 * window)

    def mark_staged_take_profit_touched(
        self,
        stage: dict[str, Any],
        *,
        touch_price: float,
        target_price: float,
    ) -> None:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        stage.setdefault("first_touched_at", now_iso)
        stage["last_touched_at"] = now_iso
        stage["touch_price"] = round(touch_price, 8)
        stage["touch_target_price"] = round(target_price, 8)
        stage["touch_source"] = "recent_1m_extreme_or_mark"
        stage["touch_pending_volume"] = True
        stage["touch_status"] = "pending_volume_confirmation"
        stage["touch_confirm_window_15m"] = max(0, int(self.config.take_profit_touch_confirm_window_15m))
        stage["touch_confirm_until"] = self.take_profit_touch_confirm_deadline(now).isoformat()

    def staged_take_profit_touch_pending_active(self, stage: dict[str, Any]) -> bool:
        if not bool(stage.get("touch_pending_volume")):
            return False
        deadline = self.parse_utc_datetime(stage.get("touch_confirm_until"))
        if deadline is None:
            return False
        now = datetime.now(timezone.utc)
        if now <= deadline:
            return True
        stage["touch_pending_volume"] = False
        stage["touch_status"] = "expired_waiting_volume_confirmation"
        stage["touch_expired_at"] = now.isoformat()
        return False

    def take_profit_volume_exit_guard(self, signal: V6Signal, side: str, stages: list[dict[str, Any]]) -> dict[str, Any]:
        enabled = bool(self.config.take_profit_volume_guard_enabled)
        volume = (signal.indicators or {}).get("tick_volume_15m") if isinstance(signal.indicators, dict) else {}
        if not isinstance(volume, dict):
            volume = {}
        key_levels = [str(stage.get("name") or "") for stage in stages if isinstance(stage, dict)]
        is_key_level = any(name.startswith("bb") or name.startswith("ma") for name in key_levels)
        ready = bool(volume.get("ready"))
        active = bool(volume.get("active"))
        allowed = (not enabled) or (not is_key_level) or (not ready) or active
        return {
            "enabled": enabled,
            "side": side,
            "is_key_level": is_key_level,
            "key_levels": key_levels,
            "volume_ready": ready,
            "volume_active": active,
            "exit_allowed": allowed,
            "decision": "close" if allowed else "hold",
            "rule": "15M tick volume expansion required at key support/resistance for staged take profit",
            "volume": volume,
        }

    def take_profit_volume_guard_allows_exit(self, signal: V6Signal, side: str, stages: list[dict[str, Any]]) -> bool:
        guard = self.take_profit_volume_exit_guard(signal, side, stages)
        for stage in stages:
            if isinstance(stage, dict):
                stage["volume_exit_guard"] = guard
        return bool(guard.get("exit_allowed"))

    def roi_take_profit_executed_fraction(self, position: dict[str, Any]) -> float:
        stages = [stage for stage in position.get("roi_take_profit_stages") or [] if isinstance(stage, dict)]
        return clamp(
            sum(
                float(stage.get("fraction_of_initial") or 0.0)
                for stage in stages
                if str(stage.get("status") or "pending") == "executed"
            ),
            0.0,
            1.0,
        )

    def current_position_roi_pct(self, signal: V6Signal, position: dict[str, Any], *, touch_price: float | None = None) -> float:
        side = str(position.get("side") or "").lower()
        price = float(touch_price if touch_price is not None else self.staged_take_profit_touch_price(signal, side))
        entry = float(position.get("entry_price") or 0.0)
        leverage = float(position.get("leverage") or self.config.default_leverage or 1.0)
        return self.position_roi_pct(side, entry, price, leverage)

    @staticmethod
    def position_opened_at(position: dict[str, Any]) -> datetime | None:
        opened_at = position.get("opened_at")
        if not opened_at:
            return None
        try:
            return datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
        except Exception:
            return None

    @classmethod
    def position_age_hours(cls, position: dict[str, Any]) -> float:
        opened = cls.position_opened_at(position)
        if opened is None:
            return 0.0
        return max(0.0, (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0)

    @staticmethod
    def selected_entry_level_from_position(position: dict[str, Any]) -> dict[str, Any]:
        risk_plan = position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {}
        context = risk_plan.get("entry_context") if isinstance(risk_plan.get("entry_context"), dict) else {}
        selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
        if selected:
            return selected
        selected = risk_plan.get("selected_entry_level") if isinstance(risk_plan.get("selected_entry_level"), dict) else {}
        if selected:
            return selected
        lots = [lot for lot in position.get("lots") or [] if isinstance(lot, dict)]
        for lot in lots:
            selected = lot.get("selected_entry_level") if isinstance(lot.get("selected_entry_level"), dict) else {}
            if selected:
                return selected
        return {}

    @classmethod
    def is_15m_bb_lower_long_position(cls, position: dict[str, Any]) -> bool:
        if str(position.get("side") or "").lower() != "long":
            return False
        selected = cls.selected_entry_level_from_position(position)
        name = str(selected.get("name") or "").lower()
        structure_signal = str(selected.get("structure_signal") or "").lower()
        timeframe = str(selected.get("timeframe") or position.get("entry_timeframe") or "").lower()
        band = str(selected.get("band") or "").lower()
        if "bb96_210" in name or "bb96_210" in structure_signal:
            return True
        if timeframe == "15m" and band in {"lower", "cluster"} and ("bb96" in name or "bb210" in name):
            return True
        return False

    @classmethod
    def is_4h_lower_long_position(cls, position: dict[str, Any]) -> bool:
        if str(position.get("side") or "").lower() != "long":
            return False
        selected = cls.selected_entry_level_from_position(position)
        name = str(selected.get("name") or "").lower()
        timeframe = str(selected.get("timeframe") or position.get("entry_timeframe") or "").lower()
        band = str(selected.get("band") or "").lower()
        structure_signal = str(selected.get("structure_signal") or "").lower()
        return bool(timeframe == "4h" and (band == "lower" or "_lower_4h" in name or "lower_support" in structure_signal))

    @staticmethod
    def h4_long_confirmed(signal: V6Signal) -> bool:
        h4 = signal.h4 if isinstance(signal.h4, dict) else {}
        right_side = h4.get("right_side") if isinstance(h4.get("right_side"), dict) else {}
        return bool(right_side.get("long_confirmed"))

    @staticmethod
    def h4_short_confirmed(signal: V6Signal) -> bool:
        h4 = signal.h4 if isinstance(signal.h4, dict) else {}
        right_side = h4.get("right_side") if isinstance(h4.get("right_side"), dict) else {}
        return bool(right_side.get("short_confirmed"))

    def update_position_profit_tracker(self, signal: V6Signal, position: dict[str, Any]) -> dict[str, Any]:
        side = str(position.get("side") or "").lower()
        roi_pct = self.current_position_roi_pct(signal, position)
        tracker = position.get("profit_guard") if isinstance(position.get("profit_guard"), dict) else {}
        previous_peak = float(tracker.get("max_roi_pct") or roi_pct)
        max_roi = max(previous_peak, roi_pct)
        tracker.update({
            "enabled": True,
            "side": side,
            "current_roi_pct": round(roi_pct, 6),
            "max_roi_pct": round(max_roi, 6),
            "giveback_pct": round(max(0.0, max_roi - roi_pct), 6),
            "price": round(float(signal.price), 8),
            "updated_at": utc_now_iso(),
        })
        if max_roi > previous_peak:
            tracker["max_roi_seen_at"] = utc_now_iso()
        tracker.setdefault("first_seen_at", utc_now_iso())
        position["profit_guard"] = tracker
        return tracker

    @staticmethod
    def profit_guard_stage_executed(position: dict[str, Any], name: str) -> bool:
        stages = [stage for stage in position.get("profit_guard_stages") or [] if isinstance(stage, dict)]
        for stage in stages:
            if str(stage.get("name") or "") == name and str(stage.get("status") or "pending") == "executed":
                return True
        return False

    def profit_guard_take_profit(
        self,
        position: dict[str, Any],
        *,
        name: str,
        fraction_of_initial: float,
        guard_type: str,
        trigger: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], float] | None:
        if self.profit_guard_stage_executed(position, name):
            return None
        remaining = clamp(float(position.get("remaining_fraction") or 1.0), 0.0, 1.0)
        if remaining <= 0.001:
            return None
        fraction = clamp(float(fraction_of_initial), 0.0, 1.0)
        if fraction <= 0:
            return None
        close_initial = min(fraction, remaining)
        close_fraction = 1.0 if close_initial >= remaining - 0.001 else clamp(close_initial / remaining, 0.0, 1.0)
        stages = [stage for stage in position.get("profit_guard_stages") or [] if isinstance(stage, dict)]
        stage = None
        for existing in stages:
            if str(existing.get("name") or "") == name:
                stage = existing
                break
        if stage is not None:
            status = str(stage.get("status") or "pending")
            if status == "retry_wait":
                retry_after = str(stage.get("retry_after") or "")
                retry_blocked = False
                if retry_after:
                    try:
                        retry_blocked = datetime.fromisoformat(retry_after.replace("Z", "+00:00")) > datetime.now(timezone.utc)
                    except Exception:
                        retry_blocked = False
                if retry_blocked:
                    return None
                stage["status"] = "pending"
                stage.pop("retry_after", None)
                stage["retry_released_at"] = utc_now_iso()
            elif status not in {"pending", ""}:
                return None
        if stage is None:
            stage = {
                "name": name,
                "status": "pending",
                "selection_strategy": guard_type,
            }
            stages.append(stage)
        stage.update({
            "fraction_of_initial": round(close_initial, 6),
            "requested_fraction_of_initial": round(fraction, 6),
            "close_fraction_of_remaining": round(close_fraction, 6),
            "trigger": trigger,
            "trigger_price": trigger.get("price"),
            "trigger_roi_pct": trigger.get("current_roi_pct"),
            "trigger_giveback_pct": trigger.get("giveback_pct"),
            "checked_at": utc_now_iso(),
        })
        position["profit_guard_stages"] = stages
        return [stage], close_fraction

    @staticmethod
    def _v5_momentum_safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if np.isfinite(parsed) else default

    @staticmethod
    def _v5_momentum_optional_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    @staticmethod
    def _v5_momentum_iso_is_fresh(value: Any, max_age_hours: float) -> bool:
        if not value:
            return False
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            return False
        return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() <= max(0.1, float(max_age_hours)) * 3600.0

    @staticmethod
    def _v5_momentum_iso_is_fresh_seconds(value: Any, max_age_seconds: float) -> bool:
        if not value:
            return False
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            return False
        return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() <= max(1.0, float(max_age_seconds))

    def v5_momentum_smart_signal_metrics(self, row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        long_score = self._v5_momentum_safe_float(row.get("long_score"), self._v5_momentum_safe_float(metadata.get("long_notional")))
        short_score = self._v5_momentum_safe_float(row.get("short_score"), self._v5_momentum_safe_float(metadata.get("short_notional")))
        long_whales = int(self._v5_momentum_safe_float(metadata.get("long_whales")))
        short_whales = int(self._v5_momentum_safe_float(metadata.get("short_whales")))
        total_whales = long_whales + short_whales
        short_to_long = short_score / long_score if long_score > 0 else (float("inf") if short_score > 0 else 0.0)
        long_to_short = long_score / short_score if short_score > 0 else (float("inf") if long_score > 0 else 0.0)
        short_whale_share = short_whales / total_whales if total_whales > 0 else 0.0
        long_whale_share = long_whales / total_whales if total_whales > 0 else 0.0
        raw_direction = str(row.get("direction") or "").lower()
        side = str(metadata.get("side") or "").upper()
        direction = raw_direction or ("short" if side == "SELL" else "long" if side == "BUY" else "")
        return {
            "provider": str(metadata.get("provider") or row.get("source") or ""),
            "direction": direction,
            "long_score": long_score,
            "short_score": short_score,
            "long_whales": long_whales,
            "short_whales": short_whales,
            "short_to_long_notional": self._v5_momentum_safe_float(metadata.get("short_to_long_notional"), short_to_long),
            "long_to_short_notional": self._v5_momentum_safe_float(metadata.get("long_to_short_notional"), long_to_short),
            "short_whale_share": self._v5_momentum_safe_float(metadata.get("short_whale_share"), short_whale_share),
            "long_whale_share": self._v5_momentum_safe_float(metadata.get("long_whale_share"), long_whale_share),
            "short_avg_entry_price": self._v5_momentum_safe_float(metadata.get("short_avg_entry_price")),
            "long_avg_entry_price": self._v5_momentum_safe_float(metadata.get("long_avg_entry_price")),
            "short_crowded": bool(metadata.get("short_crowded")) or ((short_to_long >= 2.0 and short_whale_share >= 0.55) or short_to_long >= 3.0),
            "long_crowded": bool(metadata.get("long_crowded")) or ((long_to_short >= 2.0 and long_whale_share >= 0.55) or long_to_short >= 3.0),
        }

    def v5_smart_signal_short_squeeze_context(
        self,
        *,
        panel: dict[str, Any],
        ma20_trend: dict[str, Any],
        volume_ratio: float,
        oi_context: dict[str, Any],
        price: float,
    ) -> dict[str, Any]:
        candidates = [
            source
            for source in panel.get("sources", [])
            if isinstance(source, dict) and source.get("source") in {"binance_smart_signal", "binance_smart_money"}
        ]
        crowded = next((source for source in candidates if source.get("short_crowded")), None)
        if not crowded:
            return {"active": False, "reason": "no_short_crowding"}
        price_distance = self._v5_momentum_optional_float(ma20_trend.get("price_distance_pct"))
        price_reclaimed_ma20 = bool(ma20_trend.get("bullish_trend_alive")) or (
            price_distance is not None and price_distance >= 0
        )
        volume_confirmed = volume_ratio >= 1.5
        oi_expanded = bool(oi_context.get("oi_expanded"))
        short_avg_entry = self._v5_momentum_safe_float(crowded.get("short_avg_entry_price"))
        near_or_above_short_avg = bool(short_avg_entry > 0 and price >= short_avg_entry * 0.995)
        active = bool(crowded.get("short_crowded") and price_reclaimed_ma20 and (volume_confirmed or oi_expanded or near_or_above_short_avg))
        return {
            "active": active,
            "reason": "short_crowded_reclaim_ma20" if active else "short_crowded_wait_price_confirmation",
            "price_reclaimed_ma20": price_reclaimed_ma20,
            "price_distance_ma20_15m_pct": round(price_distance, 6) if price_distance is not None else None,
            "volume_confirmed": volume_confirmed,
            "oi_expanded": oi_expanded,
            "near_or_above_short_avg_entry": near_or_above_short_avg,
            "short_to_long_notional": crowded.get("short_to_long_notional"),
            "short_whale_share_pct": round(float(crowded.get("short_whale_share") or 0.0) * 100.0, 6),
            "short_avg_entry_price": short_avg_entry,
        }

    def smart_signal_cache(self) -> dict[str, dict[str, Any]]:
        existing = getattr(self, "_smart_signal_cache", None)
        if existing is not None:
            return existing
        cache: dict[str, dict[str, Any]] = {}
        path = Path(self.config.smart_signal_file)
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception as exc:
            LOGGER.warning("V6 smart-signal cache load failed for %s: %s", path, exc)
            self._smart_signal_cache = cache
            return cache
        generated_at = payload.get("generated_at") if isinstance(payload, dict) else ""
        signals = payload.get("signals") if isinstance(payload, dict) else []
        if not isinstance(signals, list):
            self._smart_signal_cache = cache
            return cache
        max_age = float(self.config.smart_signal_snapshot_max_age_hours)
        for row in signals:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            updated_at = row.get("updated_at") or generated_at
            if updated_at and not self._v5_momentum_iso_is_fresh(updated_at, max_age):
                continue
            cache.setdefault(symbol, dict(row))
        self._smart_signal_cache = cache
        return cache

    def smart_signal_row_for_symbol(self, symbol: str) -> dict[str, Any] | None:
        symbol_upper = str(symbol or "").upper().strip()
        if not symbol_upper:
            return None
        row = self.smart_signal_cache().get(symbol_upper)
        return dict(row) if isinstance(row, dict) and row else None

    @staticmethod
    def signal_tick_volume_15m_ratio(signal: V6Signal) -> float:
        indicators = getattr(signal, "indicators", {})
        volume = (indicators or {}).get("tick_volume_15m") if isinstance(indicators, dict) else {}
        if not isinstance(volume, dict):
            return 0.0
        values: list[float] = []
        for key in ("volume_vs_ma5_ratio", "quote_volume_vs_ma5_ratio"):
            try:
                value = float(volume.get(key) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if np.isfinite(value):
                values.append(value)
        return max(values) if values else 0.0

    def v6_smart_signal_context(self, signal: V6Signal, side: str) -> dict[str, Any]:
        if not bool(self.config.smart_signal_risk_guard_enabled):
            return {"enabled": False, "active": False, "reason": "disabled"}
        side = str(side or "").lower()
        if side not in {"long", "short"}:
            return {"enabled": True, "active": False, "reason": "invalid_side"}
        row = self.smart_signal_row_for_symbol(signal.symbol)
        if not row:
            return {"enabled": True, "active": False, "has_signal": False, "reason": "no_smart_signal"}
        metrics = self.v5_momentum_smart_signal_metrics(row)
        price = float(signal.price or 0.0)
        raw_fifteen_m = getattr(signal, "fifteen_m", {})
        fifteen_m = raw_fifteen_m if isinstance(raw_fifteen_m, dict) else {}
        ma20_trend = fifteen_m.get("ma20_trend") if isinstance(fifteen_m.get("ma20_trend"), dict) else {}
        price_distance = self._v5_momentum_optional_float(ma20_trend.get("price_distance_pct"))
        volume_ratio = self.signal_tick_volume_15m_ratio(signal)
        volume_confirmed = volume_ratio >= 1.5
        short_avg_entry = self._v5_momentum_safe_float(metrics.get("short_avg_entry_price"))
        long_avg_entry = self._v5_momentum_safe_float(metrics.get("long_avg_entry_price"))
        price_reclaimed_ma20 = bool(ma20_trend.get("bullish_trend_alive")) or (
            price_distance is not None and price_distance >= 0
        )
        price_lost_ma20 = bool(ma20_trend.get("bearish_trend_alive")) or (
            price_distance is not None and price_distance <= 0
        )
        near_or_above_short_avg = bool(short_avg_entry > 0 and price >= short_avg_entry * 0.995)
        near_or_below_long_avg = bool(long_avg_entry > 0 and price <= long_avg_entry * 1.005)
        short_squeeze_active = bool(
            metrics.get("short_crowded")
            and price_reclaimed_ma20
            and (volume_confirmed or near_or_above_short_avg)
        )
        long_flush_active = bool(
            metrics.get("long_crowded")
            and price_lost_ma20
            and (volume_confirmed or near_or_below_long_avg)
        )
        adverse = bool((side == "short" and short_squeeze_active) or (side == "long" and long_flush_active))
        supportive = bool((side == "long" and short_squeeze_active) or (side == "short" and long_flush_active))
        reason = (
            "short_crowded_reclaim_ma20"
            if short_squeeze_active
            else "long_crowded_lost_ma20"
            if long_flush_active
            else "smart_signal_wait_confirmation"
        )
        return {
            "enabled": True,
            "has_signal": True,
            "active": bool(short_squeeze_active or long_flush_active),
            "side": side,
            "reason": reason,
            "adverse_to_side": adverse,
            "supportive_to_side": supportive,
            "short_squeeze_active": short_squeeze_active,
            "long_crowded_flush_active": long_flush_active,
            "price_reclaimed_ma20": price_reclaimed_ma20,
            "price_lost_ma20": price_lost_ma20,
            "volume_confirmed": volume_confirmed,
            "volume_ratio_15m": round(volume_ratio, 6),
            "price_distance_ma20_15m_pct": round(price_distance, 6) if price_distance is not None else None,
            "direction": metrics.get("direction"),
            "long_score": round(float(metrics.get("long_score") or 0.0), 6),
            "short_score": round(float(metrics.get("short_score") or 0.0), 6),
            "long_whales": metrics.get("long_whales"),
            "short_whales": metrics.get("short_whales"),
            "short_to_long_notional": metrics.get("short_to_long_notional"),
            "long_to_short_notional": metrics.get("long_to_short_notional"),
            "short_whale_share": metrics.get("short_whale_share"),
            "long_whale_share": metrics.get("long_whale_share"),
            "short_avg_entry_price": short_avg_entry,
            "long_avg_entry_price": long_avg_entry,
            "short_crowded": bool(metrics.get("short_crowded")),
            "long_crowded": bool(metrics.get("long_crowded")),
            "source": "binance_smart_signal",
        }

    def market_data_freshness_entry_block(self, signal: V6Signal, action: str) -> dict[str, Any] | None:
        if action not in {"open_short", "open_long"}:
            return None
        freshness = getattr(signal, "market_data_freshness", {})
        if not isinstance(freshness, dict) or not freshness.get("stale"):
            return None
        stale_sources = freshness.get("stale_sources") if isinstance(freshness.get("stale_sources"), list) else []
        return {
            "enabled": True,
            "blocked": True,
            "watchable": True,
            "action": action,
            "side": "short" if action == "open_short" else "long",
            "reason": "entry_blocked_stale_market_data",
            "message": "entry blocked: market data refresh failed and stale cache was used; existing position exits remain allowed",
            "market_data_freshness": {
                "stale": True,
                "stale_sources": stale_sources[-6:],
                "updated_at": freshness.get("updated_at"),
            },
        }

    def smart_signal_entry_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any]) -> dict[str, Any] | None:
        if action not in {"open_short", "open_long"}:
            return None
        side = "short" if action == "open_short" else "long"
        context = self.v6_smart_signal_context(signal, side)
        if not context.get("adverse_to_side"):
            return None
        if side == "short":
            reason = "short_blocked_by_smart_signal_short_squeeze"
            message = "entry blocked: Binance smart-signal shows crowded shorts with 15M MA20 reclaim; do not chase short into squeeze risk"
        else:
            reason = "long_blocked_by_smart_signal_long_crowding"
            message = "entry blocked: Binance smart-signal shows crowded longs with 15M MA20 loss; do not chase long into crowded flush risk"
        return {
            "enabled": True,
            "blocked": True,
            "watchable": True,
            "action": action,
            "side": side,
            "reason": reason,
            "message": message,
            "entry_context": entry_context,
            "smart_signal": context,
        }

    def v5_momentum_panel_cache(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self._v5_momentum_panel_cache is not None:
            return self._v5_momentum_panel_cache
        cache: dict[str, dict[str, dict[str, Any]]] = {"smart_money": {}, "hyper": {}}
        smart_paths = [
            Path(self.config.v5_momentum_guard_smart_money_file),
            Path("data/trading_system/skill_bridges/binance-smart-money-distill.json"),
        ]
        sources = []
        seen_paths: set[str] = set()
        for path in smart_paths:
            key = str(path)
            if key not in seen_paths:
                sources.append(("smart_money", path))
                seen_paths.add(key)
        if self.config.v5_momentum_guard_use_hyper_panel:
            sources.append(("hyper", self.config.v5_momentum_guard_hyper_whale_file))
        max_age = float(self.config.v5_momentum_guard_snapshot_max_age_hours)
        for source_name, path in sources:
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
            except Exception as exc:
                LOGGER.warning("V5 momentum panel cache load failed for %s: %s", path, exc)
                continue
            generated_at = payload.get("generated_at") if isinstance(payload, dict) else ""
            signals = payload.get("signals") if isinstance(payload, dict) else []
            if not isinstance(signals, list):
                continue
            for row in signals:
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("symbol") or "").upper().strip()
                if not symbol:
                    continue
                updated_at = row.get("updated_at") or generated_at
                if updated_at and not self._v5_momentum_iso_is_fresh(updated_at, max_age):
                    continue
                cache[source_name].setdefault(symbol, dict(row))
        self._v5_momentum_panel_cache = cache
        return cache

    def v5_momentum_panel_alignment(self, symbol: str, side: str) -> dict[str, Any]:
        symbol_upper = str(symbol or "").upper().strip()
        side = str(side or "").lower()
        ratio = max(1.0, float(self.config.v5_momentum_guard_smart_money_ratio))
        cache = self.v5_momentum_panel_cache()
        sources: list[dict[str, Any]] = []
        aligned = False
        opposed = False
        smart = cache.get("smart_money", {}).get(symbol_upper)
        if isinstance(smart, dict) and smart:
            metrics = self.v5_momentum_smart_signal_metrics(smart)
            long_score = float(metrics.get("long_score") or 0.0)
            short_score = float(metrics.get("short_score") or 0.0)
            direction = str(metrics.get("direction") or "").lower()
            smart_aligned = (
                (side == "long" and direction == "long" and long_score >= max(1.0, short_score * ratio))
                or (side == "short" and direction == "short" and short_score >= max(1.0, long_score * ratio))
            )
            smart_opposed = (
                (side == "long" and direction == "short" and short_score >= max(1.0, long_score * ratio))
                or (side == "short" and direction == "long" and long_score >= max(1.0, short_score * ratio))
            )
            aligned = aligned or smart_aligned
            opposed = opposed or smart_opposed
            sources.append({
                "source": "binance_smart_signal" if str(metrics.get("provider") or "") == "binance_smart_signal" else "binance_smart_money",
                "direction": direction,
                "long_score": round(long_score, 6),
                "short_score": round(short_score, 6),
                "long_whales": metrics.get("long_whales"),
                "short_whales": metrics.get("short_whales"),
                "short_to_long_notional": metrics.get("short_to_long_notional"),
                "short_whale_share": metrics.get("short_whale_share"),
                "short_avg_entry_price": metrics.get("short_avg_entry_price"),
                "short_crowded": metrics.get("short_crowded"),
                "aligned": smart_aligned,
                "opposed": smart_opposed,
            })
        hyper = cache.get("hyper", {}).get(symbol_upper) if self.config.v5_momentum_guard_use_hyper_panel else None
        if isinstance(hyper, dict) and hyper:
            direction = str(hyper.get("direction") or "").lower()
            metadata = hyper.get("metadata") if isinstance(hyper.get("metadata"), dict) else {}
            net_ratio = self._v5_momentum_safe_float(metadata.get("net_ratio"))
            hyper_aligned = (
                (side == "long" and (direction.startswith("long") or net_ratio >= 0.20))
                or (side == "short" and (direction.startswith("short") or net_ratio <= -0.20))
            )
            hyper_opposed = (
                (side == "long" and (direction.startswith("short") or net_ratio <= -0.20))
                or (side == "short" and (direction.startswith("long") or net_ratio >= 0.20))
            )
            aligned = aligned or hyper_aligned
            opposed = opposed or hyper_opposed
            sources.append({
                "source": "hyper",
                "direction": direction,
                "net_ratio": round(net_ratio, 6),
                "aligned": hyper_aligned,
                "opposed": hyper_opposed,
            })
        return {
            "aligned": bool(aligned),
            "opposed": bool(opposed and not aligned),
            "source_count": len(sources),
            "sources": sources,
        }

    @staticmethod
    def _nearest_value_at_or_before(points: list[tuple[int, float]], timestamp_ms: int) -> float | None:
        best: float | None = None
        for point_ts, value in points:
            if point_ts <= timestamp_ms:
                best = value
            else:
                break
        return best

    def v5_momentum_oi_context(self, position: dict[str, Any]) -> dict[str, Any]:
        guard = position.get("v5_momentum_guard") if isinstance(position.get("v5_momentum_guard"), dict) else {}
        opened = self.position_opened_at(position)
        if opened is None:
            return {**guard, "oi_status": "missing_opened_at"}
        opened = opened.astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        age_minutes = (now - opened).total_seconds() / 60.0
        if age_minutes < float(self.config.v5_momentum_guard_first_check_minutes):
            return {**guard, "oi_status": "waiting_first_check", "age_minutes": round(age_minutes, 3)}
        last_checked = guard.get("oi_last_checked_at")
        refresh_seconds = max(5.0, float(self.config.v5_momentum_guard_oi_refresh_seconds))
        if last_checked and self._v5_momentum_iso_is_fresh_seconds(last_checked, refresh_seconds):
            return dict(guard)
        symbol = str(position.get("symbol") or "").upper().strip()
        recent_base_time = max(opened, now - timedelta(minutes=30))
        start_ms = int((recent_base_time - timedelta(minutes=10)).timestamp() * 1000)
        target_ms = int(now.timestamp() * 1000)
        end_ms = target_ms
        try:
            raw = self.client.public_get(
                "/futures/data/openInterestHist",
                {"symbol": symbol, "period": "5m", "startTime": start_ms, "endTime": end_ms, "limit": 100},
            )
            points: list[tuple[int, float]] = []
            for row in raw if isinstance(raw, list) else []:
                if not isinstance(row, dict):
                    continue
                ts = int(row.get("timestamp") or 0)
                value = self._v5_momentum_safe_float(row.get("sumOpenInterestValue") or row.get("sumOpenInterest"))
                if ts > 0 and value > 0:
                    points.append((ts, value))
            points.sort(key=lambda item: item[0])
            base = self._nearest_value_at_or_before(points, int(recent_base_time.timestamp() * 1000))
            target = self._nearest_value_at_or_before(points, target_ms)
            if base is None and points:
                base = points[0][1]
            if target is None and points:
                target = points[-1][1]
            if base is None or target is None or base <= 0:
                guard.update({"oi_status": "unavailable", "oi_last_checked_at": utc_now_iso()})
            else:
                oi_30m_pct = (target - base) / base * 100.0
                guard.update({
                    "oi_status": "ready",
                    "oi_30m_pct": round(oi_30m_pct, 6),
                    "oi_expanded": bool(oi_30m_pct >= float(self.config.v5_momentum_guard_oi_expand_pct)),
                    "oi_base": round(base, 6),
                    "oi_30m": round(target, 6),
                    "oi_window_minutes": round((now - recent_base_time).total_seconds() / 60.0, 3),
                    "oi_last_checked_at": utc_now_iso(),
                    "age_minutes": round(age_minutes, 3),
                })
        except Exception as exc:
            guard.update({
                "oi_status": "error",
                "oi_error": f"{type(exc).__name__}: {exc}"[:300],
                "oi_last_checked_at": utc_now_iso(),
                "age_minutes": round(age_minutes, 3),
            })
        position["v5_momentum_guard"] = guard
        return dict(guard)

    @staticmethod
    def v5_momentum_strength_level(score: float) -> str:
        if score >= 75.0:
            return "strong"
        if score >= 55.0:
            return "medium"
        if score >= 35.0:
            return "weak"
        return "exhausted"

    @staticmethod
    def v5_momentum_strength_level_zh(level: str) -> str:
        return {
            "strong": "强",
            "medium": "中",
            "weak": "弱",
            "exhausted": "衰竭",
        }.get(str(level or ""), "未知")

    @staticmethod
    def v5_momentum_strength_tp_fraction(level: str, threshold: float) -> float:
        level = str(level or "weak")
        threshold = float(threshold or 0.0)
        table = {
            "strong": [(20.0, 0.10), (30.0, 0.10), (50.0, 0.15), (70.0, 0.15), (100.0, 0.15)],
            "medium": [(20.0, 0.20), (30.0, 0.20), (50.0, 0.25), (70.0, 0.25), (100.0, 0.10)],
            "weak": [(20.0, 0.35), (30.0, 0.30), (50.0, 0.35), (70.0, 1.00), (100.0, 1.00)],
            "exhausted": [(20.0, 0.70), (30.0, 0.20), (50.0, 0.10), (70.0, 1.00), (100.0, 1.00)],
        }.get(level, [])
        selected = 0.0
        for boundary, fraction in table:
            if threshold <= boundary + 1e-9:
                selected = fraction
                break
        if selected <= 0.0 and table:
            selected = table[-1][1]
        return clamp(selected, 0.0, 1.0)

    def v5_momentum_strength_context(
        self,
        signal: V6Signal,
        position: dict[str, Any],
        profit_tracker: dict[str, Any],
    ) -> dict[str, Any]:
        side = str(position.get("side") or "").lower()
        components: dict[str, dict[str, Any]] = {}

        volume = (signal.indicators or {}).get("tick_volume_15m") if isinstance(signal.indicators, dict) else {}
        if not isinstance(volume, dict):
            volume = {}
        volume_ratio = max(
            self._v5_momentum_safe_float(volume.get("volume_vs_ma5_ratio")),
            self._v5_momentum_safe_float(volume.get("quote_volume_vs_ma5_ratio")),
        )
        if volume_ratio >= 3.0:
            volume_score = 25.0
        elif volume_ratio >= 2.0:
            volume_score = 20.0
        elif volume_ratio >= 1.5:
            volume_score = 15.0
        elif volume_ratio >= 1.05:
            volume_score = 8.0
        else:
            volume_score = 0.0
        components["volume_15m"] = {
            "score": round(volume_score, 6),
            "ratio": round(volume_ratio, 6),
            "active": bool(volume.get("active")),
        }

        current_roi = self._v5_momentum_safe_float(profit_tracker.get("current_roi_pct"))
        max_roi = self._v5_momentum_safe_float(profit_tracker.get("max_roi_pct"), current_roi)
        giveback = max(0.0, self._v5_momentum_safe_float(profit_tracker.get("giveback_pct"), max_roi - current_roi))
        if current_roi >= 30.0 and giveback <= 3.0:
            price_score = 20.0
        elif current_roi >= 20.0 and giveback <= 5.0:
            price_score = 16.0
        elif current_roi >= 10.0 and giveback <= 8.0:
            price_score = 10.0
        elif current_roi > 0.0:
            price_score = 5.0
        else:
            price_score = 0.0
        components["price_quality"] = {
            "score": round(price_score, 6),
            "current_roi_pct": round(current_roi, 6),
            "max_roi_pct": round(max_roi, 6),
            "giveback_pct": round(giveback, 6),
        }

        fifteen_m = signal.fifteen_m if isinstance(signal.fifteen_m, dict) else {}
        ma20_trend = fifteen_m.get("ma20_trend") if isinstance(fifteen_m.get("ma20_trend"), dict) else {}
        price_distance = self._v5_momentum_safe_float(ma20_trend.get("price_distance_pct"))
        if side == "long":
            if ma20_trend.get("bullish_trend_alive"):
                ma20_score = 20.0
            elif price_distance >= 0:
                ma20_score = 10.0
            elif ma20_trend.get("bearish_trend_alive"):
                ma20_score = 0.0
            else:
                ma20_score = 5.0
        else:
            if ma20_trend.get("bearish_trend_alive"):
                ma20_score = 20.0
            elif price_distance <= 0:
                ma20_score = 10.0
            elif ma20_trend.get("bullish_trend_alive"):
                ma20_score = 0.0
            else:
                ma20_score = 5.0
        components["ma20_15m"] = {
            "score": round(ma20_score, 6),
            "state": ma20_trend.get("state"),
            "price_distance_pct": round(price_distance, 6),
        }

        oi = self.v5_momentum_oi_context(position)
        oi_expanded = bool(oi.get("oi_expanded"))
        oi_status = str(oi.get("oi_status") or "")
        if oi_status == "ready" and oi_expanded and current_roi > 0:
            oi_score = 20.0
        elif oi_status == "ready" and oi_expanded:
            oi_score = 12.0
        elif oi_status == "ready":
            oi_score = 4.0
        elif oi_status.startswith("waiting") or not oi_status:
            oi_score = 8.0
        else:
            oi_score = 5.0
        components["oi"] = {
            "score": round(oi_score, 6),
            "status": oi_status,
            "oi_30m_pct": oi.get("oi_30m_pct"),
            "oi_expanded": oi_expanded,
        }

        panel = self.v5_momentum_panel_alignment(str(position.get("symbol") or signal.symbol), side)
        squeeze = self.v5_smart_signal_short_squeeze_context(
            panel=panel,
            ma20_trend=ma20_trend,
            volume_ratio=volume_ratio,
            oi_context=oi,
            price=float(signal.price),
        )
        squeeze_active = bool(squeeze.get("active"))
        if squeeze_active and side == "long":
            panel_score = 15.0
        elif squeeze_active and side == "short":
            panel_score = 0.0
        elif panel.get("aligned"):
            panel_score = 15.0
        elif panel.get("opposed"):
            panel_score = 0.0
        elif int(panel.get("source_count") or 0) > 0:
            panel_score = 8.0
        else:
            panel_score = 5.0
        components["panel"] = {
            "score": round(panel_score, 6),
            "aligned": bool(panel.get("aligned")),
            "opposed": bool(panel.get("opposed")),
            "source_count": int(panel.get("source_count") or 0),
        }
        components["short_squeeze"] = {
            "score": 15.0 if squeeze_active and side == "long" else 0.0,
            **squeeze,
        }

        score = clamp(sum(float(item.get("score") or 0.0) for item in components.values()), 0.0, 100.0)
        level = self.v5_momentum_strength_level(score)
        return {
            "enabled": True,
            "score": round(score, 6),
            "level": level,
            "level_zh": self.v5_momentum_strength_level_zh(level),
            "components": components,
        }

    def v5_momentum_guard_candidate(
        self,
        signal: V6Signal,
        position: dict[str, Any],
        trigger: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.config.v5_momentum_guard_enabled:
            return None
        opened = self.position_opened_at(position)
        if opened is None:
            return None
        elapsed_minutes = max(0.0, (datetime.now(timezone.utc) - opened.astimezone(timezone.utc)).total_seconds() / 60.0)
        max_roi = self._v5_momentum_safe_float(trigger.get("max_roi_pct"))
        current_roi = self._v5_momentum_safe_float(trigger.get("current_roi_pct"))
        side = str(position.get("side") or "").lower()
        oi_context = self.v5_momentum_oi_context(position)
        panel = self.v5_momentum_panel_alignment(str(position.get("symbol") or signal.symbol), side)
        oi_expanded = bool(oi_context.get("oi_expanded"))
        panel_aligned = bool(panel.get("aligned"))
        any_support = oi_expanded or panel_aligned
        strong_support = oi_expanded and panel_aligned
        context = {
            **trigger,
            "source": "v5_momentum_guard",
            "elapsed_minutes": round(elapsed_minutes, 3),
            "current_roi_pct": round(current_roi, 6),
            "max_roi_pct": round(max_roi, 6),
            "oi": oi_context,
            "panel": panel,
            "oi_expanded": oi_expanded,
            "panel_aligned": panel_aligned,
            "any_support": any_support,
            "strong_support": strong_support,
        }
        checks = [
            (
                "v5_momentum_weak_120m_exit",
                float(self.config.v5_momentum_guard_final_check_minutes),
                float(self.config.v5_momentum_guard_final_min_roi_pct),
                None,
                "V5动能仓位：开仓后120分钟仍未达到18%回报，动能失败，清掉剩余仓位",
            ),
            (
                "v5_momentum_weak_60m_exit",
                float(self.config.v5_momentum_guard_mid_check_minutes),
                float(self.config.v5_momentum_guard_mid_min_roi_pct),
                "strong_support",
                "V5动能仓位：开仓后60分钟仍未达到10%回报，且OI扩张+聪明钱同向不同时满足，清掉剩余仓位",
            ),
            (
                "v5_momentum_weak_30m_reduce",
                float(self.config.v5_momentum_guard_first_check_minutes),
                float(self.config.v5_momentum_guard_first_min_roi_pct),
                "any_support",
                "V5动能仓位：开仓后30分钟仍未达到5%回报，且OI/聪明钱没有同向支持，先减仓50%",
            ),
        ]
        for name, minutes, min_roi, required_support, reason in checks:
            if elapsed_minutes < minutes or max_roi >= min_roi or self.profit_guard_stage_executed(position, name):
                continue
            if required_support == "strong_support" and strong_support:
                continue
            if required_support == "any_support" and any_support:
                continue
            remaining = clamp(float(position.get("remaining_fraction") or 1.0), 0.0, 1.0)
            if remaining <= 0.001:
                return None
            if name.endswith("30m_reduce"):
                fraction = min(remaining, float(self.config.v5_momentum_guard_first_fraction))
            else:
                fraction = remaining
            partial = self.profit_guard_take_profit(
                position,
                name=name,
                fraction_of_initial=fraction,
                guard_type="v5_momentum_guard",
                trigger={**context, "check_minutes": minutes, "required_min_roi_pct": min_roi},
            )
            if partial:
                return {
                    "kind": "partial_take_profit",
                    "type": "v5_momentum_guard",
                    "partial": partial,
                    "trigger": {**context, "check_minutes": minutes, "required_min_roi_pct": min_roi},
                    "reason": reason,
                }
        return None

    def v5_no_news_momentum_take_profit_candidate(
        self,
        signal: V6Signal,
        position: dict[str, Any],
        profit_tracker: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.is_v5_no_news_momentum_position(position):
            return None
        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"}:
            return None
        current_roi = float(profit_tracker.get("current_roi_pct") or self.current_position_roi_pct(signal, position))
        max_roi = float(profit_tracker.get("max_roi_pct") or current_roi)
        giveback = max(0.0, max_roi - current_roi)
        thresholds = [float(value) for value in self.config.v5_no_news_momentum_tp_thresholds if float(value) > 0]
        fractions = [float(value) for value in self.config.v5_no_news_momentum_tp_fractions if float(value) > 0]
        if len(thresholds) < 3 or len(fractions) < 3:
            return None
        strength = (
            self.v5_momentum_strength_context(signal, position, profit_tracker)
            if self.config.v5_momentum_strength_tp_enabled
            else {}
        )
        metadata = self.v5_no_news_momentum_position_metadata(position)
        trigger = {
            "price": round(float(signal.price), 8),
            "current_roi_pct": round(current_roi, 6),
            "max_roi_pct": round(max_roi, 6),
            "giveback_pct": round(giveback, 6),
            "source": "v5_no_news_momentum_tp",
            "momentum_strength": strength,
            "metadata": {
                "sentiment_source": metadata.get("sentiment_source") or metadata.get("source") or "",
                "sentiment_score": metadata.get("sentiment_score") or metadata.get("hotness_score"),
                "sentiment_mentions": metadata.get("sentiment_mentions") or metadata.get("mention_count"),
                "has_news_confirmation": self.candidate_has_news_confirmation(metadata),
                "price_momentum_only": self.candidate_is_price_momentum_only(metadata),
            },
        }
        stage_thresholds = list(thresholds)
        if self.config.v5_momentum_strength_tp_enabled:
            for threshold in self.config.v5_momentum_strength_extra_thresholds:
                value = float(threshold)
                if value > 0 and all(abs(value - existing) > 1e-9 for existing in stage_thresholds):
                    stage_thresholds.append(value)
        stage_thresholds = sorted(stage_thresholds)
        level = str(strength.get("level") or "") if strength else ""
        for index, threshold in enumerate(stage_thresholds):
            name = f"v5_no_news_roi_tp_{int(threshold) if float(threshold).is_integer() else threshold:g}"
            if self.config.v5_momentum_strength_tp_enabled:
                fraction = self.v5_momentum_strength_tp_fraction(level, threshold)
            else:
                if index >= len(fractions):
                    continue
                fraction = fractions[index]
            if max_roi < threshold or self.profit_guard_stage_executed(position, name):
                continue
            partial = self.profit_guard_take_profit(
                position,
                name=name,
                fraction_of_initial=fraction,
                guard_type="v5_no_news_momentum_tp",
                trigger={**trigger, "threshold_pct": threshold},
            )
            if partial:
                return {
                    "kind": "partial_take_profit",
                    "type": "v5_no_news_momentum_tp",
                    "partial": partial,
                    "trigger": {**trigger, "threshold_pct": threshold},
                    "reason": (
                        f"V5无新闻确认榜单热度仓位：8X回报达到/曾达到{threshold:.0f}%，"
                        f"动能={strength.get('level_zh', '固定') if strength else '固定'}"
                        f"({float(strength.get('score') or 0.0):.1f})，止盈{fraction * 100:.0f}%"
                    ),
                }
        momentum_guard = self.v5_momentum_guard_candidate(signal, position, trigger)
        if momentum_guard is not None:
            return momentum_guard
        if not self.profit_guard_stage_executed(position, "v5_no_news_roi_tp_50"):
            second_threshold = thresholds[1]
            final_threshold = thresholds[2]
            giveback_threshold = float(self.config.v5_no_news_momentum_final_giveback_pct)
            if max_roi >= second_threshold and max_roi < final_threshold and giveback >= giveback_threshold:
                remaining = clamp(float(position.get("remaining_fraction") or 1.0), 0.0, 1.0)
                if remaining <= 0.001:
                    return None
                partial = self.profit_guard_take_profit(
                    position,
                    name="v5_no_news_final_giveback_exit",
                    fraction_of_initial=remaining,
                    guard_type="v5_no_news_momentum_final_giveback_exit",
                    trigger={**trigger, "giveback_threshold_pct": giveback_threshold},
                )
                if partial:
                    return {
                        "kind": "partial_take_profit",
                        "type": "v5_no_news_momentum_final_giveback_exit",
                        "partial": partial,
                        "trigger": {**trigger, "giveback_threshold_pct": giveback_threshold},
                        "reason": (
                            "V5无新闻确认榜单热度仓位：最后10%未到50%目标，"
                            f"从最高回报回吐{giveback:.2f}% >= {giveback_threshold:.2f}%，清掉剩余仓位"
                        ),
                    }
        return None

    def next_roi_take_profit(self, signal: V6Signal, position: dict[str, Any]) -> tuple[list[dict[str, Any]], float] | None:
        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"}:
            return None
        stages = self.ensure_position_roi_take_profit_stages(position)
        if not stages:
            return None
        remaining = clamp(float(position.get("remaining_fraction") or 1.0), 0.0, 1.0)
        if remaining <= 0:
            return None
        touch_price = self.staged_take_profit_touch_price(signal, side)
        roi_pct = self.current_position_roi_pct(signal, position, touch_price=touch_price)
        if roi_pct <= 0:
            return None
        trigger_roi_pct = roi_pct
        trigger_source = "recent_1m_extreme_or_mark"
        profit_guard = position.get("profit_guard") if isinstance(position.get("profit_guard"), dict) else {}
        max_roi_pct = float(profit_guard.get("max_roi_pct") or 0.0)
        if max_roi_pct > trigger_roi_pct:
            trigger_roi_pct = max_roi_pct
            trigger_source = "profit_guard_max_roi_lock"
        touched_stages: list[dict[str, Any]] = []
        total_fraction = 0.0
        for stage in stages:
            status = str(stage.get("status") or "pending")
            if status == "retry_wait":
                retry_after = str(stage.get("retry_after") or "")
                retry_blocked = False
                if retry_after:
                    try:
                        retry_blocked = datetime.fromisoformat(retry_after.replace("Z", "+00:00")) > datetime.now(timezone.utc)
                    except Exception:
                        retry_blocked = False
                if retry_blocked:
                    continue
                stage["status"] = "pending"
                stage.pop("retry_after", None)
                stage["retry_released_at"] = utc_now_iso()
            elif status != "pending":
                continue
            threshold = float(stage.get("roi_threshold_pct") or 0.0)
            if threshold <= 0 or trigger_roi_pct < threshold:
                continue
            fraction = clamp(float(stage.get("fraction_of_initial") or 0.0), 0.0, 1.0)
            if fraction <= 0:
                continue
            stage["trigger_roi_pct"] = round(trigger_roi_pct, 6)
            stage["trigger_current_roi_pct"] = round(roi_pct, 6)
            stage["trigger_price"] = round(touch_price, 8)
            stage["trigger_source"] = trigger_source
            stage["checked_at"] = utc_now_iso()
            touched_stages.append(stage)
            total_fraction += fraction
        if not touched_stages:
            return None
        close_fraction = 1.0 if total_fraction >= remaining - 0.001 else clamp(total_fraction / remaining, 0.0, 1.0)
        return touched_stages, close_fraction

    def next_staged_take_profit(self, signal: V6Signal, position: dict[str, Any]) -> tuple[list[dict[str, Any]], float] | None:
        side = str(position.get("side") or "").lower()
        price = float(signal.price)
        touch_price = self.staged_take_profit_touch_price(signal, side)
        stages = self.ensure_position_take_profit_stages(position, signal)
        if side not in {"long", "short"} or not stages:
            return None
        if self.config.roi_take_profit_enabled:
            roi_pct = self.current_position_roi_pct(signal, position, touch_price=touch_price)
            if roi_pct < float(self.config.roi_take_profit_bb_resume_threshold):
                position["bb_take_profit_waiting_for_roi"] = {
                    "enabled": True,
                    "roi_pct": round(roi_pct, 6),
                    "resume_threshold_pct": float(self.config.roi_take_profit_bb_resume_threshold),
                    "resume_max_pct": float(self.config.roi_take_profit_bb_resume_max),
                    "checked_at": utc_now_iso(),
                }
                return None
        remaining = clamp(float(position.get("remaining_fraction") or 1.0), 0.0, 1.0)
        if remaining <= 0:
            return None
        touched_stages: list[dict[str, Any]] = []
        total_fraction = 0.0
        for stage in stages:
            if str(stage.get("status") or "pending") != "pending":
                continue
            target_price = float(stage.get("price") or 0.0)
            currently_touched = self.staged_take_profit_touched(side, touch_price, target_price)
            delayed_touch_active = False
            if currently_touched:
                self.mark_staged_take_profit_touched(stage, touch_price=touch_price, target_price=target_price)
                stage["exit_touch_source"] = "current_touch"
            else:
                delayed_touch_active = self.staged_take_profit_touch_pending_active(stage)
                if delayed_touch_active:
                    stage["exit_touch_source"] = "delayed_15m_volume_confirmation"
            if not currently_touched and not delayed_touch_active:
                continue
            fraction_of_initial = clamp(float(stage.get("fraction_of_initial") or 0.0), 0.0, 1.0)
            if fraction_of_initial <= 0:
                continue
            stage["exit_check_price"] = round(touch_price, 8)
            touched_stages.append(stage)
            total_fraction += fraction_of_initial
        if not touched_stages:
            return None
        if not self.take_profit_volume_guard_allows_exit(signal, side, touched_stages):
            return None
        effective_total_fraction = total_fraction
        executed_roi_fraction = self.roi_take_profit_executed_fraction(position)
        if self.config.roi_take_profit_enabled and executed_roi_fraction > 0:
            pending_total = sum(
                clamp(float(stage.get("fraction_of_initial") or 0.0), 0.0, 1.0)
                for stage in stages
                if str(stage.get("status") or "pending") == "pending"
            )
            if pending_total > 0:
                effective_total_fraction = remaining * clamp(total_fraction / pending_total, 0.0, 1.0)
                for stage in touched_stages:
                    raw_fraction = clamp(float(stage.get("fraction_of_initial") or 0.0), 0.0, 1.0)
                    stage["effective_fraction_of_initial"] = round(remaining * raw_fraction / pending_total, 6)
                    stage["bb_remaining_distribution"] = {
                        "enabled": True,
                        "remaining_fraction_before": round(remaining, 6),
                        "pending_bb_fraction_total": round(pending_total, 6),
                        "executed_roi_fraction": round(executed_roi_fraction, 6),
                    }
        close_fraction = 1.0 if effective_total_fraction >= remaining - 0.001 else clamp(effective_total_fraction / remaining, 0.0, 1.0)
        return touched_stages, close_fraction

    def opposite_exit_threshold_for_position(self, position: dict[str, Any]) -> float:
        risk_plan = position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {}
        horizon = str(risk_plan.get("horizon") or "")
        if horizon == "1d_trend":
            return float(self.config.exit_opposite_threshold_1d_trend)
        if horizon == "4h_trend":
            return float(self.config.exit_opposite_threshold_4h_trend)
        return float(self.config.exit_opposite_threshold)

    @staticmethod
    def position_max_hold_expired(position: dict[str, Any]) -> bool:
        risk_plan = position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {}
        max_days = float(risk_plan.get("max_hold_days") or position.get("max_hold_days") or 0.0)
        if max_days <= 0:
            return False
        opened_at = position.get("opened_at")
        if not opened_at:
            return False
        try:
            opened = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
        except Exception:
            return False
        return datetime.now(timezone.utc) >= opened + timedelta(days=max_days)

    def logic_weakening_exit_candidate(self, signal: V6Signal, position: dict[str, Any]) -> dict[str, Any] | None:
        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"} or not bool(self.config.logic_weakening_exit_enabled):
            return None
        h4 = signal.h4 if isinstance(signal.h4, dict) else {}
        right_side = h4.get("right_side") if isinstance(h4.get("right_side"), dict) else {}
        guard = right_side.get("logic_weakening_exit") if isinstance(right_side.get("logic_weakening_exit"), dict) else {}
        if not guard.get("enabled"):
            return None
        required = max(1, int(guard.get("required_closed_candles") or self.config.logic_weakening_exit_required_4h_candles))
        checked = int(guard.get("checked_closed_candles") or 0)
        if checked < required:
            return None
        weak_key = "long_logic_weakened" if side == "long" else "short_logic_weakened"
        weak_count_key = "long_weakened_count" if side == "long" else "short_weakened_count"
        weakened = bool(guard.get(weak_key))
        weak_count = int(guard.get(weak_count_key) or 0)
        watch = {
            "enabled": True,
            "side": side,
            "timeframe": "4h",
            "basis": "ma20",
            "required_closed_candles": required,
            "checked_closed_candles": checked,
            "weakened_count": weak_count,
            "status": "exit_ready" if weakened else "watching" if weak_count > 0 else "healthy",
            "checked_at": utc_now_iso(),
            "guard": guard,
            "h4_right_side_state": right_side.get("state"),
            "h4_long_confirmed": bool(right_side.get("long_confirmed")),
            "h4_short_confirmed": bool(right_side.get("short_confirmed")),
        }
        position["logic_weakening_watch"] = watch
        if not weakened:
            return None
        action_text = "long" if side == "long" else "short"
        opposite = "short" if side == "long" else "long"
        watch["reason"] = (
            f"{action_text} original 4H logic weakened for {required} closed 4H candles; "
            f"MA20 no longer supports original {action_text} thesis, close before stop loss"
        )
        watch["opposite_context"] = {
            "side": opposite,
            "confirmed": bool(right_side.get("short_confirmed") if side == "long" else right_side.get("long_confirmed")),
        }
        return watch

    def daily_ma7_short_reversal_exit_candidate(self, signal: V6Signal, position: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(position.get("symbol") or signal.symbol or "").upper()
        side = str(position.get("side") or "").lower()
        target_symbols = {item.upper() for item in self.config.daily_ma7_short_reversal_symbols}
        profile = symbol_bb_entry_profile(symbol)
        if (
            side != "short"
            or not bool(self.config.daily_ma7_short_reversal_exit_enabled)
            or (target_symbols and symbol not in target_symbols)
            or bool(profile.get("ignore_market_daily_ma7_reversal"))
        ):
            return None
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        market = levels.get("market_daily_ma7_reversal") if isinstance(levels.get("market_daily_ma7_reversal"), dict) else {}
        if not market.get("enabled") or not market.get("ready"):
            return None
        held = bool(market.get("reversal_confirmed"))
        held_count = int(market.get("held_count") or 0)
        ma7_held_count = int(market.get("ma7_held_count") or 0)
        required = len(market.get("market_symbols") or []) if market.get("require_all") else 1
        watch = {
            "enabled": True,
            "symbol": symbol,
            "side": side,
            "timeframe": "1d",
            "basis": "BTC/ETH daily MA7 first checkpoint + MA20 second checkpoint",
            "status": "exit_ready" if held else "watching" if ma7_held_count > 0 or int(market.get("reclaimed_count") or 0) > 0 else "healthy",
            "held_count": held_count,
            "ma7_held_count": ma7_held_count,
            "ma20_held_count": int(market.get("ma20_held_count") or 0),
            "required_closed_candles": required,
            "market_symbols": market.get("market_symbols") or [],
            "require_all": bool(market.get("require_all")),
            "checked_at": utc_now_iso(),
            "market_daily_ma7_reversal": market,
        }
        position["daily_ma7_short_reversal_watch"] = watch
        if not held:
            return None
        watch["reason"] = (
            f"{symbol} short global daily reversal: BTC/ETH 1D first reclaimed MA7 and then held above MA20; "
            f"close short because market bearish thesis has failed at the second daily checkpoint"
        )
        return watch

    def daily_ma7_short_reversal_entry_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(signal.symbol or "").upper()
        target_symbols = {item.upper() for item in self.config.daily_ma7_short_reversal_symbols}
        profile = symbol_bb_entry_profile(symbol)
        if (
            action != "open_short"
            or not bool(self.config.daily_ma7_short_reversal_exit_enabled)
            or (target_symbols and symbol not in target_symbols)
            or bool(profile.get("ignore_market_daily_ma7_reversal"))
        ):
            return None
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        market = levels.get("market_daily_ma7_reversal") if isinstance(levels.get("market_daily_ma7_reversal"), dict) else {}
        if not market.get("ready") or not market.get("reversal_confirmed"):
            return None
        return {
            "enabled": True,
            "blocked": True,
            "action": action,
            "side": "short",
            "reason": "short_blocked_market_daily_ma20_reclaim",
            "message": "entry blocked: BTC/ETH daily trend passed MA7 first checkpoint and MA20 second checkpoint; market bearish thesis is invalidated",
            "market_daily_ma7_reversal": market,
        }

    def daily_ma20_long_reversal_exit_candidate(self, signal: V6Signal, position: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(position.get("symbol") or signal.symbol or "").upper()
        side = str(position.get("side") or "").lower()
        if side != "long" or not bool(self.config.daily_ma7_short_reversal_exit_enabled):
            return None
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        daily = levels.get("daily_ma7_trend") if isinstance(levels.get("daily_ma7_trend"), dict) else {}
        if not daily.get("ready"):
            return None
        ma7_loss_held = bool(daily.get("ma7_loss_held") or (daily.get("ma7_loss_hold") or {}).get("held"))
        ma20_loss_held = bool(daily.get("ma20_loss_held") or (daily.get("ma20_loss_hold") or {}).get("held"))
        invalidated = bool(daily.get("long_reversal_on_ma20_loss") or (ma7_loss_held and ma20_loss_held))
        watch = {
            "enabled": True,
            "symbol": symbol,
            "side": side,
            "timeframe": "1d",
            "basis": "1D MA7 first checkpoint + MA20 second checkpoint",
            "status": "exit_ready" if invalidated else "watching" if ma7_loss_held or bool(daily.get("ma7_lost")) else "healthy",
            "daily_ma7_lost": bool(daily.get("ma7_lost")),
            "daily_ma7_loss_held": ma7_loss_held,
            "daily_ma20_lost": bool(daily.get("ma20_lost")),
            "daily_ma20_loss_held": ma20_loss_held,
            "checked_at": utc_now_iso(),
            "daily_ma7_trend": daily,
        }
        position["daily_ma20_long_reversal_watch"] = watch
        if not invalidated:
            return None
        watch["reason"] = (
            f"{symbol} long daily reversal: 1D first lost MA7 and then held below MA20; "
            f"close long because daily bullish thesis has failed at the second checkpoint"
        )
        return watch

    def daily_ma20_long_reversal_entry_block(self, signal: V6Signal, action: str, entry_context: dict[str, Any]) -> dict[str, Any] | None:
        if action != "open_long" or not bool(self.config.daily_ma7_short_reversal_exit_enabled):
            return None
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        daily = levels.get("daily_ma7_trend") if isinstance(levels.get("daily_ma7_trend"), dict) else {}
        if not daily.get("ready"):
            return None
        ma7_loss_held = bool(daily.get("ma7_loss_held") or (daily.get("ma7_loss_hold") or {}).get("held"))
        ma20_loss_held = bool(daily.get("ma20_loss_held") or (daily.get("ma20_loss_hold") or {}).get("held"))
        if not bool(daily.get("long_reversal_on_ma20_loss") or (ma7_loss_held and ma20_loss_held)):
            return None
        return {
            "enabled": True,
            "blocked": True,
            "action": action,
            "side": "long",
            "reason": "long_blocked_daily_ma20_loss",
            "message": "entry blocked: 1D long trend first lost MA7 and then lost MA20; daily bullish thesis is invalidated",
            "daily_ma7_trend": daily,
        }

    def smart_signal_risk_guard_candidate(
        self,
        signal: V6Signal,
        position: dict[str, Any],
        tracker: dict[str, Any],
    ) -> dict[str, Any] | None:
        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"}:
            return None
        context = self.v6_smart_signal_context(signal, side)
        if not context.get("adverse_to_side"):
            position["smart_signal_risk_watch"] = {
                "state": "watching" if context.get("has_signal") else "no_signal",
                "checked_at": utc_now_iso(),
                "context": context,
            }
            return None
        if side == "short":
            name = "v6_smart_signal_short_squeeze_reduce_30"
            reason = "short smart-signal squeeze risk: crowded shorts reclaimed 15M MA20; reduce 30%"
        else:
            name = "v6_smart_signal_long_crowded_flush_reduce_30"
            reason = "long smart-signal crowded flush risk: crowded longs lost 15M MA20; reduce 30%"
        trigger = {
            "rule": "v6_smart_signal_risk_guard",
            "symbol": signal.symbol,
            "side": side,
            "price": round(float(signal.price or 0.0), 8),
            "current_roi_pct": round(float(tracker.get("current_roi_pct") or 0.0), 6),
            "max_roi_pct": round(float(tracker.get("max_roi_pct") or 0.0), 6),
            "giveback_pct": round(float(tracker.get("giveback_pct") or 0.0), 6),
            "smart_signal": context,
        }
        position["smart_signal_risk_watch"] = {
            "state": "reduce_ready",
            "checked_at": utc_now_iso(),
            "trigger": trigger,
        }
        partial = self.profit_guard_take_profit(
            position,
            name=name,
            fraction_of_initial=float(self.config.smart_signal_risk_reduce_fraction),
            guard_type="v6_smart_signal_risk_guard",
            trigger=trigger,
        )
        if not partial:
            return None
        return {
            "kind": "partial_take_profit",
            "type": "v6_smart_signal_risk_guard",
            "name": name,
            "reason": reason,
            "partial": partial,
            "trigger": trigger,
        }

    def fifteen_m_h4_profit_giveback_candidate(
        self,
        signal: V6Signal,
        position: dict[str, Any],
        tracker: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not bool(self.config.profit_giveback_guard_enabled):
            return None
        side = str(position.get("side") or "").lower()
        if not self.is_15m_bb_lower_long_position(position):
            return None
        if side == "long" and not self.h4_short_confirmed(signal):
            return None
        roi_pct = float(tracker.get("current_roi_pct") or 0.0)
        max_roi = float(tracker.get("max_roi_pct") or 0.0)
        giveback = float(tracker.get("giveback_pct") or 0.0)
        first_threshold = float(self.config.profit_giveback_guard_first_giveback_pct)
        second_threshold = float(self.config.profit_giveback_guard_second_giveback_pct)
        entry_roi = float(self.config.profit_giveback_guard_entry_roi_pct)
        if max_roi <= entry_roi or giveback < first_threshold:
            return None
        trigger = {
            "rule": "15m_bb_long_profit_giveback_after_h4_short_flip",
            "symbol": signal.symbol,
            "side": side,
            "price": round(float(signal.price), 8),
            "current_roi_pct": round(roi_pct, 6),
            "max_roi_pct": round(max_roi, 6),
            "giveback_pct": round(giveback, 6),
            "entry_roi_exit_pct": entry_roi,
            "h4_short_confirmed": self.h4_short_confirmed(signal),
        }
        if roi_pct <= entry_roi:
            return {
                "kind": "full_exit",
                "type": "profit_giveback_guard",
                "name": "15m_h4_short_giveback_entry_buffer_close",
                "reason": (
                    f"15M BB long met 4H short reversal; profit gave back to roi={roi_pct:.2f}% "
                    f"(peak={max_roi:.2f}%, giveback={giveback:.2f}%), close remaining"
                ),
                "trigger": trigger,
            }
        first_name = "15m_h4_short_giveback_5pct_tp70"
        if not self.profit_guard_stage_executed(position, first_name):
            partial = self.profit_guard_take_profit(
                position,
                name=first_name,
                fraction_of_initial=float(self.config.profit_giveback_guard_first_fraction),
                guard_type="profit_giveback_guard",
                trigger=trigger,
            )
            if partial:
                return {
                    "kind": "partial_take_profit",
                    "type": "profit_giveback_guard",
                    "name": first_name,
                    "reason": (
                        f"15M BB long met 4H short reversal; profit giveback={giveback:.2f}% "
                        f">={first_threshold:.2f}%, take profit 70%"
                    ),
                    "partial": partial,
                    "trigger": trigger,
                }
        second_name = "15m_h4_short_giveback_10pct_tp20"
        if giveback >= second_threshold and not self.profit_guard_stage_executed(position, second_name):
            partial = self.profit_guard_take_profit(
                position,
                name=second_name,
                fraction_of_initial=float(self.config.profit_giveback_guard_second_fraction),
                guard_type="profit_giveback_guard",
                trigger=trigger,
            )
            if partial:
                return {
                    "kind": "partial_take_profit",
                    "type": "profit_giveback_guard",
                    "name": second_name,
                    "reason": (
                        f"15M BB long profit keeps giving back after 4H short reversal; "
                        f"giveback={giveback:.2f}% >={second_threshold:.2f}%, take profit 20%"
                    ),
                    "partial": partial,
                    "trigger": trigger,
                }
        return None

    def h4_lower_reversal_guard_candidate(
        self,
        signal: V6Signal,
        position: dict[str, Any],
        tracker: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not bool(self.config.h4_lower_reversal_guard_enabled):
            return None
        if not self.is_4h_lower_long_position(position):
            return None
        roi_pct = float(tracker.get("current_roi_pct") or 0.0)
        max_roi = float(tracker.get("max_roi_pct") or 0.0)
        giveback = float(tracker.get("giveback_pct") or 0.0)
        age_hours = self.position_age_hours(position)
        h4_long_seen = False
        monitor = position.get("h4_lower_reversal_watch") if isinstance(position.get("h4_lower_reversal_watch"), dict) else {}
        if monitor.get("h4_long_confirmed_seen_at"):
            h4_long_seen = True
        if self.h4_long_confirmed(signal):
            h4_long_seen = True
            monitor.setdefault("h4_long_confirmed_seen_at", utc_now_iso())
            monitor.setdefault("h4_long_confirmed_seen_age_hours", round(age_hours, 6))
        levels = signal.levels if isinstance(signal.levels, dict) else {}
        daily = levels.get("daily_ma7_trend") if isinstance(levels.get("daily_ma7_trend"), dict) else {}
        daily_ready = bool(daily.get("ready"))
        ma7_reclaimed = bool(daily.get("ma7_reclaimed") or daily.get("ma7_reclaim_held"))
        ma20_reclaimed = bool(daily.get("ma20_reclaimed"))
        monitor.update({
            "enabled": True,
            "symbol": signal.symbol,
            "side": "long",
            "age_hours": round(age_hours, 6),
            "current_roi_pct": round(roi_pct, 6),
            "max_roi_pct": round(max_roi, 6),
            "giveback_pct": round(giveback, 6),
            "h4_long_confirmed_now": self.h4_long_confirmed(signal),
            "h4_long_confirmed_seen": h4_long_seen,
            "daily_ready": daily_ready,
            "daily_ma7_reclaimed": ma7_reclaimed,
            "daily_ma20_reclaimed": ma20_reclaimed,
            "checked_at": utc_now_iso(),
        })
        position["h4_lower_reversal_watch"] = monitor
        trigger = {
            "rule": "4h_lower_reversal_guard",
            "symbol": signal.symbol,
            "side": "long",
            "price": round(float(signal.price), 8),
            "age_hours": round(age_hours, 6),
            "current_roi_pct": round(roi_pct, 6),
            "max_roi_pct": round(max_roi, 6),
            "giveback_pct": round(giveback, 6),
            "h4_long_confirmed_seen": h4_long_seen,
            "h4_long_confirmed_now": self.h4_long_confirmed(signal),
            "daily_ma7_reclaimed": ma7_reclaimed,
            "daily_ma20_reclaimed": ma20_reclaimed,
        }
        daily_guard_done = any(
            self.profit_guard_stage_executed(position, name)
            for name in ("h4_lower_1d_no_ma7_tp70", "h4_lower_1d_reclaim_tp30")
        )
        if daily_ready and roi_pct > 0 and not daily_guard_done:
            if not ma7_reclaimed:
                name = "h4_lower_1d_no_ma7_tp70"
                partial = self.profit_guard_take_profit(
                    position,
                    name=name,
                    fraction_of_initial=float(self.config.h4_lower_reversal_guard_no_1d_ma7_fraction),
                    guard_type="h4_lower_reversal_guard",
                    trigger=trigger,
                )
                if partial:
                    return {
                        "kind": "partial_take_profit",
                        "type": "h4_lower_reversal_guard",
                        "name": name,
                        "reason": "4H lower-band long is profitable but 1D has not reclaimed MA7; take profit 70%",
                        "partial": partial,
                        "trigger": trigger,
                    }
            elif ma7_reclaimed or ma20_reclaimed:
                name = "h4_lower_1d_reclaim_tp30"
                partial = self.profit_guard_take_profit(
                    position,
                    name=name,
                    fraction_of_initial=float(self.config.h4_lower_reversal_guard_1d_reclaim_fraction),
                    guard_type="h4_lower_reversal_guard",
                    trigger=trigger,
                )
                if partial:
                    return {
                        "kind": "partial_take_profit",
                        "type": "h4_lower_reversal_guard",
                        "name": name,
                        "reason": "4H lower-band long has 1D MA7/MA20 reclaim confirmation; take profit 30%, leave rest to BB targets",
                        "partial": partial,
                        "trigger": trigger,
                    }
        if (
            age_hours >= float(self.config.h4_lower_reversal_guard_48h_hours)
            and not h4_long_seen
            and roi_pct <= float(self.config.h4_lower_reversal_guard_loss_exit_roi_pct)
        ):
            return {
                "kind": "full_exit",
                "type": "h4_lower_reversal_guard",
                "name": "h4_lower_no_h4_bull_48h_loss_close",
                "reason": (
                    f"4H lower-band long has no 4H bullish flip after 48H and roi={roi_pct:.2f}% "
                    f"<= {float(self.config.h4_lower_reversal_guard_loss_exit_roi_pct):.2f}%, close"
                ),
                "trigger": trigger,
            }
        if roi_pct > 0 and not daily_guard_done:
            if h4_long_seen and age_hours <= float(self.config.h4_lower_reversal_guard_5d_hours):
                name = "h4_lower_h4_bull_5d_tp50"
                partial = self.profit_guard_take_profit(
                    position,
                    name=name,
                    fraction_of_initial=float(self.config.h4_lower_reversal_guard_h4_bull_fraction),
                    guard_type="h4_lower_reversal_guard",
                    trigger=trigger,
                )
                if partial:
                    return {
                        "kind": "partial_take_profit",
                        "type": "h4_lower_reversal_guard",
                        "name": name,
                        "reason": "4H lower-band long saw 4H bullish flip inside 5D; take profit 50%, leave rest to BB targets",
                        "partial": partial,
                        "trigger": trigger,
                    }
            if age_hours >= float(self.config.h4_lower_reversal_guard_5d_hours) and not h4_long_seen:
                name = "h4_lower_no_h4_bull_5d_tp50"
                partial = self.profit_guard_take_profit(
                    position,
                    name=name,
                    fraction_of_initial=float(self.config.h4_lower_reversal_guard_no_h4_bull_fraction),
                    guard_type="h4_lower_reversal_guard",
                    trigger=trigger,
                )
                if partial:
                    return {
                        "kind": "partial_take_profit",
                        "type": "h4_lower_reversal_guard",
                        "name": name,
                        "reason": "4H lower-band long still has no 4H bullish flip after 5D; take profit 50%",
                        "partial": partial,
                        "trigger": trigger,
                    }
        if age_hours >= float(self.config.h4_lower_reversal_guard_5d_hours) and not h4_long_seen:
            first_threshold = float(self.config.profit_giveback_guard_first_giveback_pct)
            second_threshold = float(self.config.profit_giveback_guard_second_giveback_pct)
            entry_roi = float(self.config.profit_giveback_guard_entry_roi_pct)
            if max_roi > entry_roi and giveback >= first_threshold:
                if roi_pct <= entry_roi:
                    return {
                        "kind": "full_exit",
                        "type": "h4_lower_reversal_guard",
                        "name": "h4_lower_no_h4_bull_giveback_entry_buffer_close",
                        "reason": (
                            f"4H lower-band long had no 4H bullish flip after 5D; profit gave back to roi={roi_pct:.2f}% "
                            f"(peak={max_roi:.2f}%, giveback={giveback:.2f}%), close remaining"
                        ),
                        "trigger": trigger,
                    }
                first_name = "h4_lower_no_h4_bull_giveback_5pct_tp70"
                if not self.profit_guard_stage_executed(position, first_name):
                    partial = self.profit_guard_take_profit(
                        position,
                        name=first_name,
                        fraction_of_initial=float(self.config.profit_giveback_guard_first_fraction),
                        guard_type="h4_lower_reversal_guard",
                        trigger=trigger,
                    )
                    if partial:
                        return {
                            "kind": "partial_take_profit",
                            "type": "h4_lower_reversal_guard",
                            "name": first_name,
                            "reason": "4H lower-band long no 4H bull after 5D and profit giveback >=5%; take profit 70%",
                            "partial": partial,
                            "trigger": trigger,
                        }
                second_name = "h4_lower_no_h4_bull_giveback_10pct_tp20"
                if giveback >= second_threshold and not self.profit_guard_stage_executed(position, second_name):
                    partial = self.profit_guard_take_profit(
                        position,
                        name=second_name,
                        fraction_of_initial=float(self.config.profit_giveback_guard_second_fraction),
                        guard_type="h4_lower_reversal_guard",
                        trigger=trigger,
                    )
                    if partial:
                        return {
                            "kind": "partial_take_profit",
                            "type": "h4_lower_reversal_guard",
                            "name": second_name,
                            "reason": "4H lower-band long no 4H bull after 5D and profit giveback >=10%; take profit 20%",
                            "partial": partial,
                            "trigger": trigger,
                        }
        return None

    def v5_overlay_soft_stop_candidate(
        self,
        signal: V6Signal,
        position: dict[str, Any],
        risk_plan: dict[str, Any],
    ) -> dict[str, Any] | None:
        symbol = str(position.get("symbol") or signal.symbol or "").upper()
        if not self.config.v5_overlay_rules_active_for(symbol):
            return None
        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"}:
            return None
        overlay = risk_plan.get("v5_stop_overlay") if isinstance(risk_plan.get("v5_stop_overlay"), dict) else {}
        original_stop = float(
            risk_plan.get("v5_original_structure_stop_price")
            or overlay.get("original_structure_stop_price")
            or 0.0
        )
        if original_stop <= 0:
            return None
        recent = (
            (signal.indicators or {})
            .get("recent_extremes", {})
            .get("15m_closed", {})
        )
        if not isinstance(recent, dict):
            return None
        close_15m = float(recent.get("last_close") or 0.0)
        if close_15m <= 0:
            return None
        triggered = close_15m <= original_stop if side == "long" else close_15m >= original_stop
        if not triggered:
            position["v5_soft_stop_watch"] = {
                "state": "watching",
                "side": side,
                "original_structure_stop_price": round(original_stop, 8),
                "last_closed_15m_close": round(close_15m, 8),
                "last_closed_15m_open_time": recent.get("last_open_time"),
                "hard_stop_price": round(float(position.get("stop_loss_price") or 0.0), 8),
            }
            return None
        return {
            "enabled": True,
            "side": side,
            "reason": f"{side} V5 original structure stop confirmed by 15M close",
            "original_structure_stop_price": round(original_stop, 8),
            "last_closed_15m_close": round(close_15m, 8),
            "last_closed_15m_open_time": recent.get("last_open_time"),
            "hard_stop_price": round(float(position.get("stop_loss_price") or 0.0), 8),
        }

    def exit_intent_if_needed(self, signal: V6Signal, position: dict[str, Any]) -> V6OrderIntent | None:
        side = str(position.get("side") or "").lower()
        price = float(signal.price)
        stop_price = float(position.get("stop_loss_price") or 0.0)
        take_profit_price = float(position.get("take_profit_price") or 0.0)
        should_exit = False
        reason = ""
        risk_plan = position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {}
        staged_take_profit: tuple[dict[str, Any], float] | None = None
        roi_take_profit: tuple[dict[str, Any], float] | None = None
        profit_guard_take_profit: dict[str, Any] | None = None
        profit_guard_full_exit: dict[str, Any] | None = None
        daily_ma7_short_reversal_exit: dict[str, Any] | None = None
        daily_ma20_long_reversal_exit: dict[str, Any] | None = None
        logic_weakening_exit: dict[str, Any] | None = None
        smart_signal_risk_exit: dict[str, Any] | None = None
        v5_soft_stop_exit: dict[str, Any] | None = None
        hold_expired = self.position_max_hold_expired(position)
        profit_tracker = self.update_position_profit_tracker(signal, position) if side in {"long", "short"} else {}
        if side == "short":
            if hold_expired:
                should_exit = True
                reason = "max hold days reached"
            elif stop_price > 0 and price >= stop_price:
                should_exit = True
                reason = "short stop loss touched"
            elif (v5_soft_stop_exit := self.v5_overlay_soft_stop_candidate(signal, position, risk_plan)) is not None:
                should_exit = True
                reason = str(v5_soft_stop_exit.get("reason") or "short V5 soft stop confirmed")
            elif (daily_ma7_short_reversal_exit := self.daily_ma7_short_reversal_exit_candidate(signal, position)) is not None:
                should_exit = True
                reason = str(daily_ma7_short_reversal_exit.get("reason") or "short daily MA7 reversal")
            elif (logic_weakening_exit := self.logic_weakening_exit_candidate(signal, position)) is not None:
                should_exit = True
                reason = str(logic_weakening_exit.get("reason") or "short original 4H logic weakened")
            elif (smart_signal_risk_exit := self.smart_signal_risk_guard_candidate(signal, position, profit_tracker)) is not None:
                if smart_signal_risk_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = smart_signal_risk_exit
                    smart_signal_risk_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or smart_signal_risk_exit or {}).get("reason") or "short smart-signal risk guard")
            elif (profit_guard_full_exit := self.v5_no_news_momentum_take_profit_candidate(signal, position, profit_tracker)) is not None:
                if profit_guard_full_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = profit_guard_full_exit
                    profit_guard_full_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or profit_guard_full_exit or {}).get("reason") or "V5 no-news momentum short take profit")
            elif (roi_take_profit := self.next_roi_take_profit(signal, position)) is not None:
                stages, close_fraction = roi_take_profit
                names = ",".join(str(stage.get("name") or "") for stage in stages)
                total_initial = sum(float(stage.get("fraction_of_initial") or 0.0) for stage in stages)
                max_roi = max(float(stage.get("trigger_roi_pct") or 0.0) for stage in stages)
                should_exit = True
                reason = (
                    f"ROI short take profit threshold touched: {names} "
                    f"roi={max_roi:.2f}% "
                    f"close_initial={total_initial:.2f} "
                    f"close_remaining={close_fraction:.4f}"
                )
            elif (staged_take_profit := self.next_staged_take_profit(signal, position)) is not None:
                stages, close_fraction = staged_take_profit
                names = ",".join(str(stage.get("name") or "") for stage in stages)
                total_initial = sum(
                    float(stage.get("effective_fraction_of_initial") or stage.get("fraction_of_initial") or 0.0)
                    for stage in stages
                )
                should_exit = True
                reason = (
                    f"staged short take profit touched with current/delayed 15M volume confirmation: {names} "
                    f"close_initial={total_initial:.2f} "
                    f"close_remaining={close_fraction:.4f}"
                )
            elif not self.ensure_position_take_profit_stages(position, signal) and take_profit_price > 0 and price <= take_profit_price:
                should_exit = True
                reason = "short take profit touched"
            elif (
                not self.config.technical_entry_only_for(signal.symbol)
                and signal.long_probability >= self.opposite_exit_threshold_for_position(position)
            ):
                should_exit = True
                reason = "opposite long probability threshold"
            action = "close_short"
            order_side = "BUY"
        else:
            if hold_expired:
                should_exit = True
                reason = "max hold days reached"
            elif stop_price > 0 and price <= stop_price:
                should_exit = True
                reason = "long stop loss touched"
            elif (v5_soft_stop_exit := self.v5_overlay_soft_stop_candidate(signal, position, risk_plan)) is not None:
                should_exit = True
                reason = str(v5_soft_stop_exit.get("reason") or "long V5 soft stop confirmed")
            elif (daily_ma20_long_reversal_exit := self.daily_ma20_long_reversal_exit_candidate(signal, position)) is not None:
                should_exit = True
                reason = str(daily_ma20_long_reversal_exit.get("reason") or "long daily MA20 reversal")
            elif (profit_guard_full_exit := self.v5_no_news_momentum_take_profit_candidate(signal, position, profit_tracker)) is not None:
                if profit_guard_full_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = profit_guard_full_exit
                    profit_guard_full_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or profit_guard_full_exit or {}).get("reason") or "V5 no-news momentum long take profit")
            elif (profit_guard_full_exit := self.fifteen_m_h4_profit_giveback_candidate(signal, position, profit_tracker)) is not None:
                if profit_guard_full_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = profit_guard_full_exit
                    profit_guard_full_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or profit_guard_full_exit or {}).get("reason") or "long profit guard")
            elif (profit_guard_full_exit := self.h4_lower_reversal_guard_candidate(signal, position, profit_tracker)) is not None:
                if profit_guard_full_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = profit_guard_full_exit
                    profit_guard_full_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or profit_guard_full_exit or {}).get("reason") or "long h4 lower reversal guard")
            elif (logic_weakening_exit := self.logic_weakening_exit_candidate(signal, position)) is not None:
                should_exit = True
                reason = str(logic_weakening_exit.get("reason") or "long original 4H logic weakened")
            elif (smart_signal_risk_exit := self.smart_signal_risk_guard_candidate(signal, position, profit_tracker)) is not None:
                if smart_signal_risk_exit.get("kind") == "partial_take_profit":
                    profit_guard_take_profit = smart_signal_risk_exit
                    smart_signal_risk_exit = None
                should_exit = True
                reason = str((profit_guard_take_profit or smart_signal_risk_exit or {}).get("reason") or "long smart-signal risk guard")
            elif (roi_take_profit := self.next_roi_take_profit(signal, position)) is not None:
                stages, close_fraction = roi_take_profit
                names = ",".join(str(stage.get("name") or "") for stage in stages)
                total_initial = sum(float(stage.get("fraction_of_initial") or 0.0) for stage in stages)
                max_roi = max(float(stage.get("trigger_roi_pct") or 0.0) for stage in stages)
                should_exit = True
                reason = (
                    f"ROI long take profit threshold touched: {names} "
                    f"roi={max_roi:.2f}% "
                    f"close_initial={total_initial:.2f} "
                    f"close_remaining={close_fraction:.4f}"
                )
            elif (staged_take_profit := self.next_staged_take_profit(signal, position)) is not None:
                stages, close_fraction = staged_take_profit
                names = ",".join(str(stage.get("name") or "") for stage in stages)
                total_initial = sum(
                    float(stage.get("effective_fraction_of_initial") or stage.get("fraction_of_initial") or 0.0)
                    for stage in stages
                )
                should_exit = True
                reason = (
                    f"staged long take profit touched with current/delayed 15M volume confirmation: {names} "
                    f"close_initial={total_initial:.2f} "
                    f"close_remaining={close_fraction:.4f}"
                )
            elif not self.ensure_position_take_profit_stages(position, signal) and take_profit_price > 0 and price >= take_profit_price:
                should_exit = True
                reason = "long take profit touched"
            elif (
                not self.config.technical_entry_only_for(signal.symbol)
                and signal.short_probability >= self.opposite_exit_threshold_for_position(position)
            ):
                should_exit = True
                reason = "opposite short probability threshold"
            action = "close_long"
            order_side = "SELL"
        if not should_exit:
            return None
        manual_quarantine = position.get("manual_quarantine") if isinstance(position.get("manual_quarantine"), dict) else None
        quarantine_until = self.parse_utc_datetime((manual_quarantine or {}).get("until"))
        if (manual_quarantine or {}).get("enabled") and quarantine_until and quarantine_until > datetime.now(timezone.utc):
            return None
        retry_block_until = self.parse_utc_datetime(position.get("close_retry_block_until"))
        if retry_block_until and retry_block_until > datetime.now(timezone.utc) and "stop loss touched" not in reason:
            return None
        partial_risk_plan: dict[str, Any] = {}
        close_fraction = 1.0
        target_name = ""
        if profit_guard_take_profit is not None:
            stages, close_fraction = profit_guard_take_profit["partial"]
            target_name = ",".join(str(stage.get("name") or "") for stage in stages if stage.get("name"))
            total_initial = sum(float(stage.get("fraction_of_initial") or 0.0) for stage in stages)
            partial_type = str(profit_guard_take_profit.get("type") or "profit_giveback_guard")
            partial_risk_plan = {
                "partial_take_profit": {
                    "type": partial_type,
                    "stages": stages,
                    "stage": stages[0],
                    "fraction_of_initial": total_initial,
                    "close_fraction_of_remaining": close_fraction,
                    "remaining_fraction_before": float(position.get("remaining_fraction") or 1.0),
                    "trigger_roi_pct": (profit_guard_take_profit.get("trigger") or {}).get("current_roi_pct"),
                    "trigger_giveback_pct": (profit_guard_take_profit.get("trigger") or {}).get("giveback_pct"),
                    "trigger": dict(profit_guard_take_profit.get("trigger") or {}),
                }
            }
        if staged_take_profit is not None:
            stages, close_fraction = staged_take_profit
            target_name = ",".join(str(stage.get("name") or "") for stage in stages if stage.get("name"))
            total_initial = sum(
                float(stage.get("effective_fraction_of_initial") or stage.get("fraction_of_initial") or 0.0)
                for stage in stages
            )
            partial_risk_plan = {
                "partial_take_profit": {
                    "type": "bb_take_profit",
                    "stages": stages,
                    "stage": stages[0],
                    "fraction_of_initial": total_initial,
                    "close_fraction_of_remaining": close_fraction,
                    "remaining_fraction_before": float(position.get("remaining_fraction") or 1.0),
                }
            }
        if roi_take_profit is not None:
            stages, close_fraction = roi_take_profit
            target_name = ",".join(str(stage.get("name") or "") for stage in stages if stage.get("name"))
            total_initial = sum(float(stage.get("fraction_of_initial") or 0.0) for stage in stages)
            partial_risk_plan = {
                "partial_take_profit": {
                    "type": "roi_take_profit",
                    "stages": stages,
                    "stage": stages[0],
                    "fraction_of_initial": total_initial,
                    "close_fraction_of_remaining": close_fraction,
                    "remaining_fraction_before": float(position.get("remaining_fraction") or 1.0),
                    "roi_thresholds": [float(stage.get("roi_threshold_pct") or 0.0) for stage in stages],
                    "trigger_roi_pct": max(float(stage.get("trigger_roi_pct") or 0.0) for stage in stages),
                }
            }
        if logic_weakening_exit is not None:
            partial_risk_plan = {
                "logic_weakening_exit": logic_weakening_exit,
            }
        if daily_ma7_short_reversal_exit is not None:
            partial_risk_plan = {
                "daily_ma7_short_reversal_exit": daily_ma7_short_reversal_exit,
            }
        if daily_ma20_long_reversal_exit is not None:
            partial_risk_plan = {
                "daily_ma20_long_reversal_exit": daily_ma20_long_reversal_exit,
            }
        if v5_soft_stop_exit is not None:
            partial_risk_plan = {
                "v5_soft_stop_exit": v5_soft_stop_exit,
            }
        if profit_guard_full_exit is not None:
            partial_risk_plan = {
                "profit_guard_full_exit": profit_guard_full_exit,
            }
        return V6OrderIntent(
            action=action,
            symbol=signal.symbol,
            side=order_side,
            quantity_hint_usdt=float(position.get("notional_usdt") or self.config.base_notional_usdt),
            leverage=int(position.get("leverage") or self.config.default_leverage),
            reduce_only=True,
            entry_price=price,
            stop_loss_price=stop_price,
            take_profit_price=take_profit_price,
            long_probability=signal.long_probability,
            short_probability=signal.short_probability,
            wait_probability=signal.wait_probability,
            rationale=reason,
            generated_at=utc_now_iso(),
            risk_plan=partial_risk_plan,
            close_fraction=close_fraction,
            target_name=target_name,
        )

    @staticmethod
    def is_scale_in_intent(intent: V6OrderIntent) -> bool:
        return bool((intent.risk_plan or {}).get("scale_in"))

    @staticmethod
    def scale_in_count(position: dict[str, Any]) -> int:
        counts: list[int] = []
        explicit = position.get("scale_in_count")
        if explicit is not None:
            try:
                counts.append(max(0, int(explicit)))
            except (TypeError, ValueError):
                pass
        counts.append(len([item for item in position.get("scale_in_history") or [] if isinstance(item, dict)]))
        lots = [item for item in position.get("lots") or [] if isinstance(item, dict)]
        if len(lots) > 1:
            counts.append(len(lots) - 1)
        return max(counts or [0])

    def v5_scale_in_15m_ma20_retest_candidate(self, signal: V6Signal, side: str) -> dict[str, Any] | None:
        fifteen_m = signal.fifteen_m if isinstance(signal.fifteen_m, dict) else {}
        trend = fifteen_m.get("ma20_trend") if isinstance(fifteen_m.get("ma20_trend"), dict) else {}
        if not trend.get("ready"):
            return None
        ma20 = float(trend.get("ma20") or 0.0)
        price = float(signal.price or trend.get("price") or 0.0)
        if ma20 <= 0 or price <= 0:
            return None
        side = str(side or "").lower()
        if side not in {"long", "short"}:
            return None

        tolerance_pct = max(0.0, float(trend.get("tolerance_pct") or self.config.ma20_trend_guard_tolerance_pct))
        near_pct = max(tolerance_pct, float(self.config.v5_scale_in_15m_ma20_retest_near_pct))
        max_extension_pct = max(near_pct, float(self.config.v5_scale_in_15m_ma20_max_extension_pct))
        price_distance_pct = pct_diff(price, ma20)
        candles = [item for item in trend.get("candles") or [] if isinstance(item, dict)]
        if not candles:
            return None

        if side == "long":
            trend_alive = bool(trend.get("bullish_trend_alive"))
            retest_candles = [
                item for item in candles
                if float(item.get("low_distance_pct") or 0.0) >= -tolerance_pct
                and float(item.get("low_distance_pct") or 0.0) <= near_pct
                and float(item.get("close_distance_pct") or 0.0) >= -tolerance_pct
            ]
            current_ok = -tolerance_pct <= price_distance_pct <= max_extension_pct
            structure_signal = "long_15m_ma20_pullback_hold"
            touch_price = min(float(item.get("low") or ma20) for item in retest_candles) if retest_candles else 0.0
        else:
            trend_alive = bool(trend.get("bearish_trend_alive"))
            retest_candles = [
                item for item in candles
                if float(item.get("high_distance_pct") or 0.0) <= tolerance_pct
                and float(item.get("high_distance_pct") or 0.0) >= -near_pct
                and float(item.get("close_distance_pct") or 0.0) <= tolerance_pct
            ]
            current_ok = -max_extension_pct <= price_distance_pct <= tolerance_pct
            structure_signal = "short_15m_ma20_retest_reject"
            touch_price = max(float(item.get("high") or ma20) for item in retest_candles) if retest_candles else 0.0

        if not trend_alive or not retest_candles or not current_ok:
            return None
        return {
            "name": "ma20_15m_retest",
            "price": round(ma20, 8),
            "distance_pct": round(price_distance_pct, 6),
            "abs_distance_pct": round(abs(price_distance_pct), 6),
            "timeframe": "15m",
            "band": "ma",
            "period": 20,
            "side": side,
            "structure_signal": structure_signal,
            "touch_price": round(touch_price, 8),
            "near_pct": near_pct,
            "max_extension_pct": max_extension_pct,
            "checked_candles": len(candles),
            "retest_candle_count": len(retest_candles),
            "candles": retest_candles[-3:],
        }

    def scale_in_intent_if_needed(
        self,
        signal: V6Signal,
        position: dict[str, Any],
        require_probability: bool = True,
    ) -> V6OrderIntent | None:
        if not self.config.scale_in_enabled:
            return None
        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"}:
            return None
        if self.scale_in_count(position) >= max(0, int(self.config.max_scale_ins_per_position)):
            return None
        if require_probability and signal.wait_probability > self.config.max_wait_probability_for_entry:
            return None
        if side == "short":
            action = "open_short"
        else:
            action = "open_long"
        if self.market_data_freshness_entry_block(signal, action):
            return None
        if require_probability:
            if side == "short":
                probability = float(signal.short_probability)
                threshold = self.config.short_entry_threshold_for(signal.symbol)
                opposite_probability = float(signal.long_probability)
            else:
                probability = float(signal.long_probability)
                threshold = self.config.long_entry_threshold_for(signal.symbol)
                opposite_probability = float(signal.short_probability)
            if probability < threshold or opposite_probability >= self.config.exit_opposite_threshold:
                return None

        current_price = float(signal.price)
        avg_entry = float(position.get("entry_price") or 0.0)
        if current_price <= 0 or avg_entry <= 0:
            return None
        v5_ma20_retest: dict[str, Any] | None = None
        if self.config.v5_staged_entry_enabled and self.config.v5_overlay_rules_active_for(signal.symbol):
            if self.config.v5_scale_in_require_15m_ma20_retest:
                v5_ma20_retest = self.v5_scale_in_15m_ma20_retest_candidate(signal, side)
                if not v5_ma20_retest:
                    return None
            min_hold_minutes = max(0.0, float(self.config.v5_scale_in_min_hold_minutes))
            if min_hold_minutes > 0:
                anchor_text = str(position.get("last_scale_in_at") or position.get("opened_at") or "")
                try:
                    anchor_at = datetime.fromisoformat(anchor_text.replace("Z", "+00:00"))
                except Exception:
                    anchor_at = None
                if anchor_at is not None:
                    age_minutes = (datetime.now(timezone.utc) - anchor_at).total_seconds() / 60.0
                    if age_minutes < min_hold_minutes:
                        return None
        adverse_move = pct_diff(current_price, avg_entry)
        if side == "long":
            adverse_move = -adverse_move
        if adverse_move < float(self.config.scale_in_min_adverse_move_pct):
            return None
        if self.scale_in_recent_price_duplicate(position, current_price):
            return None
        if self.indicator_confluence_hold(signal, side):
            return None
        if self.ma20_trend_entry_block(signal, action):
            return None
        entry_context = self.entry_context_for_action(action, signal.levels or {}, symbol=signal.symbol)
        if self.smart_signal_entry_block(signal, action, entry_context):
            return None
        generic_indicator_block = self.generic_indicator_entry_block(signal, action, entry_context)
        if generic_indicator_block:
            return None

        trigger = None
        if str((position.get("risk_plan") or {}).get("horizon") or position.get("horizon") or "") == "1d_trend":
            trigger = self.daily_trend_scale_in_trigger_candidate(signal, side, position)
        if not trigger:
            trigger = self.scale_in_trigger_candidate(signal, side)
        if not trigger:
            return None
        trigger_name = str(trigger.get("name") or "")
        used_names = {
            str(item.get("trigger_name") or "")
            for item in position.get("scale_in_history") or []
            if isinstance(item, dict)
        }
        if trigger_name and trigger_name in used_names:
            return None

        base_notional = max(float(self.config.base_notional_usdt), 1.0)
        add_notional = max(5.0, base_notional * max(0.0, float(self.config.scale_in_notional_fraction)))
        current_notional = float(position.get("notional_usdt") or position.get("original_notional_usdt") or base_notional)
        max_total_notional = base_notional * max(1.0, float(self.config.scale_in_max_total_notional_multiplier))
        if current_notional + add_notional > max_total_notional + 1e-9:
            return None

        stop, take_profit, risk_plan = self.stop_take_profit(signal, action)
        combined_entry = self.weighted_average_entry(avg_entry, current_notional, current_price, add_notional)
        four_h_stop = self.scale_in_4h_stop_price(signal, side, current_price, fallback=stop)
        if four_h_stop > 0:
            stop = four_h_stop
        stop_distance = abs(pct_diff(stop, combined_entry))
        if stop_distance > float(self.config.scale_in_max_4h_stop_distance_pct):
            return None

        risk_plan["stop_loss_price"] = round(float(stop), 8)
        risk_plan["default_stop_price"] = round(float(stop), 8)
        risk_plan["entry_indicators"] = dict(getattr(signal, "indicators", {}) or {})
        strategy_labels = [str(item) for item in risk_plan.get("strategy_labels") or [] if str(item or "")]
        risk_plan["scale_in"] = {
            "enabled": True,
            "side": side,
            "entry_mode": "technical_indicators_only" if not require_probability else "probability_threshold_with_indicator_trigger",
            "strategy_labels": strategy_labels,
            "strategy_label": strategy_labels[0] if strategy_labels else "",
            "scale_in_index": self.scale_in_count(position) + 1,
            "max_scale_ins": int(self.config.max_scale_ins_per_position),
            "trigger": dict(trigger),
            "trigger_name": trigger_name,
            "adverse_move_pct": round(adverse_move, 6),
            "previous_entry_price": round(avg_entry, 8),
            "previous_notional_usdt": round(current_notional, 6),
            "add_notional_usdt": round(add_notional, 6),
            "combined_entry_price_estimate": round(combined_entry, 8),
            "combined_notional_usdt_estimate": round(current_notional + add_notional, 6),
            "four_h_stop_price": round(stop, 8),
            "four_h_stop_distance_pct": round(stop_distance, 6),
            "max_4h_stop_distance_pct": float(self.config.scale_in_max_4h_stop_distance_pct),
            "max_total_notional_usdt": round(max_total_notional, 6),
        }
        if v5_ma20_retest:
            risk_plan["scale_in"]["v5_15m_ma20_retest"] = v5_ma20_retest
        rationale = (
            f"scale-in {side}: indicator setup remains active; "
            f"15M rejection_lines={int(trigger.get('line_count') or 0)} trigger={trigger_name}; "
            f"adverse_move={adverse_move:.2f}%; "
            f"4H risk stop={stop:.4f} distance={stop_distance:.2f}%"
        )
        return V6OrderIntent(
            action=action,
            symbol=signal.symbol,
            side="SELL" if action == "open_short" else "BUY",
            quantity_hint_usdt=round(add_notional, 6),
            leverage=int(position.get("leverage") or self.config.default_leverage),
            reduce_only=False,
            entry_price=current_price,
            stop_loss_price=round(float(stop), 8),
            take_profit_price=round(float(take_profit), 8),
            long_probability=signal.long_probability,
            short_probability=signal.short_probability,
            wait_probability=signal.wait_probability,
            rationale=rationale,
            generated_at=utc_now_iso(),
            risk_plan=risk_plan,
        )

    def daily_trend_scale_in_trigger_candidate(self, signal: V6Signal, side: str, position: dict[str, Any]) -> dict[str, Any] | None:
        levels = signal.levels or {}
        daily = levels.get("daily_ma7_trend") if isinstance(levels.get("daily_ma7_trend"), dict) else {}
        if not daily.get("ready"):
            return None
        if side == "long" and not daily.get("long_confirmed"):
            return None
        if side == "short" and not daily.get("short_confirmed"):
            return None
        current_price = float(signal.price or 0.0)
        recent = (
            (signal.indicators or {})
            .get("recent_extremes", {})
            .get("1m", {})
        )
        if current_price <= 0 or not isinstance(recent, dict):
            return None
        near_pct = max(float(self.config.scale_in_near_level_pct), float(self.config.daily_ma_retest_tolerance_pct))
        rows = levels.get("all_supports") if side == "long" else levels.get("all_resistances")
        touched: list[dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            name = str(row[0])
            if name not in {"ma7_1d", "ma20_1d"}:
                continue
            value = float(row[1] or 0.0)
            if value <= 0 or abs(pct_diff(current_price, value)) > near_pct:
                continue
            if side == "long":
                touch_price = float(recent.get("low") or current_price)
                if touch_price > value * (1.0 + near_pct / 100.0) or current_price < value * (1.0 - near_pct / 100.0):
                    continue
            else:
                touch_price = float(recent.get("high") or current_price)
                if touch_price < value * (1.0 - near_pct / 100.0) or current_price > value * (1.0 + near_pct / 100.0):
                    continue
            touched.append({
                "name": name,
                "price": round(value, 8),
                "distance_pct": round(pct_diff(current_price, value), 6),
                "abs_distance_pct": round(abs(pct_diff(current_price, value)), 6),
                "period": 7 if name == "ma7_1d" else 20,
                "band": "ma",
                "timeframe": "1d",
                "touch_price": round(touch_price, 8),
                "line_count": 1,
                "selection_strategy": "daily_ma7_ma20_trend_scale_in",
            })
        if not touched:
            return None
        touched.sort(key=lambda item: (0 if item["name"] == "ma7_1d" else 1, item["abs_distance_pct"]))
        trigger = touched[0]
        trigger["daily_ma7_trend"] = daily
        return trigger

    def scale_in_trigger_candidate(
        self,
        signal: V6Signal,
        side: str,
        *,
        near_pct: float | None = None,
        min_lines: int | None = None,
    ) -> dict[str, Any] | None:
        current_price = float(signal.price)
        recent = (
            (signal.indicators or {})
            .get("recent_extremes", {})
            .get("1m", {})
        )
        if not isinstance(recent, dict) or current_price <= 0:
            return None
        second_test = (signal.levels or {}).get("bb96_210_second_test_15m") if isinstance(signal.levels, dict) else {}
        if isinstance(second_test, dict):
            zone = second_test.get("upper") if side == "short" else second_test.get("lower")
            if isinstance(zone, dict) and zone.get("confirmed"):
                level_price = float(zone.get("price") or 0.0)
                touch_price = float(zone.get("recent_high") or zone.get("recent_low") or current_price)
                if level_price > 0:
                    return {
                        "name": str(zone.get("name") or f"bb96_210_{side}_second_test_15m"),
                        "price": round(level_price, 8),
                        "distance_pct": round(pct_diff(current_price, level_price), 6),
                        "abs_distance_pct": round(abs(pct_diff(current_price, level_price)), 6),
                        "timeframe": "15m",
                        "band": "cluster",
                        "period": 0,
                        "structure_signal": "short_bb96_210_upper_second_test_rejection" if side == "short" else "long_bb96_210_lower_second_test_rebound",
                        "line_count": 2,
                        "required_line_count": 2,
                        "touch_price": round(touch_price, 8),
                        "lines": [
                            {"name": "bb96_upper_15m" if side == "short" else "bb96_lower_15m", "source": "bb96_210_second_test_15m"},
                            {"name": "bb210_upper_15m" if side == "short" else "bb210_lower_15m", "source": "bb96_210_second_test_15m"},
                        ],
                        "second_test_zone": zone,
                    }
        if side == "short":
            touch_price = float(recent.get("high") or current_price)
            source_rows = (signal.levels or {}).get("all_resistances") or []
            allowed_bands = {"mid", "upper"}
            rejection_check = lambda level_price: touch_price >= level_price and current_price <= level_price
        else:
            touch_price = float(recent.get("low") or current_price)
            source_rows = (signal.levels or {}).get("all_supports") or []
            allowed_bands = {"mid", "lower"}
            rejection_check = lambda level_price: touch_price <= level_price and current_price >= level_price
        near_pct = float(self.config.scale_in_near_level_pct if near_pct is None else near_pct)
        touched: list[dict[str, Any]] = []
        for row in source_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            name = str(row[0])
            value = float(row[1] or 0.0)
            parsed = self.parse_shared_bb_name(name)
            if not parsed or parsed.get("timeframe") != "15m" or parsed.get("band") not in allowed_bands:
                continue
            if value <= 0 or abs(pct_diff(current_price, value)) > near_pct:
                continue
            if not rejection_check(value):
                continue
            distance = pct_diff(current_price, value)
            touched.append({
                "name": name,
                "price": round(value, 8),
                "distance_pct": round(distance, 6),
                "abs_distance_pct": round(abs(distance), 6),
                "period": parsed["period"],
                "band": parsed["band"],
                "timeframe": parsed["timeframe"],
                "touch_price": round(touch_price, 8),
                "structure_signal": "short_three_line_rejection" if side == "short" else "long_three_line_reclaim",
            })
        min_lines = max(1, int(self.config.scale_in_min_rejection_lines if min_lines is None else min_lines))
        if len(touched) < min_lines:
            return None
        if side == "short":
            band_rank = {"upper": 0, "mid": 1}
        else:
            band_rank = {"lower": 0, "mid": 1}
        touched.sort(key=lambda row: (
            band_rank.get(str(row.get("band") or ""), 9),
            abs(float(row.get("distance_pct") or 0.0)),
            int(row.get("period") or 0),
        ))
        selected = touched[:min_lines]
        return {
            "name": "three_line_rejection:" + ",".join(str(row.get("name") or "") for row in selected),
            "price": round(sum(float(row.get("price") or 0.0) for row in selected) / len(selected), 8),
            "distance_pct": round(sum(float(row.get("distance_pct") or 0.0) for row in selected) / len(selected), 6),
            "abs_distance_pct": round(max(float(row.get("abs_distance_pct") or 0.0) for row in selected), 6),
            "timeframe": "15m",
            "band": "cluster",
            "period": 0,
            "structure_signal": "short_three_line_rejection" if side == "short" else "long_three_line_reclaim",
            "line_count": len(touched),
            "required_line_count": min_lines,
            "touch_price": round(touch_price, 8),
            "lines": touched[:8],
        }

    def scale_in_recent_price_duplicate(self, position: dict[str, Any], price: float) -> bool:
        min_distance = float(self.config.scale_in_min_adverse_move_pct)
        for item in position.get("scale_in_history") or []:
            if not isinstance(item, dict):
                continue
            previous_price = float(item.get("entry_price") or 0.0)
            if previous_price > 0 and abs(pct_diff(price, previous_price)) < min_distance:
                return True
        return False

    @staticmethod
    def weighted_average_entry(old_entry: float, old_notional: float, add_entry: float, add_notional: float) -> float:
        total = max(0.0, old_notional) + max(0.0, add_notional)
        if total <= 0:
            return add_entry
        return (old_entry * max(0.0, old_notional) + add_entry * max(0.0, add_notional)) / total

    def scale_in_4h_stop_price(self, signal: V6Signal, side: str, price: float, *, fallback: float) -> float:
        levels = signal.levels or {}
        rows = levels.get("all_resistances") if side == "short" else levels.get("all_supports")
        candidates: list[tuple[str, float]] = []
        for row in rows or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            name = str(row[0])
            value = float(row[1] or 0.0)
            if "_4h" not in name or value <= 0:
                continue
            if side == "short" and value > price:
                candidates.append((name, value))
            if side == "long" and 0 < value < price:
                candidates.append((name, value))
        if not candidates:
            return fallback
        if side == "short":
            _, value = min(candidates, key=lambda item: item[1] - price)
            return value * 1.005
        _, value = min(candidates, key=lambda item: price - item[1])
        return value * 0.995

    def stop_take_profit(self, signal: V6Signal, action: str) -> tuple[float, float, dict[str, Any]]:
        price = float(signal.price)
        risk_plan = self.structure_risk_plan(price, action, signal.levels, symbol=signal.symbol)
        return round(float(risk_plan["stop_loss_price"]), 8), round(float(risk_plan["take_profit_price"]), 8), risk_plan

    def load_state(self) -> dict[str, Any]:
        state = load_json(self.config.state_file, default={})
        if not isinstance(state, dict):
            state = {}
        state.setdefault("schema_version", 1)
        state.setdefault("engine", "v6_execution")
        state.setdefault("positions", [])
        state.setdefault("position_lots", [])
        state.setdefault("journal", [])
        state.setdefault("entry_cooldowns", {})
        state.setdefault("v5_profit_reentry_guards", {})
        return state

    def strategy_dust_threshold(self) -> float:
        return max(
            0.0,
            float(self.config.dust_position_notional_usdt or 0.0),
            float(getattr(self.config, "strategy_dust_position_notional_usdt", 0.0) or 0.0),
        )

    @staticmethod
    def local_position_notional(position: dict[str, Any]) -> float:
        for key in ("notional_usdt", "original_notional_usdt"):
            try:
                value = abs(float(position.get(key) or 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
        try:
            qty = abs(float(position.get("position_qty") or position.get("remaining_quantity") or 0.0))
            price = float(position.get("current_price") or position.get("mark_price") or position.get("entry_price") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return abs(qty * price)

    def is_strategy_dust_position(self, position: dict[str, Any]) -> bool:
        threshold = self.strategy_dust_threshold()
        if threshold <= 0:
            return False
        try:
            qty = abs(float(position.get("position_qty") or position.get("remaining_quantity") or 0.0))
        except (TypeError, ValueError):
            qty = 0.0
        return qty > 0 and self.local_position_notional(position) < threshold

    def mark_strategy_dust_position(self, position: dict[str, Any]) -> None:
        threshold = self.strategy_dust_threshold()
        position["strategy_status"] = "dust_ignored"
        position["strategy_dust_ignored"] = True
        position["strategy_dust_notional_usdt"] = round(self.local_position_notional(position), 8)
        position["strategy_dust_threshold_usdt"] = round(threshold, 8)
        position.setdefault("strategy_dust_marked_at", utc_now_iso())

    def raw_positions(self, state: dict[str, Any], symbol: str = "") -> list[dict[str, Any]]:
        symbol_upper = str(symbol or "").upper()
        return [
            position for position in state.get("positions", [])
            if isinstance(position, dict)
            and (not symbol_upper or str(position.get("symbol") or "").upper() == symbol_upper)
        ]

    def active_positions(self, state: dict[str, Any], symbol: str = "") -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for position in self.raw_positions(state, symbol):
            if self.is_strategy_dust_position(position):
                self.mark_strategy_dust_position(position)
                continue
            active.append(position)
        return active

    def find_positions(self, state: dict[str, Any], symbol: str, *, include_strategy_dust: bool = False) -> list[dict[str, Any]]:
        if include_strategy_dust:
            return self.raw_positions(state, symbol)
        return self.active_positions(state, symbol)

    def find_position(self, state: dict[str, Any], symbol: str, side: str = "", *, include_strategy_dust: bool = False) -> dict[str, Any] | None:
        side = str(side or "").lower()
        for position in self.find_positions(state, symbol, include_strategy_dust=include_strategy_dust):
            if side and str(position.get("side") or "").lower() != side:
                continue
            return position
        return None

    @staticmethod
    def entry_cooldown_active(state: dict[str, Any], symbol: str) -> bool:
        cooldowns = state.get("entry_cooldowns") if isinstance(state.get("entry_cooldowns"), dict) else {}
        row = cooldowns.get(symbol.upper()) if isinstance(cooldowns, dict) else None
        if not isinstance(row, dict):
            return False
        until = row.get("until")
        if not until:
            return False
        try:
            return datetime.fromisoformat(str(until).replace("Z", "+00:00")) > datetime.now(timezone.utc)
        except Exception:
            return False

    @staticmethod
    def iso_after_15m_candles(candles: int) -> str:
        count = max(1, int(candles or 1))
        now = datetime.now(timezone.utc)
        base = now.replace(second=0, microsecond=0)
        minutes_to_next_close = 15 - (base.minute % 15)
        if minutes_to_next_close <= 0:
            minutes_to_next_close = 15
        first_close = base + timedelta(minutes=minutes_to_next_close)
        until = first_close + timedelta(minutes=15 * (count - 1))
        return until.isoformat()

    @staticmethod
    def set_entry_cooldown(
        state: dict[str, Any],
        symbol: str,
        *,
        hours: float,
        reason: str,
        side: str = "",
        candles_15m: int = 0,
        timeframe: str = "",
    ) -> None:
        cooldowns = state.get("entry_cooldowns") if isinstance(state.get("entry_cooldowns"), dict) else {}
        if not isinstance(cooldowns, dict):
            cooldowns = {}
        created_at = utc_now_iso()
        cooldown: dict[str, Any] = {
            "until": V6ExecutionEngine.iso_after_15m_candles(candles_15m) if int(candles_15m or 0) > 0 else iso_add_hours(created_at, hours),
            "reason": reason,
            "side": side,
            "created_at": created_at,
        }
        if int(candles_15m or 0) > 0:
            cooldown["timeframe"] = timeframe or "15m"
            cooldown["candles"] = int(candles_15m)
        cooldowns[symbol.upper()] = {
            **cooldown,
        }
        state["entry_cooldowns"] = cooldowns

    @staticmethod
    def position_reconcile_from_result(result: dict[str, Any]) -> dict[str, Any]:
        payload = result.get("position_reconcile") if isinstance(result.get("position_reconcile"), dict) else {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def exchange_after_from_result(result: dict[str, Any]) -> dict[str, Any]:
        reconcile = V6ExecutionEngine.position_reconcile_from_result(result)
        after = reconcile.get("after") if isinstance(reconcile.get("after"), dict) else {}
        return after if isinstance(after, dict) else {}

    @staticmethod
    def exchange_delta_qty_from_result(result: dict[str, Any]) -> float:
        reconcile = V6ExecutionEngine.position_reconcile_from_result(result)
        try:
            return abs(float(reconcile.get("delta_qty_abs") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def response_executed_qty(result: dict[str, Any]) -> float:
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        for key in ("executedQty", "origQty"):
            try:
                quantity = abs(float(response.get(key) or 0.0))
            except (TypeError, ValueError):
                quantity = 0.0
            if quantity > 0:
                return quantity
        return 0.0

    @staticmethod
    def order_id_from_result(result: dict[str, Any]) -> Any:
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        return response.get("orderId") or response.get("clientOrderId") if isinstance(response, dict) else None

    @staticmethod
    def compact_risk_plan_for_lot(risk_plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(risk_plan, dict):
            return {}
        context = risk_plan.get("entry_context") if isinstance(risk_plan.get("entry_context"), dict) else {}
        selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
        return {
            "mode": risk_plan.get("mode"),
            "action": risk_plan.get("action"),
            "strategy_labels": [str(item) for item in risk_plan.get("strategy_labels") or [] if str(item or "")],
            "strategy_label": risk_plan.get("strategy_label") or "",
            "horizon": risk_plan.get("horizon"),
            "entry_timeframe": risk_plan.get("entry_timeframe"),
            "max_hold_days": risk_plan.get("max_hold_days"),
            "countertrend_rebound": bool(context.get("countertrend_rebound")),
            "macro_trend_context": context.get("macro_trend_context") or "",
            "entry_context": {
                "horizon": context.get("horizon"),
                "entry_timeframe": context.get("entry_timeframe"),
                "max_hold_days": context.get("max_hold_days"),
                "countertrend_rebound": bool(context.get("countertrend_rebound")),
                "macro_trend_context": context.get("macro_trend_context") or "",
                "strategy_labels": [str(item) for item in context.get("strategy_labels") or [] if str(item or "")],
                "generic_indicator": dict(context.get("generic_indicator") or {}),
            },
            "selected_entry_level": dict(selected),
            "stop_anchor": dict(risk_plan.get("stop_anchor") or {}),
            "tp1": dict(risk_plan.get("tp1") or {}),
            "tp2": dict(risk_plan.get("tp2") or {}),
            "runner": dict(risk_plan.get("runner") or {}),
            "staged_take_profit_targets": [
                dict(stage) for stage in risk_plan.get("staged_take_profit_targets") or [] if isinstance(stage, dict)
            ],
            "scale_in": dict(risk_plan.get("scale_in") or {}),
            "v5_candidate_metadata": dict(risk_plan.get("v5_candidate_metadata") or {})
            if isinstance(risk_plan.get("v5_candidate_metadata"), dict)
            else {},
        }

    @staticmethod
    def lot_id(intent: V6OrderIntent, result: dict[str, Any], *, fill_price: float, quantity: float) -> str:
        source = "|".join([
            intent.symbol,
            "short" if intent.action == "open_short" else "long",
            str(result.get("submitted_at") or utc_now_iso()),
            str(V6ExecutionEngine.order_id_from_result(result) or ""),
            str(fill_price),
            str(quantity),
        ])
        return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]

    def lot_from_open_result(self, intent: V6OrderIntent, result: dict[str, Any], *, fill_price: float, notional: float) -> dict[str, Any]:
        delta_qty = self.exchange_delta_qty_from_result(result)
        executed_qty = self.response_executed_qty(result)
        quantity = delta_qty if delta_qty > 0 else executed_qty
        if quantity <= 0 and fill_price > 0:
            quantity = max(0.0, notional / fill_price)
        risk_plan = intent.risk_plan if isinstance(intent.risk_plan, dict) else {}
        context = risk_plan.get("entry_context") if isinstance(risk_plan.get("entry_context"), dict) else {}
        selected = context.get("selected_entry_level") if isinstance(context.get("selected_entry_level"), dict) else {}
        strategy_labels = [str(item) for item in risk_plan.get("strategy_labels") or [] if str(item or "")]
        submitted_at = result.get("submitted_at") or utc_now_iso()
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        lot = {
            "lot_id": self.lot_id(intent, result, fill_price=fill_price, quantity=quantity),
            "symbol": intent.symbol,
            "side": "short" if intent.action == "open_short" else "long",
            "status": "open",
            "opened_at": submitted_at,
            "entry_price": round(fill_price, 8),
            "quantity": round(quantity, 12),
            "remaining_quantity": round(quantity, 12),
            "notional_usdt": round(notional, 6),
            "remaining_notional_usdt": round(notional, 6),
            "leverage": intent.leverage,
            "order_id": response.get("orderId") if isinstance(response, dict) else None,
            "client_order_id": response.get("clientOrderId") if isinstance(response, dict) else None,
            "strategy_labels": strategy_labels,
            "strategy_label": strategy_labels[0] if strategy_labels else "",
            "horizon": risk_plan.get("horizon"),
            "entry_timeframe": risk_plan.get("entry_timeframe"),
            "selected_entry_level": dict(selected),
            "stop_loss_price": intent.stop_loss_price,
            "take_profit_price": intent.take_profit_price,
            "risk_plan": self.compact_risk_plan_for_lot(risk_plan),
            "position_reconcile": self.position_reconcile_from_result(result),
        }
        if self.is_scale_in_intent(intent):
            lot["kind"] = "scale_in"
            lot["scale_in"] = dict(risk_plan.get("scale_in") or {})
        else:
            lot["kind"] = "entry"
        return lot

    @staticmethod
    def normalize_lots(position: dict[str, Any]) -> list[dict[str, Any]]:
        lots = [dict(item) for item in position.get("lots") or [] if isinstance(item, dict)]
        if lots:
            return lots
        quantity = float(position.get("position_qty") or 0.0)
        if quantity <= 0 and float(position.get("entry_price") or 0.0) > 0:
            quantity = float(position.get("notional_usdt") or 0.0) / float(position.get("entry_price") or 1.0)
        if quantity <= 0:
            return []
        return [{
            "lot_id": hashlib.sha1(
                f"legacy|{position.get('symbol')}|{position.get('side')}|{position.get('opened_at')}|{position.get('entry_price')}".encode("utf-8")
            ).hexdigest()[:16],
            "symbol": str(position.get("symbol") or "").upper(),
            "side": str(position.get("side") or "").lower(),
            "status": "open",
            "opened_at": position.get("opened_at") or utc_now_iso(),
            "entry_price": float(position.get("entry_price") or 0.0),
            "quantity": round(quantity, 12),
            "remaining_quantity": round(quantity, 12),
            "notional_usdt": float(position.get("notional_usdt") or 0.0),
            "remaining_notional_usdt": float(position.get("notional_usdt") or 0.0),
            "leverage": int(position.get("leverage") or 0),
            "horizon": position.get("horizon") or (position.get("risk_plan") or {}).get("horizon"),
            "entry_timeframe": position.get("entry_timeframe") or (position.get("risk_plan") or {}).get("entry_timeframe"),
            "selected_entry_level": ((position.get("risk_plan") or {}).get("entry_context") or {}).get("selected_entry_level") or {},
            "risk_plan": V6ExecutionEngine.compact_risk_plan_for_lot(position.get("risk_plan") or {}),
            "kind": "legacy_recovered",
        }]

    @staticmethod
    def lot_remaining_quantity(lot: dict[str, Any]) -> float:
        try:
            return max(0.0, float(lot.get("remaining_quantity") if lot.get("remaining_quantity") is not None else lot.get("quantity") or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def update_position_layer_summary(self, position: dict[str, Any]) -> None:
        lots = [lot for lot in self.normalize_lots(position) if self.lot_remaining_quantity(lot) > 0]
        position["lots"] = lots
        horizons = sorted({str(lot.get("horizon") or "") for lot in lots if lot.get("horizon")})
        timeframes = sorted({str(lot.get("entry_timeframe") or "") for lot in lots if lot.get("entry_timeframe")})
        labels = sorted({
            str(label)
            for lot in lots
            for label in (lot.get("strategy_labels") or ([lot.get("strategy_label")] if lot.get("strategy_label") else []))
            if str(label or "")
        })
        position["lot_count"] = len(lots)
        position["horizons"] = horizons
        position["entry_timeframes"] = timeframes
        position["strategy_labels"] = labels
        position["strategy_label"] = labels[0] if len(labels) == 1 else "mixed" if len(labels) > 1 else position.get("strategy_label", "")
        if len(horizons) == 1:
            position["horizon"] = horizons[0]
        elif len(horizons) > 1:
            position["horizon"] = "mixed"
        if len(timeframes) == 1:
            position["entry_timeframe"] = timeframes[0]
        elif len(timeframes) > 1:
            position["entry_timeframe"] = "mixed"

    def apply_exchange_snapshot_to_position(self, position: dict[str, Any], snapshot: dict[str, Any]) -> None:
        if not isinstance(snapshot, dict) or not snapshot:
            return
        qty = abs(float(snapshot.get("position_amt") or snapshot.get("abs_position_qty") or 0.0))
        entry = float(snapshot.get("entry_price") or 0.0)
        mark = float(snapshot.get("mark_price") or 0.0)
        notional = abs(float(snapshot.get("notional") or 0.0))
        if qty > 0:
            position["position_qty"] = round(qty, 12)
            position["remaining_quantity"] = round(qty, 12)
        if entry > 0:
            if float(position.get("entry_price") or 0.0) and abs(float(position.get("entry_price") or 0.0) - entry) > 1e-8:
                position["last_local_entry_price_before_exchange_sync"] = position.get("entry_price")
            position["entry_price"] = round(entry, 8)
        if mark > 0:
            position["current_price"] = round(mark, 8)
        if notional > 0:
            position["exchange_notional_usdt"] = round(notional, 6)
            position["notional_usdt"] = round(notional, 6)
        elif qty > 0 and mark > 0:
            position["notional_usdt"] = round(qty * mark, 6)
        if snapshot.get("leverage"):
            position["leverage"] = int(snapshot.get("leverage") or position.get("leverage") or self.config.default_leverage)
        position["exchange_position"] = dict(snapshot)
        position["last_exchange_sync_at"] = utc_now_iso()

    def reconcile_lots_to_exchange_quantity(self, position: dict[str, Any], actual_qty: float, *, reason: str) -> None:
        lots = self.normalize_lots(position)
        local_qty = sum(self.lot_remaining_quantity(lot) for lot in lots)
        if actual_qty <= 0:
            for lot in lots:
                lot["remaining_quantity"] = 0.0
                lot["status"] = "closed"
                lot["closed_by_exchange_sync_at"] = utc_now_iso()
                lot["close_reason"] = reason
            position["lots"] = lots
            return
        if local_qty <= 0:
            recovered = self.recovered_lot_from_position(position, actual_qty=actual_qty, reason=reason)
            position["lots"] = [recovered]
            return
        ratio = actual_qty / local_qty
        if abs(ratio - 1.0) <= 0.001:
            position["lots"] = lots
            return
        for lot in lots:
            previous_qty = self.lot_remaining_quantity(lot)
            new_qty = max(0.0, previous_qty * ratio)
            lot["last_remaining_quantity_before_exchange_sync"] = round(previous_qty, 12)
            lot["remaining_quantity"] = round(new_qty, 12)
            lot["status"] = "open" if new_qty > 0 else "closed"
            lot["exchange_quantity_sync"] = {
                "reason": reason,
                "actual_position_qty": round(actual_qty, 12),
                "local_position_qty_before": round(local_qty, 12),
                "ratio": round(ratio, 8),
                "synced_at": utc_now_iso(),
            }
        position["lots"] = lots

    def recovered_lot_from_position(self, position: dict[str, Any], *, actual_qty: float, reason: str) -> dict[str, Any]:
        entry = float(position.get("entry_price") or 0.0)
        mark = float(position.get("current_price") or entry or 0.0)
        notional = actual_qty * (mark if mark > 0 else entry)
        return {
            "lot_id": hashlib.sha1(
                f"exchange_recovered|{position.get('symbol')}|{position.get('side')}|{utc_now_iso()}|{actual_qty}".encode("utf-8")
            ).hexdigest()[:16],
            "symbol": str(position.get("symbol") or "").upper(),
            "side": str(position.get("side") or "").lower(),
            "status": "open",
            "opened_at": position.get("opened_at") or utc_now_iso(),
            "entry_price": round(entry, 8),
            "quantity": round(actual_qty, 12),
            "remaining_quantity": round(actual_qty, 12),
            "notional_usdt": round(notional, 6),
            "remaining_notional_usdt": round(notional, 6),
            "leverage": int(position.get("leverage") or self.config.default_leverage),
            "horizon": position.get("horizon") or "exchange_recovered",
            "entry_timeframe": position.get("entry_timeframe") or "exchange_recovered",
            "selected_entry_level": {},
            "risk_plan": {"mode": "exchange_recovered", "reason": reason},
            "kind": "exchange_recovered",
        }

    def recovered_position_from_exchange_row(self, row: dict[str, Any]) -> dict[str, Any]:
        snapshot = V6BinanceClient.exchange_position_snapshot(row)
        symbol = str(snapshot.get("symbol") or "").upper()
        side = str(snapshot.get("side") or "").lower()
        entry = float(snapshot.get("entry_price") or 0.0)
        mark = float(snapshot.get("mark_price") or entry or 0.0)
        qty = abs(float(snapshot.get("position_amt") or snapshot.get("abs_position_qty") or 0.0))
        stop = entry * (1.0 + self.config.stop_loss_pct) if side == "short" else entry * (1.0 - self.config.stop_loss_pct)
        take_profit = entry * (1.0 - self.config.take_profit_pct) if side == "short" else entry * (1.0 + self.config.take_profit_pct)
        position = {
            "symbol": symbol,
            "side": side,
            "opened_at": utc_now_iso(),
            "entry_price": round(entry, 8),
            "current_price": round(mark, 8),
            "stop_loss_price": round(stop, 8),
            "take_profit_price": round(take_profit, 8),
            "position_qty": round(qty, 12),
            "remaining_quantity": round(qty, 12),
            "notional_usdt": round(abs(float(snapshot.get("notional") or qty * mark)), 6),
            "exchange_notional_usdt": round(abs(float(snapshot.get("notional") or qty * mark)), 6),
            "leverage": int(snapshot.get("leverage") or self.config.default_leverage),
            "order_mode": self.config.order_mode,
            "source": "exchange_reconciled",
            "risk_status": "missing_local_lot_recovered_from_binance_positionRisk",
            "risk_plan": {"mode": "exchange_recovered", "reason": "present_in_binance_positionRisk_but_absent_in_v6_state"},
            "horizon": "exchange_recovered",
            "entry_timeframe": "exchange_recovered",
            "original_notional_usdt": round(abs(float(snapshot.get("notional") or qty * mark)), 6),
            "remaining_fraction": 1.0,
            "take_profit_stages": [],
            "exchange_position": snapshot,
            "last_exchange_sync_at": utc_now_iso(),
        }
        position["lots"] = [self.recovered_lot_from_position(position, actual_qty=qty, reason="exchange_position_missing_from_local_state")]
        self.update_position_layer_summary(position)
        return position

    def refresh_position_lot_index(self, state: dict[str, Any]) -> None:
        index: list[dict[str, Any]] = []
        for position in state.get("positions") or []:
            if not isinstance(position, dict):
                continue
            self.update_position_layer_summary(position)
            for lot in position.get("lots") or []:
                if not isinstance(lot, dict):
                    continue
                row = dict(lot)
                row["parent_symbol"] = position.get("symbol")
                row["parent_side"] = position.get("side")
                row["parent_entry_price"] = position.get("entry_price")
                row["parent_position_qty"] = position.get("position_qty")
                index.append(row)
        state["position_lots"] = index[-500:]

    def cancel_protective_stop_if_present(self, position: dict[str, Any]) -> dict[str, Any] | None:
        order_id = position.get("protective_stop_order_id")
        symbol = str(position.get("symbol") or "").upper()
        if not order_id or not symbol:
            return None
        try:
            return self.client.cancel_order(symbol, order_id)
        except Exception as exc:
            return {
                "status": "failed",
                "order_id": order_id,
                "symbol": symbol,
                "error": f"{type(exc).__name__}: {exc}",
                "submitted_at": utc_now_iso(),
            }

    def cancel_all_protective_stops_for_position(self, position: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(position.get("symbol") or "").upper()
        side = str(position.get("side") or "").lower()
        if not symbol or side not in {"long", "short"}:
            return None
        try:
            open_orders = self.open_protective_stop_orders()
        except Exception as exc:
            return {
                "status": "failed",
                "symbol": symbol,
                "side": side,
                "error": f"{type(exc).__name__}: {exc}",
                "submitted_at": utc_now_iso(),
            }
        matched = [
            order for order in open_orders
            if str(order.get("symbol") or "").upper() == symbol
            and self.position_side_from_stop_order(order) == side
        ]
        cancellations: list[dict[str, Any]] = []
        for order in matched:
            order_id = order.get("algoId") or order.get("orderId")
            if not order_id:
                continue
            try:
                result = self.client.cancel_order(symbol, order_id)
            except Exception as exc:
                result = {
                    "status": "failed",
                    "order_id": order_id,
                    "symbol": symbol,
                    "error": f"{type(exc).__name__}: {exc}",
                    "submitted_at": utc_now_iso(),
                }
            wait = self.wait_for_protective_stop_cancel(symbol, order_id)
            if isinstance(result, dict):
                result["clear_wait"] = wait
            cancellations.append(result)
        if cancellations:
            settle_seconds = max(0.0, float(self.config.protective_stop_cancel_settle_seconds))
            if settle_seconds > 0:
                time.sleep(settle_seconds)
        return {
            "status": "submitted" if cancellations else "no_matching_orders",
            "symbol": symbol,
            "side": side,
            "matched_count": len(matched),
            "cancellations": cancellations,
            "settle_wait_seconds": max(0.0, float(self.config.protective_stop_cancel_settle_seconds)) if cancellations else 0.0,
            "submitted_at": utc_now_iso(),
        }

    def wait_for_protective_stop_cancel(
        self,
        symbol: str,
        order_id: int | str,
        *,
        attempts: int = 6,
        sleep_seconds: float = 0.5,
    ) -> dict[str, Any]:
        symbol = str(symbol or "").upper()
        order_id_text = str(order_id or "")
        if not symbol or not order_id_text:
            return {"status": "skipped", "reason": "missing_symbol_or_order_id", "submitted_at": utc_now_iso()}
        last_seen: list[dict[str, Any]] = []
        for attempt in range(1, max(1, int(attempts)) + 1):
            try:
                open_orders = self.open_protective_stop_orders()
            except Exception as exc:
                return {
                    "status": "failed",
                    "symbol": symbol,
                    "order_id": order_id_text,
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "submitted_at": utc_now_iso(),
                }
            last_seen = [
                order for order in open_orders
                if str(order.get("symbol") or "").upper() == symbol
                and str(order.get("algoId") or order.get("orderId") or "") == order_id_text
            ]
            if not last_seen:
                return {
                    "status": "cleared",
                    "symbol": symbol,
                    "order_id": order_id_text,
                    "attempt": attempt,
                    "submitted_at": utc_now_iso(),
                }
            if attempt < max(1, int(attempts)):
                time.sleep(max(0.05, float(sleep_seconds)))
        return {
            "status": "still_open",
            "symbol": symbol,
            "order_id": order_id_text,
            "attempts": max(1, int(attempts)),
            "open_order_count": len(last_seen),
            "submitted_at": utc_now_iso(),
        }

    def recent_exchange_close_order_for_missing_position(self, position: dict[str, Any]) -> dict[str, Any]:
        if self.config.order_mode == "dry_run":
            return {}
        symbol = str(position.get("symbol") or "").upper()
        side = str(position.get("side") or "").lower()
        if not symbol or side not in {"long", "short"}:
            return {}
        try:
            opened_at = datetime.fromisoformat(str(position.get("opened_at") or "").replace("Z", "+00:00"))
            start_time = int((opened_at - timedelta(minutes=5)).timestamp() * 1000)
        except Exception:
            start_time = int((datetime.now(timezone.utc) - timedelta(hours=12)).timestamp() * 1000)
        close_side = "SELL" if side == "long" else "BUY"
        try:
            expected_qty = abs(float(
                position.get("remaining_quantity")
                or position.get("position_qty")
                or position.get("quantity")
                or 0.0
            ))
        except Exception:
            expected_qty = 0.0
        try:
            orders = self.client.signed_request("GET", "/fapi/v1/allOrders", {
                "symbol": symbol,
                "startTime": str(start_time),
                "limit": "1000",
                **self.client.signed_params(),
            })
        except Exception as exc:
            LOGGER.warning("V6 recent close order lookup failed for %s: %s", symbol, exc)
            return {}
        if not isinstance(orders, list):
            return {}
        candidates = []
        for order in orders:
            if not isinstance(order, dict):
                continue
            if str(order.get("status") or "") != "FILLED":
                continue
            if str(order.get("side") or "").upper() != close_side:
                continue
            if str(order.get("reduceOnly") or "").lower() not in {"true", "1"} and order.get("reduceOnly") is not True:
                continue
            try:
                executed = abs(float(order.get("executedQty") or 0.0))
                avg_price = float(order.get("avgPrice") or 0.0)
            except Exception:
                executed = 0.0
                avg_price = 0.0
            if executed <= 0 or avg_price <= 0:
                continue
            candidates.append(order)
        if not candidates:
            return {}
        matched_candidates = candidates
        if expected_qty > 0:
            qty_tolerance = max(1e-9, expected_qty * 0.001)
            quantity_matched = []
            for order in candidates:
                try:
                    executed = abs(float(order.get("executedQty") or 0.0))
                except Exception:
                    executed = 0.0
                if abs(executed - expected_qty) <= qty_tolerance:
                    quantity_matched.append(order)
            if quantity_matched:
                matched_candidates = quantity_matched
        latest = max(matched_candidates, key=lambda item: int(item.get("updateTime") or item.get("time") or 0))
        try:
            close_time = datetime.fromtimestamp(int(latest.get("updateTime") or latest.get("time") or 0) / 1000, timezone.utc).isoformat()
        except Exception:
            close_time = utc_now_iso()
        return {
            "status": "found",
            "source": "binance_allOrders",
            "order_id": latest.get("orderId"),
            "client_order_id": latest.get("clientOrderId"),
            "side": latest.get("side"),
            "type": latest.get("type"),
            "orig_type": latest.get("origType"),
            "reduce_only": latest.get("reduceOnly"),
            "executed_qty": latest.get("executedQty"),
            "avg_price": latest.get("avgPrice"),
            "close_time": close_time,
        }

    def record_v5_profit_reentry_guard_from_exchange_close(
        self,
        state: dict[str, Any],
        position: dict[str, Any],
        close_order: dict[str, Any],
    ) -> None:
        if not self.config.v5_profit_reentry_guard_enabled:
            return
        if not isinstance(close_order, dict) or close_order.get("status") != "found":
            return
        symbol = str(position.get("symbol") or "").upper()
        side = str(position.get("side") or "").lower()
        if not symbol or side not in {"long", "short"}:
            return
        try:
            entry = float(position.get("entry_price") or 0.0)
            close_price = float(close_order.get("avg_price") or 0.0)
            leverage = float(position.get("leverage") or self.config.default_leverage or 1.0)
        except Exception:
            return
        roi_pct = self.position_roi_pct(side, entry, close_price, leverage)
        if roi_pct <= 0:
            return
        guards = state.get("v5_profit_reentry_guards") if isinstance(state.get("v5_profit_reentry_guards"), dict) else {}
        guards[symbol] = {
            "symbol": symbol,
            "side": side,
            "exit_price": round(close_price, 8),
            "entry_price": round(entry, 8),
            "roi_pct": round(roi_pct, 6),
            "event": "exchange_reconcile_close",
            "order_status": "reconciled_missing_exchange_position",
            "order_id": close_order.get("order_id"),
            "client_order_id": close_order.get("client_order_id"),
            "created_at": close_order.get("close_time") or utc_now_iso(),
            "pullback_pct": float(self.config.v5_profit_reentry_pullback_pct),
            "ma20_near_pct": float(self.config.v5_profit_reentry_ma20_near_pct),
            "source": "v5_profit_reentry_guard_exchange_reconcile",
        }
        state["v5_profit_reentry_guards"] = guards

    def enqueue_reconciled_closed_trade_review(
        self,
        position: dict[str, Any],
        *,
        close_reason: str,
        close_order: dict[str, Any] | None = None,
    ) -> None:
        symbol = str(position.get("symbol") or "").upper()
        if not symbol:
            return
        close_price = float(
            (close_order or {}).get("avg_price")
            or position.get("stop_loss_price")
            or position.get("current_price")
            or position.get("entry_price")
            or 0.0
        )
        closed_at = str((close_order or {}).get("close_time") or utc_now_iso())
        review = {
            "event": "v6_closed_trade_review_due",
            "status": "pending",
            "review_delay_hours": 12,
            "closed_at": closed_at,
            "review_due_at": iso_add_hours(closed_at, 12),
            "symbol": symbol,
            "side": str(position.get("side") or ""),
            "opened_at": position.get("opened_at"),
            "entry_price": float(position.get("entry_price") or 0.0),
            "close_price": close_price,
            "leverage": int(position.get("leverage") or self.config.default_leverage),
            "original_notional_usdt": float(position.get("original_notional_usdt") or position.get("notional_usdt") or self.config.base_notional_usdt),
            "close_reason": close_reason,
            "target_name": "",
            "risk_plan": position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {},
            "take_profit_stages": [dict(stage) for stage in position.get("take_profit_stages") or [] if isinstance(stage, dict)],
            "roi_take_profit_stages": [dict(stage) for stage in position.get("roi_take_profit_stages") or [] if isinstance(stage, dict)],
            "partial_take_profit_history": [dict(item) for item in position.get("partial_take_profit_history") or [] if isinstance(item, dict)],
            "closing_intent": {},
            "closing_order": {
                "status": "reconciled_missing_exchange_position",
                "protective_stop_order_id": position.get("protective_stop_order_id"),
                "protective_stop_client_order_id": position.get("protective_stop_client_order_id"),
                "exchange_close_order": close_order or {},
            },
        }
        append_text_line(self.config.closed_trade_review_queue_file, json.dumps(review, ensure_ascii=False))

    def reconcile_state_with_exchange(
        self,
        state: dict[str, Any],
        actual_rows: list[dict[str, Any]] | None = None,
        sync_exchange_side_effects: bool = True,
    ) -> dict[str, Any] | None:
        if self.config.order_mode == "dry_run":
            return None
        positions = [item for item in state.get("positions", []) if isinstance(item, dict)]
        try:
            if actual_rows is None:
                actual_rows = self.client.actual_position_rows()
            ignored_dust_positions: list[dict[str, Any]] = []
            ignored_dust_keys: set[tuple[str, str]] = set()
            actual_by_key: dict[tuple[str, str], dict[str, Any]] = {}
            for row in actual_rows:
                snapshot = V6BinanceClient.exchange_position_snapshot(row)
                key = self.position_key_from_risk_row(row)
                if self.client.is_dust_exchange_snapshot(snapshot):
                    ignored_dust_keys.add(key)
                    ignored_dust_positions.append({
                        "symbol": snapshot.get("symbol"),
                        "side": snapshot.get("side"),
                        "position_qty": snapshot.get("abs_position_qty"),
                        "notional": round(self.client.exchange_snapshot_notional(snapshot), 8),
                    })
                    continue
                actual_by_key[key] = row
            actual_keys = set(actual_by_key)
            kept = []
            updated = []
            v5_overlay_migrations = []
            for item in positions:
                key = self.position_key_from_state(item)
                if key not in actual_keys:
                    continue
                snapshot = V6BinanceClient.exchange_position_snapshot(actual_by_key[key])
                self.apply_exchange_snapshot_to_position(item, snapshot)
                self.reconcile_lots_to_exchange_quantity(
                    item,
                    abs(float(snapshot.get("position_amt") or snapshot.get("abs_position_qty") or 0.0)),
                    reason="scheduled_exchange_position_reconcile",
                )
                migration = self.ensure_v5_overlay_stop_on_position(item)
                if migration:
                    v5_overlay_migrations.append(migration)
                self.update_position_layer_summary(item)
                kept.append(item)
                updated.append({
                    "symbol": item.get("symbol"),
                    "side": item.get("side"),
                    "position_qty": item.get("position_qty"),
                    "entry_price": item.get("entry_price"),
                    "stop_loss_price": item.get("stop_loss_price"),
                    "lot_count": item.get("lot_count"),
                    "v5_overlay_migrated": bool(migration),
                })
            removed = [item for item in positions if self.position_key_from_state(item) not in actual_keys]
            local_keys = {self.position_key_from_state(item) for item in positions}
            recovered = [
                self.recovered_position_from_exchange_row(row)
                for key, row in actual_by_key.items()
                if key not in local_keys
            ]
            reconciled_positions = kept + recovered
            stop_sync = (
                self.sync_protective_stops_with_exchange(reconciled_positions, actual_by_key)
                if sync_exchange_side_effects
                else []
            )
            if not removed and not recovered:
                state["positions"] = reconciled_positions
                self.refresh_position_lot_index(state)
                return {
                    "event": "exchange_position_reconcile",
                    "status": "ok",
                    "removed": [],
                    "recovered": [],
                    "ignored_dust_positions": ignored_dust_positions,
                    "updated": updated,
                    "actual_positions": sorted(f"{symbol}:{side}" for symbol, side in actual_keys),
                    "protective_stop_sync": stop_sync,
                    "exchange_side_effects_enabled": sync_exchange_side_effects,
                    "v5_overlay_migrations": v5_overlay_migrations,
                }
            cancellations = []
            local_dust_removed = []
            for position in removed:
                symbol = str(position.get("symbol") or "").upper()
                side = str(position.get("side") or "")
                key = self.position_key_from_state(position)
                is_dust_cleanup = self.is_strategy_dust_position(position) or key in ignored_dust_keys
                if is_dust_cleanup:
                    self.mark_strategy_dust_position(position)
                    local_dust_removed.append({
                        "symbol": symbol,
                        "side": side,
                        "position_qty": position.get("position_qty"),
                        "notional_usdt": position.get("notional_usdt"),
                        "strategy_dust_notional_usdt": position.get("strategy_dust_notional_usdt"),
                        "strategy_dust_threshold_usdt": position.get("strategy_dust_threshold_usdt"),
                    })
                    if sync_exchange_side_effects:
                        cancellation = self.cancel_protective_stop_if_present(position)
                        if cancellation:
                            cancellations.append(cancellation)
                    continue
                close_order = (
                    self.recent_exchange_close_order_for_missing_position(position)
                    if sync_exchange_side_effects
                    else {
                        "status": "skipped",
                        "reason": "exchange_side_effects_disabled",
                        "source": "local_reconcile",
                    }
                )
                self.enqueue_reconciled_closed_trade_review(
                    position,
                    close_reason="exchange position missing after protective stop or manual close",
                    close_order=close_order,
                )
                self.record_v5_profit_reentry_guard_from_exchange_close(state, position, close_order)
                if symbol:
                    cooldown_candles = (
                        int(self.config.v5_reentry_wait_15m_candles or 0)
                        if self.config.v5_overlay_rules_active_for(symbol)
                        else 0
                    )
                    self.set_entry_cooldown(
                        state,
                        symbol,
                        hours=1,
                        reason="recent exchange close detected by reconcile",
                        side=side,
                        candles_15m=cooldown_candles,
                        timeframe="15m" if cooldown_candles > 0 else "",
                    )
                if sync_exchange_side_effects:
                    cancellation = self.cancel_protective_stop_if_present(position)
                    if cancellation:
                        cancellations.append(cancellation)
            state["positions"] = reconciled_positions
            self.refresh_position_lot_index(state)
            journal = list(state.get("journal") or [])
            for position in removed:
                key = self.position_key_from_state(position)
                if self.is_strategy_dust_position(position) or key in ignored_dust_keys:
                    journal.append({
                        "timestamp": utc_now_iso(),
                        "event": "reconcile_strategy_dust_position_ignored",
                        "symbol": position.get("symbol"),
                        "side": position.get("side"),
                        "reason": "local_or_exchange_position_below_strategy_dust_threshold",
                        "notional_usdt": position.get("notional_usdt"),
                        "strategy_dust_notional_usdt": position.get("strategy_dust_notional_usdt"),
                        "strategy_dust_threshold_usdt": position.get("strategy_dust_threshold_usdt"),
                    })
                    continue
                journal.append({
                    "timestamp": utc_now_iso(),
                    "event": "reconcile_missing_exchange_position",
                    "symbol": position.get("symbol"),
                    "side": position.get("side"),
                    "reason": "present_in_v6_state_but_absent_or_dust_in_binance_positionRisk",
                })
            for position in recovered:
                journal.append({
                    "timestamp": utc_now_iso(),
                    "event": "reconcile_recovered_exchange_position",
                    "symbol": position.get("symbol"),
                    "side": position.get("side"),
                    "reason": "present_in_binance_positionRisk_but_absent_in_v6_state",
                    "position_qty": position.get("position_qty"),
                    "entry_price": position.get("entry_price"),
                })
            state["journal"] = journal[-1000:]
            return {
                "event": "exchange_position_reconcile",
                "status": "ok",
                "removed": [position.get("symbol") for position in removed],
                "ignored_local_dust_positions": local_dust_removed,
                "recovered": [f"{position.get('symbol')}:{position.get('side')}" for position in recovered],
                "ignored_dust_positions": ignored_dust_positions,
                "updated": updated,
                "actual_positions": sorted(f"{symbol}:{side}" for symbol, side in actual_keys),
                "cancelled_protective_stops": cancellations,
                "protective_stop_sync": stop_sync,
                "exchange_side_effects_enabled": sync_exchange_side_effects,
                "v5_overlay_migrations": v5_overlay_migrations,
            }
        except Exception as exc:
            LOGGER.warning("V6 exchange position reconcile failed: %s", exc)
            return {
                "event": "exchange_position_reconcile",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def refresh_protective_stop_after_partial_take_profit(self, position: dict[str, Any]) -> dict[str, Any] | None:
        if self.config.order_mode == "dry_run":
            return None
        try:
            actual_rows = self.client.actual_position_rows(symbol=str(position.get("symbol") or ""))
            actual_by_key = {self.position_key_from_risk_row(row): row for row in actual_rows}
            actions = self.sync_protective_stops_with_exchange([position], actual_by_key)
            return {
                "event": "partial_take_profit_protective_stop_sync",
                "status": "ok",
                "symbol": position.get("symbol"),
                "side": position.get("side"),
                "actions": actions,
            }
        except Exception as exc:
            LOGGER.warning("V6 partial take-profit protective stop sync failed: %s", exc)
            return {
                "event": "partial_take_profit_protective_stop_sync",
                "status": "failed",
                "symbol": position.get("symbol"),
                "side": position.get("side"),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def ensure_v5_overlay_stop_on_position(self, position: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(position.get("symbol") or "").upper()
        if not self.config.v5_overlay_rules_active_for(symbol):
            return None
        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"}:
            return None
        risk_plan = position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {}
        overlay = risk_plan.get("v5_stop_overlay") if isinstance(risk_plan.get("v5_stop_overlay"), dict) else {}
        forced_stop_pct = max(0.0, float(self.config.v5_forced_stop_loss_pct or 0.0))
        if overlay.get("enabled") and risk_plan.get("v5_original_structure_stop_price") and forced_stop_pct <= 0:
            return None
        try:
            entry = float(position.get("entry_price") or 0.0)
            original_stop = float(position.get("stop_loss_price") or 0.0)
        except (TypeError, ValueError):
            return None
        if entry <= 0 or original_stop <= 0:
            return None
        if side == "long" and original_stop >= entry:
            return None
        if side == "short" and original_stop <= entry:
            return None
        if forced_stop_pct > 0:
            forced_stop = entry * (1.0 - forced_stop_pct) if side == "long" else entry * (1.0 + forced_stop_pct)
            if forced_stop <= 0:
                return None
            if abs(float(position.get("stop_loss_price") or 0.0) - forced_stop) <= max(entry * 0.000001, 0.00000001):
                return None
            now = utc_now_iso()
            risk_plan["v5_original_structure_stop_price"] = round(original_stop, 8)
            risk_plan["v5_forced_stop_loss_pct"] = round(forced_stop_pct, 8)
            risk_plan["v5_stop_overlay"] = {
                "enabled": False,
                "mode": "forced_hard_stop",
                "migrated_existing_position": True,
                "migrated_at": now,
                "forced_stop_loss_pct": round(forced_stop_pct * 100.0, 6),
                "hard_stop_distance_pct": round(abs(pct_diff(entry, forced_stop)), 6),
                "note": "Existing V5 position migrated to forced hard stop.",
            }
            risk_plan["stop_loss_price"] = round(forced_stop, 8)
            position["risk_plan"] = risk_plan
            position["v5_original_structure_stop_price"] = round(original_stop, 8)
            position["v5_overlay_migrated_at"] = now
            position["stop_loss_price"] = round(forced_stop, 8)
            return {
                "event": "v5_forced_stop_migrated",
                "symbol": symbol,
                "side": side,
                "original_stop_price": round(original_stop, 8),
                "forced_stop_price": round(forced_stop, 8),
                "forced_stop_loss_pct": round(forced_stop_pct * 100.0, 6),
            }
        stop_multiplier = max(1.0, float(self.config.v5_disaster_stop_multiplier or 1.0))
        if stop_multiplier <= 1.0:
            return None
        stop_distance = abs(entry - original_stop)
        disaster_stop = entry - stop_distance * stop_multiplier if side == "long" else entry + stop_distance * stop_multiplier
        if disaster_stop <= 0:
            return None
        now = utc_now_iso()
        risk_plan["v5_original_structure_stop_price"] = round(original_stop, 8)
        risk_plan["v5_stop_overlay"] = {
            "enabled": True,
            "mode": "disaster_hard_stop",
            "migrated_existing_position": True,
            "migrated_at": now,
            "disaster_stop_multiplier": stop_multiplier,
            "original_structure_stop_price": round(original_stop, 8),
            "original_structure_stop_distance_pct": round(abs(pct_diff(entry, original_stop)), 6),
            "hard_stop_distance_pct": round(abs(pct_diff(entry, disaster_stop)), 6),
            "soft_stop_note": "Existing V5 position migrated: original stop is now checked by closed 15M candle; exchange stop is widened as disaster stop.",
        }
        risk_plan["stop_loss_price"] = round(disaster_stop, 8)
        position["risk_plan"] = risk_plan
        position["v5_original_structure_stop_price"] = round(original_stop, 8)
        position["v5_overlay_migrated_at"] = now
        position["stop_loss_price"] = round(disaster_stop, 8)
        return {
            "event": "v5_overlay_stop_migrated",
            "symbol": symbol,
            "side": side,
            "original_structure_stop_price": round(original_stop, 8),
            "disaster_stop_price": round(disaster_stop, 8),
            "disaster_stop_multiplier": stop_multiplier,
        }

    @staticmethod
    def position_key_from_state(position: dict[str, Any]) -> tuple[str, str]:
        return str(position.get("symbol") or "").upper(), str(position.get("side") or "").lower()

    @staticmethod
    def position_key_from_risk_row(row: dict[str, Any]) -> tuple[str, str]:
        return str(row.get("symbol") or "").upper(), V6BinanceClient.position_side_from_risk_row(row)

    @staticmethod
    def position_side_from_stop_order(order: dict[str, Any]) -> str:
        side = str(order.get("side") or "").upper()
        if side == "BUY":
            return "short"
        if side == "SELL":
            return "long"
        return ""

    def sync_protective_stops_with_exchange(
        self,
        positions: list[dict[str, Any]],
        actual_by_key: dict[tuple[str, str], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not positions:
            return []
        actions: list[dict[str, Any]] = []
        open_stops = self.open_protective_stop_orders()
        stops_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for order in open_stops:
            key = (str(order.get("symbol") or "").upper(), self.position_side_from_stop_order(order))
            stops_by_key.setdefault(key, []).append(order)
        for position in positions:
            key = self.position_key_from_state(position)
            actual = actual_by_key.get(key)
            if not actual:
                continue
            expected_quantity = self.quantity_from_position_amount(str(position.get("symbol") or ""), actual)
            expected_trigger = self.client.resolve_price(str(position.get("symbol") or ""), float(position.get("stop_loss_price") or 0.0))
            current_stops = stops_by_key.get(key, [])
            matching = [order for order in current_stops if self.protective_stop_matches(order, expected_quantity, expected_trigger)]
            if len(current_stops) == 1 and matching:
                self.record_protective_stop_on_position(position, matching[0])
                continue
            for order in current_stops:
                order_id = order.get("algoId") or order.get("orderId")
                if order_id:
                    actions.append({"action": "cancel_stale_protective_stop", "symbol": position.get("symbol"), "order_id": order_id, "result": self.client.cancel_order(str(position.get("symbol") or ""), order_id)})
            intent = self.intent_from_state_position(position)
            precheck = self.client.protective_stop_precheck(intent)
            if precheck.get("would_immediately_trigger"):
                position["protective_stop_submit_status"] = "skipped_immediate_trigger"
                position["protective_stop_skip_reason"] = "would_immediately_trigger"
                position["protective_stop_precheck"] = precheck
                actions.append({
                    "action": "skip_protective_stop",
                    "symbol": position.get("symbol"),
                    "side": position.get("side"),
                    "quantity": expected_quantity,
                    "trigger_price": expected_trigger,
                    "reason": "would_immediately_trigger",
                    "precheck": precheck,
                })
                continue
            try:
                submitted = self.client.submit_protective_stop(intent, quantity=expected_quantity)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                if "-2021" in error_text or "Order would immediately trigger" in error_text:
                    position["protective_stop_submit_status"] = "skipped_immediate_trigger"
                    position["protective_stop_skip_reason"] = "exchange_rejected_immediate_trigger"
                    position["protective_stop_last_error"] = error_text
                    position["protective_stop_precheck"] = precheck
                    actions.append({
                        "action": "skip_protective_stop",
                        "symbol": position.get("symbol"),
                        "side": position.get("side"),
                        "quantity": expected_quantity,
                        "trigger_price": expected_trigger,
                        "reason": "exchange_rejected_immediate_trigger",
                        "error": error_text,
                        "precheck": precheck,
                    })
                    continue
                raise
            response = submitted.get("response") if isinstance(submitted, dict) else {}
            if isinstance(response, dict):
                self.record_protective_stop_on_position(position, response)
            actions.append({
                "action": "submit_protective_stop",
                "symbol": position.get("symbol"),
                "side": position.get("side"),
                "quantity": expected_quantity,
                "trigger_price": expected_trigger,
                "result_status": submitted.get("status") if isinstance(submitted, dict) else "",
            })
        return actions

    def open_protective_stop_orders(self) -> list[dict[str, Any]]:
        payload = self.client.signed_request("GET", "/fapi/v1/openAlgoOrders", self.client.signed_params())
        rows = payload if isinstance(payload, list) else []
        return [
            row for row in rows
            if isinstance(row, dict)
            and str(row.get("orderType") or row.get("type") or "") == "STOP_MARKET"
            and str(row.get("algoStatus") or row.get("status") or "").upper() in {"NEW", "PARTIALLY_FILLED"}
        ]

    def quantity_from_position_amount(self, symbol: str, row: dict[str, Any]) -> str:
        amount = abs(float(row.get("positionAmt") or 0.0))
        step, _ = self.client.quantity_rules(symbol)
        quantity = V6BinanceClient.round_down_to_step(amount, step)
        if quantity <= 0:
            raise V6ExecutionError(f"resolved zero protective stop quantity for {symbol}")
        return V6BinanceClient.format_decimal(quantity)

    @staticmethod
    def protective_stop_matches(order: dict[str, Any], expected_quantity: str, expected_trigger: str) -> bool:
        quantity = str(order.get("quantity") or order.get("origQty") or "")
        trigger = str(order.get("triggerPrice") or order.get("stopPrice") or "")
        reduce_only = str(order.get("reduceOnly") or "").lower() == "true" or order.get("reduceOnly") is True
        try:
            quantity_matches = Decimal(quantity) == Decimal(expected_quantity)
            trigger_matches = Decimal(trigger) == Decimal(expected_trigger)
        except Exception:
            quantity_matches = quantity == expected_quantity
            trigger_matches = trigger == expected_trigger
        return quantity_matches and trigger_matches and reduce_only

    def intent_from_state_position(self, position: dict[str, Any]) -> V6OrderIntent:
        side = str(position.get("side") or "").lower()
        action = "open_short" if side == "short" else "open_long"
        return V6OrderIntent(
            action=action,
            symbol=str(position.get("symbol") or "").upper(),
            side="SELL" if action == "open_short" else "BUY",
            quantity_hint_usdt=float(position.get("notional_usdt") or self.config.base_notional_usdt),
            leverage=int(position.get("leverage") or self.config.default_leverage),
            reduce_only=False,
            entry_price=float(position.get("entry_price") or 0.0),
            stop_loss_price=float(position.get("stop_loss_price") or 0.0),
            take_profit_price=float(position.get("take_profit_price") or 0.0),
            long_probability=0.0,
            short_probability=0.0,
            wait_probability=0.0,
            rationale="sync exchange protective stop",
            generated_at=utc_now_iso(),
        )

    @staticmethod
    def record_protective_stop_on_position(position: dict[str, Any], order: dict[str, Any]) -> None:
        order_id = order.get("algoId") or order.get("orderId")
        if order_id:
            position["protective_stop_order_id"] = order_id
        client_id = order.get("clientAlgoId") or order.get("clientOrderId")
        if client_id:
            position["protective_stop_client_order_id"] = client_id
        status = order.get("algoStatus") or order.get("status")
        if status:
            position["protective_stop_status"] = status
        position["protective_stop_submit_status"] = "submitted"

    def apply_lot_close_allocation(
        self,
        position: dict[str, Any],
        intent: V6OrderIntent,
        result: dict[str, Any],
        *,
        fallback_fraction: float,
    ) -> None:
        lots = self.normalize_lots(position)
        total_qty = sum(self.lot_remaining_quantity(lot) for lot in lots)
        if total_qty <= 0:
            position["lots"] = lots
            return
        close_qty = self.response_executed_qty(result)
        if close_qty <= 0:
            close_qty = total_qty * clamp(float(fallback_fraction or 0.0), 0.0, 1.0)
        close_qty = min(total_qty, max(0.0, close_qty))
        if close_qty <= 0:
            position["lots"] = lots
            return
        close_ratio = close_qty / total_qty
        allocations: list[dict[str, Any]] = []
        for lot in lots:
            previous_qty = self.lot_remaining_quantity(lot)
            lot_close_qty = min(previous_qty, previous_qty * close_ratio)
            remaining_qty = max(0.0, previous_qty - lot_close_qty)
            lot["remaining_quantity"] = round(remaining_qty, 12)
            if remaining_qty <= 0.000000001:
                lot["status"] = "closed"
                lot["closed_at"] = result.get("submitted_at") or utc_now_iso()
                lot["close_reason"] = intent.rationale
            allocations.append({
                "lot_id": lot.get("lot_id"),
                "previous_quantity": round(previous_qty, 12),
                "closed_quantity": round(lot_close_qty, 12),
                "remaining_quantity": round(remaining_qty, 12),
            })
        history = [dict(item) for item in position.get("lot_close_history") or [] if isinstance(item, dict)]
        history.append({
            "submitted_at": result.get("submitted_at") or utc_now_iso(),
            "action": intent.action,
            "target_name": intent.target_name,
            "rationale": intent.rationale,
            "close_quantity": round(close_qty, 12),
            "total_quantity_before": round(total_qty, 12),
            "close_ratio": round(close_ratio, 8),
            "allocations": allocations,
            "order_id": self.order_id_from_result(result),
        })
        position["lot_close_history"] = history[-50:]
        position["lots"] = [lot for lot in lots if self.lot_remaining_quantity(lot) > 0]

    @staticmethod
    def submitted_order_has_no_confirmed_position_delta(result: dict[str, Any]) -> bool:
        status = str(result.get("status") or "")
        if status not in {"submitted", "submitted_unprotected", "submitted_no_position_delta"}:
            return False
        reconcile = result.get("position_reconcile")
        if not isinstance(reconcile, dict):
            return False
        try:
            delta_qty = abs(float(reconcile.get("delta_qty_abs") or 0.0))
        except Exception:
            delta_qty = 0.0
        return delta_qty <= 0.0

    def apply_partial_take_profit_to_position(self, position: dict[str, Any], intent: V6OrderIntent, result: dict[str, Any]) -> dict[str, Any] | None:
        partial = (intent.risk_plan or {}).get("partial_take_profit") or {}
        partial_type = str(partial.get("type") or "bb_take_profit")
        if partial_type == "roi_take_profit":
            stage_key = "roi_take_profit_stages"
            stages = self.ensure_position_roi_take_profit_stages(position)
        elif partial_type in {
            "profit_giveback_guard",
            "h4_lower_reversal_guard",
            "v5_no_news_momentum_tp",
            "v5_no_news_momentum_final_giveback_exit",
            "v5_momentum_guard",
        }:
            stage_key = "profit_guard_stages"
            stages = [stage for stage in position.get(stage_key) or [] if isinstance(stage, dict)]
        else:
            stage_key = "take_profit_stages"
            stages = self.ensure_position_take_profit_stages(position)
        target_names = [name for name in str(intent.target_name or "").split(",") if name]
        matched_stages = [
            stage for stage in stages
            if str(stage.get("name") or "") in target_names and str(stage.get("status") or "pending") == "pending"
        ]
        if not matched_stages:
            return position
        if self.submitted_order_has_no_confirmed_position_delta(result):
            retry_after = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
            history = list(position.get("partial_take_profit_skipped_history") or [])
            history.append({
                "type": partial_type,
                "target_name": ",".join(target_names),
                "submitted_at": result.get("submitted_at") or utc_now_iso(),
                "result_status": result.get("status"),
                "reason": "exchange_position_delta_not_confirmed",
                "retry_after": retry_after,
                "position_reconcile": result.get("position_reconcile"),
                "order_id": self.order_id_from_result(result),
            })
            position["partial_take_profit_skipped_history"] = history[-20:]
            for matched_stage in matched_stages:
                matched_stage["status"] = "retry_wait"
                matched_stage["retry_after"] = retry_after
                matched_stage["retry_reason"] = "exchange_position_delta_not_confirmed"
                matched_stage["last_failed_order_id"] = self.order_id_from_result(result)
                matched_stage["last_failed_at"] = result.get("submitted_at") or utc_now_iso()
            return position

        fraction_of_initial = clamp(
            float(partial.get("fraction_of_initial") or sum(float(stage.get("fraction_of_initial") or 0.0) for stage in matched_stages)),
            0.0,
            1.0,
        )
        previous_remaining = clamp(float(position.get("remaining_fraction") or 1.0), 0.0, 1.0)
        remaining = max(0.0, previous_remaining - fraction_of_initial)
        submitted_at = result.get("submitted_at") or utc_now_iso()
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        per_stage_close_fraction = round(float(intent.close_fraction or 1.0), 6)
        for matched_stage in matched_stages:
            stage_executed_fraction = clamp(
                float(matched_stage.get("effective_fraction_of_initial") or matched_stage.get("fraction_of_initial") or 0.0),
                0.0,
                1.0,
            )
            matched_stage.update({
                "status": "executed",
                "executed_at": submitted_at,
                "trigger_price": intent.entry_price,
                "close_fraction_of_remaining": per_stage_close_fraction,
                "remaining_fraction_before": round(previous_remaining, 6),
                "remaining_fraction_after": round(remaining, 6),
                "result_status": result.get("status"),
                "partial_take_profit_type": partial_type,
                "executed_fraction_of_initial": round(stage_executed_fraction, 6),
                "touch_pending_volume": False,
                "touch_status": "executed",
            })
            if isinstance(response, dict):
                order_id = response.get("orderId") or response.get("algoId")
                if order_id:
                    matched_stage["order_id"] = order_id
                if response.get("executedQty") is not None:
                    matched_stage["executed_qty"] = response.get("executedQty")
                if response.get("avgPrice") is not None:
                    matched_stage["avg_price"] = response.get("avgPrice")

        position[stage_key] = stages
        position["remaining_fraction"] = round(remaining, 6)
        original_notional = float(position.get("original_notional_usdt") or position.get("notional_usdt") or self.config.base_notional_usdt)
        position["original_notional_usdt"] = original_notional
        position["notional_usdt"] = round(original_notional * remaining, 6)
        position["current_price"] = intent.entry_price
        history = list(position.get("partial_take_profit_history") or [])
        history.append({
            "type": partial_type,
            "target_name": ",".join(target_names),
            "trigger_price": intent.entry_price,
            "fraction_of_initial": round(fraction_of_initial, 6),
            "close_fraction_of_remaining": round(float(intent.close_fraction or 1.0), 6),
            "trigger_roi_pct": partial.get("trigger_roi_pct"),
            "roi_thresholds": partial.get("roi_thresholds"),
            "trigger_giveback_pct": partial.get("trigger_giveback_pct"),
            "submitted_at": submitted_at,
            "result_status": result.get("status"),
        })
        position["partial_take_profit_history"] = history[-20:]
        self.apply_lot_close_allocation(
            position,
            intent,
            result,
            fallback_fraction=fraction_of_initial,
        )
        after = self.exchange_after_from_result(result)
        if after:
            self.apply_exchange_snapshot_to_position(position, after)
            self.reconcile_lots_to_exchange_quantity(
                position,
                abs(float(after.get("position_amt") or after.get("abs_position_qty") or 0.0)),
                reason="partial_take_profit_after_order_reconcile",
            )
        else:
            total_remaining_qty = sum(self.lot_remaining_quantity(lot) for lot in self.normalize_lots(position))
            if total_remaining_qty > 0:
                position["position_qty"] = round(total_remaining_qty, 12)
                position["remaining_quantity"] = round(total_remaining_qty, 12)
        self.update_position_layer_summary(position)
        return None if remaining <= 0.001 else position

    def enqueue_closed_take_profit_review(self, position: dict[str, Any], intent: V6OrderIntent, result: dict[str, Any]) -> None:
        if not self.is_staged_take_profit_intent(intent) and "take profit" not in str(intent.rationale or "").lower():
            return
        submitted_at = result.get("submitted_at") or utc_now_iso()
        review = {
            "event": "v6_closed_trade_review_due",
            "status": "pending",
            "review_delay_hours": 12,
            "closed_at": submitted_at,
            "review_due_at": iso_add_hours(submitted_at, 12),
            "symbol": intent.symbol,
            "side": str(position.get("side") or ("short" if intent.action == "close_short" else "long")),
            "opened_at": position.get("opened_at"),
            "entry_price": float(position.get("entry_price") or 0.0),
            "close_price": float(intent.entry_price or 0.0),
            "leverage": int(position.get("leverage") or intent.leverage or self.config.default_leverage),
            "original_notional_usdt": float(position.get("original_notional_usdt") or position.get("notional_usdt") or intent.quantity_hint_usdt),
            "close_reason": intent.rationale,
            "target_name": intent.target_name,
            "risk_plan": position.get("risk_plan") if isinstance(position.get("risk_plan"), dict) else {},
            "take_profit_stages": [dict(stage) for stage in position.get("take_profit_stages") or [] if isinstance(stage, dict)],
            "roi_take_profit_stages": [dict(stage) for stage in position.get("roi_take_profit_stages") or [] if isinstance(stage, dict)],
            "partial_take_profit_history": [dict(item) for item in position.get("partial_take_profit_history") or [] if isinstance(item, dict)],
            "closing_intent": intent.to_dict(),
            "closing_order": result,
        }
        append_text_line(self.config.closed_trade_review_queue_file, json.dumps(review, ensure_ascii=False))

    @staticmethod
    def filled_entry_price_for_state(intent: V6OrderIntent, result: dict[str, Any]) -> float:
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        for key in ("avgPrice", "price"):
            value = response.get(key) if isinstance(response, dict) else None
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = 0.0
            if parsed > 0:
                return parsed
        try:
            executed_qty = float(response.get("executedQty") or 0.0)
            cum_quote = float(response.get("cumQuote") or 0.0)
        except (TypeError, ValueError):
            executed_qty = 0.0
            cum_quote = 0.0
        if executed_qty > 0 and cum_quote > 0:
            return cum_quote / executed_qty
        return float(intent.entry_price or 0.0)

    @staticmethod
    def filled_notional_for_state(intent: V6OrderIntent, result: dict[str, Any], entry_price: float) -> float:
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        if isinstance(response, dict):
            try:
                cum_quote = float(response.get("cumQuote") or 0.0)
            except (TypeError, ValueError):
                cum_quote = 0.0
            if cum_quote > 0:
                return cum_quote
            try:
                executed_qty = float(response.get("executedQty") or response.get("origQty") or 0.0)
            except (TypeError, ValueError):
                executed_qty = 0.0
            if executed_qty > 0 and entry_price > 0:
                return executed_qty * entry_price
        return float(intent.quantity_hint_usdt or 0.0)

    def apply_scale_in_to_position(self, position: dict[str, Any], intent: V6OrderIntent, result: dict[str, Any]) -> dict[str, Any]:
        fill_price = self.filled_entry_price_for_state(intent, result)
        add_notional = self.filled_notional_for_state(intent, result, fill_price)
        old_entry = float(position.get("entry_price") or 0.0)
        old_notional = float(position.get("notional_usdt") or position.get("original_notional_usdt") or 0.0)
        old_original = float(position.get("original_notional_usdt") or old_notional)
        combined_notional = old_notional + add_notional
        combined_original = old_original + add_notional
        combined_entry = self.weighted_average_entry(old_entry, old_notional, fill_price, add_notional)
        scale_meta = (intent.risk_plan or {}).get("scale_in") if isinstance(intent.risk_plan, dict) else {}
        strategy_labels = [str(item) for item in (intent.risk_plan or {}).get("strategy_labels") or [] if str(item or "")]
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        history = [dict(item) for item in position.get("scale_in_history") or [] if isinstance(item, dict)]
        history.append({
            "index": len(history) + 1,
            "submitted_at": result.get("submitted_at") or utc_now_iso(),
            "entry_price": round(fill_price, 8),
            "notional_usdt": round(add_notional, 6),
            "strategy_labels": strategy_labels,
            "strategy_label": strategy_labels[0] if strategy_labels else "",
            "trigger_name": str((scale_meta or {}).get("trigger_name") or ""),
            "trigger": dict((scale_meta or {}).get("trigger") or {}),
            "adverse_move_pct": (scale_meta or {}).get("adverse_move_pct"),
            "order_id": response.get("orderId") if isinstance(response, dict) else None,
            "result_status": result.get("status"),
        })
        stages = [dict(stage) for stage in intent.risk_plan.get("staged_take_profit_targets") or [] if isinstance(stage, dict)]
        lots = self.normalize_lots(position)
        lots.append(self.lot_from_open_result(intent, result, fill_price=fill_price, notional=add_notional))
        position.setdefault("initial_entry_price", old_entry)
        position["entry_price"] = round(combined_entry, 8)
        position["current_price"] = fill_price
        position["stop_loss_price"] = intent.stop_loss_price
        position["take_profit_price"] = intent.take_profit_price
        position["notional_usdt"] = round(combined_notional, 6)
        position["original_notional_usdt"] = round(combined_original, 6)
        position["remaining_fraction"] = round(combined_notional / combined_original, 6) if combined_original > 0 else 1.0
        position["risk_plan"] = intent.risk_plan
        position["take_profit_stages"] = stages
        position["scale_in_count"] = len(history)
        position["scale_in_history"] = history[-10:]
        position["last_scale_in_at"] = result.get("submitted_at") or utc_now_iso()
        position["lots"] = lots
        after = self.exchange_after_from_result(result)
        if after:
            self.apply_exchange_snapshot_to_position(position, after)
            self.reconcile_lots_to_exchange_quantity(
                position,
                abs(float(after.get("position_amt") or after.get("abs_position_qty") or 0.0)),
                reason="scale_in_after_order_reconcile",
            )
        self.update_position_layer_summary(position)
        return position

    def apply_intent_to_state(self, state: dict[str, Any], intent: V6OrderIntent, result: dict[str, Any]) -> None:
        positions = [item for item in state.get("positions", []) if isinstance(item, dict)]
        if intent.action in {"open_long", "open_short"} and self.submitted_order_has_no_confirmed_position_delta(result):
            expected_side = "short" if intent.action == "open_short" else "long"
            skip_payload = {
                "timestamp": result.get("submitted_at") or utc_now_iso(),
                "intent": intent.to_dict(),
                "result_status": result.get("status"),
                "reason": "exchange_position_delta_not_confirmed",
                "position_reconcile": result.get("position_reconcile"),
                "order_id": self.order_id_from_result(result),
            }
            matched_existing_position = False
            for position in positions:
                if (
                    str(position.get("symbol") or "").upper() == intent.symbol.upper()
                    and str(position.get("side") or "").lower() == expected_side
                ):
                    matched_existing_position = True
                    history_key = "scale_in_skipped_history" if self.is_scale_in_intent(intent) else "open_skipped_history"
                    history = [dict(item) for item in position.get(history_key) or [] if isinstance(item, dict)]
                    history.append(skip_payload)
                    position[history_key] = history[-20:]
            if not matched_existing_position:
                history = [dict(item) for item in state.get("open_skipped_history") or [] if isinstance(item, dict)]
                history.append(skip_payload)
                state["open_skipped_history"] = history[-50:]
                self.set_entry_cooldown(
                    state,
                    intent.symbol,
                    hours=0.5,
                    reason="exchange_position_delta_not_confirmed",
                    side=expected_side,
                )
            state["positions"] = positions
            self.refresh_position_lot_index(state)
            journal = list(state.get("journal") or [])
            journal.append({
                "timestamp": utc_now_iso(),
                "intent": intent.to_dict(),
                "result_status": result.get("status"),
                "order_mode": result.get("order_mode"),
                "state_mutation": "skipped_no_exchange_position_delta",
            })
            state["journal"] = journal[-1000:]
            return
        if intent.action in {"close_long", "close_short"}:
            if self.is_partial_take_profit_intent(intent):
                updated_positions: list[dict[str, Any]] = []
                for position in positions:
                    if str(position.get("symbol") or "").upper() != intent.symbol.upper() or str(position.get("side") or "").lower() != local_side_for_action(intent.action):
                        updated_positions.append(position)
                        continue
                    updated = self.apply_partial_take_profit_to_position(position, intent, result)
                    if not self.submitted_order_has_no_confirmed_position_delta(result):
                        self.record_v5_profit_reentry_guard(
                            state,
                            position,
                            intent,
                            result,
                            event="partial_take_profit",
                        )
                    if updated is not None:
                        updated_positions.append(updated)
                positions = updated_positions
            else:
                updated_positions = []
                for position in positions:
                    if str(position.get("symbol") or "").upper() != intent.symbol.upper() or str(position.get("side") or "").lower() != local_side_for_action(intent.action):
                        updated_positions.append(position)
                        continue
                    if self.submitted_order_has_no_confirmed_position_delta(result):
                        history = list(position.get("close_skipped_history") or [])
                        history.append({
                            "submitted_at": result.get("submitted_at") or utc_now_iso(),
                            "result_status": result.get("status"),
                            "reason": "exchange_position_delta_not_confirmed",
                            "position_reconcile": result.get("position_reconcile"),
                            "order_id": self.order_id_from_result(result),
                            "intent": intent.to_dict(),
                        })
                        position["close_skipped_history"] = history[-20:]
                        updated_positions.append(position)
                        continue
                    self.record_v5_profit_reentry_guard(
                        state,
                        position,
                        intent,
                        result,
                        event="full_close",
                    )
                    stop_loss_cooldown_minutes = max(0.0, float(self.config.v5_stop_loss_cooldown_minutes or 0.0))
                    close_reason_text = str(intent.rationale or "").lower()
                    if (
                        stop_loss_cooldown_minutes > 0
                        and self.config.v5_overlay_rules_active_for(intent.symbol)
                        and (
                            "stop loss" in close_reason_text
                            or "soft stop" in close_reason_text
                            or "structure stop" in close_reason_text
                        )
                    ):
                        self.set_entry_cooldown(
                            state,
                            intent.symbol,
                            hours=stop_loss_cooldown_minutes / 60.0,
                            reason=f"v5_stop_loss_cooldown_{stop_loss_cooldown_minutes:.0f}m",
                            side=local_side_for_action(intent.action),
                        )
                    logic_cooldown_minutes = max(0, int(self.config.logic_weakening_reentry_cooldown_minutes or 0))
                    if logic_cooldown_minutes > 0 and (
                        "original 4h logic weakened" in close_reason_text
                        or "ma20 no longer supports original" in close_reason_text
                    ):
                        self.set_entry_cooldown(
                            state,
                            intent.symbol,
                            hours=logic_cooldown_minutes / 60.0,
                            reason=f"v6_logic_weakening_reentry_cooldown_{logic_cooldown_minutes}m",
                            side=local_side_for_action(intent.action),
                        )
                    self.enqueue_closed_take_profit_review(position, intent, result)
                positions = updated_positions
        elif intent.action in {"open_long", "open_short"}:
            expected_side = "short" if intent.action == "open_short" else "long"
            if result.get("status") == "closed_after_protective_stop_failed":
                positions = [
                    item for item in positions
                    if not (
                        str(item.get("symbol") or "").upper() == intent.symbol.upper()
                        and str(item.get("side") or "").lower() == expected_side
                    )
                ]
                self.set_entry_cooldown(
                    state,
                    intent.symbol,
                    hours=1,
                    reason="protective_stop_failed_emergency_closed",
                    side=expected_side,
                )
                state["positions"] = positions
                self.refresh_position_lot_index(state)
                journal = list(state.get("journal") or [])
                journal.append({
                    "timestamp": utc_now_iso(),
                    "intent": intent.to_dict(),
                    "result_status": result.get("status"),
                    "order_mode": result.get("order_mode"),
                })
                state["journal"] = journal[-1000:]
                return
            if self.is_scale_in_intent(intent):
                updated_positions = []
                applied = False
                for position in positions:
                    if str(position.get("symbol") or "").upper() == intent.symbol.upper() and str(position.get("side") or "").lower() == expected_side:
                        updated_positions.append(self.apply_scale_in_to_position(position, intent, result))
                        applied = True
                    else:
                        updated_positions.append(position)
                if applied:
                    state["positions"] = updated_positions
                    self.refresh_position_lot_index(state)
                    journal = list(state.get("journal") or [])
                    journal.append({
                        "timestamp": utc_now_iso(),
                        "intent": intent.to_dict(),
                        "result_status": result.get("status"),
                        "order_mode": result.get("order_mode"),
                    })
                    state["journal"] = journal[-1000:]
                    return
            positions = [item for item in positions if not (str(item.get("symbol") or "").upper() == intent.symbol.upper() and str(item.get("side") or "").lower() == expected_side)]
            fill_price = self.filled_entry_price_for_state(intent, result)
            fill_notional = self.filled_notional_for_state(intent, result, fill_price)
            after_snapshot = self.exchange_after_from_result(result)
            actual_qty = abs(float(after_snapshot.get("position_amt") or after_snapshot.get("abs_position_qty") or 0.0)) if after_snapshot else 0.0
            actual_entry = float(after_snapshot.get("entry_price") or 0.0) if after_snapshot else 0.0
            actual_mark = float(after_snapshot.get("mark_price") or 0.0) if after_snapshot else 0.0
            actual_notional = abs(float(after_snapshot.get("notional") or 0.0)) if after_snapshot else 0.0
            protective_stop = result.get("protective_stop") if isinstance(result.get("protective_stop"), dict) else {}
            protective_response = protective_stop.get("response") if isinstance(protective_stop, dict) else {}
            stages = [dict(stage) for stage in intent.risk_plan.get("staged_take_profit_targets") or [] if isinstance(stage, dict)]
            lot = self.lot_from_open_result(intent, result, fill_price=fill_price, notional=fill_notional)
            strategy_labels = [str(item) for item in (intent.risk_plan or {}).get("strategy_labels") or [] if str(item or "")]
            position = {
                "symbol": intent.symbol,
                "side": "short" if intent.action == "open_short" else "long",
                "opened_at": result.get("submitted_at") or utc_now_iso(),
                "entry_price": round(actual_entry or fill_price or intent.entry_price, 8),
                "current_price": round(actual_mark or fill_price or intent.entry_price, 8),
                "stop_loss_price": intent.stop_loss_price,
                "take_profit_price": intent.take_profit_price,
                "position_qty": round(actual_qty or self.response_executed_qty(result), 12),
                "remaining_quantity": round(actual_qty or self.response_executed_qty(result), 12),
                "notional_usdt": round(actual_notional or fill_notional or intent.quantity_hint_usdt, 6),
                "leverage": intent.leverage,
                "order_mode": result.get("order_mode"),
                "source": "v6_execution",
                "strategy_labels": strategy_labels,
                "strategy_label": strategy_labels[0] if strategy_labels else "",
                "risk_plan": intent.risk_plan,
                "horizon": (intent.risk_plan or {}).get("horizon"),
                "entry_timeframe": (intent.risk_plan or {}).get("entry_timeframe"),
                "max_hold_days": (intent.risk_plan or {}).get("max_hold_days"),
                "original_notional_usdt": round(actual_notional or fill_notional or intent.quantity_hint_usdt, 6),
                "remaining_fraction": 1.0,
                "take_profit_stages": stages,
                "lots": [lot],
            }
            if after_snapshot:
                self.apply_exchange_snapshot_to_position(position, after_snapshot)
                self.reconcile_lots_to_exchange_quantity(
                    position,
                    abs(float(after_snapshot.get("position_amt") or after_snapshot.get("abs_position_qty") or 0.0)),
                    reason="open_after_order_reconcile",
                )
            protective_id = protective_response.get("algoId") or protective_response.get("orderId") if isinstance(protective_response, dict) else None
            if isinstance(protective_response, dict) and protective_id:
                position["protective_stop_order_id"] = protective_id
                position["protective_stop_client_order_id"] = protective_response.get("clientAlgoId") or protective_response.get("clientOrderId")
                position["protective_stop_status"] = protective_response.get("algoStatus") or protective_response.get("status")
            if isinstance(protective_stop, dict) and protective_stop.get("status"):
                position["protective_stop_submit_status"] = protective_stop.get("status")
            if result.get("status") == "submitted_unprotected":
                position["risk_status"] = "unprotected_protective_stop_failed"
                emergency_close = result.get("emergency_close") if isinstance(result.get("emergency_close"), dict) else {}
                if emergency_close:
                    position["emergency_close_status"] = emergency_close.get("status")
            self.update_position_layer_summary(position)
            positions.append(position)
        state["positions"] = positions
        self.refresh_position_lot_index(state)
        journal = list(state.get("journal") or [])
        journal.append({
            "timestamp": utc_now_iso(),
            "intent": intent.to_dict(),
            "result_status": result.get("status"),
            "order_mode": result.get("order_mode"),
        })
        state["journal"] = journal[-1000:]


def price_position_pct(closes: list[float], price: float, period: int) -> float:
    window = closes[-period:] if len(closes) >= period else closes
    if not window:
        return 50.0
    high = max(window)
    low = min(window)
    if high <= low:
        return 50.0
    return (price - low) / (high - low) * 100.0


def compression_score(closes: list[float], period: int = 20) -> float:
    window = closes[-period:] if len(closes) >= period else closes
    if not window:
        return 0.0
    mid = sum(window) / len(window)
    span = max(window) - min(window)
    if mid <= 0:
        return 0.0
    return clamp(1.0 - span / mid, 0.0, 1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent V6 signal + execution layer")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols; overrides TS_V6_SYMBOLS")
    parser.add_argument("--order-mode", default="", help="dry_run, testnet_order/demo, live_order")
    parser.add_argument("--submit", action="store_true", help="Submit orders for generated intents")
    parser.add_argument("--commit-dry-run-state", action="store_true", help="Write dry-run open/close intents into V6 state")
    parser.add_argument("--json", action="store_true", help="Print full JSON summary")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = V6ExecutionConfig.from_env()
    if args.symbols:
        config.symbols = parse_symbols(args.symbols)
    if args.order_mode:
        config.order_mode = canonical_order_mode(args.order_mode)
    engine = V6ExecutionEngine(config)
    try:
        summary = engine.run_once(submit=bool(args.submit), commit_dry_run_state=bool(args.commit_dry_run_state))
    finally:
        engine.close()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "generated_at": summary["generated_at"],
                    "order_mode": summary["order_mode"],
                    "submit": summary["submit"],
                    "signal_count": len(summary["signals"]),
                    "intent_count": len(summary["intents"]),
                    "result_count": len(summary["results"]),
                    "latest_summary_file": str(config.latest_summary_file),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0
