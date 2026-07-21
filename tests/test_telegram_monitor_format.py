from __future__ import annotations

import time
from types import SimpleNamespace

import scripts.telegram_monitor_v5_v6 as tg
from scripts.telegram_monitor_v5_v6 import (
    execution_status_text,
    format_compact_execution_result,
    format_report_v6_position_line,
    format_v5_paper_journal_event,
    format_v5_trade_activity_details,
    format_v5_trade_request,
    format_v6_activity_details,
)


def test_report_command_ignores_natural_language_symbol_placeholder() -> None:
    assert tg.normalize_symbol("相关币种") == ""
    assert tg.parse_report_command_text("report 相关币种") == ["BTCUSDT", "ETHUSDT"]
    assert tg.parse_report_command_text("report btc eth sol hype") == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "HYPEUSDT",
    ]


def test_execution_status_text_uses_icons() -> None:
    assert execution_status_text("submitted") == "✅"
    assert execution_status_text("filled") == "✅"
    assert execution_status_text("failed") == "❌"
    assert execution_status_text("rejected") == "❌"


def test_compact_execution_result_puts_execution_icon_last() -> None:
    text = format_compact_execution_result(
        {
            "status": "submitted",
            "order_id": "abc123",
            "executed_qty": "0.01",
            "avg_price": "65000",
        }
    )

    assert text == "成交=0.01@65000.00 订单=abc123 执行=✅"


def test_failed_compact_execution_result_puts_failed_icon_last() -> None:
    text = format_compact_execution_result({"status": "failed", "error": "Binance execution failed"})

    assert text == "错误=Binance执行失败 执行=❌"


def test_v5_trade_request_puts_execution_icon_after_reason() -> None:
    text = format_v5_trade_request(
        {
            "symbol": "ethusdt",
            "action": "open_long",
            "status": "failed",
            "entry_price": 3000,
            "rationale": "manual-check",
        }
    )

    assert text.endswith("理由=manual-check 执行=❌")


def test_v5_paper_journal_event_marks_success_at_end() -> None:
    text = format_v5_paper_journal_event(
        {
            "symbol": "btcusdt",
            "action": "open_long",
            "entry_price": 65000,
            "reason": "paper-fill",
        }
    )

    assert text.endswith("理由=paper-fill 执行=✅")


def test_v5_v6_execution_titles_use_phone_emoji() -> None:
    assert format_v6_activity_details({"intents": [], "results": []}).splitlines()[0] == "☎️ [Somnus执行] V6"
    assert format_v5_trade_activity_details(
        {
            "trade_requests": [
                {
                    "symbol": "btcusdt",
                    "action": "open_long",
                    "status": "submitted",
                    "entry_price": 65000,
                }
            ]
        }
    ).splitlines()[0] == "☎️ [Somnus执行] V5"


def test_v6_activity_pairs_results_by_symbol_and_action_not_list_index() -> None:
    intents = [
        {"symbol": "BTCUSDT", "action": "open_long", "risk_plan": {}, "rationale": "btc"},
        {"symbol": "ETHUSDT", "action": "open_short", "risk_plan": {}, "rationale": "eth"},
    ]
    results = [
        {"symbol": "ETHUSDT", "action": "open_short", "status": "submitted", "order_id": "eth-order"},
        {"symbol": "BTCUSDT", "action": "open_long", "status": "submitted", "order_id": "btc-order"},
    ]

    message = format_v6_activity_details({"intents": intents, "results": results})
    lines = message.splitlines()

    assert "BTCUSDT" in lines[1]
    assert "btc-order" in lines[2]
    assert "ETHUSDT" in lines[3]
    assert "eth-order" in lines[4]


def test_v6_activity_pairs_real_nested_execution_results_and_shows_fill() -> None:
    intents = [
        {"symbol": "BTCUSDT", "action": "open_long", "risk_plan": {}, "rationale": "btc"},
        {"symbol": "ETHUSDT", "action": "open_short", "risk_plan": {}, "rationale": "eth"},
    ]
    results = [
        {
            "status": "submitted",
            "intent": {"symbol": "ETHUSDT", "action": "open_short"},
            "response": {"orderId": "eth-order", "executedQty": "0.25", "avgPrice": "2000"},
        },
        {
            "status": "submitted",
            "intent": {"symbol": "BTCUSDT", "action": "open_long"},
            "response": {"orderId": "btc-order", "executedQty": "0.01", "avgPrice": "65000"},
        },
    ]

    message = format_v6_activity_details({"intents": intents, "results": results})
    lines = message.splitlines()

    assert "BTCUSDT" in lines[1]
    assert lines[2] == "成交=0.01@65000.00 订单=btc-order 执行=✅"
    assert "ETHUSDT" in lines[3]
    assert lines[4] == "成交=0.25@2000.00 订单=eth-order 执行=✅"


def test_report_position_line_outputs_only_plain_pnl_ratio_not_source_or_amount() -> None:
    text = format_report_v6_position_line(
        "BTCUSDT",
        [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "entry_price": 100,
                "mark_price": 110,
                "notional_usdt": 1000,
                "position_initial_margin_usdt": 100,
                "unrealized_pnl_usdt": 73.36,
                "exchange_source": "binance_demo_positionRisk",
            }
        ],
    )

    assert "盈亏=73%" in text
    assert "交易所实际" not in text
    assert "+73" not in text
    assert "U /" not in text
    assert "73.36U" not in text
    assert "+10.00%" not in text


def test_report_position_line_marks_missing_exchange_pnl_instead_of_estimating() -> None:
    text = format_report_v6_position_line(
        "BTCUSDT",
        [
            {
                "symbol": "BTCUSDT",
                "side": "long",
                "entry_price": 100,
                "mark_price": 110,
                "notional_usdt": 1000,
            }
        ],
        pnl_error_note="交易所读取失败：timeout",
    )

    assert "未获取到交易所实际盈亏" in text
    assert "timeout" in text
    assert "+10.00%" not in text


def test_current_report_v6_positions_prefers_exchange_positions(monkeypatch) -> None:
    local = [{"symbol": "BTCUSDT", "side": "long", "entry_price": 100, "mark_price": 110}]

    def fake_fetch(local_positions):
        assert local_positions == local
        return [dict(local[0], unrealized_pnl_usdt=7.89, exchange_source="binance_demo_positionRisk")], "binance_demo_positionRisk"

    monkeypatch.setattr(tg, "current_v6_summary_positions", lambda: local)
    monkeypatch.setattr(tg, "fetch_v6_exchange_positions", fake_fetch)

    positions, source, error_note = tg.current_report_v6_positions()

    assert positions[0]["unrealized_pnl_usdt"] == 7.89
    assert source == "binance_demo_positionRisk"
    assert error_note == ""


def test_current_report_v6_positions_reports_exchange_failure(monkeypatch) -> None:
    local = [{"symbol": "ETHUSDT", "side": "short", "entry_price": 3000, "mark_price": 2900}]

    def fake_fetch(local_positions):
        raise RuntimeError("timeout")

    monkeypatch.setattr(tg, "current_v6_summary_positions", lambda: local)
    monkeypatch.setattr(tg, "fetch_v6_exchange_positions", fake_fetch)

    positions, source, error_note = tg.current_report_v6_positions()

    assert positions == local
    assert source == "本地V6状态"
    assert "未获取到交易所实际盈亏" in error_note
    assert "timeout" in error_note


def test_fetch_command_klines_refreshes_api_instead_of_command_cache(monkeypatch) -> None:
    symbol = "BTCUSDT"
    interval = "15m"
    max_period = 10
    bars = max_period + 20
    fresh_rows = [
        {
            "open_time": 2,
            "open": 2.0,
            "high": 2.0,
            "low": 2.0,
            "close": 2.0,
            "volume": 1.0,
            "binance_symbol": symbol,
            "interval": interval,
        }
    ]
    tg.COMMAND_KLINE_MEMORY_CACHE.clear()
    tg.COMMAND_KLINE_MEMORY_CACHE[(symbol, interval, bars)] = (
        time.time(),
        [dict(fresh_rows[0], open_time=1, close=1.0)],
    )
    calls: list[dict[str, object]] = []

    class FakeLabeler:
        def __init__(self, cache_dir: str):
            self.cache_dir = cache_dir

        def fetch_klines(self, fetch_symbol, fetch_interval, start, end, **kwargs):
            calls.append(
                {
                    "symbol": fetch_symbol,
                    "interval": fetch_interval,
                    "force_refresh": kwargs.get("force_refresh"),
                    "start": start,
                    "end": end,
                }
            )
            return list(fresh_rows)

    monkeypatch.setattr(tg, "CoinScout4HLabeler", FakeLabeler)
    monkeypatch.setattr(
        tg,
        "load_recent_command_kline_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("disk cache used")),
    )

    rows = tg.fetch_command_klines(symbol, interval, max_period)

    assert rows == fresh_rows
    assert calls and calls[0]["force_refresh"] is True


def test_v5_v6_bots_are_push_only_even_when_command_flag_is_enabled() -> None:
    class PushOnlyClient:
        def get_updates(self, *args, **kwargs):
            raise AssertionError("V5/V6 bot must not poll interactive Telegram updates")

    for scope in ("v5", "v6"):
        args = SimpleNamespace(scope=scope, disable_commands=False, dry_run=False)
        assert tg.handle_telegram_commands(args, PushOnlyClient(), {}) == 0
