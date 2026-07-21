from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from scripts import hermes_somnus_command as command
from scripts import telegram_monitor_v5_v6 as tg


def test_report_appends_crypto_only_whale_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(command.tg, "build_report_response", lambda symbols: f"report:{','.join(symbols)}")
    monkeypatch.setattr(command, "_telegram_whale_summary", lambda: "\n🐋 whale summary")

    assert command._report(["BTC"]) == 0
    assert capsys.readouterr().out == "report:BTCUSDT\n\n🐋 whale summary\n"


def test_whale_summary_formats_completed_and_imminent_flows(monkeypatch, tmp_path) -> None:
    bridge = tmp_path / "telegram_whale_crypto.json"
    bridge.write_text(
        """{
          "windows": {
            "60m": {
              "event_count": 3,
              "totals": {
                "liquidated_long_usd": 1200000,
                "liquidated_short_usd": 200000,
                "net_liquidation_pressure_usd": -1000000
              },
              "symbols": [{
                "symbol": "BTC",
                "liquidated_long_usd": 1200000,
                "liquidated_short_usd": 200000,
                "imminent_long_usd": 5000000,
                "imminent_short_usd": 0
              }]
            }
          }
        }""",
        encoding="utf-8",
    )
    monkeypatch.setattr(command, "TELEGRAM_WHALE_BRIDGE_FILE", bridge)

    text = command._telegram_whale_summary()

    assert "Hyperdash 加密资金流" in text
    assert "净压力=偏空 1.00M" in text
    assert "BTC：多爆=1.20M｜空爆=200.00K｜临近多爆=5.00M" in text


def test_whale_summary_falls_back_to_24h_when_60m_is_empty(monkeypatch, tmp_path) -> None:
    bridge = tmp_path / "telegram_whale_crypto.json"
    bridge.write_text(
        """{
          "windows": {
            "60m": {"event_count": 0, "totals": {}, "symbols": []},
            "24h": {
              "event_count": 1,
              "totals": {
                "liquidated_long_usd": 100000,
                "liquidated_short_usd": 0,
                "net_liquidation_pressure_usd": -100000
              },
              "symbols": [{"symbol": "ETH", "liquidated_long_usd": 100000}]
            }
          }
        }""",
        encoding="utf-8",
    )
    monkeypatch.setattr(command, "TELEGRAM_WHALE_BRIDGE_FILE", bridge)

    text = command._telegram_whale_summary()

    assert "近24小时=1条" in text
    assert "净压力=偏空 100.00K" in text


def test_bare_p_routes_to_v6_demo_only(monkeypatch, capsys) -> None:
    status = {"v6": {"positions": []}}
    calls: list[tuple[str, dict, dict]] = []
    monkeypatch.setattr(command.tg, "collect_status", lambda _args: status)

    def fake_response(target: str, received_status: dict, **kwargs) -> str:
        calls.append((target, received_status, kwargs))
        return "demo positions"

    monkeypatch.setattr(command.tg, "build_position_response", fake_response)

    assert command._positions(SimpleNamespace(), []) == 0
    assert capsys.readouterr().out == "demo positions\n"
    assert calls == [("v6", status, {"v6_title": "[Somnus持仓] V6 Demo"})]


def test_p_momo_routes_to_v6_live_only(monkeypatch, capsys) -> None:
    config = object()
    status = {"v6": {"positions": [{"symbol": "BTCUSDT"}]}}
    calls: list[tuple[str, dict, dict]] = []
    monkeypatch.setattr(command, "_live_v6_config", lambda: (config, "momo"))
    monkeypatch.setattr(command, "_v6_status_from_files", lambda *_args: status)
    monkeypatch.setattr(command, "_live_v6_balance_text", lambda _config: "💰 当前余额=998.12U")

    def fake_response(target: str, received_status: dict, **kwargs) -> str:
        calls.append((target, received_status, kwargs))
        return "live positions"

    monkeypatch.setattr(command.tg, "build_position_response", fake_response)

    assert command._positions(SimpleNamespace(), ["momo"]) == 0
    assert capsys.readouterr().out == "live positions\n💰 当前余额=998.12U\n"
    assert calls == [
        (
            "v6",
            status,
            {
                "v6_config": config,
                "v6_title": "[Somnus持仓] V6-momo",
                "v6_exchange_source": "binance_live_positionRisk",
            },
        )
    ]


def test_p_momo2_uses_account_specific_v6_label(monkeypatch, capsys) -> None:
    config = object()
    status = {"v6": {"positions": []}}
    calls: list[dict] = []

    def fake_config(account: str = "momo"):
        assert account == "momo2"
        return config, "momo2"

    monkeypatch.setattr(command, "_live_v6_config", fake_config)
    monkeypatch.setattr(command, "_v6_status_from_files", lambda *_args: status)
    monkeypatch.setattr(command, "_live_v6_balance_text", lambda _config: "💰 当前余额=990.00U")

    def fake_response(_target: str, _status: dict, **kwargs) -> str:
        calls.append(kwargs)
        return "momo2 positions"

    monkeypatch.setattr(command.tg, "build_position_response", fake_response)

    assert command._positions(SimpleNamespace(), ["momo2"]) == 0
    assert capsys.readouterr().out == "momo2 positions\n💰 当前余额=990.00U\n"
    assert calls == [{
        "v6_config": config,
        "v6_title": "[Somnus持仓] V6-momo2",
        "v6_exchange_source": "binance_live_positionRisk",
    }]


def test_live_v6_balance_uses_total_margin_balance(monkeypatch) -> None:
    closed = False

    class FakeClient:
        def __init__(self, config) -> None:
            assert config == "live-config"

        @staticmethod
        def signed_params() -> dict:
            return {"timestamp": 1}

        @staticmethod
        def signed_request(method: str, endpoint: str, params: dict) -> dict:
            assert (method, endpoint) == ("GET", "/fapi/v2/account")
            assert params == {"timestamp": 1}
            return {
                "totalMarginBalance": "993.74044760",
                "totalWalletBalance": "990.00000000",
            }

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(command.tg, "V6BinanceClient", FakeClient)

    assert command._live_v6_balance_text("live-config") == "💰 当前余额=993.74U"
    assert closed is True


def test_p_momo_keeps_positions_when_balance_read_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(command, "_live_v6_config", lambda: (object(), "momo"))
    monkeypatch.setattr(command, "_v6_status_from_files", lambda *_args: {"v6": {"positions": []}})
    monkeypatch.setattr(command.tg, "build_position_response", lambda *_args, **_kwargs: "live positions")

    def fail_balance(_config) -> str:
        raise RuntimeError("network down")

    monkeypatch.setattr(command, "_live_v6_balance_text", fail_balance)

    assert command._positions(SimpleNamespace(), ["momo"]) == 0
    assert capsys.readouterr().out == "live positions\n💰 当前余额=读取失败\n"


def test_position_response_uses_explicit_v6_exchange_config(monkeypatch) -> None:
    config = object()
    captured: dict = {}

    def fake_fetch(local_positions, *, config=None, source=""):
        captured["local_positions"] = local_positions
        captured["config"] = config
        captured["source"] = source
        return [], source

    monkeypatch.setattr(tg, "fetch_v6_exchange_positions", fake_fetch)

    text = tg.build_position_response(
        "v6",
        {"v6": {"positions": []}},
        v6_config=config,
        v6_title="[Somnus持仓] V6 LIVE · momo",
        v6_exchange_source="binance_live_positionRisk",
    )

    assert text == "[Somnus持仓] V6 LIVE · momo\n当前无持仓"
    assert captured == {
        "local_positions": [],
        "config": config,
        "source": "binance_live_positionRisk",
    }


def test_position_parser_accepts_momo_alias() -> None:
    parser = command.build_parser()
    assert parser.parse_args(["p"]).targets == []
    assert parser.parse_args(["p", "momo"]).targets == ["momo"]


def test_p_momo_24h_routes_to_live_history_window() -> None:
    assert command._live_history_hours(["momo", "24h"]) == 24
    assert command._live_history_hours(["momo", "history"]) == 24
    assert command._live_history_hours(["v6"]) is None


def test_live_history_report_uses_exchange_fills_and_income() -> None:
    trades = [
        {
            "symbol": "HYPEUSDT",
            "positionSide": "SHORT",
            "side": "SELL",
            "price": "58.600",
            "qty": "2.6",
            "realizedPnl": "0",
            "commission": "0.07618",
            "commissionAsset": "USDT",
            "orderId": 100,
            "id": 1,
            "time": 1,
        },
        {
            "symbol": "HYPEUSDT",
            "positionSide": "SHORT",
            "side": "BUY",
            "price": "58.700",
            "qty": "1.0",
            "realizedPnl": "-0.100",
            "commission": "0.02935",
            "commissionAsset": "USDT",
            "orderId": 200,
            "id": 2,
            "time": 2,
        },
        {
            "symbol": "HYPEUSDT",
            "positionSide": "SHORT",
            "side": "BUY",
            "price": "58.700",
            "qty": "1.6",
            "realizedPnl": "-0.160",
            "commission": "0.04696",
            "commissionAsset": "USDT",
            "orderId": 200,
            "id": 3,
            "time": 2,
        },
    ]
    income = [
        {"symbol": "HYPEUSDT", "asset": "USDT", "incomeType": "REALIZED_PNL", "income": "-0.260"},
        {"symbol": "HYPEUSDT", "asset": "USDT", "incomeType": "COMMISSION", "income": "-0.15249"},
    ]

    report = command._format_live_history_report(
        trades,
        income,
        hours=24,
        label="momo",
        generated_at=datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
    )

    assert "[Somnus历史盈亏] V6 LIVE · momo" in report
    assert "HYPEUSDT" in report
    assert "开仓均价=58.600" in report
    assert "平仓均价=58.700" in report
    assert "平仓订单=1" in report
    assert "已实现=-0.260U" in report
    assert "手续费=0.152U" in report
    assert "币种总盈利=-0.412U" in report


def test_chart_parser_accepts_momo_prefix(monkeypatch) -> None:
    monkeypatch.setattr(command, "_live_v6_label", lambda: "momo")
    parser = command.build_parser()
    parsed = parser.parse_args(["fc", "momo", "HYPE", "15m", "bb672", "--json"])

    assert parsed.tokens == ["momo", "HYPE", "15m", "bb672"]
    assert parsed.json is True
    assert command._parse_chart_tokens(parsed.tokens) == ("HYPE", "15m", "bb672", "live", "momo")


def test_chart_parser_keeps_legacy_demo_syntax() -> None:
    assert command._parse_chart_tokens(["BTC", "4h"]) == ("BTC", "4h", None, "default", "")


def test_chart_momo_forwards_live_scope(monkeypatch, capsys, tmp_path) -> None:
    image = tmp_path / "chart.png"
    image.write_bytes(b"png")
    captured: dict = {}

    def fake_build(symbol, interval, focus, **kwargs):
        captured.update({"symbol": symbol, "interval": interval, "focus": focus, **kwargs})
        return {
            "output": str(image),
            "symbol": symbol,
            "interval": interval,
            "focus_period": focus,
            "execution_scope": kwargs["execution_scope"],
            "account_label": kwargs["account_label"],
        }

    monkeypatch.setattr(command.tg, "build_chart_image", fake_build)
    monkeypatch.setattr(command.tg, "format_chart_caption", lambda payload: f"scope={payload['execution_scope']}")

    assert (
        command._chart("HYPE", "15m", "bb672", as_json=True, execution_scope="live", account_label="momo")
        == 0
    )
    output = capsys.readouterr().out

    assert '"execution_scope": "live"' in output
    assert '"account_label": "momo"' in output
    assert captured == {
        "symbol": "HYPEUSDT",
        "interval": "15m",
        "focus": 672,
        "execution_scope": "live",
        "account_label": "momo",
    }


def test_empty_live_state_does_not_fall_back_to_stale_summary(tmp_path) -> None:
    state_file = tmp_path / "state.json"
    summary_file = tmp_path / "latest_run_summary.json"
    state_file.write_text('{"positions": []}', encoding="utf-8")
    summary_file.write_text('{"positions": [{"symbol": "STALEUSDT"}]}', encoding="utf-8")

    assert command._v6_status_from_files(state_file, summary_file) == {"v6": {"positions": []}}
