from scripts.xuejian_proactive_push import event_theme_fingerprint, format_geo_push_batch, geo_news_to_chinese
from scripts import xuejian_proactive_push


def test_format_geo_push_wraps_chinese_news_and_removes_raw_notice() -> None:
    long_note = (
        "info_stream/news: 【金十整理：黄金跌破4000美元的六大因素】"
        + "美债收益率上升削弱黄金吸引力，美元走强对黄金施加压力。" * 10
        + ", signal=event, source=jin10"
    )

    text = format_geo_push_batch(
        [
            {
                "symbol": "SOLUSDT",
                "hotness_score": 62.0,
                "notes": long_note,
                "metadata": {"matched_source": "jin10"},
            }
        ]
    )

    lines = text.splitlines()
    assert "ℹ️ 说明" not in text
    assert "不是自动下单信号" not in text
    assert "原文" not in text
    assert "signal=event" not in text
    assert "source=jin10" not in text
    assert "📰 中文快讯：" in lines
    assert "判断：" in lines
    assert "来源=金十" in text
    assert max(len(line) for line in lines) <= 64


def test_format_geo_push_translates_english_news_without_showing_original() -> None:
    english_note = (
        "info_stream/news: Iran’s FM Deputy Says Any Framework for Passage "
        "Through Strait Must Be in Coordination with Iran or Outcome Will Be "
        "Suspension of Designated Route, signal=event, source=reuters"
    )

    text = format_geo_push_batch(
        [
            {
                "symbol": "BTCUSDT",
                "hotness_score": 62.0,
                "notes": english_note,
                "metadata": {"matched_source": "reuters"},
            }
        ]
    )

    assert "📰 中文快讯：" in text
    assert "来源=路透" in text
    assert "伊朗副外长表示" in text
    assert "指定航线可能暂停" in text
    assert "Iran" not in text
    assert "Framework" not in text
    assert "Strait" not in text
    assert "signal=event" not in text
    assert "source=reuters" not in text
    assert "原文" not in text


def test_geo_news_to_chinese_fallback_never_returns_raw_english() -> None:
    translated = geo_news_to_chinese("Oil conflict risk rises after missile attack")

    assert translated.startswith("外媒消息指向")
    assert "能源与油价" in translated
    assert "军事冲突风险" in translated
    assert "Oil" not in translated
    assert "missile" not in translated


def test_event_theme_fingerprint_dedupes_same_geo_theme_across_symbols() -> None:
    btc_item = {
        "symbol": "BTCUSDT",
        "notes": "info_stream/news: Reuters: Iran says Strait of Hormuz passage risk may affect oil and energy markets, signal=event",
        "metadata": {"matched_source": "reuters", "text_original": "Iran Hormuz oil energy safe passage"},
    }
    fuel_item = {
        "symbol": "FUELUSDT",
        "notes": "info_stream/news: 6551news: Iran/Hormuz news points to oil and energy market risk",
        "metadata": {"matched_source": "6551news", "text_original": "Iran Hormuz crude energy route"},
    }

    assert event_theme_fingerprint(btc_item) == event_theme_fingerprint(fuel_item)


def test_binance_session_alert_is_sent_once_and_resets_after_recovery(tmp_path, monkeypatch) -> None:
    alert_file = tmp_path / "binance_session_alert.txt"
    alert_file.write_text(
        "\n".join(
            [
                "time=2026-07-19T00:00:00Z",
                "mode=Binance API",
                "detail=traders=48, ok_private=0, auth_required=48, error_private=0, new_operations=0, new_history=0",
            ]
        ),
        encoding="utf-8",
    )
    sent: list[str] = []
    monkeypatch.setattr(xuejian_proactive_push, "BINANCE_SESSION_ALERT_FILE", alert_file)
    monkeypatch.setattr(xuejian_proactive_push, "send_telegram", lambda text, dry_run=False: sent.append(text))
    monkeypatch.setattr(xuejian_proactive_push, "log", lambda _message: None)
    state: dict[str, object] = {}

    assert xuejian_proactive_push.maybe_send_binance_session_alert(state) is True
    assert len(sent) == 1
    assert "认证失效=48" in sent[0]
    assert xuejian_proactive_push.maybe_send_binance_session_alert(state) is False
    assert len(sent) == 1

    alert_file.unlink()
    assert xuejian_proactive_push.maybe_send_binance_session_alert(state) is False
    assert state["binance_session_alert_active"] is False
    assert state["last_binance_session_alert_fingerprint"] == ""

    alert_file.write_text(
        "time=2026-07-20T00:00:00Z\nmode=Binance API\ndetail=traders=48, auth_required=48, new_operations=0, new_history=0",
        encoding="utf-8",
    )
    assert xuejian_proactive_push.maybe_send_binance_session_alert(state) is True
    assert len(sent) == 2
