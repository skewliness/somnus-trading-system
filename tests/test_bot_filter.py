from trading_system.bot_filter import decide_square_post, is_valid_tradable_symbol


def test_valid_symbol_filter() -> None:
    assert is_valid_tradable_symbol("BTCUSDT")
    assert not is_valid_tradable_symbol("币安人生USDT")


def test_bot_filter_drops_low_quality_post() -> None:
    decision = decide_square_post(
        {
            "username": "Square-Creator-abc",
            "display_name": "Square-Creator-abc",
            "body_text": "",
            "view_count": 20000,
            "trading_pairs": ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT"],
            "bot_features": {"engagement_view_ratio": 0.0001, "engagement_total": 1},
        },
        author_post_count=4,
    )
    assert not decision.keep
    assert decision.suspicion_score >= 3.5


def test_bot_filter_drops_repeated_template_posts() -> None:
    decision = decide_square_post(
        {
            "username": "templater",
            "display_name": "templater",
            "body_text": "BUY NOW GREAT SIGNAL",
            "view_count": 500,
            "trading_pairs": ["BTCUSDT"],
            "bot_features": {
                "engagement_view_ratio": 0.002,
                "engagement_total": 1,
                "repeated_text_count": 3,
                "author_unique_text_ratio": 0.25,
            },
        },
        author_post_count=4,
    )
    assert not decision.keep
    assert "duplicate_body_pattern" in decision.reasons
