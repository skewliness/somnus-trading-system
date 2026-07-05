from unittest.mock import patch

import os
import pytest

from trading_system import cli


def test_build_parser_defaults_to_live_mode() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["scan"])

    assert args.mode == "live"


def test_build_parser_uses_repo_local_square_curl_file_default() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["collect-square-feed"])

    assert args.square_curl_file == "data/trading_system/binance_square_request.txt"


def test_build_parser_allows_env_override_for_square_curl_file() -> None:
    with patch.dict(os.environ, {"TS_SQUARE_CURL_FILE": "/tmp/custom-square.txt"}, clear=False):
        parser = cli.build_parser()
        args = parser.parse_args(["collect-square-feed"])

    assert args.square_curl_file == "/tmp/custom-square.txt"


def test_rebalance_rejects_replay_mode() -> None:
    with patch("sys.argv", ["trading-system", "rebalance", "--mode", "replay"]):
        with pytest.raises(SystemExit, match="requires --mode live"):
            cli.main()
