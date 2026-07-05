import json
from pathlib import Path

from trading_system.skill_adapters import SkillBridgeSync, import_execution_result, import_skill_output


def test_skill_bridge_sync_converts_raw_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    bridge_dir = tmp_path / "bridge"
    raw_dir.mkdir()
    (raw_dir / "crypto-market-rank.json").write_text(
        json.dumps({"ranks": [{"symbol": "AIXBTUSDT", "social_hype_score": 90, "discuss_count": 11}]}),
        encoding="utf-8",
    )
    summary = SkillBridgeSync(raw_dir, bridge_dir).sync()
    assert summary[0]["signal_count"] == 1
    payload = json.loads((bridge_dir / "crypto-market-rank.json").read_text(encoding="utf-8"))
    assert payload["signals"][0]["symbol"] == "AIXBTUSDT"
    assert payload["capability_category"] == "market_data"
    assert payload["parallel_group"] == "binance"


def test_skill_bridge_sync_converts_okx_signal_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    bridge_dir = tmp_path / "bridge"
    raw_dir.mkdir()
    (raw_dir / "okx-dex-signal.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "tokenSymbol": "RAVEUSDT",
                        "signal_strength": 87,
                        "wallet_count": 4,
                        "wallet_type": "smart_money",
                        "chain": "solana",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = SkillBridgeSync(raw_dir, bridge_dir).sync()
    assert summary[0]["bridge_name"] == "okx-dex-signal"
    payload = json.loads((bridge_dir / "okx-dex-signal.json").read_text(encoding="utf-8"))
    assert payload["signals"][0]["symbol"] == "RAVEUSDT"
    assert payload["signals"][0]["mention_count"] == 4
    assert payload["capability_category"] == "signal"
    assert payload["parallel_group"] == "okx"


def test_import_skill_output_writes_raw_and_bridge(tmp_path: Path) -> None:
    input_file = tmp_path / "market_rank.json"
    input_file.write_text(
        json.dumps({"ranks": [{"symbol": "TSTUSDT", "social_hype_score": 88, "discuss_count": 19}]}),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    bridge_dir = tmp_path / "bridge"
    summary = import_skill_output(input_file, "crypto-market-rank", raw_dir, bridge_dir)
    assert summary["bridge_name"] == "crypto-market-rank"
    assert (raw_dir / "crypto-market-rank.json").exists()
    payload = json.loads((bridge_dir / "crypto-market-rank.json").read_text(encoding="utf-8"))
    assert payload["signals"][0]["symbol"] == "TSTUSDT"


def test_import_execution_result_copies_file(tmp_path: Path) -> None:
    input_file = tmp_path / "result.json"
    input_file.write_text(
        json.dumps({"skill_name": "derivatives-trading-usds-futures", "status": "submitted"}),
        encoding="utf-8",
    )
    result_dir = tmp_path / "results"
    summary = import_execution_result(input_file, result_dir, label="latest_execution")
    assert summary["status"] == "submitted"
    assert (result_dir / "latest_execution.json").exists()


def test_import_execution_result_allows_same_source_and_target(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    input_file = result_dir / "smoke_test_result.json"
    input_file.write_text(
        json.dumps({"skill_name": "derivatives-trading-usds-futures", "status": "submitted"}),
        encoding="utf-8",
    )

    summary = import_execution_result(input_file, result_dir, label="smoke_test_result")

    assert summary["status"] == "submitted"
    assert summary["result_file"] == str(input_file)
