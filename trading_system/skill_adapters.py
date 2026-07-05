from __future__ import annotations

import json
from pathlib import Path
import shutil


class SkillBridgeSync:
    def __init__(self, raw_dir: Path, bridge_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.bridge_dir = bridge_dir
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

    def sync(self) -> list[dict]:
        summaries: list[dict] = []
        if not self.raw_dir.exists():
            return summaries
        for path in sorted(self.raw_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            bridge_name = path.stem
            converter = _CONVERTERS.get(bridge_name, _generic_converter)
            output = converter(payload, bridge_name)
            output.setdefault("capability_category", _CAPABILITY_CATEGORIES.get(bridge_name, "sentiment"))
            output.setdefault("parallel_group", _PARALLEL_GROUPS.get(bridge_name, "external"))
            out_path = self.bridge_dir / f"{bridge_name}.json"
            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            summaries.append(
                {
                    "raw_file": str(path),
                    "bridge_file": str(out_path),
                    "bridge_name": output.get("bridge_name", bridge_name),
                    "signal_count": len(output.get("signals", [])),
                    "capability_category": output.get("capability_category", "sentiment"),
                    "parallel_group": output.get("parallel_group", "external"),
                }
            )
        return summaries


def import_skill_output(input_path: Path, bridge_name: str, raw_dir: Path, bridge_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    bridge_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    raw_path = raw_dir / f"{bridge_name}.json"
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = SkillBridgeSync(raw_dir, bridge_dir).sync()
    match = next((item for item in summary if item["bridge_name"] == bridge_name), None)
    return match or {
        "raw_file": str(raw_path),
        "bridge_file": str(bridge_dir / f"{bridge_name}.json"),
        "bridge_name": bridge_name,
        "signal_count": 0,
    }


def import_execution_result(input_path: Path, result_dir: Path, label: str | None = None) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{label}.json" if label else input_path.name
    target_path = result_dir / target_name
    if input_path.resolve() != target_path.resolve():
        shutil.copyfile(input_path, target_path)
    payload = json.loads(target_path.read_text(encoding="utf-8-sig"))
    return {
        "result_file": str(target_path),
        "status": str(payload.get("status") or "unknown"),
        "skill_name": str(payload.get("skill_name") or ""),
        "endpoint": str(payload.get("endpoint") or ""),
    }


def _generic_converter(payload: dict, bridge_name: str) -> dict:
    signals = []
    for item in payload.get("signals", payload.get("data", [])):
        symbol = str(item.get("symbol") or item.get("code") or "").strip().upper()
        if not symbol:
            continue
        signals.append(
            {
                "symbol": symbol,
                "hotness_score": float(item.get("hotness_score") or item.get("score") or 0.0),
                "mention_count": int(float(item.get("mention_count") or item.get("mentions") or 0)),
                "source": bridge_name,
                "notes": str(item.get("notes") or item.get("label") or ""),
            }
        )
    return {"bridge_name": bridge_name, "signals": signals}


def _crypto_market_rank_converter(payload: dict, bridge_name: str) -> dict:
    signals = []
    for item in payload.get("ranks", payload.get("data", [])):
        symbol = str(item.get("symbol") or item.get("code") or "").strip().upper()
        if not symbol:
            continue
        hotness = float(item.get("social_hype_score") or item.get("hotness_score") or item.get("score") or 0.0)
        mentions = int(float(item.get("mention_count") or item.get("discuss_count") or item.get("mentions") or 0))
        signals.append(
            {
                "symbol": symbol,
                "hotness_score": hotness,
                "mention_count": mentions,
                "source": bridge_name,
                "notes": str(item.get("narrative") or item.get("notes") or "market rank bridge"),
            }
        )
    return {"bridge_name": bridge_name, "signals": signals}


def _meme_rush_converter(payload: dict, bridge_name: str) -> dict:
    signals = []
    for item in payload.get("tokens", payload.get("data", [])):
        symbol = str(item.get("symbol") or item.get("code") or "").strip().upper()
        if not symbol:
            continue
        rush_score = float(item.get("rush_score") or item.get("hotness_score") or item.get("score") or 0.0)
        signals.append(
            {
                "symbol": symbol,
                "hotness_score": rush_score,
                "mention_count": int(float(item.get("mention_count") or item.get("mentions") or 0)),
                "source": bridge_name,
                "notes": str(item.get("topic") or item.get("notes") or "meme rush bridge"),
            }
        )
    return {"bridge_name": bridge_name, "signals": signals}


def _trading_signal_converter(payload: dict, bridge_name: str) -> dict:
    signals = []
    for item in payload.get("signals", payload.get("data", [])):
        symbol = str(item.get("symbol") or item.get("code") or "").strip().upper()
        if not symbol:
            continue
        strength = float(item.get("signal_strength") or item.get("score") or item.get("hotness_score") or 0.0)
        signals.append(
            {
                "symbol": symbol,
                "hotness_score": strength,
                "mention_count": int(float(item.get("signal_count") or item.get("mention_count") or 1)),
                "source": bridge_name,
                "notes": str(item.get("signal_type") or item.get("notes") or "trading signal bridge"),
            }
        )
    return {"bridge_name": bridge_name, "signals": signals}


def _okx_wallet_activity_converter(payload: dict, bridge_name: str) -> dict:
    signals = []
    entries = payload.get("signals", payload.get("data", payload.get("items", [])))
    for item in entries:
        symbol = str(item.get("symbol") or item.get("tokenSymbol") or item.get("code") or "").strip().upper()
        if not symbol:
            continue
        score = float(
            item.get("signal_strength")
            or item.get("hotness_score")
            or item.get("score")
            or item.get("amount_score")
            or 0.0
        )
        mentions = int(float(item.get("wallet_count") or item.get("signal_count") or item.get("mention_count") or 1))
        note_parts = [
            str(item.get("wallet_type") or item.get("tracker_type") or "").strip(),
            str(item.get("chain") or item.get("chainName") or "").strip(),
            str(item.get("notes") or item.get("label") or item.get("signal_type") or "").strip(),
        ]
        signals.append(
            {
                "symbol": symbol,
                "hotness_score": score,
                "mention_count": mentions,
                "source": bridge_name,
                "notes": " | ".join(part for part in note_parts if part),
            }
        )
    return {"bridge_name": bridge_name, "signals": signals}


_CAPABILITY_CATEGORIES = {
    "crypto-market-rank": "market_data",
    "meme-rush": "market_data",
    "trading-signal": "signal",
    "lanaaielsa-distill": "sentiment",
    "lanaai-square-distill": "sentiment",
    "okx-dex-market": "market_data",
    "okx-dex-token": "market_data",
    "okx-dex-signal": "signal",
    "okx-dex-ws": "ws_realtime",
    "okx-security": "security",
    "okx-defi-invest": "defi",
    "okx-defi-portfolio": "defi",
    "okx-wallet-portfolio": "portfolio",
    "okx-onchain-gateway": "execution_gateway",
    "okx-dex-trenches": "market_data",
}


_PARALLEL_GROUPS = {
    "crypto-market-rank": "binance",
    "meme-rush": "binance",
    "trading-signal": "binance",
    "lanaaielsa-distill": "binance",
    "lanaai-square-distill": "binance",
    "okx-dex-market": "okx",
    "okx-dex-token": "okx",
    "okx-dex-signal": "okx",
    "okx-dex-ws": "okx",
    "okx-security": "okx",
    "okx-defi-invest": "okx",
    "okx-defi-portfolio": "okx",
    "okx-wallet-portfolio": "okx",
    "okx-onchain-gateway": "okx",
    "okx-dex-trenches": "okx",
}


_CONVERTERS = {
    "crypto-market-rank": _crypto_market_rank_converter,
    "meme-rush": _meme_rush_converter,
    "trading-signal": _trading_signal_converter,
    "okx-dex-market": _generic_converter,
    "okx-dex-token": _generic_converter,
    "okx-dex-signal": _okx_wallet_activity_converter,
    "okx-dex-ws": _okx_wallet_activity_converter,
    "okx-security": _generic_converter,
    "okx-defi-invest": _generic_converter,
    "okx-defi-portfolio": _generic_converter,
    "okx-wallet-portfolio": _generic_converter,
    "okx-onchain-gateway": _generic_converter,
    "okx-dex-trenches": _generic_converter,
}
