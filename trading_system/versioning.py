from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SENSITIVE_KEY_PARTS = (
    "api_key",
    "api_secret",
    "secret_key",
    "secret",
    "token",
    "cookie",
    "authorization",
    "password",
    "signature",
    "confirmation",
)
NON_PORTABLE_FIELD_SUFFIXES = ("_file", "_dir", "_path", "_base_url", "_proxy_url")
NON_PORTABLE_FIELDS = {
    "instance_name",
    "binance_proxy_url",
    "public_base_url",
    "demo_base_url",
    "live_base_url",
}


def _is_excluded_field(name: str) -> bool:
    lowered = str(name or "").lower()
    return (
        lowered in NON_PORTABLE_FIELDS
        or any(part in lowered for part in SENSITIVE_KEY_PARTS)
        or lowered.endswith(NON_PORTABLE_FIELD_SUFFIXES)
    )


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not _is_excluded_field(str(key))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonical_value(item) for item in value]
    return str(value)


def config_fingerprint_payload(config: Any) -> dict[str, Any]:
    """Return a portable, credential-free projection of runtime configuration."""

    if is_dataclass(config):
        names = [item.name for item in fields(config)]
    else:
        names = sorted(name for name in vars(config) if not name.startswith("_"))
    return {
        name: _canonical_value(getattr(config, name))
        for name in names
        if hasattr(config, name) and not _is_excluded_field(name)
    }


def config_fingerprint(config: Any) -> str:
    encoded = json.dumps(
        config_fingerprint_payload(config),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()[:16]}"


def model_artifact_version(model_path: Path | str, explicit_version: str = "") -> str:
    explicit = str(explicit_version or "").strip()
    if explicit:
        return explicit[:160]
    path = Path(model_path)
    if not path.is_file():
        return f"unavailable:{path.name or 'model'}"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return f"unreadable:{path.name or 'model'}"
    return f"sha256:{digest.hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class RuntimeVersionInfo:
    strategy_version: str
    config_version: str
    model_version: str
    config_hash: str

    @classmethod
    def from_config(cls, config: Any) -> "RuntimeVersionInfo":
        strategy_version = str(
            getattr(config, "strategy_version", "")
            or getattr(config, "trade_event_strategy_version", "")
            or "6.0-baseline"
        ).strip()
        config_version = str(getattr(config, "config_version", "") or "v6.5").strip()
        model_version = model_artifact_version(
            getattr(config, "model_path", Path("model.pkl")),
            getattr(config, "model_version", ""),
        )
        return cls(
            strategy_version=strategy_version[:160],
            config_version=config_version[:160],
            model_version=model_version,
            config_hash=config_fingerprint(config),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "strategy_version": self.strategy_version,
            "config_version": self.config_version,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
        }
