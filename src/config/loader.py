"""Configuration loader with validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Immutable config container with dot-style access."""

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", self._deep_freeze(data))

    @staticmethod
    def _deep_freeze(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: Config._deep_freeze(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return tuple(Config._deep_freeze(x) for x in obj)
        return obj

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name in data:
            v = data[name]
            return Config(v) if isinstance(v, dict) else v
        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        raise RuntimeError("Config is immutable")

    def get(self, key: str, default: Any = None) -> Any:
        """Get key with optional default."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Return raw dict (copy of internal)."""
        return dict(object.__getattribute__(self, "_data"))


def load_config(path: str | Path) -> Config:
    """Load YAML config from path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError("Config file is empty")
    return Config(data)
