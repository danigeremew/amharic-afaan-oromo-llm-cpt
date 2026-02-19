from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from config_schema import TrainConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping (top-level object). Got: {type(raw)}")
    return raw


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object (top-level mapping). Got: {type(raw)}")
    return raw


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = _load_yaml(config_path)
    elif suffix == ".json":
        raw = _load_json(config_path)
    else:
        raise ValueError("Config must be .yaml/.yml or .json")

    return TrainConfig.model_validate(raw)

