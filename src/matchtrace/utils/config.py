from __future__ import annotations

import os
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    source_root = repository_root()
    if config_path is not None:
        path = Path(config_path).resolve()
        root = path.parent.parent if path.parent.name == "configs" else path.parent
    else:
        repository_config = source_root / "configs" / "default.toml"
        if repository_config.is_file():
            path = repository_config
            root = source_root
        else:
            path = Path(str(files("matchtrace").joinpath("default.toml")))
            root = Path.cwd().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("rb") as handle:
        settings = tomllib.load(handle)

    required_sections = {
        "project",
        "paths",
        "synthetic",
        "model",
        "detector",
        "tracker",
        "motion",
        "pitch",
        "analytics",
        "render",
        "api",
        "audit",
    }
    missing = sorted(required_sections.difference(settings))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    settings["_root"] = root
    model_override = os.getenv("MATCHTRACE_MODEL_PATH")
    if model_override:
        settings["detector"]["model_path"] = model_override
    return settings


def resolve_path(settings: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(settings["_root"]) / path
    return path.resolve()


def configured_path(settings: dict[str, Any], key: str) -> Path:
    try:
        value = settings["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Unknown configured path: {key}") from exc
    return resolve_path(settings, value)
