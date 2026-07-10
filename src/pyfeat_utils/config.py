from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

SUPPORTED_PROCESS_TYPES = {"image", "video"}


class ConfigError(ValueError):
    """Raised when a pyfeat-utils config cannot be used."""


@dataclass(frozen=True)
class ProcessingConfig:
    data_dir: Path
    process_types: tuple[str, ...] = ("image", "video")
    video_skip_frames: int = 20


def default_config_path() -> Path:
    return Path.home() / "Documents" / "pyfeat-utils" / "config.json"


def _expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _get_data_processing(payload: dict[str, Any]) -> dict[str, Any]:
    data_processing = payload.get("data_processing")
    if not isinstance(data_processing, dict):
        raise ConfigError("Config must contain a 'data_processing' object.")
    return data_processing


def load_config(path: Path) -> ProcessingConfig:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    data_processing = _get_data_processing(payload)

    raw_data_dir = data_processing.get("data_dir") or data_processing.get(
        "data_pyfeat-utils"
    )
    if not isinstance(raw_data_dir, str) or not raw_data_dir.strip():
        raise ConfigError("Config data_processing.data_dir must be a non-empty path.")

    raw_process_types = data_processing.get("process_types") or data_processing.get(
        "process_type", ["image", "video"]
    )
    if not isinstance(raw_process_types, list) or not raw_process_types:
        raise ConfigError("Config process_types must be a non-empty list.")

    process_types = tuple(str(item).lower() for item in raw_process_types)
    unsupported = sorted(set(process_types) - SUPPORTED_PROCESS_TYPES)
    if unsupported:
        raise ConfigError(f"Unsupported process type(s): {', '.join(unsupported)}")

    video_skip_frames = int(data_processing.get("video_skip_frames", 20))
    if video_skip_frames < 1:
        raise ConfigError("video_skip_frames must be at least 1.")

    return ProcessingConfig(
        data_dir=_expand_path(raw_data_dir),
        process_types=process_types,
        video_skip_frames=video_skip_frames,
    )


def write_default_config(
    config_path: Path | None = None, data_dir: Path | None = None
) -> Path:
    config_path = config_path or default_config_path()
    data_dir = data_dir or config_path.parent / "data"

    config_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "data_processing": {
            "data_dir": str(data_dir),
            "process_types": ["image", "video"],
            "video_skip_frames": 20,
        }
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path
