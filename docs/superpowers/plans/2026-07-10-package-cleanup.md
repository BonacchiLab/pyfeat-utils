# pyfeat-utils Package Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `pyfeat-utils` into a clean Python package with modern py-feat-compatible dependency metadata, testable internals, and real CLI entry points.

**Architecture:** Keep py-feat processing behavior but move script logic into focused modules. The CLI will be thin and call config, file discovery, processing, and statistics functions. Heavy py-feat imports will be lazy so tests and `--help` do not download or initialize models.

**Tech Stack:** Python 3.11+, uv, hatchling, pytest, ruff, pandas, matplotlib/seaborn for optional plotting, py-feat for runtime detection.

## Global Constraints

- Use `pyproject.toml` as the primary source of truth for package metadata and dependencies.
- Use `requires-python = ">=3.11"` to match py-feat's documented supported range.
- Keep `.python-version` at `3.13` for local development.
- Keep `py-feat` as a normal runtime dependency.
- Remove `pip` from runtime dependencies unless there is a concrete runtime reason.
- Keep development tools in a dev dependency group: `pytest`, `ruff`, and `ipykernel`.
- CLI help must not import or initialize py-feat models.
- Ordinary unit tests must not require py-feat model downloads.
- Preserve workflow-level behavior: process media files, produce CSVs, and compute descriptive summaries.

---

## File Structure

- Modify `pyproject.toml`: project metadata, dependencies, CLI entry point, ruff/pytest settings.
- Modify `.python-version`: keep local interpreter version at `3.13`.
- Create `src/pyfeat_utils/config.py`: config defaults, loading, path expansion, validation, and init-file writing.
- Create `src/pyfeat_utils/files.py`: media/statistics file discovery and deterministic output paths.
- Create `src/pyfeat_utils/processing.py`: lazy detector creation, image/video processing, CSV writing.
- Create `src/pyfeat_utils/statistics.py`: load output tables and compute emotion/AU summaries.
- Create `src/pyfeat_utils/cli.py`: argparse-based command entry points.
- Replace `src/pyfeat_utils/pyfeat_processor.py`: compatibility wrapper that delegates to `cli`.
- Replace `src/pyfeat_utils/descriptive_statistics.py`: compatibility wrapper that delegates to `cli`.
- Modify `src/pyfeat_utils/__init__.py`: expose package version only.
- Remove or ignore `src/pyfeat_utils/init.py`: superseded by `pyfeat-utils init`.
- Modify `README.md`: update installation and usage.
- Create tests under `tests/`: config, files, statistics, processing, and CLI smoke tests.

---

### Task 1: Package Metadata And CLI Skeleton

**Files:**
- Modify: `pyproject.toml`
- Modify: `.python-version`
- Modify: `src/pyfeat_utils/__init__.py`
- Create: `src/pyfeat_utils/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `pyfeat_utils.cli.build_parser() -> argparse.ArgumentParser`
- Produces: `pyfeat_utils.cli.main(argv: list[str] | None = None) -> int`
- Produces: console script `pyfeat-utils = pyfeat_utils.cli:main`

- [ ] **Step 1: Write failing CLI smoke tests**

Create `tests/test_cli.py`:

```python
from pyfeat_utils.cli import main


def test_cli_help_does_not_import_pyfeat(capsys):
    result = main(["--help"])

    captured = capsys.readouterr()
    assert result == 0
    assert "pyfeat-utils" in captured.out
    assert "process" in captured.out
    assert "stats" in captured.out


def test_process_requires_existing_config(tmp_path, capsys):
    missing_config = tmp_path / "missing.json"

    result = main(["process", "--config", str(missing_config)])

    captured = capsys.readouterr()
    assert result == 2
    assert "Config file not found" in captured.err
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run --with pytest pytest tests/test_cli.py -v`

Expected: FAIL because `pyfeat_utils.cli` does not exist.

- [ ] **Step 3: Update package metadata and CLI skeleton**

Set `.python-version` to:

```text
3.13
```

Update `pyproject.toml`:

```toml
[project]
name = "pyfeat-utils"
version = "0.1.0"
description = "Utilities for batch py-feat processing and descriptive analysis"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "py-feat",
]

[project.scripts]
pyfeat-utils = "pyfeat_utils.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "ipykernel",
    "pytest>=8.3.5",
    "ruff>=0.11.4",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py311"
```

Replace `src/pyfeat_utils/__init__.py`:

```python
"""Utilities for batch py-feat processing and descriptive analysis."""

__version__ = "0.1.0"
```

Create `src/pyfeat_utils/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyfeat-utils",
        description="Batch py-feat processing and descriptive analysis utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a starter config.")
    init_parser.add_argument("--config", type=Path, default=None)
    init_parser.add_argument("--data-dir", type=Path, default=None)

    process_parser = subparsers.add_parser("process", help="Process images/videos.")
    process_parser.add_argument("--config", type=Path, required=True)
    process_parser.add_argument("--visualize", action="store_true")

    stats_parser = subparsers.add_parser("stats", help="Compute descriptive stats.")
    stats_parser.add_argument("--config", type=Path, required=True)
    stats_parser.add_argument("--plots", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"process", "stats"} and not args.config.exists():
        parser.exit(2, f"Config file not found: {args.config}\n")

    if args.command == "init":
        return 0

    return 0
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run --with pytest pytest tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add .python-version pyproject.toml src/pyfeat_utils/__init__.py src/pyfeat_utils/cli.py tests/test_cli.py
git commit -m "feat: add package cli skeleton"
```

---

### Task 2: Config And File Discovery

**Files:**
- Create: `src/pyfeat_utils/config.py`
- Create: `src/pyfeat_utils/files.py`
- Modify: `src/pyfeat_utils/cli.py`
- Test: `tests/test_config.py`
- Test: `tests/test_files.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `pyfeat_utils.cli.main(argv: list[str] | None = None) -> int`
- Produces: `ProcessingConfig(data_dir: Path, process_types: tuple[str, ...], video_skip_frames: int)`
- Produces: `default_config_path() -> Path`
- Produces: `load_config(path: Path) -> ProcessingConfig`
- Produces: `write_default_config(config_path: Path | None = None, data_dir: Path | None = None) -> Path`
- Produces: `discover_media_files(data_dir: Path, process_types: tuple[str, ...]) -> list[Path]`
- Produces: `discover_table_files(data_dir: Path) -> list[Path]`
- Produces: `prediction_output_path(input_path: Path, output_dir: Path | None = None) -> Path`

- [ ] **Step 1: Write failing config and file tests**

Create `tests/test_config.py`:

```python
import json

import pytest

from pyfeat_utils.config import ConfigError, load_config, write_default_config


def test_load_config_expands_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_processing": {
                    "data_dir": str(data_dir),
                    "process_types": ["image"],
                    "video_skip_frames": 7,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data_dir == data_dir
    assert config.process_types == ("image",)
    assert config.video_skip_frames == 7


def test_load_config_rejects_unknown_process_type(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_processing": {
                    "data_dir": str(data_dir),
                    "process_types": ["audio"],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Unsupported process type"):
        load_config(config_path)


def test_write_default_config_creates_config_and_data_dir(tmp_path):
    config_path = tmp_path / "config.json"
    data_dir = tmp_path / "data"

    result = write_default_config(config_path=config_path, data_dir=data_dir)

    assert result == config_path
    assert config_path.exists()
    assert data_dir.exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["data_processing"]["data_dir"] == str(data_dir)
```

Create `tests/test_files.py`:

```python
from pyfeat_utils.files import (
    discover_media_files,
    discover_table_files,
    prediction_output_path,
)


def test_discover_media_files_filters_by_process_type(tmp_path):
    image = tmp_path / "face.png"
    video = tmp_path / "clip.mp4"
    table = tmp_path / "face.png.csv"
    image.write_text("image", encoding="utf-8")
    video.write_text("video", encoding="utf-8")
    table.write_text("csv", encoding="utf-8")

    files = discover_media_files(tmp_path, ("image",))

    assert files == [image]


def test_discover_table_files_skips_prediction_csvs_for_images(tmp_path):
    video_csv = tmp_path / "clip.mp4.csv"
    image_csv = tmp_path / "face.png.csv"
    summary = tmp_path / "manual.csv"
    video_csv.write_text("frame,happiness\n1,0.9\n", encoding="utf-8")
    image_csv.write_text("happiness\n0.8\n", encoding="utf-8")
    summary.write_text("happiness\n0.7\n", encoding="utf-8")

    files = discover_table_files(tmp_path)

    assert files == [video_csv, summary]


def test_prediction_output_path_keeps_original_suffix_in_name(tmp_path):
    input_path = tmp_path / "clip.mp4"

    output_path = prediction_output_path(input_path)

    assert output_path == tmp_path / "clip.mp4.csv"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run --with pytest pytest tests/test_config.py tests/test_files.py -v`

Expected: FAIL because `config.py` and `files.py` do not exist.

- [ ] **Step 3: Implement config and file modules**

Create `src/pyfeat_utils/config.py`:

```python
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
```

Create `src/pyfeat_utils/files.py`:

```python
from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
TABLE_EXTENSIONS = {".csv", ".txt"}


def discover_media_files(data_dir: Path, process_types: tuple[str, ...]) -> list[Path]:
    suffixes: set[str] = set()
    if "image" in process_types:
        suffixes.update(IMAGE_EXTENSIONS)
    if "video" in process_types:
        suffixes.update(VIDEO_EXTENSIONS)

    files = [
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in suffixes
        and not path.name.lower().endswith(".csv")
    ]
    return sorted(files)


def discover_table_files(data_dir: Path) -> list[Path]:
    files = []
    image_prediction_markers = tuple(f"{ext}.csv" for ext in IMAGE_EXTENSIONS)
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TABLE_EXTENSIONS:
            continue
        if path.name.lower().endswith(image_prediction_markers):
            continue
        files.append(path)
    return files


def prediction_output_path(input_path: Path, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or input_path.parent
    return output_dir / f"{input_path.name}.csv"
```

- [ ] **Step 4: Wire `init` and config errors into CLI**

Update `src/pyfeat_utils/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from pyfeat_utils.config import ConfigError, load_config, write_default_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyfeat-utils",
        description="Batch py-feat processing and descriptive analysis utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a starter config.")
    init_parser.add_argument("--config", type=Path, default=None)
    init_parser.add_argument("--data-dir", type=Path, default=None)

    process_parser = subparsers.add_parser("process", help="Process images/videos.")
    process_parser.add_argument("--config", type=Path, required=True)
    process_parser.add_argument("--visualize", action="store_true")

    stats_parser = subparsers.add_parser("stats", help="Compute descriptive stats.")
    stats_parser.add_argument("--config", type=Path, required=True)
    stats_parser.add_argument("--plots", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            config_path = write_default_config(args.config, args.data_dir)
            print(f"Created config: {config_path}")
            return 0

        config = load_config(args.config)
    except ConfigError as exc:
        parser.exit(2, f"{exc}\n")

    if args.command in {"process", "stats"}:
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run --with pytest pytest tests/test_cli.py tests/test_config.py tests/test_files.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/pyfeat_utils/config.py src/pyfeat_utils/files.py src/pyfeat_utils/cli.py tests/test_config.py tests/test_files.py tests/test_cli.py
git commit -m "feat: add config and file discovery"
```

---

### Task 3: Descriptive Statistics Library

**Files:**
- Create: `src/pyfeat_utils/statistics.py`
- Modify: `src/pyfeat_utils/cli.py`
- Replace: `src/pyfeat_utils/descriptive_statistics.py`
- Test: `tests/test_statistics.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ProcessingConfig`
- Consumes: `discover_table_files(data_dir: Path) -> list[Path]`
- Produces: `FileStatistics(source: Path, row_count: int, emotion_columns: tuple[str, ...], most_common_emotion: str, emotion_counts: dict[str, int], emotion_summary: dict[str, dict[str, float]], au_summary: dict[str, dict[str, float]])`
- Produces: `compute_file_statistics(path: Path) -> FileStatistics`
- Produces: `compute_directory_statistics(data_dir: Path) -> list[FileStatistics]`
- Produces: `write_statistics_summary(statistics: list[FileStatistics], output_path: Path) -> Path`

- [ ] **Step 1: Write failing statistics tests**

Create `tests/test_statistics.py`:

```python
import pandas as pd

from pyfeat_utils.statistics import (
    compute_directory_statistics,
    compute_file_statistics,
    write_statistics_summary,
)


def test_compute_file_statistics_for_emotions_and_aus(tmp_path):
    path = tmp_path / "clip.mp4.csv"
    pd.DataFrame(
        {
            "frame": [1, 2, 3],
            "happiness": [0.9, 0.2, 0.4],
            "sadness": [0.1, 0.8, 0.3],
            "AU01": [0.2, 0.4, 0.6],
        }
    ).to_csv(path, index=False)

    stats = compute_file_statistics(path)

    assert stats.source == path
    assert stats.row_count == 3
    assert stats.emotion_columns == ("happiness", "sadness")
    assert stats.most_common_emotion == "happiness"
    assert stats.emotion_counts == {"happiness": 2, "sadness": 1}
    assert stats.emotion_summary["happiness"]["mean"] == 0.5
    assert stats.au_summary["AU01"]["median"] == 0.4


def test_compute_directory_statistics_uses_table_discovery(tmp_path):
    video_csv = tmp_path / "clip.mp4.csv"
    image_csv = tmp_path / "face.png.csv"
    pd.DataFrame({"frame": [1], "neutral": [1.0]}).to_csv(video_csv, index=False)
    pd.DataFrame({"neutral": [1.0]}).to_csv(image_csv, index=False)

    stats = compute_directory_statistics(tmp_path)

    assert [item.source for item in stats] == [video_csv]


def test_write_statistics_summary_creates_csv(tmp_path):
    source = tmp_path / "clip.mp4.csv"
    pd.DataFrame({"frame": [1], "neutral": [1.0]}).to_csv(source, index=False)
    stats = [compute_file_statistics(source)]
    output = tmp_path / "statistics_summary.csv"

    result = write_statistics_summary(stats, output)

    assert result == output
    summary = pd.read_csv(output)
    assert summary.loc[0, "source"] == "clip.mp4.csv"
    assert summary.loc[0, "most_common_emotion"] == "neutral"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run --with pytest pytest tests/test_statistics.py -v`

Expected: FAIL because `statistics.py` does not exist.

- [ ] **Step 3: Implement statistics module**

Create `src/pyfeat_utils/statistics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyfeat_utils.files import discover_table_files

EMOTION_COLUMNS = ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")


class StatisticsError(ValueError):
    """Raised when py-feat output statistics cannot be computed."""


@dataclass(frozen=True)
class FileStatistics:
    source: Path
    row_count: int
    emotion_columns: tuple[str, ...]
    most_common_emotion: str
    emotion_counts: dict[str, int]
    emotion_summary: dict[str, dict[str, float]]
    au_summary: dict[str, dict[str, float]]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".txt":
        try:
            return pd.read_csv(path, delimiter="\t")
        except Exception:
            return pd.read_csv(path, delimiter=",")
    return pd.read_csv(path)


def _numeric_columns(df: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        column
        for column in candidates
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    )


def _summary(df: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        column: {
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "q1": float(df[column].quantile(0.25)),
            "q3": float(df[column].quantile(0.75)),
        }
        for column in columns
    }


def compute_file_statistics(path: Path) -> FileStatistics:
    df = _read_table(path)
    if df.empty:
        raise StatisticsError(f"No rows found in {path}")

    emotion_columns = _numeric_columns(df, EMOTION_COLUMNS)
    if not emotion_columns:
        raise StatisticsError(f"No numeric emotion columns found in {path}")

    row_emotions = df[list(emotion_columns)].idxmax(axis=1)
    emotion_counts = {
        str(index): int(value) for index, value in row_emotions.value_counts().items()
    }
    most_common_emotion = str(row_emotions.value_counts().idxmax())
    au_columns = tuple(
        column
        for column in df.columns
        if column.startswith("AU") and pd.api.types.is_numeric_dtype(df[column])
    )

    return FileStatistics(
        source=path,
        row_count=int(len(df)),
        emotion_columns=emotion_columns,
        most_common_emotion=most_common_emotion,
        emotion_counts=emotion_counts,
        emotion_summary=_summary(df, emotion_columns),
        au_summary=_summary(df, au_columns),
    )


def compute_directory_statistics(data_dir: Path) -> list[FileStatistics]:
    return [compute_file_statistics(path) for path in discover_table_files(data_dir)]


def write_statistics_summary(
    statistics: list[FileStatistics], output_path: Path
) -> Path:
    rows = []
    for item in statistics:
        rows.append(
            {
                "source": item.source.name,
                "row_count": item.row_count,
                "emotion_columns": ",".join(item.emotion_columns),
                "most_common_emotion": item.most_common_emotion,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
```

- [ ] **Step 4: Wire stats CLI and compatibility wrapper**

Update `src/pyfeat_utils/cli.py` so the `stats` branch imports lazily:

```python
    if args.command == "stats":
        from pyfeat_utils.statistics import (
            compute_directory_statistics,
            write_statistics_summary,
        )

        statistics = compute_directory_statistics(config.data_dir)
        output_path = config.data_dir / "statistics_summary.csv"
        write_statistics_summary(statistics, output_path)
        print(f"Wrote statistics summary: {output_path}")
        return 0
```

Replace `src/pyfeat_utils/descriptive_statistics.py`:

```python
from __future__ import annotations

from pyfeat_utils.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["stats"]))
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run --with pytest pytest tests/test_statistics.py tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/pyfeat_utils/statistics.py src/pyfeat_utils/cli.py src/pyfeat_utils/descriptive_statistics.py tests/test_statistics.py tests/test_cli.py
git commit -m "feat: add descriptive statistics library"
```

---

### Task 4: Processing Library With Lazy py-feat

**Files:**
- Create: `src/pyfeat_utils/processing.py`
- Modify: `src/pyfeat_utils/cli.py`
- Replace: `src/pyfeat_utils/pyfeat_processor.py`
- Test: `tests/test_processing.py`

**Interfaces:**
- Consumes: `ProcessingConfig`
- Consumes: `discover_media_files(data_dir: Path, process_types: tuple[str, ...]) -> list[Path]`
- Consumes: `prediction_output_path(input_path: Path, output_dir: Path | None = None) -> Path`
- Produces: `ProcessingResult(input_path: Path, output_path: Path, media_type: str)`
- Produces: `create_detector() -> object`
- Produces: `process_media_file(input_path: Path, detector: object, video_skip_frames: int, output_dir: Path | None = None) -> ProcessingResult`
- Produces: `process_directory(config: ProcessingConfig, detector: object | None = None) -> list[ProcessingResult]`

- [ ] **Step 1: Write failing processing tests**

Create `tests/test_processing.py`:

```python
import pandas as pd

from pyfeat_utils.config import ProcessingConfig
from pyfeat_utils.processing import process_directory, process_media_file


class FakePrediction(pd.DataFrame):
    @property
    def _constructor(self):
        return FakePrediction


class FakeDetector:
    def __init__(self):
        self.calls = []

    def detect_image(self, path, data_type="image"):
        self.calls.append(("image", path, data_type))
        return FakePrediction({"happiness": [0.9]})

    def detect_video(self, path, data_type="video", skip_frames=20):
        self.calls.append(("video", path, data_type, skip_frames))
        return FakePrediction({"frame": [1], "neutral": [0.8]})


def test_process_media_file_writes_image_prediction(tmp_path):
    image = tmp_path / "face.png"
    image.write_text("image", encoding="utf-8")
    detector = FakeDetector()

    result = process_media_file(image, detector, video_skip_frames=20)

    assert result.media_type == "image"
    assert result.output_path == tmp_path / "face.png.csv"
    assert result.output_path.exists()
    assert detector.calls == [("image", image, "image")]


def test_process_media_file_writes_video_prediction_with_skip_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_text("video", encoding="utf-8")
    detector = FakeDetector()

    result = process_media_file(video, detector, video_skip_frames=7)

    assert result.media_type == "video"
    assert result.output_path == tmp_path / "clip.mp4.csv"
    assert detector.calls == [("video", video, "video", 7)]


def test_process_directory_discovers_configured_media(tmp_path):
    image = tmp_path / "face.png"
    video = tmp_path / "clip.mp4"
    image.write_text("image", encoding="utf-8")
    video.write_text("video", encoding="utf-8")
    detector = FakeDetector()
    config = ProcessingConfig(data_dir=tmp_path, process_types=("video",), video_skip_frames=5)

    results = process_directory(config, detector=detector)

    assert [result.input_path for result in results] == [video]
    assert detector.calls == [("video", video, "video", 5)]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run --with pytest pytest tests/test_processing.py -v`

Expected: FAIL because `processing.py` does not exist.

- [ ] **Step 3: Implement processing module**

Create `src/pyfeat_utils/processing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyfeat_utils.config import ProcessingConfig
from pyfeat_utils.files import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    discover_media_files,
    prediction_output_path,
)


class ProcessingError(ValueError):
    """Raised when media processing cannot continue."""


@dataclass(frozen=True)
class ProcessingResult:
    input_path: Path
    output_path: Path
    media_type: str


def create_detector() -> Any:
    from feat import Detector

    return Detector()


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ProcessingError(f"Unsupported media file: {path}")


def process_media_file(
    input_path: Path,
    detector: Any,
    video_skip_frames: int,
    output_dir: Path | None = None,
) -> ProcessingResult:
    media_type = _media_type(input_path)
    if media_type == "image":
        prediction = detector.detect_image(input_path, data_type="image")
    else:
        prediction = detector.detect_video(
            input_path, data_type="video", skip_frames=video_skip_frames
        )

    output_path = prediction_output_path(input_path, output_dir)
    prediction.to_csv(output_path, index=False)
    return ProcessingResult(input_path, output_path, media_type)


def process_directory(
    config: ProcessingConfig, detector: Any | None = None
) -> list[ProcessingResult]:
    detector = detector or create_detector()
    files = discover_media_files(config.data_dir, config.process_types)
    return [
        process_media_file(
            path,
            detector,
            video_skip_frames=config.video_skip_frames,
            output_dir=config.data_dir,
        )
        for path in files
    ]
```

- [ ] **Step 4: Wire process CLI and compatibility wrapper**

Update `src/pyfeat_utils/cli.py` so the `process` branch imports lazily:

```python
    if args.command == "process":
        from pyfeat_utils.processing import process_directory

        results = process_directory(config)
        for result in results:
            print(f"Processed {result.media_type}: {result.input_path} -> {result.output_path}")
        print(f"Processed {len(results)} file(s).")
        return 0
```

Replace `src/pyfeat_utils/pyfeat_processor.py`:

```python
from __future__ import annotations

from pyfeat_utils.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["process"]))
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run --with pytest pytest tests/test_processing.py tests/test_cli.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/pyfeat_utils/processing.py src/pyfeat_utils/cli.py src/pyfeat_utils/pyfeat_processor.py tests/test_processing.py
git commit -m "feat: add lazy processing library"
```

---

### Task 5: Documentation, Lockfile, And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `uv.lock`
- Modify or remove: `requirements.in`
- Modify or remove: `requirements-dev.in`
- Modify or remove: `requirements.txt`
- Modify or remove: `requirements-dev.txt`

**Interfaces:**
- Consumes: all earlier CLI and package interfaces.
- Produces: README commands that match implemented behavior.
- Produces: verified lockfile for package metadata.

- [ ] **Step 1: Update README**

Replace README content with installation and usage that includes:

```markdown
# pyfeat-utils

`pyfeat-utils` provides command-line utilities for batch py-feat processing of images and videos, plus descriptive summaries of generated facial emotion and Action Unit CSV outputs.

## Requirements

py-feat currently requires Python 3.11+. This repository develops against Python 3.13 via `.python-version`.

## Installation

```powershell
uv sync
```

If uv fails because the global cache path is not usable on Windows, use a project-local cache:

```powershell
$env:UV_CACHE_DIR = ".\.uv-cache"
uv sync
```

## Usage

Create a starter config and data directory:

```powershell
uv run pyfeat-utils init
```

Process configured image/video files:

```powershell
uv run pyfeat-utils process --config "$HOME\Documents\pyfeat-utils\config.json"
```

Compute descriptive statistics:

```powershell
uv run pyfeat-utils stats --config "$HOME\Documents\pyfeat-utils\config.json"
```
```

- [ ] **Step 2: Regenerate/check lockfile**

Run:

```powershell
$env:UV_CACHE_DIR = ".\.uv-cache"
uv lock
```

Expected: lockfile resolves for Python 3.11+ metadata.

- [ ] **Step 3: Run full verification**

Run:

```powershell
$env:UV_CACHE_DIR = ".\.uv-cache"
uv lock --check
uv sync --dry-run
uv run --with pytest pytest -v
uv run --with ruff ruff check
uv run pyfeat-utils --help
```

Expected: all commands pass. `pyfeat-utils --help` prints help without loading py-feat models.

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md uv.lock requirements.in requirements-dev.in requirements.txt requirements-dev.txt
git commit -m "docs: update installation workflow"
```

---

## Self-Review

Spec coverage:

- Python/uv dependency cleanup is covered by Tasks 1 and 5.
- CLI entry points are covered by Tasks 1, 3, and 4.
- Config generation and loading are covered by Task 2.
- File discovery and deterministic outputs are covered by Tasks 2 and 4.
- Statistics are covered by Task 3.
- Lazy py-feat imports and no model-download tests are covered by Tasks 1 and 4.
- README and verification are covered by Task 5.

Placeholder scan:

- No TBD/TODO placeholders are intentionally left.
- All tasks include exact files, interfaces, test commands, and expected outcomes.

Type consistency:

- `ProcessingConfig`, `ProcessingResult`, `FileStatistics`, and CLI signatures are consistently named across tasks.
