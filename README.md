# pyfeat-utils

`pyfeat-utils` provides command-line utilities for batch py-feat processing of
images and videos, plus descriptive summaries of generated facial emotion and
Action Unit CSV outputs.

## Requirements

py-feat currently requires Python 3.11+. This repository develops against
Python 3.13 via `.python-version`.

The project uses `pyproject.toml` and `uv.lock` as the dependency source of
truth. The lock uses `py-feat>=2.0.3` and uv overrides for `pandas>=2.2.3,<3`
and `scikit-image>=0.25.2,<0.27` because `py-feat 2.0.3` still resolves older
releases of those packages that do not ship usable Python 3.13 Windows wheels.

## Installation

Create or sync the project environment:

```powershell
uv sync
```

If uv fails because the global cache path is not usable on Windows, use a
project-local cache:

```powershell
$env:UV_CACHE_DIR = ".\.uv-cache"
uv sync
```

## Usage

Create a starter config and data directory:

```powershell
uv run pyfeat-utils init
```

By default this creates:

```text
~/Documents/pyfeat-utils/config.json
~/Documents/pyfeat-utils/data/
```

Process configured image and video files:

```powershell
uv run pyfeat-utils process --config "$HOME\Documents\pyfeat-utils\config.json"
```

Compute descriptive statistics from generated py-feat CSV/TXT outputs:

```powershell
uv run pyfeat-utils stats --config "$HOME\Documents\pyfeat-utils\config.json"
```

The statistics command writes `statistics_summary.csv` in the configured data
directory.

## Config

The generated config uses this shape:

```json
{
  "data_processing": {
    "data_dir": "C:\\Users\\you\\Documents\\pyfeat-utils\\data",
    "process_types": ["image", "video"],
    "video_skip_frames": 20
  }
}
```

Supported image formats are `.jpg`, `.jpeg`, and `.png`. Supported video
formats are `.mp4`, `.avi`, and `.mov`.

## Development

Run the unit tests without forcing a full py-feat environment sync:

```powershell
$env:UV_CACHE_DIR = ".\.uv-cache"
$env:PYTHONPATH = "src"
uv run --no-project --with pytest --with pandas pytest -v
```

Run linting:

```powershell
uv run --with ruff ruff check
```

Full environment checks:

```powershell
$env:UV_CACHE_DIR = ".\.uv-cache"
uv lock --check
uv sync --dry-run
uv run pyfeat-utils --help
```
