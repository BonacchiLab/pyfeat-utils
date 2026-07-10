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
