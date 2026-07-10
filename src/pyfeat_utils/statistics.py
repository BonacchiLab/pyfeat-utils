from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyfeat_utils.files import discover_table_files

EMOTION_COLUMNS = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
)


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


def write_statistics_summary(statistics: list[FileStatistics], output_path: Path) -> Path:
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
