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
    try:
        from feat import Detector
    except ImportError:
        from feat import Detectorv1 as Detector

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
