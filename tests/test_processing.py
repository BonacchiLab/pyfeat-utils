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


class FakeUnifiedDetector:
    def __init__(self):
        self.calls = []

    def detect(self, path, data_type="image", skip_frames=None, progress_bar=True):
        self.calls.append((path, data_type, skip_frames, progress_bar))
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


def test_process_media_file_supports_unified_pyfeat_detector_api(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_text("video", encoding="utf-8")
    detector = FakeUnifiedDetector()

    result = process_media_file(video, detector, video_skip_frames=7)

    assert result.media_type == "video"
    assert result.output_path == tmp_path / "clip.mp4.csv"
    assert detector.calls == [(video, "video", 7, False)]


def test_process_directory_discovers_configured_media(tmp_path):
    image = tmp_path / "face.png"
    video = tmp_path / "clip.mp4"
    image.write_text("image", encoding="utf-8")
    video.write_text("video", encoding="utf-8")
    detector = FakeDetector()
    config = ProcessingConfig(
        data_dir=tmp_path, process_types=("video",), video_skip_frames=5
    )

    results = process_directory(config, detector=detector)

    assert [result.input_path for result in results] == [video]
    assert detector.calls == [("video", video, "video", 5)]
