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
