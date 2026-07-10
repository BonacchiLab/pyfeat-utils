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
