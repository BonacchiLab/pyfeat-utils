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
