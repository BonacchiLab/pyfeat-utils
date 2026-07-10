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
