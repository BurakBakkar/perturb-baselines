from pbench.cli import main


def test_cli_help_exits_zero(capsys):
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    assert "download" in capsys.readouterr().out
