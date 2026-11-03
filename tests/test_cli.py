import pytest

from armforge.cli import _overrides, main


def test_overrides_parse_yaml_values():
    assert _overrides(["train.total_steps=500", "env.scene.tray=true", "algo.hidden=[32,32]", "name=abc"]) == {
        "train.total_steps": 500, "env.scene.tray": True, "algo.hidden": [32, 32], "name": "abc"}


def test_bad_override_exits():
    with pytest.raises(SystemExit):
        _overrides(["nonsense"])


@pytest.mark.parametrize("cmd", ["doctor", "scene", "expert", "train", "eval", "publish", "readme"])
def test_subcommand_help(cmd, capsys):
    with pytest.raises(SystemExit) as exc:
        main([cmd, "--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_readme_command_regenerates(tmp_path):
    (tmp_path / "legacy").mkdir()
    (tmp_path / "legacy" / "README.md").write_text("# Old\n\n![x](projects/a.png)\n")
    assert main(["readme", "--repo", str(tmp_path)]) == 0
    out = (tmp_path / "README.md").read_text()
    assert out.startswith("# ArmForge") and "](legacy/projects/a.png)" in out
