"""Tests for the unified CLI (help, doctor, headless demos, evaluate/report)."""

import importlib.util
from pathlib import Path

import pytest

from armlab.cli import build_parser, main


class TestParser:
    def test_top_level_help(self):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["--help"])
        assert exc.value.code == 0

    def test_subcommand_help(self):
        for argv in (["evaluate", "--help"], ["train", "dqn", "--help"], ["benchmark", "--help"]):
            with pytest.raises(SystemExit) as exc:
                build_parser().parse_args(argv)
            assert exc.value.code == 0

    def test_unknown_command_fails(self):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["fly"])
        assert exc.value.code == 2

    def test_evaluate_requires_scenario(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["evaluate", "--controller", "ik"])

    def test_train_device_flags(self):
        assert build_parser().parse_args(["train", "dqn", "--device", "cuda"]).device == "cuda"
        assert build_parser().parse_args(["train", "sac", "--device", "cpu"]).device == "cpu"

    def test_resolve_device_override_and_auto(self):
        from armlab.cli import _resolve_device

        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("cuda") == "cuda"
        assert _resolve_device("auto") in {"cpu", "cuda"}


class TestCommands:
    def test_doctor(self, capsys):
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "armlab" in out
        assert "torch device" in out

    def test_demo_ik_headless(self, capsys):
        assert main(["demo", "ik", "--trials", "2", "--headless", "--seed", "1"]) == 0
        out = capsys.readouterr().out
        assert "reached" in out

    def test_demo_physics_headless(self, capsys):
        result = main(["demo", "physics", "--steps", "10", "--headless"])
        out = capsys.readouterr().out
        if result == 3:
            assert "MuJoCo is not installed" in out
        else:
            assert "final ee" in out

    def test_evaluate_and_report_and_replay(self, tmp_path, capsys):
        code = main(
            [
                "evaluate",
                "--scenario",
                "static_reaching",
                "--controller",
                "ik",
                "--episodes",
                "2",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert code == 0
        runs = list(tmp_path.glob("*/summary.json"))
        assert len(runs) == 1
        run_dir = runs[0].parent

        assert main(["report", "--run", str(run_dir)]) == 0
        assert (run_dir / "plots" / "trajectory_3d.png").is_file()

        assert main(["replay", "--run", str(run_dir), "--episode", "0"]) == 0

    def test_dashboard_reports_missing_ui(self, capsys):
        if importlib.util.find_spec("dash") is not None:
            pytest.skip("dash installed; dashboard would launch a server")
        assert main(["dashboard"]) == 3
        assert "requires dash" in capsys.readouterr().out

    def test_error_returns_nonzero(self, capsys):
        code = main(
            [
                "evaluate",
                "--scenario",
                "not_a_scenario",
                "--controller",
                "ik",
                "--output-dir",
                str(Path.cwd() / "unused"),
            ]
        )
        assert code == 1
        assert "error:" in capsys.readouterr().out
