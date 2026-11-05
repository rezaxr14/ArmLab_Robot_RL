import zipfile

import pytest

from armforge.archive import pack_run, restore_run


def _make_run(tmp_path, with_buffer=True):
    run = tmp_path / "runs" / "pickplace-s0"
    (run / "milestones").mkdir(parents=True)
    (run / "latest.pt").write_bytes(b"latest")
    (run / "best.pt").write_bytes(b"best")
    (run / "milestones" / "m_000050000.pt").write_bytes(b"m")
    (run / "metrics.csv").write_text("steps\n1\n")
    (run / "latest.pt.tmp").write_bytes(b"half-written")
    if with_buffer:
        (run / "latest_buffer.pt").write_bytes(b"B" * 1000)
    repo = tmp_path / "repo"
    (repo / "docs" / "media" / "pickplace").mkdir(parents=True)
    (repo / "docs" / "media" / "pickplace" / "progression.gif").write_bytes(b"gif")
    return run, repo


def test_pack_then_restore_roundtrip(tmp_path):
    run, repo = _make_run(tmp_path)
    info = pack_run(run, tmp_path / "out.zip", repo_root=repo)
    assert info["has_latest"]
    names = set(zipfile.ZipFile(tmp_path / "out.zip").namelist())
    assert "runs/pickplace-s0/latest.pt" in names and "runs/pickplace-s0/milestones/m_000050000.pt" in names
    assert "docs/media/pickplace/progression.gif" in names
    assert not any(n.endswith(".tmp") for n in names), "half-written checkpoints must not be packed"
    assert "runs/pickplace-s0/latest_buffer.pt" not in names, "the replay buffer is opt-in"

    fresh_runs, fresh_repo = tmp_path / "fresh" / "runs", tmp_path / "fresh" / "repo"
    out = restore_run(tmp_path / "out.zip", fresh_runs, repo_root=fresh_repo)
    assert out["runs"] == ["pickplace-s0"] and out["media_files"] == 1
    assert (fresh_runs / "pickplace-s0" / "latest.pt").read_bytes() == b"latest"
    assert (fresh_repo / "docs" / "media" / "pickplace" / "progression.gif").read_bytes() == b"gif"


def test_buffer_is_included_on_request(tmp_path):
    run, repo = _make_run(tmp_path)
    pack_run(run, tmp_path / "out.zip", repo_root=repo, include_buffer=True)
    assert "runs/pickplace-s0/latest_buffer.pt" in zipfile.ZipFile(tmp_path / "out.zip").namelist()


def test_packing_an_empty_run_fails_loudly(tmp_path):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "train.log").write_text("x")
    with pytest.raises(FileNotFoundError):
        pack_run(tmp_path / "r", tmp_path / "o.zip")


def test_restore_rejects_path_traversal(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("runs/x/../../escape.txt", "nope")
    with pytest.raises(ValueError):
        restore_run(bad, tmp_path / "runs")
    assert not (tmp_path / "escape.txt").exists()
