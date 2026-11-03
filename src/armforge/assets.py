"""Fetching the robot models from MuJoCo Menagerie (pinned to a commit).

The UR5e and Robotiq 2F-85 meshes are *not* vendored into this repository: they are fetched
once into a cache directory (about 45 MB, sparse checkout, a few seconds) and reused. Menagerie
keeps each model's own licence next to it (BSD-style for the UR5e and the 2F-85).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Commit the model composition in scene.py was written and checked against.
MENAGERIE_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
REQUIRED_DIRS = ("universal_robots_ur5e", "robotiq_2f85")
REQUIRED_FILES = ("universal_robots_ur5e/ur5e.xml", "robotiq_2f85/2f85.xml")


def cache_root() -> Path:
    """Directory for downloaded assets and generated scene XML (override: ``ARMFORGE_CACHE``)."""
    root = Path(os.environ.get("ARMFORGE_CACHE", Path.home() / ".cache" / "armforge"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_complete(path: Path) -> bool:
    return all((path / rel).is_file() for rel in REQUIRED_FILES)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _sparse_fetch(dest: Path) -> None:
    """Fetch exactly ``MENAGERIE_COMMIT`` and only the two robot folders."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _run(["git", "init", "-q", "."], cwd=dest)
    _run(["git", "remote", "add", "origin", MENAGERIE_URL], cwd=dest)
    _run(["git", "config", "core.sparseCheckout", "true"], cwd=dest)
    (dest / ".git" / "info" / "sparse-checkout").write_text("\n".join(f"{d}/" for d in REQUIRED_DIRS) + "\n")
    _run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_COMMIT], cwd=dest)
    _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)


def _shallow_clone(dest: Path) -> None:
    """Fallback when fetching by commit is refused: newest default branch (not pinned)."""
    if dest.exists():
        shutil.rmtree(dest)
    _run(["git", "clone", "-q", "--depth", "1", MENAGERIE_URL, str(dest)])


def ensure_menagerie(path: str | Path | None = None) -> Path:
    """Return a directory containing ``universal_robots_ur5e/`` and ``robotiq_2f85/``.

    Resolution order: the ``path`` argument, the ``ARMFORGE_MENAGERIE`` environment variable
    (point either at an existing Menagerie checkout), then a pinned download into the cache.
    """
    for candidate in (path, os.environ.get("ARMFORGE_MENAGERIE")):
        if candidate:
            p = Path(candidate).expanduser()
            if not _is_complete(p):
                raise FileNotFoundError(
                    f"{p} is not a Menagerie checkout: expected {', '.join(REQUIRED_FILES)}"
                )
            return p

    dest = cache_root() / f"mujoco_menagerie-{MENAGERIE_COMMIT[:10]}"
    if _is_complete(dest):
        return dest
    try:
        _sparse_fetch(dest)
    except (subprocess.CalledProcessError, FileNotFoundError) as first_error:
        try:
            _shallow_clone(dest)
        except (subprocess.CalledProcessError, FileNotFoundError) as second_error:
            raise RuntimeError(
                "Could not download MuJoCo Menagerie. Clone "
                f"{MENAGERIE_URL} yourself and set ARMFORGE_MENAGERIE=/path/to/mujoco_menagerie.\n"
                f"pinned fetch error: {first_error}\nfallback clone error: {second_error}"
            ) from second_error
    if not _is_complete(dest):
        raise RuntimeError(f"Menagerie download at {dest} is missing {REQUIRED_FILES}")
    return dest
