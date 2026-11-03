import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def menagerie():
    """Path to a Menagerie checkout (env ARMFORGE_MENAGERIE, else the pinned download)."""
    from armforge.assets import ensure_menagerie

    try:
        return ensure_menagerie()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Menagerie assets unavailable: {exc}")


@pytest.fixture
def small_cfg():
    from armforge.config import EnvConfig, from_dict

    return from_dict(EnvConfig, {"task": {"name": "reach", "horizon": 10}})
