"""Pack / restore a training run as one zip, so checkpoints survive a Colab runtime reset.

``pack_run`` stores, in one archive:

* ``runs/<name>/``   every checkpoint and log of the run: ``latest.pt`` (full training state), ``best.pt``,
                     ``final.pt``, ``milestones/*.pt`` (the policy snapshots), ``metrics.csv``, ``config.yaml``,
                     ``train.log``... The replay buffer (``latest_buffer.pt``, hundreds of MB) is optional.
* ``docs/media/``    the published GIFs, stills and plots.

``restore_run`` puts them back so ``armforge train --resume`` continues the run. Extraction only accepts those two
prefixes and rejects any path that would escape its target directory.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

BUFFER_FILE = "latest_buffer.pt"


def pack_run(run_dir: str | Path, out_zip: str | Path, repo_root: str | Path | None = None, include_buffer: bool = False) -> dict:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run directory at {run_dir}")
    files: list[tuple[Path, str]] = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file() or p.name.endswith(".tmp") or (p.name == BUFFER_FILE and not include_buffer):
            continue
        files.append((p, f"runs/{run_dir.name}/{p.relative_to(run_dir).as_posix()}"))
    if repo_root is not None and (Path(repo_root) / "docs" / "media").is_dir():
        media = Path(repo_root) / "docs" / "media"
        files += [(p, f"docs/media/{p.relative_to(media).as_posix()}") for p in sorted(media.rglob("*")) if p.is_file()]
    if not any(arc.endswith("latest.pt") or "/milestones/" in arc for _, arc in files):
        raise FileNotFoundError(f"{run_dir} holds no checkpoints (latest.pt / milestones/) - nothing to pack")
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p, arc in files:
            zf.write(p, arc)
    return {"zip": str(out_zip), "files": len(files), "bytes": out_zip.stat().st_size, "has_latest": any(a.endswith("/latest.pt") for _, a in files)}


def _safe_target(base: Path, rel: str) -> Path:
    target = (base / rel).resolve()
    if base.resolve() != target and base.resolve() not in target.parents:
        raise ValueError(f"unsafe path in archive: {rel!r}")
    return target


def restore_run(zip_path: str | Path, out_dir: str | Path, repo_root: str | Path | None = None) -> dict:
    """Extract ``runs/<name>/...`` into ``out_dir/<name>/`` and ``docs/media/...`` into ``repo_root`` (when given)."""
    out_dir = Path(out_dir)
    runs, media = set(), 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith("runs/"):
                run, _, rest = name[len("runs/"):].partition("/")
                if not run or not rest:
                    continue
                target = _safe_target(out_dir / run, rest)
                runs.add(run)
            elif name.startswith("docs/media/") and repo_root is not None:
                target = _safe_target(Path(repo_root) / "docs" / "media", name[len("docs/media/"):])
                media += 1
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
    return {"runs": sorted(runs), "media_files": media, "out_dir": str(out_dir)}
