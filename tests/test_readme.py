import json
import re
from pathlib import Path

from armforge.readme import algo_table, build_readme, physics_table, top_section
from armforge.report import LEGACY_NOTE, legacy_section, relocate_links

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy" / "README.md"


def test_relocate_links_only_touches_relative_targets():
    md = "![a](projects/x.png) [b](https://e.com/x) [c](#sec) [d](docs/m.md) [e](/abs) [f](mailto:a@b.c)"
    out = relocate_links(md, "legacy/")
    assert "](legacy/projects/x.png)" in out and "](legacy/docs/m.md)" in out
    assert "](https://e.com/x)" in out and "](#sec)" in out and "](/abs)" in out and "](mailto:a@b.c)" in out


def test_legacy_readme_is_embedded_verbatim_except_links():
    legacy = LEGACY.read_text(encoding="utf-8")
    readme = build_readme(legacy, ROOT / "docs" / "media")
    section = legacy_section(legacy)
    assert LEGACY_NOTE in section
    body = section.split(LEGACY_NOTE, 1)[1].lstrip("\n")
    # undoing the link prefix must reproduce the original README exactly (no truncation, no edits)
    assert body.replace("](legacy/", "](").rstrip() == legacy.rstrip()
    assert readme.count("\n") > legacy.count("\n")
    assert "legacy/projects/screenshots/inv_kin_Figure_2.png" in readme
    # nothing relative is left un-relocated inside the legacy block
    block = readme[readme.index("# Legacy: ArmLab RL"):]
    bad = [t for t in re.findall(r"\]\(([^)\s]+)\)", block) if not re.match(r"(https?://|#|legacy/|mailto:|/)", t)]
    assert not bad, bad


def test_no_media_means_honest_placeholder(tmp_path):
    text = top_section(tmp_path)
    assert "No training run has been published" in text
    assert text.count("No published run for") == 3
    assert ".gif" not in text.split("## The robot and the physics")[0].split("[![Open in Colab]")[0]


def test_published_media_shows_real_numbers(tmp_path):
    d = tmp_path / "reach"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "task": "reach", "run": "r", "total_steps": 1000,
        "stages": [
            {"step": 0, "label": "untrained", "success_rate": 0.03, "mean_final_distance": 0.31, "episodes": 30, "files": {"hero": "stage_0.gif"}},
            {"step": 1000, "label": "final", "success_rate": 0.9, "mean_final_distance": 0.012, "episodes": 30, "files": {"hero": "stage_1.gif"}},
        ]}))
    (d / "progression.gif").write_bytes(b"GIF89a")
    text = top_section(tmp_path)
    assert "Untrained policy: 3% success" in text and "1,000 steps: 90%" in text
    assert "| 1,000 | final | 90% | 1.2 cm |" in text
    assert "docs/media/reach/progression.gif" in text
    assert "No published run for `reach`" not in text


def test_tables_are_generated_from_config_defaults():
    assert "elliptic" in physics_table() and "2F-85" in physics_table()
    assert "TQC" in algo_table() and "future" in algo_table()


def test_relocated_legacy_links_resolve():
    legacy = LEGACY.read_text(encoding="utf-8")
    block = build_readme(legacy, ROOT / "docs" / "media")
    targets = set(re.findall(r"\]\((legacy/[^)\s]+)\)", block[block.index("# Legacy: ArmLab RL"):]))
    assert targets and all((ROOT / t).exists() for t in targets), [t for t in targets if not (ROOT / t).exists()]
