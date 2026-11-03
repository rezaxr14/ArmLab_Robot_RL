import numpy as np
from PIL import Image

from armforge.media import caption, filmstrip, hold, pick_frames, write_gif


def _frames(n=6, h=120, w=160):
    rng = np.random.default_rng(0)
    return [(rng.integers(0, 255, (h, w, 3))).astype(np.uint8) for _ in range(n)]


def test_caption_keeps_shape_and_changes_pixels():
    f = _frames(1)[0]
    out = caption(f, "reach | 0 steps", "eval success 0%", 0.123, False, 0.03)
    assert out.shape == f.shape and out.dtype == np.uint8 and not np.array_equal(out, f)
    out2 = caption(f, "t", "", 0.01, True, 0.03)
    assert not np.array_equal(out, out2)  # success tag differs


def test_gif_has_all_frames_and_loops(tmp_path):
    frames = hold(_frames(5), 3)
    p = write_gif(frames, tmp_path / "a.gif", fps=12.5)
    with Image.open(p) as im:
        assert im.info.get("loop") == 0
        total = 0
        for i in range(im.n_frames):
            im.seek(i)
            total += im.info["duration"]
        # identical consecutive frames are merged by PIL, but the hold time must be preserved
        assert total == 8 * 80


def test_filmstrip_and_pick_frames(tmp_path):
    fr = _frames(4)
    p = filmstrip(fr, list("abcd"), tmp_path / "s.png")
    with Image.open(p) as im:
        assert im.size[0] == 4 * 160 + 3 * 4 and im.size[1] == 120 + 34
    picked = pick_frames(_frames(11), 3)
    assert len(picked) == 3
