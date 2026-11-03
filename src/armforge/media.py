"""Rollout recording, captions, GIF/MP4 writers and still-image strips."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import EVAL_SEED_BASE, RunConfig

GOOD, BAD, INK, BAR = (46, 160, 90), (214, 80, 60), (245, 245, 245), (20, 22, 28)


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size) if hasattr(ImageFont, "load_default") else ImageFont.load_default()


# ----------------------------------------------------------------------------- policies
def load_policy(path: str | Path, cfg: RunConfig, device: str = "cpu"):
    """Rebuild an agent from a policy snapshot (actor + normalisers); critics stay untrained."""
    import torch

    from .rl.agent import SACAgent, resolve_device

    state = torch.load(path, map_location="cpu")
    obs_dim, goal_dim, act_dim = state["dims"]
    agent = SACAgent(obs_dim, goal_dim, act_dim, cfg.algo, resolve_device(device))
    agent.load_policy_state(state)
    agent.snapshot_extra = state.get("extra", {})
    return agent


# ----------------------------------------------------------------------------- recording
def record_episode(env, act_fn, seed: int, camera: str, size=(480, 360), render_every: int = 2, goal_scale: float = 1.0):
    """Run one episode and return per-frame images with the live distance/success."""
    obs = env.reset(seed=seed, goal_scale=goal_scale)
    w, h = size
    frames = [env.render(camera, w, h)]
    dists = [float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))]
    succ = [dists[0] <= env.cfg.task.success_threshold]
    done, step, info = False, 0, {"is_success": False}
    while not done:
        obs, _, done, info = env.step(act_fn(obs))
        step += 1
        if step % render_every == 0 or done:
            frames.append(env.render(camera, w, h))
            dists.append(info["distance"])
            succ.append(info["is_success"])
    return {"frames": frames, "dist": dists, "success": succ, "final_success": bool(info["is_success"]), "seed": seed}


def caption(frame: np.ndarray, title: str, subtitle: str = "", dist: float | None = None, success: bool | None = None, threshold: float | None = None) -> np.ndarray:
    """Overlay a title bar (top) and a status strip (bottom) on an RGB frame."""
    img = Image.fromarray(frame).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img, "RGBA")
    big, small = _font(max(14, h // 20)), _font(max(11, h // 28))
    d.rectangle([0, 0, w, int(h * 0.105)], fill=BAR + (215,))
    d.text((10, int(h * 0.02)), title, fill=INK, font=big)
    if subtitle:
        tw = d.textlength(subtitle, font=small)
        d.text((w - tw - 10, int(h * 0.032)), subtitle, fill=(200, 205, 215), font=small)
    if dist is not None:
        d.rectangle([0, h - int(h * 0.085), w, h], fill=BAR + (215,))
        label = f"goal distance {dist * 100:5.1f} cm"
        d.text((10, h - int(h * 0.07)), label, fill=INK, font=small)
        if success is not None:
            tag, col = ("GOAL REACHED", GOOD) if success else ("not at goal", BAD)
            tw = d.textlength(tag, font=small)
            d.rounded_rectangle([w - tw - 22, h - int(h * 0.076), w - 8, h - 6], radius=6, fill=col + (235,))
            d.text((w - tw - 15, h - int(h * 0.07)), tag, fill=(255, 255, 255), font=small)
    return np.asarray(img)


# ----------------------------------------------------------------------------- writers
def write_gif(frames: list[np.ndarray], path: str | Path, fps: float = 12.5, colors: int = 96) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(f).convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(round(1000 / fps)), loop=0, optimize=False, disposal=1)
    return path


def write_mp4(frames: list[np.ndarray], path: str | Path, fps: float = 25.0) -> Path:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1, pixelformat="yuv420p")
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()
    return path


def hold(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    return frames + [frames[-1]] * n


def filmstrip(images: list[np.ndarray], labels: list[str], path: str | Path, label_height: int = 34) -> Path:
    """Horizontal strip of stills, each with a caption underneath."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = images[0].shape[:2]
    canvas = Image.new("RGB", (w * len(images) + 4 * (len(images) - 1), h + label_height), (252, 252, 251))
    d = ImageDraw.Draw(canvas)
    font = _font(15)
    for i, (img, lab) in enumerate(zip(images, labels)):
        x = i * (w + 4)
        canvas.paste(Image.fromarray(img), (x, 0))
        d.text((x + 8, h + 8), lab, fill=(30, 30, 30), font=font)
    canvas.save(path)
    return path


def pick_frames(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def make_env(cfg: RunConfig, seed: int = EVAL_SEED_BASE):
    from .env.arm_env import ArmForgeEnv

    return ArmForgeEnv(replace(cfg.env, seed=seed))
