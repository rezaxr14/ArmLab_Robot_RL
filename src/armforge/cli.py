"""``armforge`` command line: doctor, scene, expert, train, eval, publish, readme."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .config import load_config


def _overrides(pairs: list[str] | None) -> dict:
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def _cfg(args):
    cfg = load_config(args.config, _overrides(args.set))
    if getattr(args, "name", None):
        cfg.name = args.name
    if getattr(args, "out_dir", None):
        cfg.out_dir = args.out_dir
    return cfg


def cmd_doctor(args) -> int:
    ok = True

    def line(name, status, detail=""):
        print(f"[{'ok' if status else 'FAIL'}] {name} {detail}")

    try:
        import mujoco

        line("mujoco", True, mujoco.__version__)
    except Exception as exc:  # noqa: BLE001
        line("mujoco", False, str(exc))
        return 1
    try:
        import torch

        line("torch", True, f"{torch.__version__} cuda={torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    except Exception as exc:  # noqa: BLE001
        line("torch", False, str(exc))
        ok = False
    try:
        from .assets import ensure_menagerie

        line("menagerie", True, str(ensure_menagerie()))
    except Exception as exc:  # noqa: BLE001
        line("menagerie", False, str(exc))
        return 1
    try:
        from .env.arm_env import ArmForgeEnv

        cfg = load_config(None)
        env = ArmForgeEnv(cfg.env)
        env.reset(seed=0)
        line("env", True, f"obs_dim={env.obs_dim} act_dim={env.action_dim} ncon@reset={env.data.ncon}")
        try:
            img = env.render("front", 320, 240)
            line("render", True, f"{img.shape} (MUJOCO_GL={__import__('os').environ.get('MUJOCO_GL')})")
        except Exception as exc:  # noqa: BLE001
            line("render", False, f"{exc}  -> try MUJOCO_GL=osmesa")
            ok = False
        env.close()
    except Exception as exc:  # noqa: BLE001
        line("env", False, repr(exc))
        ok = False
    return 0 if ok else 1


def cmd_scene(args) -> int:
    from .report import render_scene_gallery

    cfg = load_config(args.config, {"env.task.name": args.task, **_overrides(args.set)})
    out = render_scene_gallery(cfg, args.out)
    print(out)
    return 0


def cmd_expert(args) -> int:
    from dataclasses import replace

    from .env.arm_env import ArmForgeEnv
    from .rl.expert import rollout_expert, summarize_episode

    cfg = _cfg(args)
    env = ArmForgeEnv(replace(cfg.env, seed=args.seed))
    wins, dists, traces = 0, [], []
    for i in range(args.episodes):
        ep = rollout_expert(env, seed=args.seed + i)
        wins += int(ep["success"])
        dists.append(float(((ep["ag"][-1] - ep["goal"]) ** 2).sum() ** 0.5))
        print(summarize_episode(ep))
        if args.trace_out and not ep["success"]:
            traces.append({k: ep[k] for k in ("seed", "phases", "trace", "goal_height")})
    if args.trace_out:
        Path(args.trace_out).write_text(json.dumps(traces))
        print(f"wrote {len(traces)} failing-episode traces to {args.trace_out}")
    print(json.dumps({"task": cfg.env.task.name, "episodes": args.episodes, "success_rate": wins / args.episodes, "mean_final_distance": sum(dists) / len(dists)}, indent=2))
    env.close()
    return 0


def cmd_train(args) -> int:
    from .rl.trainer import Trainer

    cfg = _cfg(args)
    trainer = Trainer(cfg, resume=args.resume)
    try:
        summary = trainer.train()
    finally:
        trainer.close()
    print(json.dumps(summary, indent=2))
    return 0


def cmd_eval(args) -> int:
    from .media import load_policy
    from .rl.evaluate import agent_policy, evaluate_policy

    run = Path(args.run)
    cfg = load_config(run / "config.yaml", _overrides(args.set))
    ckpt = Path(args.checkpoint) if args.checkpoint else run / "best.pt"
    agent = load_policy(ckpt, cfg, args.device)
    res = evaluate_policy(cfg.env, agent_policy(agent), args.episodes)
    res.pop("rows")
    print(json.dumps(res, indent=2))
    return 0


def cmd_publish(args) -> int:
    from .report import publish_task

    manifest = publish_task(args.run, args.repo, cameras=tuple(args.cameras), clips_per_stage=args.clips, stats_episodes=args.episodes)
    print(json.dumps({k: manifest[k] for k in ("task", "run", "total_steps")}, indent=2))
    return cmd_readme(argparse.Namespace(repo=args.repo))


def cmd_readme(args) -> int:
    from .readme import build_readme

    repo = Path(args.repo)
    legacy = (repo / "legacy" / "README.md").read_text(encoding="utf-8")
    (repo / "README.md").write_text(build_readme(legacy, repo / "docs" / "media"), encoding="utf-8")
    print(repo / "README.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="armforge", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_cfg(p, with_run=False):
        p.add_argument("--config", help="YAML run config (defaults if omitted)")
        p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config fields, e.g. train.total_steps=50000")
        if with_run:
            p.add_argument("--name")
            p.add_argument("--out-dir")

    sub.add_parser("doctor", help="check mujoco, torch, Menagerie download and rendering").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("scene", help="render a gallery of scene cameras")
    add_cfg(p)
    p.add_argument("--task", default="pickplace")
    p.add_argument("--out", default="docs/media/scene_gallery.png")
    p.set_defaults(fn=cmd_scene)

    p = sub.add_parser("expert", help="run the scripted expert and report its success rate")
    add_cfg(p)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1_000_000)
    p.add_argument("--trace-out", help="write a per-step trace of every failing episode to this JSON file")
    p.set_defaults(fn=cmd_expert)

    p = sub.add_parser("train", help="train a policy")
    add_cfg(p, with_run=True)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval", help="evaluate a saved policy on held-out seeds")
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--set", nargs="*")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("publish", help="render media for a run into docs/media and regenerate README.md")
    p.add_argument("--run", required=True)
    p.add_argument("--repo", default=".")
    p.add_argument("--cameras", nargs="+", default=["hero"])
    p.add_argument("--clips", type=int, default=2)
    p.add_argument("--episodes", type=int, default=30)
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("readme", help="regenerate README.md from docs/media and legacy/README.md")
    p.add_argument("--repo", default=".")
    p.set_defaults(fn=cmd_readme)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
