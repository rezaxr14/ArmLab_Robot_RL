"""Unified ``armlab`` command-line interface (implementation plan Phase I1).

Commands::

    armlab doctor
    armlab demo ik [--headless]
    armlab demo physics [--headless]
    armlab train dqn  --config configs/train/dqn.yaml [--max-steps N] [--seed S]
    armlab train sac  --config configs/train/sac.yaml [--total-timesteps N]
    armlab evaluate   --scenario static_reaching --controller ik [--config ...]
    armlab benchmark  --suite default [--config configs/benchmark/default.yaml]
    armlab report     --run results/<run-id> | --results results [--train-dir DIR]
    armlab replay     --run results/<run-id> [--episode 0] [--gif]
    armlab dashboard  [--results results] [--port 8050]

All paths are relative to the current working directory; no command depends on
a developer's absolute path. Optional dependencies are imported lazily and
produce a clear error message when a requested feature is unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

DEFAULT_ROBOT_CONFIG = "configs/robot/three_link.yaml"
DEFAULT_ENV_CONFIG = "configs/env/reaching.yaml"
DEFAULT_DQN_CONFIG = "configs/train/dqn.yaml"
DEFAULT_SAC_CONFIG = "configs/train/sac.yaml"
DEFAULT_BENCHMARK_CONFIG = "configs/benchmark/default.yaml"


# --------------------------------------------------------------- helpers
def _existing(path: str | None) -> str | None:
    return path if path and Path(path).is_file() else None


def _robot_params(path: str | None):
    from armlab.config import RobotParams
    from armlab.configio import load_robot_params

    resolved = _existing(path) or _existing(DEFAULT_ROBOT_CONFIG)
    if resolved is None:
        return RobotParams()
    return load_robot_params(resolved)


def _env_config(path: str | None):
    from armlab.config import EnvParams, RewardParams
    from armlab.configio import load_env_config

    resolved = _existing(path) or _existing(DEFAULT_ENV_CONFIG)
    if resolved is None:
        return EnvParams(), RewardParams(), {}
    return load_env_config(resolved)


def _print(message: str = "") -> None:
    print(message)


# ---------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    import armlab

    _print(f"armlab {armlab.__version__}")
    _print(f"python {sys.version.split()[0]} on {platform.platform()}")
    _print(f"executable: {sys.executable}")
    _print("\noptional dependencies:")
    for name, extra in (
        ("yaml", "base"),
        ("torch", "rl"),
        ("stable_baselines3", "rl"),
        ("mujoco", "physics"),
        ("dash", "ui"),
        ("plotly", "ui"),
    ):
        found = importlib.util.find_spec(name) is not None
        status = "available" if found else f"MISSING (install with .[{extra}])"
        _print(f"  {name:20s} {status}")

    _print("\ntorch device:")
    if importlib.util.find_spec("torch") is not None:
        import torch

        _print(f"  cuda available: {torch.cuda.is_available()}")
        _print(f"  selected device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    else:
        _print("  torch not installed")

    _print("\nrendering:")
    headless = not sys.platform.startswith("win") and not __import__("os").environ.get("DISPLAY")
    _print(f"  GUI display likely available: {not headless}")
    _print("  use --headless for all commands that may open a window")
    return 0


# ------------------------------------------------------------------ demo
def cmd_demo_ik(args: argparse.Namespace) -> int:
    import numpy as np

    from armlab.config import EnvParams
    from armlab.controllers import JointPositionController
    from armlab.envs import KinematicReachingEnv
    from armlab.kinematics import RobotKinematics
    from armlab.targets import sample_uniform_reachable

    robot = _robot_params(args.robot_config)
    kin = RobotKinematics(robot)
    rng = np.random.default_rng(args.seed)
    render_mode = None if args.headless else "human"
    env = KinematicReachingEnv(robot, EnvParams(control_mode="continuous"), render_mode=render_mode)
    controller = JointPositionController(kin, kp=0.6, max_dq=0.3)

    _print("ArmLab analytical IK demo")
    reached = 0
    for trial in range(args.trials):
        target = sample_uniform_reachable(robot, env.env_params, rng, kinematics=kin)
        env.reset(seed=int(rng.integers(0, 2**31 - 1)), options={"target": target})
        best = float(np.linalg.norm(kin.end_effector(env.q) - target))
        for _ in range(env.env_params.max_episode_steps):
            action = np.clip(controller.act(env.q, target), -1.0, 1.0)
            _, _, terminated, truncated, info = env.step(action)
            best = min(best, float(info["distance"]))
            if terminated or truncated:
                break
        ok = best < env.reward_params.success_threshold
        reached += int(ok)
        _print(f"  trial {trial + 1}: min_error={best:.4f} success={ok}")
    env.close()
    _print(
        f"\nreached {reached}/{args.trials} targets (threshold {env.reward_params.success_threshold})"
    )
    return 0


def cmd_demo_physics(args: argparse.Namespace) -> int:
    from armlab.physics import MUJOCO_AVAILABLE

    if not MUJOCO_AVAILABLE:
        _print("MuJoCo is not installed. Install with: pip install -e '.[physics]'")
        return 3
    import numpy as np

    from armlab.physics import PhysicsReachingEnv

    robot = _robot_params(args.robot_config)
    render_mode = None if args.headless else "human"
    env = PhysicsReachingEnv(robot_params=robot, render_mode=render_mode)
    obs, info = env.reset(seed=args.seed)
    _print(f"ArmLab physics demo: initial ee={np.round(info['ee_position'], 4)}")
    for step in range(args.steps):
        action = np.zeros(3, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            _print(
                f"  episode ended at step {step}: reason={info.get('termination_reason', 'n/a')}"
            )
            break
    _print(f"final ee={np.round(info['ee_position'], 4)} distance={info['distance']:.4f}")
    _print(f"finite state: {bool(np.all(np.isfinite(obs)))}")
    env.close()
    return 0


# ----------------------------------------------------------------- train
def _resolve_output_dir(base: str | None, default: str = "results") -> Path:
    return Path(base or default)


def cmd_train_dqn(args: argparse.Namespace) -> int:
    import gymnasium as gym

    from armlab.config import RLConfig
    from armlab.configio import load_rl_config
    from armlab.envs import register_envs  # noqa: F401 - registers env ids
    from armlab.rl.agent import DQNAgent
    from armlab.rl.training import train_dqn

    resolved = _existing(args.config) or _existing(DEFAULT_DQN_CONFIG)
    data: dict[str, Any] = load_rl_config(resolved) if resolved else {}
    seed = args.seed if args.seed is not None else int(data.get("seed", 0))
    env_id = args.env_id or data.get("env_id", "ArmLabKinematic-v0")
    agent_cfg = data.get("agent") or RLConfig()
    n_episodes = args.episodes if args.episodes is not None else int(data.get("n_episodes", 200))
    eval_every = args.eval_every if args.eval_every is not None else int(data.get("eval_every", 25))
    eval_episodes = int(data.get("eval_episodes", 5))
    max_steps = args.max_steps if args.max_steps is not None else None

    out_dir = _resolve_output_dir(args.output_dir or data.get("output_dir"))
    run_dir = out_dir / f"train_dqn_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "latest.pt"

    register_envs()
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    agent = DQNAgent(state_dim, action_dim, config=agent_cfg, seed=seed)

    _print(
        f"Training DQN: env={env_id} seed={seed} episodes={n_episodes} "
        f"max_steps={max_steps} out={run_dir}"
    )
    try:
        result = train_dqn(
            agent,
            env,
            n_episodes=n_episodes,
            max_steps_per_episode=int(data.get("max_steps_per_episode", 250)),
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_env=eval_env,
            eval_seed=args.eval_seed,
            checkpoint_path=str(checkpoint),
            output_dir=str(run_dir),
            max_steps=max_steps,
            seed=seed,
            log_fn=_print,
        )
    finally:
        env.close()
        eval_env.close()

    _print(
        f"Done: env_steps={result.total_env_steps} best_success_rate={result.best_success_rate:.2f} "
        f"elapsed={result.elapsed_seconds:.1f}s"
    )
    _print(f"checkpoints: {result.checkpoint_path}")
    if result.best_checkpoint_path:
        _print(f"best checkpoint: {result.best_checkpoint_path}")
    if max_steps is not None or result.total_env_steps < 20_000:
        _print("NOTE: this is a smoke/development run, not evidence of convergence.")
    return 0


def cmd_train_sac(args: argparse.Namespace) -> int:
    from armlab.configio import load_rl_config
    from armlab.envs import register_envs  # noqa: F401
    from armlab.rl.sac import SAC_AVAILABLE, train_sac

    if not SAC_AVAILABLE:
        _print("SAC requires stable-baselines3 and torch. Install with: pip install -e '.[rl]'")
        return 3

    resolved = _existing(args.config) or _existing(DEFAULT_SAC_CONFIG)
    data: dict[str, Any] = load_rl_config(resolved) if resolved else {}
    seed = args.seed if args.seed is not None else int(data.get("seed", 0))
    env_id = args.env_id or data.get("env_id", "ArmLabKinematicContinuous-v0")
    timesteps = (
        args.total_timesteps
        if args.total_timesteps is not None
        else int(data.get("total_timesteps", 50_000))
    )
    out_dir = _resolve_output_dir(args.output_dir or data.get("output_dir"))
    register_envs()
    payload = train_sac(
        env_id=env_id,
        output_dir=out_dir,
        seed=seed,
        total_timesteps=timesteps,
        sac_params=data.get("sac"),
        eval_episodes=int(data.get("eval_episodes", 5)),
        eval_seed=args.eval_seed,
        deterministic_eval=bool(data.get("deterministic_eval", True)),
        log_fn=_print,
    )
    _print(f"model saved: {payload['model_path']}")
    return 0


# -------------------------------------------------------------- evaluate
def cmd_evaluate(args: argparse.Namespace) -> int:
    from armlab.evaluation.benchmark import evaluate_run

    robot = _robot_params(args.robot_config)
    result = evaluate_run(
        scenario=args.scenario,
        controller=args.controller,
        seed=args.seed,
        n_episodes=args.episodes,
        base_eval_seed=args.base_seed,
        output_dir=args.output_dir,
        robot=robot,
        checkpoint=args.checkpoint,
        smoke=not args.final,
    )
    summary = result["summary"]
    _print(f"run directory: {result['run_dir']}")
    _print(
        f"scenario={summary['scenario']} controller={summary['controller']} "
        f"episodes={summary['n_episodes']} success_rate={summary['success_rate']:.3f} "
        f"success_count={summary['success_count']}"
    )
    if summary.get("smoke"):
        _print("NOTE: smoke/development result (pass --final to label a full benchmark).")
    return 0


# ------------------------------------------------------------- benchmark
def cmd_benchmark(args: argparse.Namespace) -> int:
    from armlab.configio import load_benchmark_config
    from armlab.evaluation.benchmark import run_benchmark
    from armlab.evaluation.plots import plot_controller_comparison, plot_error_distributions

    resolved = _existing(args.config) or _existing(DEFAULT_BENCHMARK_CONFIG)
    if resolved is None and args.suite != "default":
        _print(f"Benchmark config not found: {args.config}")
        return 2
    if resolved is not None:
        config = load_benchmark_config(resolved)
    else:
        from armlab.configio import BenchmarkConfig

        config = BenchmarkConfig()

    robot = _robot_params(args.robot_config)
    result = run_benchmark(config, output_dir=args.output_dir, robot=robot)
    _print(f"benchmark suite={result['suite']} runs={result['n_runs']}")
    for cell in result["results"]:
        _print(
            f"  {cell['scenario']:20s} {cell['controller']:10s} "
            f"success_rate={cell['success_rate']:.3f} (n={cell['n_episodes']})"
        )
    for failure in result["failures"]:
        _print(f"  SKIPPED {failure['scenario']}/{failure['controller']}: {failure['reason']}")

    if result["runs"]:
        try:
            comparison = plot_controller_comparison(
                result["runs"], Path(args.output_dir or config.output_dir) / "benchmark_success.png"
            )
            _print(f"comparison plot: {comparison}")
            errors = plot_error_distributions(
                result["runs"], Path(args.output_dir or config.output_dir) / "benchmark_errors.png"
            )
            _print(f"error plot: {errors}")
        except Exception as exc:
            _print(f"plot generation skipped: {exc}")
    if config.smoke:
        _print("NOTE: smoke suite. Do not report these as final benchmark numbers.")
    return 0


# ----------------------------------------------------------------- report
def cmd_report(args: argparse.Namespace) -> int:
    from armlab.evaluation.artifacts import discover_runs
    from armlab.evaluation.plots import generate_run_report, plot_training_curves

    produced = []
    if args.run:
        produced.extend(generate_run_report(args.run))
        training = plot_training_curves(args.run)
        if training:
            produced.append(training)
    if args.results:
        for run_dir in discover_runs(args.results):
            produced.extend(generate_run_report(run_dir))
    if args.train_dir:
        training = plot_training_curves(args.train_dir)
        if training:
            produced.append(training)

    if not produced:
        _print("No artifacts found to report on.")
        return 1
    for path in produced:
        _print(f"wrote {path}")
    return 0


# ----------------------------------------------------------------- replay
def cmd_replay(args: argparse.Namespace) -> int:
    from armlab.visualization.replay import load_trajectory, render_gif

    trajectory = load_trajectory(args.run, args.episode)
    _print(
        f"episode {trajectory.episode_index}: steps={trajectory.steps} "
        f"min_error={trajectory.min_error:.4f} final_error={trajectory.final_error:.4f}"
    )
    if args.gif:
        path = render_gif(args.run, args.episode)
        if path is None:
            _print("GIF rendering unavailable (install Pillow); NPZ trajectory remains available.")
            return 1
        _print(f"wrote {path}")
    return 0


# ------------------------------------------------------------- dashboard
def cmd_dashboard(args: argparse.Namespace) -> int:
    from armlab.dashboard.app import UI_AVAILABLE
    from armlab.dashboard.app import main as dashboard_main

    if not UI_AVAILABLE:
        _print("Dashboard requires dash and plotly. Install with: pip install -e '.[ui]'")
        return 3
    dashboard_main(results_root=args.results, port=args.port)
    return 0


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="armlab",
        description="ArmLab: reproducible robotics control, simulation, and RL benchmark.",
    )
    parser.add_argument("--version", action="version", version="armlab 0.2.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="report environment and optional-dependency status")
    p_doctor.set_defaults(func=cmd_doctor)

    p_demo = sub.add_parser("demo", help="run a short demo")
    demo_sub = p_demo.add_subparsers(dest="demo", required=True)
    p_demo_ik = demo_sub.add_parser("ik", help="analytical IK demo")
    p_demo_ik.add_argument("--robot-config", default=None)
    p_demo_ik.add_argument("--trials", type=int, default=5)
    p_demo_ik.add_argument("--seed", type=int, default=0)
    p_demo_ik.add_argument("--headless", action="store_true")
    p_demo_ik.set_defaults(func=cmd_demo_ik)
    p_demo_phys = demo_sub.add_parser("physics", help="MuJoCo demo (requires physics extra)")
    p_demo_phys.add_argument("--robot-config", default=None)
    p_demo_phys.add_argument("--steps", type=int, default=50)
    p_demo_phys.add_argument("--seed", type=int, default=0)
    p_demo_phys.add_argument("--headless", action="store_true")
    p_demo_phys.set_defaults(func=cmd_demo_physics)

    p_train = sub.add_parser("train", help="train an RL agent")
    train_sub = p_train.add_subparsers(dest="algo", required=True)
    p_dqn = train_sub.add_parser("dqn", help="train Dueling Double DQN")
    p_dqn.add_argument("--config", default=DEFAULT_DQN_CONFIG)
    p_dqn.add_argument("--env-id", default=None)
    p_dqn.add_argument("--seed", type=int, default=None)
    p_dqn.add_argument("--episodes", type=int, default=None)
    p_dqn.add_argument("--max-steps", type=int, default=None)
    p_dqn.add_argument("--eval-every", type=int, default=None)
    p_dqn.add_argument("--eval-seed", type=int, default=10_000)
    p_dqn.add_argument("--output-dir", default=None)
    p_dqn.set_defaults(func=cmd_train_dqn)
    p_sac = train_sub.add_parser("sac", help="train SAC (requires rl extra)")
    p_sac.add_argument("--config", default=DEFAULT_SAC_CONFIG)
    p_sac.add_argument("--env-id", default=None)
    p_sac.add_argument("--seed", type=int, default=None)
    p_sac.add_argument("--total-timesteps", type=int, default=None)
    p_sac.add_argument("--eval-seed", type=int, default=100_000)
    p_sac.add_argument("--output-dir", default=None)
    p_sac.set_defaults(func=cmd_train_sac)

    p_eval = sub.add_parser("evaluate", help="evaluate a controller or policy")
    p_eval.add_argument("--scenario", required=True)
    p_eval.add_argument("--controller", required=True)
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--episodes", type=int, default=20)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--base-seed", type=int, default=100_000)
    p_eval.add_argument("--output-dir", default="results")
    p_eval.add_argument("--robot-config", default=None)
    p_eval.add_argument(
        "--final", action="store_true", help="label as a full benchmark rather than smoke"
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_bench = sub.add_parser("benchmark", help="run a benchmark suite")
    p_bench.add_argument("--suite", default="default")
    p_bench.add_argument("--config", default=DEFAULT_BENCHMARK_CONFIG)
    p_bench.add_argument("--output-dir", default=None)
    p_bench.add_argument("--robot-config", default=None)
    p_bench.set_defaults(func=cmd_benchmark)

    p_report = sub.add_parser("report", help="generate plots from saved artifacts")
    p_report.add_argument("--run", default=None)
    p_report.add_argument("--results", default=None)
    p_report.add_argument("--train-dir", default=None)
    p_report.set_defaults(func=cmd_report)

    p_replay = sub.add_parser("replay", help="replay a saved trajectory")
    p_replay.add_argument("--run", required=True)
    p_replay.add_argument("--episode", type=int, default=0)
    p_replay.add_argument("--gif", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    p_dash = sub.add_parser("dashboard", help="launch the optional dashboard")
    p_dash.add_argument("--results", default="results")
    p_dash.add_argument("--port", type=int, default=8050)
    p_dash.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        _print("interrupted")
        return 130
    except Exception as exc:
        _print(f"error: {exc}")
        if "--debug" in (argv or sys.argv):
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
