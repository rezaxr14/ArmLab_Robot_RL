"""End-to-end DQN training smoke test (CPU only)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from armlab.config import EnvParams, RLConfig
from armlab.envs import KinematicReachingEnv
from armlab.rl import DQNAgent
from armlab.rl.training import train_dqn


def _tiny_agent():
    return DQNAgent(
        state_dim=9,
        action_dim=6,
        config=RLConfig(
            batch_size=8, min_buffer_samples=8, learn_every=1, buffer_size=500, epsilon_decay=0.99
        ),
        seed=0,
    )


class TestTrainingLoop:
    def test_smoke_run_writes_metrics_and_checkpoints(self, tmp_path):
        env = KinematicReachingEnv(env_params=EnvParams(max_episode_steps=25))
        eval_env = KinematicReachingEnv(env_params=EnvParams(max_episode_steps=25))
        agent = _tiny_agent()
        ckpt = tmp_path / "latest.pt"
        result = train_dqn(
            agent,
            env,
            n_episodes=4,
            max_steps_per_episode=25,
            eval_every=2,
            eval_episodes=2,
            eval_env=eval_env,
            eval_seed=500,
            checkpoint_path=str(ckpt),
            output_dir=str(tmp_path),
            seed=1,
            log_fn=lambda _: None,
        )
        env.close()
        eval_env.close()

        assert result.total_env_steps > 0
        assert len(result.episode_scores) == 4
        assert ckpt.is_file()
        assert (tmp_path / "training_metrics.json").is_file()
        assert (tmp_path / "episodes.csv").is_file()
        assert (tmp_path / "evaluations.csv").is_file()

        payload = json.loads((tmp_path / "training_metrics.json").read_text(encoding="utf-8"))
        assert payload["seed"] == 1
        assert payload["metrics"]["total_env_steps"] == result.total_env_steps

    def test_checkpoint_reloads_in_fresh_agent(self, tmp_path):
        agent = _tiny_agent()
        rng = np.random.default_rng(0)
        for i in range(6):
            agent.step(
                rng.standard_normal(9).astype(np.float32),
                i % 6,
                1.0,
                rng.standard_normal(9).astype(np.float32),
                False,
                False,
            )
        path = tmp_path / "ckpt.pt"
        agent.save_checkpoint(path)

        fresh = _tiny_agent()
        fresh.load_checkpoint(path)
        assert fresh.env_steps == agent.env_steps
        state = np.zeros(9, dtype=np.float32)
        assert fresh.choose_action(state, greedy=True) == agent.choose_action(state, greedy=True)
