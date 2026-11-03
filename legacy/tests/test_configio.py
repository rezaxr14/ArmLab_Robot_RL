"""Tests for YAML configuration loading and validation."""

from pathlib import Path

import pytest

from armlab.config import EnvParams, RewardParams, RobotParams
from armlab.configio import (
    BenchmarkConfig,
    config_hash,
    dump_yaml,
    load_benchmark_config,
    load_env_config,
    load_rl_config,
    load_robot_params,
    load_yaml,
    save_resolved_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


class TestRobotConfig:
    def test_loads_example_config(self):
        params = load_robot_params(CONFIGS / "robot" / "three_link.yaml")
        assert isinstance(params, RobotParams)
        assert params.link_lengths == (5.0, 10.0, 8.0)

    def test_unknown_key_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        dump_yaml({"link_lengths": [1, 2, 3], "nope": 1}, path)
        with pytest.raises(ValueError, match="Unknown key"):
            load_robot_params(path)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_yaml("does-not-exist.yaml")


class TestEnvConfig:
    def test_loads_env_and_reward(self):
        env, reward, meta = load_env_config(CONFIGS / "env" / "reaching.yaml")
        assert isinstance(env, EnvParams)
        assert isinstance(reward, RewardParams)
        assert meta["scenario"] == "static_reaching"
        assert reward.success_threshold == pytest.approx(1.0)

    def test_obstacle_config_metadata(self):
        _, _, meta = load_env_config(CONFIGS / "env" / "obstacle_reaching.yaml")
        assert len(meta["obstacles"]) == 2
        assert meta["link_radius"] == pytest.approx(0.3)


class TestTrainConfig:
    def test_loads_dqn_and_builds_agent_config(self):
        data = load_rl_config(CONFIGS / "train" / "dqn.yaml")
        assert data["seed"] == 7
        assert data["agent"].batch_size == 128

    def test_loads_sac(self):
        data = load_rl_config(CONFIGS / "train" / "sac.yaml")
        assert data["total_timesteps"] > 0


class TestBenchmarkConfig:
    def test_loads_default_suite(self):
        config = load_benchmark_config(CONFIGS / "benchmark" / "default.yaml")
        assert isinstance(config, BenchmarkConfig)
        assert "static_reaching" in config.scenarios
        assert config.smoke is True

    def test_empty_scenarios_rejected(self):
        with pytest.raises(ValueError):
            BenchmarkConfig(scenarios=())


class TestHelpers:
    def test_config_hash_stable(self):
        assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})

    def test_save_resolved_config(self, tmp_path):
        path = save_resolved_config(tmp_path, {"robot": {"link_lengths": [1, 2, 3]}})
        assert path.is_file()
        assert load_yaml(path)["robot"]["link_lengths"] == [1, 2, 3]
