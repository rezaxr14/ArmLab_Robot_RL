"""SAC baseline tests (skipped when the rl extra is unavailable)."""

import pytest

pytest.importorskip("torch")
sb3 = pytest.importorskip("stable_baselines3")

from armlab.rl.sac import SAC_AVAILABLE, evaluate_sac, load_sac, train_sac


@pytest.mark.skipif(not SAC_AVAILABLE, reason="stable-baselines3 not installed")
class TestSAC:
    def test_tiny_training_and_reload(self, tmp_path):
        payload = train_sac(
            env_id="ArmLabKinematicContinuous-v0",
            output_dir=tmp_path,
            seed=0,
            total_timesteps=150,
            sac_params={"learning_starts": 20, "batch_size": 32, "net_arch": [32, 32]},
            eval_episodes=2,
            eval_seed=900,
            log_fn=lambda _: None,
        )
        assert payload["smoke"] is True
        assert (tmp_path / "sac_metrics_seed0.json").is_file()
        model = load_sac(payload["model_path"])
        assert model is not None

    def test_evaluate_reports_measured_metrics(self, tmp_path):
        payload = train_sac(
            env_id="ArmLabKinematicContinuous-v0",
            output_dir=tmp_path,
            seed=1,
            total_timesteps=120,
            sac_params={"learning_starts": 20, "batch_size": 32, "net_arch": [32, 32]},
            eval_episodes=2,
            eval_seed=1000,
            log_fn=lambda _: None,
        )
        model = load_sac(payload["model_path"])
        import gymnasium as gym

        from armlab.envs import register_envs

        register_envs()
        env = gym.make("ArmLabKinematicContinuous-v0")
        try:
            metrics = evaluate_sac(model, env, n_episodes=2, base_seed=2000)
        finally:
            env.close()
        assert metrics["n_episodes"] == 2
        assert 0.0 <= metrics["success_rate"] <= 1.0
        assert metrics["final_error_mean"] >= 0.0
