"""Tests for the scenario framework (Phase F)."""

import numpy as np
import pytest

from armlab.scenarios import list_scenarios, make_scenario


class TestScenarioRegistry:
    def test_all_scenarios_registered(self):
        names = list_scenarios()
        assert names == ["boundary_reaching", "collision_avoidance",
                         "moving_target", "static_reaching"]

    def test_unknown_scenario(self):
        with pytest.raises(KeyError):
            make_scenario("teleportation")


class TestStaticReaching:
    def test_builds_and_runs(self):
        bundle = make_scenario("static_reaching", seed=1)
        assert bundle.name == "static_reaching"
        obs, info = bundle.env.reset(seed=1)
        for _ in range(10):
            obs, r, term, trunc, info = bundle.env.step(0)
        assert np.isfinite(r)

    def test_reproducible(self):
        def rollout():
            b = make_scenario("static_reaching", seed=5)
            b.env.reset(seed=5)
            return [b.env.step(a)[0].copy() for a in [0, 2, 4, 1]]

        assert all(np.allclose(a, b) for a, b in zip(rollout(), rollout()))


class TestBoundaryReaching:
    def test_targets_near_boundary(self):
        b = make_scenario("boundary_reaching", seed=2)
        kin = b.env.unwrapped.kin
        shoulder = np.array([0.0, 0.0, b.env.unwrapped.robot_params.base_height])
        for seed in range(10):
            _, info = b.env.reset(seed=seed)
            dist = np.linalg.norm(info["target_position"] - shoulder)
            # Near the boundary: between 70% and 100% of max reach.
            assert 0.7 * kin.l2 + kin.l3 * 0.7 <= dist <= kin.l2 + kin.l3 + 1e-9


class TestMovingTarget:
    def test_reproducible_trajectory(self):
        def rollout():
            b = make_scenario("moving_target", seed=3)
            b.env.reset(seed=3)
            return [b.env.step(a)[0].copy() for a in [0, 2, 4]]

        assert all(np.allclose(a, b) for a, b in zip(rollout(), rollout()))

    def test_no_early_success_termination(self):
        b = make_scenario("moving_target", seed=4)
        b.env.reset(seed=4)
        # Success should not terminate in moving_target mode.
        assert b.env.unwrapped.reward_params.success_terminates is False


class TestCollisionAvoidance:
    def test_has_obstacles(self):
        b = make_scenario("collision_avoidance", seed=6)
        assert len(b.obstacles) == 2
        assert b.metadata["link_radius"] == 0.3

    def test_env_runs(self):
        b = make_scenario("collision_avoidance", seed=6)
        b.env.reset(seed=6)
        obs, r, term, trunc, info = b.env.step(0)
        assert np.isfinite(r)
