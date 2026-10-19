"""Tests for the corrected Dueling Double DQN agent (no GPU required)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from armlab.config import RLConfig
from armlab.rl import DQNAgent, DuelingQNetwork, ReplayBuffer


@pytest.fixture
def agent():
    return DQNAgent(state_dim=9, action_dim=6, config=RLConfig(batch_size=16), seed=0)


class TestNetwork:
    def test_output_shape(self):
        net = DuelingQNetwork(9, 6, hidden_sizes=(32, 32), stream_hidden_size=16)
        out = net(torch.zeros(4, 9))
        assert out.shape == (4, 6)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            DuelingQNetwork(0, 6)
        with pytest.raises(ValueError):
            DuelingQNetwork(9, 6, hidden_sizes=())


class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(capacity=100, batch_size=8, seed=0)
        for i in range(20):
            buf.add(np.full(9, i, dtype=np.float32), i % 6, 1.0,
                    np.full(9, i + 1, dtype=np.float32), False, False)
        assert len(buf) == 20
        states, actions, rewards, next_states, term, trunc = buf.sample()
        assert states.shape == (8, 9)
        assert actions.shape[0] == 8
        assert term.shape[0] == 8

    def test_capacity_wrap(self):
        buf = ReplayBuffer(capacity=10, batch_size=4, seed=0)
        for i in range(25):
            buf.add(np.zeros(3, dtype=np.float32), 0, 0.0,
                    np.zeros(3, dtype=np.float32), False, False)
        assert len(buf) == 10

    def test_seeded_sampling_reproducible(self):
        def sample_first():
            buf = ReplayBuffer(capacity=50, batch_size=8, seed=42)
            for i in range(30):
                buf.add(np.full(3, i, dtype=np.float32), 0, 0.0,
                        np.zeros(3, dtype=np.float32), False, False)
            return buf.sample()[0].numpy()

        np.testing.assert_allclose(sample_first(), sample_first())

    def test_insufficient_samples(self):
        buf = ReplayBuffer(capacity=10, batch_size=8, seed=0)
        buf.add(np.zeros(3, dtype=np.float32), 0, 0.0,
                np.zeros(3, dtype=np.float32), False, False)
        with pytest.raises(ValueError):
            buf.sample()

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=0, batch_size=4)
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=4, batch_size=8)

class TestAgent:
    def test_target_initialized_equal_to_local(self, agent):
        for tp, lp in zip(agent.qnetwork_target.parameters(),
                          agent.qnetwork_local.parameters()):
            assert torch.equal(tp, lp)

    def test_epsilon_schedule(self):
        cfg = RLConfig(epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.9)
        agent = DQNAgent(9, 6, config=cfg, seed=0)
        assert agent.epsilon == 1.0
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.9)
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon == 0.1  # clamped

    def test_choose_action_single_and_batch(self, agent):
        a = agent.choose_action(np.zeros(9, dtype=np.float32), greedy=True)
        assert isinstance(a, int) and 0 <= a < 6
        batch = agent.choose_action(np.zeros((5, 9), dtype=np.float32), greedy=True)
        assert batch.shape == (5,) and batch.dtype == np.int64

    def test_greedy_deterministic(self, agent):
        s = np.arange(9, dtype=np.float32)
        assert agent.choose_action(s, greedy=True) == agent.choose_action(s, greedy=True)

    def test_invalid_state_shape(self, agent):
        with pytest.raises(ValueError):
            agent.choose_action(np.zeros(5, dtype=np.float32))

    def test_hard_update_copies_weights(self):
        cfg = RLConfig(target_update_mode="hard", target_update_every=2)
        agent = DQNAgent(9, 6, config=cfg, seed=0)
        with torch.no_grad():
            for p in agent.qnetwork_local.parameters():
                p.add_(1.0)
        agent.learn_steps = 2  # multiple of target_update_every
        agent._maybe_update_target()
        for tp, lp in zip(agent.qnetwork_target.parameters(),
                          agent.qnetwork_local.parameters()):
            assert torch.equal(tp, lp)

    def test_soft_update_polyak(self):
        cfg = RLConfig(target_update_mode="soft", tau=0.5)
        agent = DQNAgent(9, 6, config=cfg, seed=0)
        t_before = [p.clone() for p in agent.qnetwork_target.parameters()]
        l_before = [p.clone() for p in agent.qnetwork_local.parameters()]
        with torch.no_grad():
            for p in agent.qnetwork_local.parameters():
                p.add_(1.0)
        agent._maybe_update_target()
        for tp, t0, l1 in zip(agent.qnetwork_target.parameters(), t_before, l_before):
            expected = 0.5 * (l1 + 1.0) + 0.5 * t0  # local was perturbed +1.0
            torch.testing.assert_close(tp, expected)

    def test_learn_updates_without_error(self, agent):
        rng = np.random.default_rng(0)
        for _ in range(32):
            agent.memory.add(rng.standard_normal(9).astype(np.float32),
                             int(rng.integers(0, 6)), float(rng.standard_normal()),
                             rng.standard_normal(9).astype(np.float32), False, False)
        assert np.isfinite(agent.learn())

    def test_learn_separated_terminated_truncated(self, agent):
        rng = np.random.default_rng(1)
        for i in range(32):
            term = i % 2 == 0
            agent.memory.add(rng.standard_normal(9).astype(np.float32), i % 6, 0.0,
                             rng.standard_normal(9).astype(np.float32), term, not term)
        assert np.isfinite(agent.learn())

    def test_gradient_clip_applied(self, monkeypatch):
        # Adam is scale-invariant, so instead of checking parameter deltas we
        # verify clip_grad_norm_ is invoked with the configured max norm.
        cfg = RLConfig(gradient_clip=0.5, batch_size=8)
        agent = DQNAgent(9, 6, config=cfg, seed=0)
        rng = np.random.default_rng(2)
        for _ in range(16):
            agent.memory.add(rng.standard_normal(9).astype(np.float32), 0, 100.0,
                             rng.standard_normal(9).astype(np.float32), False, False)
        recorded = {}
        import torch.nn.utils as tnu

        original = tnu.clip_grad_norm_

        def spy(params, max_norm, **kw):
            recorded["max_norm"] = max_norm
            return original(params, max_norm, **kw)

        monkeypatch.setattr(tnu, "clip_grad_norm_", spy)
        agent.learn()
        assert recorded.get("max_norm") == 0.5

    def test_step_learns_periodically(self):
        cfg = RLConfig(learn_every=4, batch_size=8, min_buffer_samples=8)
        agent = DQNAgent(9, 6, config=cfg, seed=0)
        rng = np.random.default_rng(3)
        for i in range(16):
            agent.step(rng.standard_normal(9).astype(np.float32), i % 6, 1.0,
                       rng.standard_normal(9).astype(np.float32), False, False)
        assert agent.learn_steps >= 1

    def test_checkpoint_round_trip(self, agent, tmp_path):
        rng = np.random.default_rng(4)
        for i in range(10):
            agent.step(rng.standard_normal(9).astype(np.float32), i % 6, 0.5,
                       rng.standard_normal(9).astype(np.float32), False, False)
        path = tmp_path / "ckpt.pt"
        agent.save_checkpoint(path)

        agent2 = DQNAgent(9, 6, config=RLConfig(batch_size=16), seed=99)
        agent2.load_checkpoint(path)
        assert agent2.env_steps == agent.env_steps
        for p1, p2 in zip(agent.qnetwork_local.parameters(),
                          agent2.qnetwork_local.parameters()):
            assert torch.equal(p1, p2)

    def test_checkpoint_dim_mismatch(self, agent, tmp_path):
        path = tmp_path / "ckpt.pt"
        agent.save_checkpoint(path)
        other = DQNAgent(6, 4, seed=0)
        with pytest.raises(ValueError):
            other.load_checkpoint(path)
