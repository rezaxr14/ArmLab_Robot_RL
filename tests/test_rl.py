"""Learning-stack tests (need `torch`; the trainer smoke test also needs `mujoco`)."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from armforge.config import AlgoConfig, RunConfig, load_config  # noqa: E402
from armforge.rl.agent import SACAgent  # noqa: E402
from armforge.rl.buffer import EpisodeBuffer  # noqa: E402
from armforge.rl.networks import QuantileCritic, TanhGaussianActor  # noqa: E402
from armforge.rl.normalizer import RunningNorm  # noqa: E402

DEV = torch.device("cpu")


def tiny_algo(**kw):
    base = dict(hidden=(32, 32), critic_ensemble=3, n_quantiles=7, top_quantiles_to_drop=1, batch_size=32)
    base.update(kw)
    return AlgoConfig(**base)


def fake_buffer(n=20, T=6, od=5, gd=3, ad=2):
    buf = EpisodeBuffer(64, T, od, gd, ad, DEV, 0.05, "sparse")
    rng = np.random.default_rng(0)
    ag = rng.normal(size=(n, T + 1, gd)).astype(np.float32)
    buf.add(rng.normal(size=(n, T + 1, od)), ag, ag[:, -1] + 5.0, rng.uniform(-1, 1, (n, T, ad)), np.zeros((n, T)))
    return buf, ag


def test_running_norm_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, (500, 4)).astype(np.float32)
    rn = RunningNorm(4, DEV)
    for chunk in np.array_split(data, 7):
        rn.update(chunk)
    assert np.allclose(rn.mean.numpy(), data.mean(0), atol=1e-3) and np.allclose(rn.var.numpy(), data.var(0), rtol=1e-2)
    assert rn(torch.tensor(data)).abs().max() <= 5.0


def test_buffer_relabelling_uses_future_achieved_goals():
    buf, ag = fake_buffer()
    torch.manual_seed(0)
    b = buf.sample(2000, her_prob=1.0)
    assert b["relabelled"].all()
    # every relabelled goal must equal the achieved goal at some step of some episode
    flat = torch.from_numpy(ag.reshape(-1, 3))
    d = torch.cdist(b["goal"], flat).min(1).values
    assert d.max() < 1e-5
    # reward is -1 unless the *next* achieved goal is within the threshold of the (relabelled) goal
    assert set(torch.unique(b["reward"]).tolist()) <= {-1.0, 0.0}
    assert (b["reward"] == 0).any(), "hindsight goals must produce some successes"


def test_no_relabel_keeps_original_goal_and_rewards_failures():
    buf, _ = fake_buffer()
    b = buf.sample(500, her_prob=0.0)
    assert not b["relabelled"].any() and (b["reward"] == -1).all()  # goals were put 5 units away


def test_buffer_future_index_never_leaves_episode():
    buf, _ = fake_buffer(T=3)
    for _ in range(50):
        b = buf.sample(256, her_prob=1.0)
        assert torch.isfinite(b["goal"]).all()


def test_tanh_logprob_matches_change_of_variables():
    torch.manual_seed(0)
    actor = TanhGaussianActor(6, 3, (16,), False)
    s = torch.randn(64, 6)
    a, logp, mu = actor(s)
    assert a.abs().max() <= 1.0 and logp.shape == (64,) and torch.isfinite(logp).all()
    h = actor.body(s)
    log_std = actor.log_std(h).clamp(-5, 2)
    u = torch.atanh(a.clamp(-0.999999, 0.999999))
    base = torch.distributions.Normal(actor.mu(h), log_std.exp()).log_prob(u).sum(-1)
    jac = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
    assert torch.allclose(logp, base - jac, atol=5e-2)


def test_critic_ensemble_members_are_independent_and_shaped():
    c = QuantileCritic(8, 2, (16, 16), ensemble=4, n_quantiles=5, layer_norm=True)
    q = c(torch.randn(10, 8), torch.rand(10, 2))
    assert q.shape == (10, 4, 5)
    assert not torch.allclose(q[:, 0], q[:, 1])  # different initialisations


def test_quantile_huber_is_zero_at_target_and_grows_with_error():
    agent = SACAgent(5, 3, 2, tiny_algo(), DEV)
    t = torch.zeros(8, 15)
    assert agent._quantile_huber(torch.zeros(8, 3, 7), t) == 0
    assert agent._quantile_huber(torch.full((8, 3, 7), 2.0), t) > agent._quantile_huber(torch.full((8, 3, 7), 0.5), t) > 0


def test_twin_critic_target_is_the_pairwise_minimum():
    agent = SACAgent(5, 3, 2, tiny_algo(critic="twin"), DEV)
    assert (agent.n_critics, agent.n_quantiles, agent.drop_total) == (2, 1, 1)
    z = torch.randn(16, 2)
    kept = torch.sort(z, dim=1).values[:, : 2 - agent.drop_total]
    assert torch.equal(kept.squeeze(1), z.min(1).values)


def test_update_runs_changes_weights_and_stays_finite():
    torch.manual_seed(0)
    buf, _ = fake_buffer()
    agent = SACAgent(5, 3, 2, tiny_algo(), DEV, return_range=(-50.0, 0.0))
    before = [p.detach().clone() for p in agent.actor.parameters()]
    alpha0 = agent.alpha.item()
    for _ in range(5):
        m = agent.update(buf.sample(32, 0.8))
    assert all(math.isfinite(v) for v in m.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert agent.alpha.item() != alpha0 and agent.updates == 5


def test_critic_learns_a_known_value():
    """With a constant reward of -1 and gamma=0 every quantile must converge to -1."""
    torch.manual_seed(0)
    cfg = tiny_algo(gamma=0.0, critic_lr=3e-3, layer_norm=False)
    agent = SACAgent(5, 3, 2, cfg, DEV, return_range=None)
    batch = {"obs": torch.randn(64, 5), "act": torch.rand(64, 2) * 2 - 1, "reward": -torch.ones(64), "next_obs": torch.randn(64, 5), "goal": torch.randn(64, 3)}
    for _ in range(400):
        agent.update(batch)
    q = agent.critic(agent._state(batch["obs"], batch["goal"]), batch["act"])
    assert (q.mean() + 1.0).abs() < 0.15


def test_bc_update_fits_a_simple_expert():
    torch.manual_seed(0)
    agent = SACAgent(4, 3, 2, tiny_algo(actor_lr=3e-3), DEV)
    obs, goal = torch.randn(256, 4), torch.randn(256, 3)
    target = torch.tanh(obs[:, :2])
    first = agent.bc_update(obs, goal, target)
    for _ in range(300):
        last = agent.bc_update(obs, goal, target)
    assert last < 0.3 * first


def test_policy_roundtrip(tmp_path):
    torch.manual_seed(0)
    a = SACAgent(5, 3, 2, tiny_algo(), DEV)
    a.obs_norm.update(torch.randn(100, 5))
    a.save(tmp_path / "p.pt", policy_only=True)
    b = SACAgent(5, 3, 2, tiny_algo(), DEV)
    b.load_policy_state(torch.load(tmp_path / "p.pt"))
    o, g = np.random.randn(5).astype(np.float32), np.random.randn(3).astype(np.float32)
    assert np.allclose(a.act(o, g, deterministic=True), b.act(o, g, deterministic=True), atol=1e-6)


def test_invalid_drop_is_rejected():
    with pytest.raises(ValueError):
        SACAgent(5, 3, 2, tiny_algo(top_quantiles_to_drop=7), DEV)


def test_milestone_steps():
    from armforge.rl.trainer import milestone_steps

    assert milestone_steps((0.0, 0.1, 1.0, 5000), 10_000) == [0, 1000, 5000, 10_000]


# ---- end to end (simulator) ----------------------------------------------------------
def test_trainer_smoke_run_writes_everything(tmp_path, menagerie, monkeypatch):
    pytest.importorskip("mujoco")
    monkeypatch.setenv("ARMFORGE_MENAGERIE", str(menagerie))
    from pathlib import Path

    from armforge.rl.trainer import Trainer

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml", {"out_dir": str(tmp_path), "train.total_steps": 400, "train.eval_every": 200, "train.log_every": 100, "train.checkpoint_every": 200, "train.start_steps": 100, "train.warmup_updates_after": 100})
    t = Trainer(cfg)
    summary = t.train()
    t.close()
    run = tmp_path / cfg.name
    for f in ("config.yaml", "metrics.csv", "latest.pt", "best.pt", "final.pt", "summary.json"):
        assert (run / f).exists(), f
    assert len(list((run / "milestones").glob("step_*.pt"))) == 3
    assert summary["total_steps"] >= 400
    # resume continues without error and keeps the step counter
    t2 = Trainer(cfg, resume=True)
    assert t2.steps >= 400
    t2.close()
