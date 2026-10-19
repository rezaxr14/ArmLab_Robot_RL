# Dueling Double DQN demo

The original reinforcement-learning demo. `agent.py` defines the Dueling
Double DQN (local/target networks, replay buffer), `robot_env.py` wraps the
3-link arm as a Gymnasium environment, and `train.py` / `test.py` train and
render the policy.

```bash
cd projects/rl_double_dueling_dqn
python train.py     # writes qnetwork.pth
python test.py      # loads qnetwork.pth and renders the policy
```

The demo uses `torch` and a CUDA device when available. The integrated,
checkpointed version with evaluation artifacts is `armlab.rl.dqn`; see the
[top-level README](../../README.md) for the GPU training results.
