"""Reinforcement learning: Dueling Double DQN (requires torch).

Importing this package raises ImportError with installation guidance when
torch is not available; the rest of armlab (kinematics, controllers,
kinematic env, evaluation) works without it.
"""

from armlab.rl.agent import DQNAgent
from armlab.rl.networks import DuelingQNetwork
from armlab.rl.replay import ReplayBuffer

__all__ = ["DQNAgent", "DuelingQNetwork", "ReplayBuffer"]
