# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DuelingQNetwork(nn.Module):
    """
    The Dueling DQN neural network architecture.
    It has two streams: one for the state value and one for the action advantages.
    """
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        
        # Common feature learning layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),      # Increased from 256
            nn.Linear(256, 256), nn.ReLU()          # Increased from 256
        )
        # You would also need to adjust the input size of the subsequent layers
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),         # Adjusted input, increased hidden layer
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),         # Adjusted input, increased hidden layer
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        """Defines the forward pass of the network."""
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage streams to get final Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class ReplayBuffer:
    """A memory buffer to store and sample past experiences."""
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """The main agent class that interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # --- Q-Networks: Local and Target ---
        self.qnetwork_local = DuelingQNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)

        self.memory = ReplayBuffer(buffer_size=100000, batch_size=128, device=self.device)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        # Slower decay to encourage more exploration over time
        self.epsilon_decay = 0.999
        
        self.learn_every = 4
        self.step_count = 0
        self.tau = 1e-3 # For soft updating the target network

    def choose_action(self, state):
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval() # Set network to evaluation mode
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train() # Set network back to training mode
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        if len(self.memory) < self.memory.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        # --- Double DQN Implementation ---
        # 1. Get the best action from the LOCAL network
        best_actions_next = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # 2. Get the Q-value for that action from the TARGET network
        q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions_next)
        
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = nn.MSELoss()(q_expected, q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Soft update target network ---
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.step_count = (self.step_count + 1) % self.learn_every
        if self.step_count == 0:
            if len(self.memory) > self.memory.batch_size:
                self.learn()
