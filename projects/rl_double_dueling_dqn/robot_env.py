# robot_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from visualizer import RobotVisualizer 

class RobotEnv(gym.Env):
    """
    A custom Reinforcement Learning Environment for a 3-link robot arm.
    """
    metadata = {'render_modes': ['human'], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(RobotEnv, self).__init__()
        self.link_lengths = np.array([5, 10, 8])
        self.thetas = np.zeros(3)
        self.action_space = spaces.Discrete(6)
        self.action_step_size = np.deg2rad(5)

        # --- NEW: Observation space now includes end-effector position ---
        # State: [theta1, theta2, theta3, ee_x, ee_y, ee_z, target_x, target_y, target_z]
        # Total of 9 values.
        max_reach = sum(self.link_lengths)
        low_obs = np.array([-np.pi]*3 + [-max_reach]*3 + [-max_reach]*3)
        high_obs = np.array([np.pi]*3 + [max_reach]*3 + [max_reach]*3)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.target_pos = self._generate_random_target()
        self.render_mode = render_mode
        self.visualizer = None
        if self.render_mode == "human":
            self.visualizer = RobotVisualizer(self.link_lengths)
        
        self.max_episode_steps = 250
        self.current_step = 0

    def _forward_kinematics(self, thetas):
        l1, l2, l3 = self.link_lengths
        theta1, theta2, theta3 = thetas
        x = np.cos(theta1) * (l2 * np.sin(theta2) + l3 * np.sin(theta2 + theta3))
        y = np.sin(theta1) * (l2 * np.sin(theta2) + l3 * np.sin(theta2 + theta3))
        z = l1 + l2 * np.cos(theta2) + l3 * np.cos(theta2 + theta3)
        return np.array([x, y, z])

    def _generate_random_target(self):
        radius = np.random.uniform(5, 15)
        angle = np.random.uniform(0, 2 * np.pi)
        height = np.random.uniform(5, 20)
        return np.array([radius * np.cos(angle), radius * np.sin(angle), height])

    def _get_obs(self):
        """Calculates the observation, now including the end-effector position."""
        end_effector_pos = self._forward_kinematics(self.thetas)
        return np.concatenate([self.thetas, end_effector_pos, self.target_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.thetas = np.zeros(3)
        self.target_pos = self._generate_random_target()
        self.current_step = 0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        joint_to_move = action // 2
        direction = 1 if action % 2 == 0 else -1
        
        self.thetas[joint_to_move] += direction * self.action_step_size
        
        new_end_effector_pos = self._forward_kinematics(self.thetas)
        distance_to_target = np.linalg.norm(new_end_effector_pos - self.target_pos)
        
        # A strong, direct reward based on the inverse of the distance.
        reward = -distance_to_target * 0.1

        terminated = False
        if distance_to_target < 1.0:
            # Huge bonus for reaching the goal
            reward += 100
            terminated = True
            
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human" and self.visualizer:
            l1, l2, l3 = self.link_lengths
            t1, t2, t3 = self.thetas
            p0 = np.array([0,0,0])
            p1 = np.array([0,0,l1])
            p2 = np.array([l2*np.cos(t1)*np.sin(t2), l2*np.sin(t1)*np.sin(t2), l1+l2*np.cos(t2)])
            p3 = self._forward_kinematics(self.thetas)
            joint_positions = [p0, p1, p2, p3]
            self.visualizer.plot_robot(joint_positions, self.target_pos)

    def close(self):
        if self.visualizer:
            self.visualizer.close()
