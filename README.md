# 3D Robotic Arm Control: From Inverse Kinematics to Reinforcement Learning

This project provides a comprehensive simulation and control suite for a 3-link robotic arm in a 3D environment. It explores two distinct methods for controlling the robot to reach a target: a precise **Analytical Inverse Kinematics (IK)** solver and a modern **Deep Q-Network (DQN) Reinforcement Learning (RL)** agent.

---

### 🎬 Project Demonstration

Watch the robot arm in action! The video below showcases both the analytical solver tracking a moving target and the trained reinforcement learning agent successfully reaching its goals.

[![RL Robot Arm Demo](https://github.com/rezaxr14/3D-Robot-Control-RL-Analytical/blob/main/Screenshots/inv_kin_Figure_2.png)](https://youtu.be/OU_E-vsCCfU "RL Robot Arm Demo")

---

### 📸 Screenshots

| Analytical IK in Action | RL Agent Reaching Target |
| :---------------------: | :----------------------: |
| ![IK Tracking](https://github.com/rezaxr14/3D-Robot-Control-RL-Analytical/blob/main/Screenshots/inv_kin_Figure_2.png) | ![RL Success](https://github.com/rezaxr14/3D-Robot-Control-RL-Analytical/blob/main/Screenshots/RL_Figure_1.png) |
| Analytical IK in Action | RL Agent Reaching Target |
| :--------------------------: | :---------------------: |
| ![Starting Game Background 1](https://github.com/rezaxr14/3D-Robot-Control-RL-Analytical/blob/main/Screenshots/inv_kin_Figure_1.png) | ![RL Agent Reaching Target](https://github.com/rezaxr14/3D-Robot-Control-RL-Analytical/blob/main/Screenshots/RL_Figure_4.png) |

---

## ✨ Features

This project is divided into two main parts, each demonstrating a different control paradigm.

### Part 1: Analytical Inverse Kinematics
- **`robot.py`**: Defines the robot's physical properties and contains the core mathematical solvers.
- **`forward_kinematics`**: Calculates the end-effector's 3D position from given joint angles.
- **`inverse_kinematics`**: Analytically calculates the required joint angles to reach a specific `(x, y, z)` target.
  - **Solution Selection**: Intelligently calculates both "elbow up" and "elbow down" solutions and chooses the one that requires the smallest change, preventing unnatural "flipping."
  - **Singularity Handling**: Gracefully manages the singularity at the robot's base (`z-axis`), preventing the arm from "resetting" when the target is directly overhead.
- **`visualizer.py`**: A clean `matplotlib`-based 3D visualizer to plot the robot's movement in real-time.
- **`main.py`**: A controller script that demonstrates the IK solver by having the arm smoothly track a moving target.

### Part 2: Reinforcement Learning (Deep Q-Network)
- **`robot_env.py`**: A custom environment built following the `Gymnasium` (formerly OpenAI Gym) API standard. It defines the state space, action space, and reward function.
- **`agent.py`**: Implements a **Deep Q-Network (DQN)** agent using PyTorch.
  - **Q-Network**: A neural network that learns to predict the expected long-term reward for each action.
  - **Replay Buffer**: Stores past experiences to stabilize and improve learning.
  - **Epsilon-Greedy Policy**: Balances exploration (trying new things) and exploitation (using known good actions).
- **`train.py`**: The main training script that allows the agent to interact with the environment for thousands of episodes to learn a control policy.
- **`test.py`**: A script to load the trained agent and watch it perform the task.
- **GPU Acceleration**: Automatically detects and utilizes a CUDA-enabled GPU to dramatically speed up the training process.

---

## 🛠️ Technologies Used

- **Python 3**
- **NumPy**: For numerical operations and vector math.
- **Matplotlib**: For 3D visualization of the robot and environment.
- **PyTorch**: For building and training the Deep Q-Network.
- **Gymnasium**: For providing the standardized environment structure for reinforcement learning.

---

## 🚀 Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/rezaxr14/3D-Robot-Control-RL-Analytical.git](https://github.com/rezaxr14/3D-Robot-Control-RL-Analytical.git)
    cd 3D-Robot-Control-RL-Analytical
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install numpy matplotlib torch gymnasium
    ```
    *(**Note:** If you have a CUDA-enabled GPU, make sure to install the appropriate version of PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/)).*

---

## 💡 How to Use

### Running the Inverse Kinematics Demo

To see the analytical solver in action, navigate to the `Analytical_Inverse_Kinematics` directory and run `main.py`.

```bash
cd Analytical_Inverse_Kinematics
python main.py
```
A `matplotlib` window will appear showing the robot arm smoothly tracking a moving red star.

### Training the Reinforcement Learning Agent

To train the RL agent, navigate to the `RL_Double_Dueling_DQN` directory and run the `train.py` script.

```bash
cd RL_Double_Dueling_DQN
python train.py
```
- Training will begin and run significantly faster if a GPU is detected.
- Progress will be printed to the console.
- Once complete, the trained model will be saved as `qnetwork.pth`, and a plot of the training performance will be displayed.

### Testing the Trained Agent

After training is complete, you can watch your smart agent in action by running the `test.py` script.

```bash
python test.py
```
A `matplotlib` window will appear, and you will see the agent efficiently guiding the robot arm to the target in each new episode.

---

## 🔮 Future Improvements That Can Be Made

- **Continuous Action Space**: Implement a more advanced RL agent (like DDPG or SAC) that can output continuous joint angles instead of discrete steps.
- **Obstacle Avoidance**: Add obstacles to the environment and modify the reward function to teach the agent to avoid collisions.
- **Joint Limits**: Add physical constraints to the robot's joints to make the simulation more realistic.
- **Hyperparameter Tuning**: Experiment with different neural network architectures, learning rates, and other hyperparameters to optimize training performance.
