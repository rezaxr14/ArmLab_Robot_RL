# train.py
import torch
from collections import deque
import numpy as np
import time
import os
from robot_env import RobotEnv
from agent import DQNAgent

def train(agent, env, n_episodes=10000, max_t=1000):
    """The main training loop for the DQN agent."""
    scores = []
    scores_window = deque(maxlen=100)
    
    print(f"Starting training on {agent.device}...")
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if env.render_mode == "human":
                time.sleep(0.02)

            if done:
                break
                
        scores_window.append(score)
        scores.append(score)
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            
    # Always save the model after training completes
    torch.save(agent.qnetwork_local.state_dict(), 'qnetwork.pth')
    print(f"\nTraining complete. Model saved to qnetwork.pth")
    return scores

if __name__ == '__main__':
    # --- Detect Device and Initialize ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env = RobotEnv(render_mode=None) 
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, device=device)

    # --- Load existing model if it exists ---
    model_path = 'qnetwork.pth'
    if os.path.exists(model_path):
        try:
            agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
            # When continuing training, you might want to adjust the exploration rate
            # For example, set it to a lower value if the model is already well-trained
            agent.epsilon = 0.1
            print("Existing model loaded. Continuing training.")
        except Exception as e:
            print(f"Could not load model. Starting from scratch. Error: {e}")
    else:
        print("No existing model found. Starting training from scratch.")


    scores = train(agent, env)

    env.close()
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Training Performance')
    plt.show()
