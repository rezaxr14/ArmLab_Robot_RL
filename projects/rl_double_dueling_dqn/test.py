# test.py
import torch
import time
from robot_env import RobotEnv
from agent import DQNAgent

def test(agent, env, n_episodes=10):
    """
    Tests a trained DQN agent.
    
    Args:
        agent (DQNAgent): The trained agent.
        env (RobotEnv): The environment to test in.
        n_episodes (int): The number of episodes to run for testing.
    """
    print(f"Starting testing on {agent.device}...")
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        while True:
            # Agent chooses the best action based on its learned policy
            action = agent.choose_action(state)
            
            # Action is performed in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            
            # Only sleep if we are rendering for a human to watch
            if env.render_mode == "human":
                # Add a small delay to make the visualization easier to follow
                time.sleep(0.02)
            
            if done:
                break
        
        print(f'Episode {i_episode}\tScore: {score:.2f}')
            
    env.close()
    print("\nTesting complete.")

if __name__ == '__main__':
    # --- Detect Device and Initialize ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = RobotEnv(render_mode="human") 
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # Pass the detected device to the agent
    agent = DQNAgent(state_size=state_size, action_size=action_size, device=device)

    # --- Load the Trained Weights ---
    try:
        # Load the model onto the correct device
        agent.qnetwork_local.load_state_dict(torch.load('qnetwork.pth', map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: 'qnetwork.pth' not found. Please run train.py first.")
        exit()

    # --- Set Agent to Evaluation Mode ---
    agent.qnetwork_local.eval()
    agent.epsilon = 0.0 

    # --- Start Testing ---
    test(agent, env)
