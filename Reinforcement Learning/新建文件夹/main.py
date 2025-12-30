import gymnasium as gym
from qlearn import Agent
from utils import plotLearning
import numpy as np
import torch
import pickle
import multiprocessing

# Function to save the Agent instance
def save_agent(agent, filename):
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)

# Function to load the Agent instance
def load_agent(filename):
    with open(filename, 'rb') as f:
        agent = pickle.load(f)
    return agent

def train_agent(env_name, render=False):
    env = gym.make(env_name, render_mode='human' if render else 'rgb_array')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 100

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(np.array(observation))
            observation_, reward, done, _, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
            if render:
                env.render()
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}')

    return scores, eps_history

if __name__ == '__main__':
    env_names = ['LunarLander-v3'] * 3  # Example with 3 environments
    render_modes = [True, False, False]  # Render only the first environment

    processes = []
    for env_name, render in zip(env_names, render_modes):
        process = multiprocessing.Process(target=train_agent, args=(env_name, render))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
