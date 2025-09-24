import numpy as np
import gymnasium as gym
from agent import VPGAgent

env = gym.make("CartPole-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = VPGAgent(state_dim, action_dim)
max_episodes = 1000

returns = []
for episode in range(max_episodes) :
    state, _ = env.reset()
    rewards = 0
    done = False 
    while not done :
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.memory.store(state, action, reward, done, log_prob, value)
        rewards += reward
        state = next_state
    agent.update()
    returns.append(rewards)
    print(f"Episode {episode + 1}: Total Reward = {rewards}")