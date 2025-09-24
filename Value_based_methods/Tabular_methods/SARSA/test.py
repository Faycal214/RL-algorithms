import numpy as np
import gymnasium as gym
from agent import SARSAagent

# define the parameters 
alpha = 0.1 # learning rate
epsilon = 1
gamma = 0.5 # discount factor
episodes = 100000 # nomber of episodes
env = gym.make('CliffWalking-v1', render_mode= 'rgb_array')
agent = SARSAagent(env = env, alpha = alpha, gamma = gamma, epsilon = epsilon, episodes= episodes)

agent.training()