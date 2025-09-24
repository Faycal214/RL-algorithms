import numpy as np
import sys
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

sys.path.append("/home/billel/Desktop/ai_projects/RL-algorithms/DQN")
from agent import dqn
from env_processing import *

env = gym_super_mario_bros.make('SuperMarioBros2-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
dqn_agent = dqn(state_dim= 84*84, action_dim=len(SIMPLE_MOVEMENT), 
                learning_rate=1e-4, gamma=0.99, epsilon=1.0, batch_size=64, 
                max_memory_size=100000)

returns = []

for episode in range(1000):
    state = env.reset()
    done = False
    score = 0
    while not done:
        processed = process_state(state)
        action = dqn_agent.choose_action(processed)
        next_state, reward, done, info= env.step(action)
        score += reward
        dqn_agent.memory.store_transition(processed, action, reward, process_state(next_state), done)
        dqn_agent.learn()
        state = next_state
        env.render()
    
    returns.append(score)
    print(f"Episode {episode} - Total Rewards: {score} - Average Rewards: {np.mean(returns[-100:])}")

env.close()