import numpy as np 
import gymnasium as gym 
from algorithm import SimulatedAnnealingPolicyOptimizer

# Run SA on a discrete RL environment
env = gym.make("CartPole-v1")  # Changed to CartPole (FrozenLake needs tabular methods)
agent = SimulatedAnnealingPolicyOptimizer(env)
optimal_policy, policy_performance = agent.run()

print("Optimal Policy performance :")
print(policy_performance)