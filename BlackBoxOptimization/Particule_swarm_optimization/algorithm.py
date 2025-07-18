import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import cma

# Define your policy NN
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, action_dim)
    
    def forward(self, x):
        return torch.tanh(self.fc2(self.relu(self.fc1(x))))

    def get_weights(self):
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_weights(self, flat_weights):
        pointer = 0
        for p in self.parameters():
            size = np.prod(p.shape)
            p.data = torch.tensor(flat_weights[pointer:pointer+size].reshape(p.shape), dtype=torch.float32)
            pointer += size

# Evaluation function
def evaluate(policy, env, episodes=3):
    total_reward = 0
    for _ in range(episodes):
        obs = env.reset()[0]
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = policy(obs_tensor).detach().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    return -total_reward  # CMA-ES minimizes

# Setup
env = gym.make("LunarLanderContinuous-v3")
policy = Policy(env.observation_space.shape[0], env.action_space.shape[0])
params_dim = len(policy.get_weights())

# CMA-ES optimizer
es = cma.CMAEvolutionStrategy(params_dim * [0], 0.5)

for gen in range(100):
    solutions = es.ask()
    fitnesses = []
    for s in solutions:
        policy.set_weights(s)
        fitnesses.append(evaluate(policy, env))
    es.tell(solutions, fitnesses)
    es.disp()

# Final result
best = es.result.xbest
policy.set_weights(best)
