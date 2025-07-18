import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from memory import VPGMemory
from policy import Policy_network, value_network

class VPGAgent :
    def __init__(self, state_dim, action_dim, learning_rate = 0.001 , gamma = 0.99):
        self.policy = Policy_network(state_dim, action_dim)
        self.value = value_network(state_dim)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=learning_rate)
        self.memory = VPGMemory()
        self.gamma = gamma
    
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(state_tensor)
        return action.item(), log_prob, value
    
    def update(self):
        states, actions, rewards, dones, log_probs, _ = self.memory.get()
    
        # Compute returns
        n = len(rewards)
        returns = torch.zeros(n, dtype=torch.float32)
        R = 0
        for i in reversed(range(n)):
            if dones[i]:
                R = 0
            R = rewards[i] + self.gamma * R
            returns[i] = R

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Recompute value predictions for states
        values = self.value(states).squeeze()

        # Compute advantage
        advantages = returns - values.detach()

        # Update policy
        policy_loss = -(log_probs * advantages).mean()
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # Update value network
        value_loss = nn.functional.mse_loss(values, returns)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # Clear memory
        self.memory.clear()
