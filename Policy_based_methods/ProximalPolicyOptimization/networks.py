import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from memory import PPOMemory

class Policy_network(nn.Module) :
    def __init__(self, n_actions, input_dims, alpha, 
                 fc1_dims = 256, fc2_dims = 256) :
        super(Policy_network, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim = -1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr= alpha)
    
    def forward(self, state) :
        state = torch.tensor(state, dtype = torch.float32)
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class ValueNetwork(nn.Module) :
    def __init__(self, input_dims, alpha, 
                 fc1_dims = 256, fc2_dims = 256) :
        super(ValueNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
    
    def forward(self, state) :
        state = torch.tensor(state, dtype = torch.float32)
        value = self.critic(state)
        return value
    
