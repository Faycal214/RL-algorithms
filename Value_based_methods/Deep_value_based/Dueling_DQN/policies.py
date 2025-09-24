import torch as T
import torch.nn as nn
import torch.nn.functional as F

class DuelingDeepQNetwork(nn.Module) :
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims) :
        super(DuelingDeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.lr = lr
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Value stream
        self.value = nn.Linear(self.fc2_dims, 1)

        # Advantage stream
        self.advantage = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = T.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        
    def forward(self, state) :
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        value = self.value(x)
        advantage = self.advantage(x)

        # Combine value and advantage to get Q-values
        # q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return value, advantage