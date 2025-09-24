import torch

class VPGMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(torch.tensor(value, dtype = torch.float32))

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get(self):
        return (
            torch.stack(self.states),
            torch.tensor(self.actions, dtype=torch.long),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.bool),
            torch.stack(self.log_probs),
            torch.stack(self.values)
        )
