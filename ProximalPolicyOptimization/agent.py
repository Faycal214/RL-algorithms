import numpy as np
import torch
from memory import PPOMemory
from networks import Policy_network, ValueNetwork

class PPOAgent :
    def __init__(self, input_dims, n_actions, gamma = 0.99, alpha = 1e-3, gae_lambda = 0.95,
                 policy_clip = 0.3, batch_size = 32, N = 2048, n_epochs = 10) :
        
        self.gamma = gamma
        self.policy_clip = policy_clip 
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.input_dims = input_dims
        
        self.policy_net = Policy_network(n_actions, self.input_dims, alpha)
        self.value_net = ValueNetwork(self.input_dims, alpha)
        self.memory = PPOMemory(batch_size)
    
    def remember(self, state, action, probs, vals, reward, done) :
        self.memory.store_memory(state, action, reward, probs, vals, done)
    
    def choose_action(self, observation) :
        state = torch.tensor(observation, dtype = torch.float)
        
        dist = self.policy_net(state)
        value = self.value_net(state)
        action = dist.sample()
        
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        
        return action, probs, value
    
    def learn(self) :
        for _ in range(self.n_epochs) :
            state_arr, action_arr, old_probs_arr, vals_arr,\
                reward_arr, done_arr, batches = \
                    self.memory.generate_batchs()
            
            values = vals_arr
            advantages = np.zeros(len(reward_arr), dtype = np.float32)
            
            for t in range(len(reward_arr) - 1) :
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr )- 1) :
                    a_t += discount * (reward_arr[k] + self.gamma*values[k+1])*\
                        (1 - int(done_arr[k]) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantages[t] = a_t
            advantages = torch.tensor(advantages)
            
            values = torch.tensor(values)
            for batch in batches :
                states = torch.tensor(state_arr[batch], dtype = torch.float)
                old_probs = torch.tensor(old_probs_arr[batch])
                actions = torch.tensor(action_arr[batch])
                
                dist = self.policy_net.forward(states)
                value_net_value = self.value_net.forward(states)
                value_net_value = torch.squeeze(value_net_value)
                
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantages[batch]
                policy_net_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantages[batch] + values[batch]
                value_net_loss = (returns - value_net_value)**2
                value_net_loss = value_net_loss.mean()
                
                total_loss = policy_net_loss + 0.5*value_net_loss
                self.policy_net.optimizer.zero_grad()
                self.value_net.optimizer.zero_grad()
                total_loss.backward()
                self.policy_net.optimizer.step()
                self.value_net.optimizer.step()
        
        self.memory.clear_memory()
        
