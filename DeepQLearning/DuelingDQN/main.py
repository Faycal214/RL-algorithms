import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class ReplayBuffer(object) :
    def __init__(self, max_size, Input_shape, n_actions) :
        self.memory_size = max_size # maximum size of the memory
        self.memory_cntr = 0 # memory counter
        self.state_memory = np.zeros((self.memory_size, *Input_shape),
                                    dtype= np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *Input_shape),
                                    dtype= np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype= np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype = np.uint8)
    
    def store_transition(self, state, action, reward, next_state, done) :
        index = self.memory_cntr % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.memory_cntr += 1
    
    def sample_buffer(self, batch_size) :
        max_memory = min(self.memory_cntr, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace= False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, next_states, dones
    

class DuelingDeepQNetwork(nn.Module) :
    def __init__(self, Alpha, n_actions, name, input_dims, chkpt_dir = 'tmp/dqn') :
        super(DuelingDeepQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr= Alpha)
        self.loss = nn.MSELoss()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"dueling_dqn")
    
    def forward(self, state) :
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        V = self.V(l2)
        A = self.A(l2)
        
        return V, A
    
    def save_checkpoint(self) :
        print("======saving checkpoint======")
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self) :
        print("======loading checkpoint======")
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object) :
    def __init__(self, gamma, epsilon, alpha, n_actions, input_dims,
                memory_size, batch_size, eps_min = 0.01, eps_dec= 5e-7,
                replace= 1000, chkpt_dir= "tmp/dueling_dqn") :
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.replace_target_cntr = replace
        self.batch_size = batch_size
        
        self.memory = ReplayBuffer(memory_size, input_dims, n_actions)
        
        self.Q_eval = DuelingDeepQNetwork(alpha, n_actions,
                                          input_dims = input_dims,
                                          name= "Q_eval",
                                          chkpt_dir = chkpt_dir)
        
        self.Q_next = DuelingDeepQNetwork(alpha, n_actions,
                                          input_dims = input_dims,
                                          name= "Q_eval",
                                          chkpt_dir = chkpt_dir) 
    
    def store_transition(self, state, action, reward, new_state, done) :
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def choose_action(self, observation) :
        if np.random.random() > self.epsilon :
            # observation = observation[np.newaxis, :]
            state = torch.tensor(observation)
            _, advantage = self.Q_eval.forward(state)
            action = torch.argmax(advantage).item()
        else :
            action = np.random.choice(self.action_space)
        return action
    
    def replace_target_network(self) :
        if self.replace_target_cntr is not None and self.learn_step_counter % self.replace_target_cntr == 0 :
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
            self.learn_step_counter = 0
        
    def decrement_epsilon(self) :
        if self.epsilon > self.eps_min :
            self.epsilon = self.epsilon - self.eps_dec
        else :
            self.epsilon = self.eps_min
    
    def learn(self) :
        if self.memory.memory_cntr < self.memory.memory_size :
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        self.replace_target_network()
        
        # batch_size = min(self.memory.memory_cntr, self.memory.memory_size)
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        dones = torch.tensor(dones)
        
        V_s, A_s = self.Q_eval.forward(states)
        V_s_, A_s_ = self.Q_next.forward(next_states)
        
        q_pred = torch.add(V_s, (A_s - A_s.mean(dim= 1, keepdim = True))).gather(1, 
                                                actions.unsqueeze(1))
        
        q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim= 1, keepdim = True)))
        
        q_target = rewards + self.gamma * (torch.max(q_next, dim = 1)[0].detach())
        q_target[dones.bool()] = 0.0
        
        loss = self.Q_eval.loss(q_target, q_pred.squeeze())
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
        
        
    def save_models(self) :
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()
    
    
    def load_models(self) :
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()
