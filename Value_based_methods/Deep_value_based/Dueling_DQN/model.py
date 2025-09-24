import torch as T
import numpy as np
from memory import ReplayBuffer
from policies import DuelingDeepQNetwork

class DuelingDQNAgent :
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, replace=1000) :
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(max_mem_size, input_dims, n_actions)

        self.q_eval = DuelingDeepQNetwork(lr, n_actions, input_dims, 256, 256)
        self.q_next = DuelingDeepQNetwork(lr, n_actions, input_dims, 256, 256)

    def choose_action(self, observation) :
        if np.random.random() > self.epsilon :
            state = T.tensor([observation], dtype=T.float32)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else :
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done) :
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self) :
        if self.learn_step_counter % self.replace_target_cnt == 0 and \
            self.replace_target_cnt is not None:

            self.q_next.load_state_dict(self.q_eval.state_dict())
        
    def decrement_epsilon(self) :
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon >  self.eps_min else self.eps_min
    
    def learn(self) :
        if self.memory.mem_cntr < self.batch_size :
            return 
        
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(states)
        actions = T.tensor(actions, dtype=T.long).unsqueeze(1)
        rewards = T.tensor(rewards)
        states_ = T.tensor(states_)
        dones = T.tensor(dones)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1, actions).squeeze(1)
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim= True)))

        Q_target = rewards + self.gamma * T.max(q_next, dim= 1)[0].detach()
        Q_target[dones] = 0.0

        loss = self.q_eval.loss(q_pred, Q_target)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        print("V_s shape:", V_s.shape)
        print("A_s shape:", A_s.shape)
        print("actions shape:", actions.shape)
