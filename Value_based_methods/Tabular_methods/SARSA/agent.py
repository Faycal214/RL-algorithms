import numpy as np
import random

class SARSAagent :
    def __init__(self, env, alpha, gamma, epsilon, episodes) :
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon 
        self.episodes = episodes
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        np.random.seed(42)
        self.q_table = np.random.uniform(-10, 10, (self.state_space, self.action_space))
    
    def select_action(self, state) :
        random_number = random.random()
        if  random_number < self.epsilon :
            return random.randrange(self.action_space)
        else :
            return np.argmax(self.q_table[state])

    def update_Q(self, state, action, reward, next_state, done) :
        if done :
            self.q_table[state, action] = self.q_table[state, action] + self.alpha * (reward - self.q_table[state, action])
        else : 
            next_action = self.select_action(next_state)
            self.q_table[state, action] = self.q_table[state, action] + self.alpha * (reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action])
    
    def training(self) :
        for episode in range(self.episodes) :
            total_return = 0
            (state, _) = self.env.reset()
            done = False 
            while not done :
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.update_Q(state, action, reward, next_state, done)
                total_return += reward
                state = next_state
            # Decay epsilon after each episode
            if self.epsilon > 0.01:   # prevent it from going to zero
                self.epsilon *= 0.9995
            print(f"Episode {episode} | total return {total_return} | epsilon {self.epsilon}")
