import numpy as np
import random 

class QLearning :
    def __init__(self, env, alpha, gamma, epsilon, episodes) :
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.stateDimension = self.env.observation_space.n
        self.actionDimension = self.env.action_space.n
        np.random.seed(42)
        self.q_table = np.random.uniform(0, 10, (self.stateDimension, self.actionDimension))
    
    def select_action(self, state, index) :
        if index < 1000 :
            return random.randrange(self.actionDimension)
        if index > 3000 :
            self.epsilon = self.epsilon * 0.99
        random_number = random.random()
        if random_number < self.epsilon :
            return random.randrange(self.actionDimension)
        else :
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, current_state, action, reward, next_state, done) :
        if done :
            self.q_table[current_state][action] = self.q_table[current_state][action] + self.alpha * (reward - self.q_table[current_state][action])
        else :
            max_next_q = np.max(self.q_table[next_state])
            self.q_table[current_state][action] = self.q_table[current_state][action] + self.alpha * (reward + self.gamma * max_next_q - self.q_table[current_state][action])
    
    def training_process(self) :
        sumRewardEpisodes = []
        for episode in range(self.episodes) :
            rewardEpisode = 0
            (current_state, _) = self.env.reset()
            done = False
            while not done :
                action = self.select_action(current_state, episode)
                (next_state, reward, done, _, _) = self.env.step(action)
                self.update_q_table(current_state, action, reward, next_state, done)
                rewardEpisode += reward
                current_state = next_state
            sumRewardEpisodes.append(rewardEpisode)
            if rewardEpisode != 0 :
                print(f"Episode {episode} : {rewardEpisode}")
        
        