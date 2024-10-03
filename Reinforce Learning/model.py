import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


class ReinforceLearning :
    def __init__(self, env, nb_episodes, alpha, gamma, epsilon) :
        self.env= env
        self.nb_episodes = nb_episodes
        self.alpha= alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = self.PolicyNetwork(self.env)
        self.optimizer = Adam(learning_rate= self.alpha)
        self.rewards_per_episode = []

    class PolicyNetwork(tf.keras.Model) :
        def __init__(self, env) :
            super(ReinforceLearning.PolicyNetwork, self).__init__()  # Proper inheritance
            self.dense1= Dense(units= 24, activation = 'relu')
            self.dense2 = Dense(units= 24, activation= 'relu')
            self.logits = Dense(units= env.action_space.n)
        
        def __call__(self, state) :
            state = tf.expand_dims(state, axis= 0)
            x = self.dense1(state)
            x = self.dense2(x)
            return self.logits(x)
    
    def choose_action(self, state) :
        random_number = np.random.rand()
        if random_number < self.epsilon : # if the epsilon is still high
            return np.random.choice(self.env.action_space.n)
        else :
            # pass the state vector as an input to the model and return the output
            logits = self.model(state)
            # convert the output to a distribution of probabilities 
            logits = tf.nn.softmax(logits)
            # choose the action with the highest probability
            return np.argmax(logits.numpy())
    
    def compute_returns(self, rewards) :
        returns = []
        discounted_sum = 0
        for i, r in enumerate(reversed(rewards)) :
            # discounted_sum *= r + (self.gamma) ** (i)
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return returns
    
    def train_agent(self) :
        for episode in range(self.nb_episodes) :
            (current_state, _) = self.env.reset()
            rewards = []
            actions = []
            states = []
            done = False
            while not done :
                # choose an action with epsilon-greedy approach
                action = self.choose_action(tf.convert_to_tensor(current_state, dtype= tf.float32))
                # apply the action in the environment and observ the next state, rawards 
                (next_state, reward, done, _, _) = self.env.step(action)
                # save the state, action and reward
                states.append(current_state)
                actions.append(action)
                rewards.append(reward)
                # update the state
                current_state = next_state
            
            # calculer les retours G_t
            returns = self.compute_returns(rewards)
            total_rewards = sum(rewards)
            self.rewards_per_episode.append(total_rewards)

            if episode > 300 :
                self.epsilon = self.epsilon * 0.9995

            # update the policy 
            with tf.GradientTape() as tape :
                loss = 0
                for i in range(len(states)) :
                    state = tf.convert_to_tensor(states[i], dtype= tf.float32)
                    logits = self.model(state)
                    action_proba = tf.nn.softmax(logits)
                    action_one_hot = tf.one_hot(actions[i], self.env.action_space.n)
                    selected_action_proba = tf.reduce_sum(action_proba * action_one_hot)
                    loss -= tf.math.log(selected_action_proba) * returns[i]
                
                # calculate the gradients 
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


                print(f"\nEpisode {episode + 1}/{self.nb_episodes} , Rewrad: {total_rewards}, Epsilon : {self.epsilon}, Loss : {loss.numpy()}")
        
        return self.model
    
    def plot_rewards(self, save_path=None):
        # Tracer les récompenses cumulées par épisode
        plt.plot(self.rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('rewards')
        plt.title('Récompense par épisode')

        if save_path:
            plt.savefig(save_path)  # Save the plot to a file
            print(f"Plot saved as {save_path}")
        else:
            plt.show()  # Display the plot

