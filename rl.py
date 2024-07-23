import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from IPython import display
from tensorflow import keras
import numpy as np
import random
import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow import gather_nd
from collections import deque
from tensorflow.keras.saving import register_keras_serializable


class DeepQLearning :
    def __init__(self, env, gamma, epsilon, nbEpisodes) :
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.nbEpisodes = nbEpisodes
        
        # dimension d'etats
        self.stateDimension = self.env.observation_space.shape[0]
        # dimension d'actions
        self.actionDimension = self.env.action_space.n
        # this is the maximum size of the replay buffer
        self.ReplayBufferSize = 400
        # this is the size of the training batch that is randomly sampled from the replay buffers
        self.batchReplayBufferSize = 100
        
        # number of training episodes it takes to update the target q network parameters
        # that is every UpdateTargetQNetworkPeriod we update the target network parameters
        # c-a-d : every 100 steps we update the weights of the traget network
        self.UpdateTargetQNetworkPeriod = 100
        
        # this is the counter for updating the target network
        # if this counter exceeds (UpdateTargetQNetworkPeriod - 1) we update the network
        # parameters and reset the counter to zero
        self.CounterUpdateTargetQNetwork = 0
        
        # this sum is used to store the sum of rewards obtained during each training episode
        self.sumRewardsEpisode = []
        
        # replay buffer 
        self.replayBuffer = deque(maxlen = self.ReplayBufferSize)
        
        # this is the main network 
        self.DeepQNetwork = self.create_network()
        
        # this is the target network 
        self.TargetQNetwork = self.create_network()
        
        # copy the inital weights to TargetNetwork
        self.TargetQNetwork.set_weights(self.DeepQNetwork.get_weights())
        
        # this list is used in the cost function to select certain entries of the
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend = []
    
    def create_network(self) :
        model = Sequential([
            Dense(units = 64,input_dim = self.stateDimension, activation = 'relu'),
            Dense(units = 64, activation = 'relu'),
            Dense(units = self.actionDimension, activation = 'linear'),
        ])
        
        model.compile(
            optimizer = RMSprop(),
            loss = self.my_loss,
            metrics = ['accuracy'],
        )
        return model
    
    def training_episodes(self) :
        # here we loop through the episodes (keep in trace the rewards -total return- of each episode)
        for indexEpisode in range(self.nbEpisodes) :
            
            # list that stores rewards per episode 
            rewardsEpisode = []
            print(f'\nsimulating episode {indexEpisode}')
            
            # reset the enviroment at the beginning of every episode
            (currentState, _) = self.env.reset()
            
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState = False
            
            while not terminalState :
                
                # select an action on the basis of the current state and the episode indix
                action = self.selectAction(currentState, indexEpisode)
                
                # here we step and return the state, reward, and boolean denoting if the terminal state is reached
                (nextState, reward, terminalState, _, _) = self.env.step(action)
                rewardsEpisode.append(reward)
                
                #add current state, action, reward, next state, and terminal flag 
                self.replayBuffer.append((currentState, action, reward, nextState, terminalState))
                
                # train network
                self.trainNetwork()
                
                # set the current state for the next state
                currentState = nextState
        
            print(f"Sum of rewards {np.sum(rewardsEpisode)}")
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))
    
    def selectAction(self, state, index) :
        #first index episodes we select completely random actions to have some exploration 
        if index < 1 :
            return np.random.choice(self.actionDimension)
        
        # returns  a random real number in the half-open interval [0, 1)
        # this number is used for the epsilon greedy policy approach
        randomNumber = np.random.random()
        
        # after index episodes , we slowly start to decrease the epsilon parameter
        # by the time goes by , the actions computed in the greedy strategy are more optimal
        if index > 40 :
            self.epsilon = 0.99 * self.epsilon
        
        #if this condition is satisfied, we are exploring, that is we select random actions 
        if randomNumber < self.epsilon :
            return np.random.choice(self.actionDimension)
        # otherwise we are selecting greedy actions 
        else :
            # return the index where Qvalues[state, :] has the max value
            # that is since the index denotes an action we select greedy actions
            
            Qvalues = self.DeepQNetwork.predict(state.reshape(1, 4))
            return np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0 ,:]))[0])
            # here we need to return the minimum index since it can happen 
            # that there are several identical maximal entries, for exemple 
            # a = [0, 1 ,1, 0]
            # np.where(a == np.max(a))
            # this will return [1, 2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a == np.max(a))[0])
            # note that the zero has to be added here since np.where() returns a tuple
    
    def trainNetwork(self) :
        # if the replay buffer has at least batchReplayBufferSize elements
        # then train the model
        # otherwise wait until the size of the elements exceeds batchReplayBufferSize
        
        if (len(self.replayBuffer) > self.batchReplayBufferSize) :
            
            # sample a batch from the replay buffer
            randomSampleBatch = random.sample(self.replayBuffer, k= self.batchReplayBufferSize)
            
            # here we form current state batch
            # and next state batch
            # they are used as inputs for predictions
            currentStateBatch = np.zeros(shape = (self.batchReplayBufferSize, 4))
            nextStateBatch = np.zeros(shape = (self.batchReplayBufferSize, 4))
            
            # this will enumerate the tuple entries of the randomSampleBatch
            # index will loop through the number of tuples
            for index, tuples in enumerate(randomSampleBatch) :
                # first entry of the tuple is the current state
                currentStateBatch[index, :] = tuples[0]
                # fourth entry of the tuple is the next state
                nextStateBatch[index, :] = tuples[3]
            
            # here use the target network to predict Q-values
            QnextStateTargetNetwork = self.TargetQNetwork.predict(nextStateBatch)
            # here use the Q network to predict Q-values
            QcurrentStateQNetwork = self.DeepQNetwork.predict(currentStateBatch)
            
            # now we form batches for training 
            # input for training
            inputNetwork = currentStateBatch
            # output for training
            outputNetwork = np.zeros(shape = (self.batchReplayBufferSize, 2))
            
            # this list will contain the actions that are selected from the batch
            # this list is used in my_loss to define the loss-function
            self.actionsAppend = [] # Reset actionsAppend for each batch
            for index, (currentState, action, reward, nextState, terminated) in enumerate(randomSampleBatch) :
                # if the next state is the terminal state 
                if terminated :
                    y= reward
                # if the next state is not the terminal state j
                else :
                    y = reward + (self.gamma * np.max(QnextStateTargetNetwork[index]))
                
                # this actually does not matter since we do not use all the entries in the cost function
                outputNetwork[index] = QcurrentStateQNetwork[index]
                # this is what matters
                outputNetwork[index, action] = y
                
                # append the action to the list
                self.actionsAppend.append(action) 
            
            # here we train the network
            self.DeepQNetwork.fit(inputNetwork, outputNetwork, batch_size = self.batchReplayBufferSize, verbose = 0, epochs = 100)
            
            #  after updateTargetQNetworkPeriod training sessions, update the coeficients
            # of the target network 
            # increase the counter for training the target network
            self.CounterUpdateTargetQNetwork += 1
            if (self.CounterUpdateTargetQNetwork > (self.UpdateTargetQNetworkPeriod - 1)) :
                # copy the weights to the target netwrok
                self.TargetQNetwork.set_weights(self.DeepQNetwork.get_weights())
                print(f"Target naetwork updated !")
                print(f"counter value {self.CounterUpdateTargetQNetwork}")
                # reset the counter 
                self.CounterUpdateTargetQNetwork = 0

    def save_model(self, filepath):
        self.DeepQNetwork.save(filepath)
        print(f"Modèle sauvegardé sous {filepath}")

@register_keras_serializable()
class CustomDeepQLearning(DeepQLearning):
    def __init__(self, env, gamma, epsilon, nbEpisodes):
        super().__init__(env, gamma, epsilon, nbEpisodes)

    def my_loss(self, y_true, y_pred):
        s1, s2 = y_true.shape.as_list()
        indices = tf.zeros(shape=(s1, s2), dtype=tf.int32)
        indices = tf.stack([tf.range(s1), self.actionsAppend], axis=1)

        loss = mean_squared_error(
            tf.gather_nd(y_true, indices),
            tf.gather_nd(y_pred, indices)
        )

        return loss

    def get_config(self):
        return {}