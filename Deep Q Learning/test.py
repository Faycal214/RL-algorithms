from rl import CustomDeepQLearning
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
gamma = 0.6
epsilon = 0.5
nbEpisode = 50
dql = CustomDeepQLearning(env, gamma, epsilon, nbEpisode)
dql.training_episodes()
dql.DeepQNetwork.summary()


plt.plot(range(dql.nbEpisodes), dql.sumRewardsEpisode)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.show()