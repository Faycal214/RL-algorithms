from main import QLearning
import matplotlib.pyplot as  plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import gymnasium as gym
import numpy as np
import matplotlib.animation as animation


# define the parameters 
alpha = 0.1 # learning rate
epsilon = 0.7
gamma = 0.5 # discount factor
episodes = 25000 # nomber of episodes
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=6), render_mode= 'rgb_array')
agent = QLearning(env = env, alpha = alpha, gamma = gamma, epsilon = epsilon, episodes= episodes)

# training the algorithm 
agent.training_process()

# simulate the envorinment
(current_state, _) = env.reset()
done = False
frames = [] 
while not done :
    action = np.argmax(agent.q_table[current_state])
    (next_state, reward, done, _, _) = env.step(action)

    frames.append(env.render())
    current_state = next_state

env.close()
# Convert frames to a video and display it
plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
patch = plt.imshow(frames[0])
plt.axis('off')

def animate(i):
    patch.set_data(frames[i])

anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=100)
plt.show()

# Save the frames as a video
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim.save('agent_solution.mp4', writer=writer)