from test import dql
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

env = gym.make('CartPole-v1', render_mode = 'rgb_array')

# Reset the environment
(current_state, _) = env.reset()
terminal_state = False
frames = [] 


while not terminal_state:
    # Select action using the trained model
    # Replace with your action selection logic using loaded_model.predict()
    Qvalues = dql.DeepQNetwork.predict(current_state.reshape(1, dql.stateDimension))
    # select the action that gives the max Qvalue
    action = np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0])

    # Apply the action
    (next_state, reward, terminal_state, _, _) = env.step(action)
    
    frames.append(env.render())

    # Update current state
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

anim.save('DQN_solution.mp4', writer=writer)