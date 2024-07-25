from test import dql
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v1', render_mode = 'rgb_array')

# Reset the environment
(current_state, _) = env.reset()
sum_obtained_reward = 0
terminal_state = False

# Render the initial frame 
prev_screen = env.render()
plt.imshow(prev_screen)
plt.axis('off')
plt.show()

while not terminal_state:
    # Select action using the trained model
    # Replace with your action selection logic using loaded_model.predict()
    Qvalues = dql.DeepQNetwork.predict(current_state.reshape(1, dql.stateDimension))
    # select the action that gives the max Qvalue
    action = np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0])

    # Apply the action
    (next_state, reward, terminal_state, _, _) = env.step(action)
    sum_obtained_reward += reward

    # Render the environment
    screen = env.render()

    # Display the updated frame
    plt.imshow(screen)
    plt.axis('off')
    plt.show()

    # Pause briefly to slow down the visualization
    plt.pause(0.01)

    # Update current state
    current_state = next_state

env.close()
print(f"Total obtained reward: {sum_obtained_reward}")