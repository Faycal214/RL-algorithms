import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import tensorflow as tf
from model import ReinforceLearning
import cv2

env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -10.0,
    enable_wind  = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
    render_mode= 'rgb_array'
)

# parameters 
alpha = 0.1
gamma = 0.99
epsilon = 1
nb_episodes = 2500

agent = ReinforceLearning(env= env, nb_episodes= nb_episodes, alpha= alpha, gamma= gamma, epsilon= epsilon)
agent.train_agent()

# save the plot
agent.plot_rewards(save_path= "rewards_plot.png")

# simulate the envirnmont 
(observation, _) = env.reset()
done = False
frames = []
while not done :
    predictions = agent.model(tf.expand_dims(observation, axis= 0))
    predictions = tf.nn.softmax(predictions)
    action = np.argmax(predictions.numpy())
    (next_observation, reward, done, _, _) = env.step(action)
    # frames.append(env.render())
    frames.append(env.render())
    observation = next_observation

env.close()
# save the frames as a video using OpenCv
height, width, layers = frames[0].shape
video_name = "agent_simulation.mp4"

# initialize the video writer object
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

#convert RGB frames to RGB and write to video
for frame in frames :
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
# release the video writer
out.release()

# play the video using OpenCv (optional)
for frame in frames :
    cv2.imshow('Simulation', cv2.cvtColor(frame, cv2.COLOR_RGB2RGB))
    # display in RGB format 
    if cv2.waitKey(300) & 0xFF == ord('q') :
        # press "q" to quit the display 
        break

cv2.destroyAllWindows()