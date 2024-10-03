from main import Genetic_algo
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import cv2


generation = int(input("Give the nomber of generations : "))
individus = int(input("Give the lenght of generations : "))

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=6), render_mode= 'rgb_array')
agent = Genetic_algo(nb_individus= individus, nb_generation= generation, env= env)

q_table, reward = agent.training_process()
print(f"the best q_table is : {q_table} \n the total rewards : {reward}")

# simulate the envorinment
(current_state, _) = env.reset()
done = False
frames = [] 
while not done :
    action = np.argmax(q_table[current_state])
    (next_state, reward, done, _, _) = env.step(action)

    frames.append(env.render())
    current_state = next_state

env.close()
# Save the frames as a video using OpenCV
height, width, layers = frames[0].shape
video_name = 'agent_solution.mp4'

# Initialize the video writer object
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

# Convert RGB frames to BGR and write to video
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR format

out.release()  # Release the video writer
# Play the video using OpenCV (optional)
for frame in frames:
    cv2.imshow('Simulation', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Display in BGR format
    if cv2.waitKey(300) & 0xFF == ord('q'):  # Press 'q' to quit the display
        break

cv2.destroyAllWindows()  # Close the OpenCV window