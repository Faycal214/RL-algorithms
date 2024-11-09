import gymnasium as gym
from main import Agent
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

# creating the environment
env = gym.make("LunarLander-v3", continuous=False, render_mode='rgb_array')
# creating the agent 
agent = Agent(gamma = 0.99, epsilon = 1.0, alpha = 5e-4, input_dims= [env.observation_space.shape[0]],
            n_actions= env.action_space.n, memory_size = 1000, eps_min = 0.1, 
            batch_size = 64, eps_dec = 1e-5, replace = 1000)
# other parameters
num_episode = 2000
load_chekpoint = False

if load_chekpoint :
    agent.load_models()

file_name = "MountainCarContinuous_dueling.png"
video_filename = "lunar_lander_video.avi"

scores = []
avg_scores = []  # list to store average scoresstory = []

for i in range(num_episode) :
    (current_state, _) = env.reset()
    score = 0
    done = False
    while not done :
        action = agent.choose_action(current_state)
        next_state, reward, done, _, _ = env.step(action)
        score += reward
        agent.store_transition(current_state, action, reward, next_state, int(done))
        agent.learn()
        current_state = next_state
    
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)
    print(f"episode {i} | score {scores[i]} | avg score {avg_score} | epsilon {agent.epsilon}")

env.close()

# Plotting the scores and average scores
x = [i+1 for i in range(num_episode)]  # x-axis (episodes)
# Tracer les scores
plt.figure(figsize=(10, 6))
plt.plot(x, scores, label='Scores', alpha=0.6)  # plot the individual episode scores
plt.plot(x, avg_scores, label='Average Score (last 100 episodes)', color='orange', linewidth=2)  # plot the moving average
plt.xlabel("Episodes")
plt.ylabel("Scores")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.savefig("scores_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Initialisation pour l'enregistrement de la vidéo avec OpenCV
frame_rate = 30  # FPS de la vidéo
frame_width = 600  # Largeur de la vidéo
frame_height = 400  # Hauteur de la vidéo
out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))

(current_state, _) = env.reset()
done = False
cuurent_state = torch.tensor(current_state)
while not done :
    _, advantage = agent.Q_eval.forward(current_state)
    action = torch.argmax(advantage).item()
    action = agent.choose_action(current_state)
    next_state, reward, done, _, _ = env.step(action)
    current_state = next_state
    current_state = torch.tensor(current_state)
    
    # Capture d'une frame de l'environnement et ajout dans la vidéo
    frame = env.render()
    frame_resized = cv2.resize(frame, (frame_width, frame_height))  # Redimensionner la frame
    out.write(frame_resized)  # Ajouter la frame à la vidéo

# Libérer la ressource de la vidéo
out.release()
env.close()
