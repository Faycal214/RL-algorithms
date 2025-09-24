import gym
import numpy as np

np.bool8 = np.bool_
# ==== Define the Environment ====
env = gym.make('CartPole-v1', render_mode = "rgb_array")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# ==== PSO Parameters ====
n_particles = 30
n_dimensions = n_states
blo, bup = -1.0, 1.0
w = 0.5            # inertia
phi_p = 1.5        # cognitive
phi_g = 1.5        # social
max_iter = 500

# ==== Fitness Function: Total Reward ====
def evaluate_policy(weights):
    total_reward = 0
    obs = env.reset()[0] 
    done = False
    while not done:
        action = 1 if np.dot(weights, obs) > 0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    return total_reward

# ==== Initialize PSO ====
particles = np.random.uniform(blo, bup, (n_particles, n_dimensions))
velocities = np.random.uniform(-abs(bup-blo), abs(bup-blo), (n_particles, n_dimensions))
p_best = particles.copy()
p_best_scores = np.array([evaluate_policy(p) for p in particles])
g_best = p_best[np.argmax(p_best_scores)]
g_best_score = np.max(p_best_scores)

# ==== PSO Main Loop ====
for iteration in range(max_iter):
    for i in range(n_particles):
        r_p = np.random.rand(n_dimensions)
        r_g = np.random.rand(n_dimensions)

        velocities[i] = (
            w * velocities[i]
            + phi_p * r_p * (p_best[i] - particles[i])
            + phi_g * r_g * (g_best - particles[i])
        )
        particles[i] += velocities[i]

        score = evaluate_policy(particles[i])
        if score > p_best_scores[i]:
            p_best[i] = particles[i]
            p_best_scores[i] = score
            if score > g_best_score:
                g_best = particles[i]
                g_best_score = score

    print(f"Iteration {iteration+1} - Best Score: {g_best_score}")

# ==== Final Run ====
print("\nRunning final policy...")
final_score = evaluate_policy(g_best)
print("Final Score:", final_score)
env.close()
