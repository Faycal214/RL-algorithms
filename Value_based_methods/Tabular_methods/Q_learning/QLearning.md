# Q-Learning Algorithm Explanation and Demonstration

This repository contains an implementation of the Q-learning algorithm applied to solve a reinforcement learning problem. Q-learning is a model-free reinforcement learning technique used to find an optimal action-selection policy for any given finite Markov decision process (MDP). In this demonstration, we apply Q-learning to train an agent to solve a simple environment.

## Q-Learning Overview

Q-learning involves learning an action-value function $Q(S, a)$, where:
- $Q(S, a)$ represents the expected cumulative reward when taking action $a$ in state $S$.

### Key Concepts

- **Q-table**: The Q-table is a data structure where each entry $Q(S, a)$ represents the learned value of taking action $a$ in state $S$. Initially, Q-values are initialized arbitrarily or to zeros. During training, Q-values are updated iteratively based on the agent's interactions with the environment.

- **Epsilon-Greedy Policy**: To balance exploration and exploitation, the agent uses an epsilon-greedy policy:
  - With probability $\epsilon$, the agent selects a random action (exploration).
  - With probability $1 - \epsilon$, the agent selects the action with the highest Q-value for the current state (exploitation).

- **Parameters**:
  - **Alpha $\alpha$**: Learning rate that determines how much new information overrides old information during Q-value updates.
  - **Gamma $\gamma$**: Discount factor that determines the importance of future rewards relative to immediate rewards.
  - **Epsilon $\epsilon$**: Exploration-exploitation trade-off parameter controlling the rate of random action selection versus exploitation of learned Q-values.


## Demonstration Video

Watch the video below to see how the agent learns to solve the problem using Q-learning:

https://github.com/user-attachments/assets/d587b245-4380-423b-b579-41dbbfb65859

Click on the image above to watch the full video on YouTube.

## Contents

- `q_learning.py`: Python script containing the Q-learning algorithm implementation.
- `environment.py`: Python script defining the environment where the agent interacts.
- `README.md`: This file, explaining the Q-learning algorithm and demonstrating the video.


