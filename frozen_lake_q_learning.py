
# first attempt for the Frozen Lake v1 (v0 is deprecated)

# Basic imports

import numpy as np
import gym
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

rng = np.random.default_rng(seed=179)
# even with a set seed there is still variability in the results. WHY?

# Create the environment from gym along with the state & action space

env = gym.make("FrozenLake-v1")

state_size = env.observation_space.n
action_size = env.action_space.n

# Create and initialise with zeros the Q table with dimensions = [states, actions]

q_values = np.zeros((state_size, action_size))
print(q_values)

# Hyper - parameters

total_episodes = 15000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005             # Exponential decay rate for exploration prob

NUM_RUNS = 20


def epsilon_greedy(q_values, state, epsilon):
    """
    Def tha chooses the action for the agent by the exploration- exploitation criterion
    :param q_values:
    :param state:
    :param epsilon:
    :return:
    """
    coin = rng.uniform(0, 1)
    if coin <= epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_values[state, :])
    return action


def q_learning(q_values, state, next_state, action, learning_rate, reward, gamma):
    """
    Updates the Q value for one step
    :param q_values:
    :param state:
    :param next_state:
    :param action:
    :param learning_rate:
    :param reward:
    :param gamma:
    :return:
    """
    q_values[state, action] = q_values[state, action] * (1 - learning_rate) + \
                              learning_rate * (reward + gamma * np.max(q_values[next_state, :]))

    return None  # This function shouldn't return anything


def run_episode(epsilon=epsilon):
    """
    Returns the reward for one episode
    :return:episode_rewards
    """
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        action = epsilon_greedy(q_values, state, epsilon)     # choose actions

        new_state, reward, done, info = env.step(action)       # observe next state (take a step)

        q_learning(q_values, state, new_state, action, learning_rate, reward, gamma)        # update Q value

        total_rewards += reward

        state = new_state         # Reset the state for the next step
        # If done (if we're dead) : finish episode
        if done:
            break

    return total_rewards


def run():
    """
    runs 1 epoch consisting of num_episodes episodes where every
    episode consists of max_steps time-steps
    :return: the list of rewards for all the episodes
    """
    global epsilon
    epoch_rewards = 0
    pbar = tqdm(range(total_episodes), ncols=100)

    for episode in pbar:
        epoch_rewards += (run_episode(epsilon))
        pbar.set_description(
            "Avg reward: {:0.6f}".format(
                (epoch_rewards/total_episodes)))
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    return epoch_rewards


if __name__ == "__main__":

    rewards = []
    t0 = time.time()
    for _ in range(NUM_RUNS):
        rewards.append(run())

    rewards = np.array(rewards)
    rewards_per_episode = rewards/total_episodes
    average_rewards = np.mean(rewards_per_episode)
    tend = time.time() - t0
    ttotal = time.strftime("%H:%M:%S", time.gmtime(tend))
    print(f'Running Time: {ttotal} h:m:s')

    fig, axis = plt.subplots()
    x = np.arange(NUM_RUNS)
    axis.plot(x, rewards_per_episode)
    axis.set_xlabel('# Epochs')
    axis.set_ylabel('Reward per epoch')
    axis.set_title('Q - learning algorithm Frozen Lake v1')
    plt.show()
