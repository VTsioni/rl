# DQN implementation with OOP
# Basic Imports

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Make sure GPU is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 2000

# Class Experience
Experience = namedtuple(
    "Experience",
    ("state", "action", "next_state", "reward")
)

# Class DQN
class DQN(nn.Module):
    def __init__(self, num_obs, num_actions):
        """
        The constructor function for the DQN with 3 layers
        :param num_obs: the observations (the experience tuple?)
        :param num_actions: (the 2 actions possible - left or right)
        """
        super(DQN, self).__init__()    # so that it does not override the constructor from nn.Module

        # Construct the linear (or fully connected layers)
        self.fc1 = nn.Linear(in_features=num_obs, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=num_actions)

    def forward(self, tensor):
        """
        The forward propagation with relu activation functions for the first
         two layers and the identity function for the output
        :param tensor:
        :return: tensor
        """
        tensor = F.relu(self.fc1(tensor))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.out(tensor)
        return tensor


# Class Replay Memory
class ReplayMemory(object):
    """
    ReplayMemory - a cyclic buffer of bounded size that holds the
    experiences observed recently. It also implements a .sample() method
    for selecting a random batch of transitions for training.
    Makes use of deques so that if the length exceeds the capacity the
    oldest experiences will be overridden by the newest.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):  # if its not necessary delete it
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)


# Class Epsilon Greedy Strategy
class EpsilonGreedyStrategy(object):
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        exploration_rate = self.end + (self.start - self.end) * \
                           math.exp(-1.0 * current_step / self.decay)
        return exploration_rate


# Class Agent
class Agent(object):
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        coin = random.random()

        if coin > rate:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)  # exploit
            # exploitation
        else:
            # exploration
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def plot_durations(show_result=False):
    global episode_durations
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    # pause a bit so that plots are updated


def train():
    for episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = agent.select_action(state=state, policy_net=policy_net)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimisation
            optimize(batch_size=Batch_size, gamma=Gamma)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * Tau +\
                                             target_net_state_dict[key] * (1 - Tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
    return

################################################## 
# Hyper-parameters (needs fixing)
Batch_size = 128
Gamma = 0.99
Eps_start = 1.0
Eps_end = 0.00
Eps_decay = 1000
Tau = 0.005
Learning_Rate = 1e-4
Memory_size = 10_000

####################################################
# Import the environment form gym
env = gym.make("CartPole-v1")

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# Create the instances of the classes that are going to be used
# 2 DQNs one policy, one target net, make sure the target net is only for evaluation
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ReplayMemory(Memory_size)
strategy = EpsilonGreedyStrategy(Eps_start, Eps_end, Eps_decay)
agent = Agent(strategy, n_actions, device)

optimizer = optim.AdamW(policy_net.parameters(), lr=Learning_Rate, amsgrad=True)


def optimize(batch_size=Batch_size, gamma=Gamma):
    # Sample random batch from replay memory.
    if memory.can_provide_sample(batch_size):
        experiences = memory.sample(batch_size)

        # Preprocess states from batch.
        batch = Experience(*zip(*experiences))

        # Compute a mask of non - final states and concatenate the batch elements
        # a final state would have been the one after which simulation ended
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device,
                                      dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        current_q_values = policy_net(state_batch).gather(1, action_batch)

        next_q_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_q_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        # calculate the target Q-values (aka. Bellman Optimality Equation)
        target_q_values = reward_batch + (gamma * next_q_values)

        # select the loss function (other choice could be mse_loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, target_q_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
    return
###################################################################


episode_durations = []
train()
plot_durations(show_result=True)
plt.ioff()
plt.show()
