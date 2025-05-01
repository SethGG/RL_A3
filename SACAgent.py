import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, device, hidden_dim):
        super(CriticNetwork, self).__init__()
        # Q1 layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # Q2 layers
        self.fc4 = nn.Linear(input_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        # Move the model to the specified device (CPU or GPU)
        self.to(device)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=-1)
        # Forward pass for Q1
        q1 = torch.relu(self.fc1(sa))
        q1 = torch.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        # Forward pass for Q2
        q2 = torch.relu(self.fc4(sa))
        q2 = torch.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim):
        super(PolicyNetwork, self).__init__()
        # Define a two-layer hidden network with one output layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # Move the model to the specified device (CPU or GPU)
        self.to(device)

    def forward(self, state):
        # Forward pass through the network:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)  # apply softmax for probability distribution

        return x


class SACAgent:
    def __init__(self, state_dim, n_actions, learning_rate, gamma, hidden_dim, temperature, memory_size, batch_size, update_freq):
        self.gamma = gamma
        self.temperature = temperature
        self.batch_size = batch_size

        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network: outputs a probability distribution over actions
        self.pi = PolicyNetwork(state_dim, n_actions, self.device, hidden_dim)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=learning_rate)

        # Q networks (double-Q)
        self.Q = CriticNetwork(state_dim + 1, self.device, hidden_dim)
        self.optimizer_Q = optim.Adam(self.Q.parameters(), lr=learning_rate)

        # Q target networks
        self.Q_target = CriticNetwork(state_dim + 1, self.device, hidden_dim)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=memory_size)
        self.update_count = 0
        self.update_freq = update_freq

    def select_action_sample(self, state):
        # Convert state to tensor and forward through policy network to get probabilities
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action_probs = self.pi.forward(state)
        # Create a categorical distribution from the probabilities and sample an action
        m = Categorical(action_probs)
        action = m.sample().item()
        return action

    def select_action_greedy(self, state):
        # For evaluation, choose the action with the highest probability
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action_probs = self.pi.forward(state)
        action = torch.argmax(action_probs).item()
        return action

    def update(self, state, action, reward, next_state, done):
        self.update_count += 1
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < self.batch_size * 2:
            return

        if self.update_count < self.update_freq:
            return

        self.update_count = 0

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = (np.array(x) for x in zip(*batch))

        # Convert numpy arrays to torch tensors
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.int, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.int, device=self.device)

        next_action_probs = self.pi(next_state)
        m = Categorical(next_action_probs)
        next_action = m.sample()
        next_action = next_action.unsqueeze(-1)
        q_values = self.Q_target(next_state, next_action)

        breakpoint()

        # y = reward + self.gamma * (1 - done) * \
        #    (torch.min(torch.tensor([self.Q1_target(next_action), self.Q2_target(next_action)]), dim=0).values -)


if __name__ == "__main__":
    test = SACAgent(state_dim=4, n_actions=2, learning_rate=0.001, gamma=1,
                    hidden_dim=128, temperature=1, memory_size=6, batch_size=3, update_freq=1)
    while True:
        test.update(np.random.sample(4), 1, 1, np.random.sample(4), 0)
