import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim):
        super(NeuralNet, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Move the model to the specified device (CPU or GPU)
        self.to(device)

    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, n_actions, n_states, alpha, gamma, update_freq, hidden_dim, tn=False, er=False):
        self.n_actions = n_actions  # Number of possible actions
        self.n_states = n_states  # Number of states
        self.gamma = gamma  # Discount factor
        self.update_freq = update_freq  # Frequency of updates

        self.tn = tn  # Flag for target network
        self.er = er  # Flag for experience replay
        self.memory_size = 10_000  # Size of the replay buffer
        self.batch_size = 32  # Batch size for training
        self.learning_starts = 1000  # Number of steps before learning starts

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = nn.MSELoss().to(self.device)  # Mean Squared Error loss function
        self.Q = NeuralNet(n_states, n_actions, self.device, hidden_dim)  # Q-network
        self.optimizer = optim.Adam(self.Q.parameters(), lr=alpha)  # Adam optimizer
        self.update_count = 0  # Counter for updates

        if tn:
            # Initialize target network if tn is True
            self.Q_target = NeuralNet(n_states, n_actions, self.device, hidden_dim)
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.Q_target.eval()
        if er:
            # Initialize replay buffer if er is True
            self.replay_buffer = deque(maxlen=self.memory_size)
        else:
            # Initialize update buffer if er is False
            self.update_buffer = []

    def select_action(self, state, epsilon):  # ϵ-greedy policy
        if np.random.random() > epsilon:
            # Select action with highest Q-value
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            with torch.no_grad():
                q_values = self.Q.forward(state)
            action = torch.argmax(q_values).item()
        else:
            # Select random action
            action = np.random.choice(self.n_actions)
        return action

    def update(self, state, action, reward, next_state, done):  # Q-Learning update equation
        self.update_count += 1
        if self.er:
            # Add experience to replay buffer
            self.replay_buffer.append((state, action, reward, next_state, done))
            if len(self.replay_buffer) < self.learning_starts:
                return
        else:
            # Add experience to update buffer
            self.update_buffer.append((state, action, reward, next_state, done))
        if self.update_count < self.update_freq:
            return

        self.update_count = 0

        if self.er:
            # Sample a batch from the replay buffer
            batch = random.sample(self.replay_buffer, self.batch_size)
            state, action, reward, next_state, done = (np.array(x) for x in zip(*batch))
        else:
            # Use the entire update buffer as the batch
            state, action, reward, next_state, done = (np.array(x) for x in zip(*self.update_buffer))
            self.update_buffer.clear()

        # Convert numpy arrays to torch tensors
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.int, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.int, device=self.device)

        # Compute Q-values for the current state-action pairs
        q_value = self.Q.forward(state).gather(1, action).squeeze(1)

        with torch.no_grad():
            if self.tn:
                # Use target network to compute Q-values for the next state
                q_next = self.Q_target.forward(next_state).max(1)[0]
            else:
                # Use current network to compute Q-values for the next state
                q_next = self.Q.forward(next_state).max(1)[0]

            # Compute target Q-values
            q_target = reward + self.gamma * q_next * (1 - done)

        # Compute loss
        q_loss = self.loss_function(q_value, q_target)

        # Perform gradient descent
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

    def update_tn(self):
        if self.tn:
            # Update target network with the weights of the current network
            self.Q_target.load_state_dict(self.Q.state_dict())
