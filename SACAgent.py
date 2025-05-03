import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim, log_softmax=False):
        super().__init__()
        self.log_softmax = log_softmax
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state):
        logits = self.net(state)
        if self.log_softmax:
            return torch.log_softmax(logits, dim=-1)
        else:
            return logits


class SACAgent:
    def __init__(self, state_dim, n_actions, lr, gamma, hidden_dim, alpha, buffer_size, batch_size, tau,
                 full_expectation, double_q):
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.full_expectation = full_expectation
        self.double_q = double_q

        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network: outputs a probability distribution over actions
        self.pi = NeuralNetwork(state_dim, n_actions, hidden_dim, log_softmax=True).to(self.device)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=lr)

        # Q network and target network
        self.Q1 = NeuralNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.Q1_optim = optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q1_target = NeuralNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.Q1_target.load_state_dict(self.Q1.state_dict())

        # Double Q trick
        if self.double_q:
            self.Q2 = NeuralNetwork(state_dim, n_actions, hidden_dim).to(self.device)
            self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=lr)
            self.Q2_target = NeuralNetwork(state_dim, n_actions, hidden_dim).to(self.device)
            self.Q2_target.load_state_dict(self.Q2.state_dict())

        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.update_count = 0

    def select_action_sample(self, state):
        # Convert state to tensor and forward through policy network to get probabilities
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action_probs = self.pi(state)
        # Create a categorical distribution from the probabilities and sample an action
        m = Categorical(action_probs)
        action = m.sample().item()
        return action

    def select_action_greedy(self, state):
        # For evaluation, choose the action with the highest probability
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action_probs = self.pi(state)
        action = torch.argmax(action_probs).item()
        return action

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Convert numpy arrays to torch tensors
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(-1)
        reward = torch.tensor(reward, dtype=torch.int, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.int, device=self.device)

        with torch.no_grad():
            q1_next_state = self.Q1_target(next_state)
            if self.double_q:
                q2_next_state = self.Q2_target(next_state)
                q_next_state = torch.min(q1_next_state, q2_next_state)
            else:
                q_next_state = q1_next_state

            log_probs_next_action = self.pi(next_state)
            probs_next_action = torch.exp(log_probs_next_action)
            if self.full_expectation:
                v_next_action = torch.sum(probs_next_action * (q_next_state - self.alpha * log_probs_next_action),
                                          dim=1, keepdim=True)
            else:
                next_action_sample = torch.multinomial(probs_next_action, num_samples=1)
                log_prob_next_action_sample = torch.gather(log_probs_next_action, 1, next_action_sample)
                q_next_state_action = torch.gather(q_next_state, 1, next_action_sample)
                v_next_action = q_next_state_action - self.alpha * log_prob_next_action_sample

            q_target = reward + self.gamma * (1 - done) * v_next_action

        q1_values = self.Q1(state)
        q1_current = torch.gather(q1_values, 1, action)

        q1_loss = F.mse_loss(q1_current, q_target)
        self.Q1_optim.zero_grad()
        q1_loss.backward()
        self.Q1_optim.step()

        if self.double_q:
            q2_values = self.Q2(state)
            q2_current = torch.gather(q2_values, 1, action)

            q2_loss = F.mse_loss(q2_current, q_target)
            self.Q2_optim.zero_grad()
            q2_loss.backward()
            self.Q2_optim.step()


if __name__ == "__main__":
    test = SACAgent(state_dim=4, n_actions=2, lr=0.001, gamma=1, hidden_dim=128, alpha=0.2, buffer_size=100000,
                    batch_size=5, tau=0.005, full_expectation=True, double_q=True)
    while True:
        test.add_experience(np.random.sample(4), 1, 1, np.random.sample(4), 0)
        test.update()
