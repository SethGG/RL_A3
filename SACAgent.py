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
        self.tau = tau

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

    def select_action_sample(self, s):
        # Convert state to tensor and forward through policy network to get probabilities
        s = torch.tensor(s, dtype=torch.float, device=self.device)
        with torch.no_grad():
            log_probs_s = self.pi(s)
        probs_s = log_probs_s.exp()
        a = torch.multinomial(probs_s, num_samples=1).item()
        return a

    def select_action_greedy(self, s):
        # For evaluation, choose the action with the highest probability
        s = torch.tensor(s, dtype=torch.float, device=self.device)
        with torch.no_grad():
            log_probs_s = self.pi(s)
        a = torch.argmax(log_probs_s).item()
        return a

    def add_experience(self, s, a, r, next_s, d):
        self.replay_buffer.append((s, a, r, next_s, d))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        s, a, r, next_s, d = zip(*batch)

        # Convert numpy arrays to torch tensors
        s = torch.tensor(s, dtype=torch.float, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(-1)
        r = torch.tensor(r, dtype=torch.int, device=self.device).unsqueeze(-1)
        next_s = torch.tensor(next_s, dtype=torch.float, device=self.device)
        d = torch.tensor(d, dtype=torch.int, device=self.device).unsqueeze(-1)

        # Calculate Q-network targets
        with torch.no_grad():
            q1_next_s = self.Q1_target(next_s)
            if self.double_q:
                q2_next_s = self.Q2_target(next_s)
                q_next_s = torch.min(q1_next_s, q2_next_s)
            else:
                q_next_s = q1_next_s

            log_probs_next_s = self.pi(next_s)
            probs_next_s = log_probs_next_s.exp()
            if self.full_expectation:
                v_next_s = (probs_next_s * (q_next_s - self.alpha * log_probs_next_s)).sum(dim=1, keepdim=True)
            else:
                sampled_next_a = torch.multinomial(probs_next_s, num_samples=1)
                log_probs_next_sa = log_probs_next_s.gather(1, sampled_next_a)
                q_next_sa = q_next_s.gather(1, sampled_next_a)
                v_next_s = q_next_sa - self.alpha * log_probs_next_sa

            q_target = r + self.gamma * (1 - d) * v_next_s

        # Gather current Q-netwok values
        q1_s = self.Q1(s)
        q1_sa = q1_s.gather(1, a)

        # Update Q-networks
        q1_loss = F.mse_loss(q1_sa, q_target)
        self.Q1_optim.zero_grad()
        q1_loss.backward()
        self.Q1_optim.step()

        if self.double_q:
            q2_s = self.Q2(s)
            q2_sa = q2_s.gather(1, a)

            q2_loss = F.mse_loss(q2_sa, q_target)
            self.Q2_optim.zero_grad()
            q2_loss.backward()
            self.Q2_optim.step()

        # Policy update
        log_probs_s = self.pi(s)
        probs_s = log_probs_s.exp()
        q1_s = self.Q1(s)
        if self.double_q:
            q2_s = self.Q2(s)
            q_s = torch.min(q1_s, q2_s)
        else:
            q_s = q1_s

        if self.full_expectation:
            policy_loss = (probs_s * (self.alpha * log_probs_s - q_s)).sum(dim=1).mean()
        else:
            sampled_a = torch.multinomial(probs_s, num_samples=1)
            log_probs_sa = log_probs_s.gather(1, sampled_a)
            q_sa = q_s.gather(1, sampled_a)
            policy_loss = (self.alpha * log_probs_sa - q_sa).mean()

        self.optimizer_pi.zero_grad()
        policy_loss.backward()
        self.optimizer_pi.step()

        # Target network update
        for p, tp in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        if self.double_q:
            for p, tp in zip(self.Q2.parameters(), self.Q2_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
