import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.amp import autocast, GradScaler


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
      return torch.log_softmax(logits, dim=-1) if self.log_softmax else logits


class SACAgent:
   def __init__(self, state_dim, n_actions, lr, gamma, hidden_dim, alpha, buffer_size, batch_size,
                learning_starts, tau, full_expectation, double_q, update_freq, update_num):
      self.gamma = gamma
      self.alpha = alpha
      self.buffer_size = buffer_size
      self.batch_size = batch_size
      self.learning_starts = learning_starts
      self.tau = tau

      self.full_expectation = full_expectation
      self.double_q = double_q

      # Set device (GPU if available, else CPU)
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.use_amp = not torch.cuda.amp.amp_definitely_not_available()

      # Policy network: outputs a probability distribution over actions
      self.pi = NeuralNetwork(state_dim, n_actions, hidden_dim, log_softmax=True).to(self.device)
      self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=lr)

      # Q network and target network
      self.Q1 = NeuralNetwork(state_dim, n_actions, hidden_dim).to(self.device)
      self.Q1_optim = optim.Adam(self.Q1.parameters(), lr=lr)
      self.Q1_target = NeuralNetwork(state_dim, n_actions, hidden_dim).to(self.device)
      self.Q1_target.load_state_dict(self.Q1.state_dict())

      # AMP Gradscalers
      self.scaler_Q = GradScaler()
      self.scaler_pi = GradScaler()

      # Double Q trick
      if self.double_q:
         self.Q2 = NeuralNetwork(state_dim, n_actions, hidden_dim).to(self.device)
         self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=lr)
         self.Q2_target = NeuralNetwork(state_dim, n_actions, hidden_dim).to(self.device)
         self.Q2_target.load_state_dict(self.Q2.state_dict())

      # Initialize replay buffer
      self.replay_buffer = {
         "s": torch.empty((self.buffer_size, state_dim), dtype=torch.float, device=self.device),
         "a": torch.empty((self.buffer_size, 1), dtype=torch.int64, device=self.device),
         "r": torch.empty((self.buffer_size, 1), dtype=torch.float, device=self.device),
         "next_s": torch.empty((self.buffer_size, state_dim), dtype=torch.float,
                               device=self.device),
         "d": torch.empty((self.buffer_size, 1), dtype=torch.float, device=self.device),
      }
      self.buffer_index = 0
      self.buffer_update_count = 0

   def add_experience(self, s, a, r, next_s, d):
      idx = self.buffer_index % self.buffer_size
      if not torch.is_tensor(s):
         s = torch.as_tensor(s, dtype=torch.float, device=self.device)
      if not torch.is_tensor(a):
         a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
      if not torch.is_tensor(r):
         r = torch.as_tensor(r, dtype=torch.float, device=self.device)
      if not torch.is_tensor(next_s):
         next_s = torch.as_tensor(next_s, dtype=torch.float, device=self.device)
      if not torch.is_tensor(d):
         d = torch.as_tensor(d, dtype=torch.float, device=self.device)
      self.replay_buffer["s"][idx].copy_(s)
      self.replay_buffer["a"][idx].copy_(a)
      self.replay_buffer["r"][idx].copy_(r)
      self.replay_buffer["next_s"][idx].copy_(next_s)
      self.replay_buffer["d"][idx].copy_(d)
      self.buffer_index += 1
      self.buffer_update_count += 1

   def sample_batch(self):
      indices = torch.randint(0, min(self.buffer_index, self.buffer_size), (self.batch_size,),
                              device=self.device)
      return {key: value[indices] for key, value in self.replay_buffer.items()}

   def select_action_sample(self, s):
      # Convert state to tensor and forward through policy network to get probabilities
      if not torch.is_tensor(s):
         s = torch.tensor(s, dtype=torch.float, device=self.device)
      with torch.no_grad():
         log_probs_s = self.pi(s)
      probs_s = log_probs_s.exp()
      return torch.multinomial(probs_s, num_samples=1).item()

   def select_action_greedy(self, s):
      # For evaluation, choose the action with the highest probability
      if not torch.is_tensor(s):
         s = torch.tensor(s, dtype=torch.float, device=self.device)
      with torch.no_grad():
         log_probs_s = self.pi(s)
      return torch.argmax(log_probs_s).item()

   def update(self):
      if self.buffer_update_count < self.learning_starts:
         return

      batch = self.sample_batch()
      s = batch["s"]
      a = batch["a"]
      r = batch["r"]
      next_s = batch["next_s"]
      d = batch["d"]

      q_target = self.compute_Q_target(next_s, r, d)
      self.update_Q_networks(s, a, q_target)
      self.update_policy(s)
      self.update_target_networks()

   def compute_Q_target(self, next_s, r, d):
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
            v_next_s = (probs_next_s * (q_next_s - self.alpha * log_probs_next_s)).sum(dim=1,
                                                                                       keepdim=True)
         else:
            sampled_next_a = torch.multinomial(probs_next_s, num_samples=1)
            log_probs_next_sa = log_probs_next_s.gather(1, sampled_next_a)
            q_next_sa = q_next_s.gather(1, sampled_next_a)
            v_next_s = q_next_sa - self.alpha * log_probs_next_sa

         q_target = r + self.gamma * (1 - d) * v_next_s
      return q_target

   def update_Q_networks(self, s, a, q_target):
      self.Q1_optim.zero_grad()
      with autocast(device_type=self.device.type):
         q1_s = self.Q1(s)
         q1_sa = q1_s.gather(1, a)
         q1_loss = F.mse_loss(q1_sa, q_target)
      self.scaler_Q.scale(q1_loss).backward()
      self.scaler_Q.step(self.Q1_optim)
      self.scaler_Q.update()

      if self.double_q:
         self.Q2_optim.zero_grad()
         with autocast(device_type=self.device.type):
            q2_s = self.Q2(s)
            q2_sa = q2_s.gather(1, a)
            q2_loss = F.mse_loss(q2_sa, q_target)
         self.scaler_Q.scale(q2_loss).backward()
         self.scaler_Q.step(self.Q2_optim)
         self.scaler_Q.update()

   def update_policy(self, s):
      self.optimizer_pi.zero_grad()
      with autocast(device_type=self.device.type):
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
            policy_loss = (log_probs_sa * (self.alpha * log_probs_sa - q_sa)).mean()
      self.scaler_pi.scale(policy_loss).backward()
      self.scaler_pi.step(self.optimizer_pi)
      self.scaler_pi.update()

   def update_target_networks(self):
      for p, tp in zip(self.Q1.parameters(), self.Q1_target.parameters()):
         tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
      if self.double_q:
         for p, tp in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
