import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim, softmax_output):
        super(NeuralNet, self).__init__()
        self.softmax_output = softmax_output
        # Define a two-layer hidden network with one output layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # Move the model to the specified device (CPU or GPU)
        self.to(device)

    def forward(self, x):
        # Forward pass through the network:
        # 1. Pass through first fully connected layer then apply ReLU
        x = torch.relu(self.fc1(x))
        # 2. Pass through second fully connected layer then apply ReLU
        x = torch.relu(self.fc2(x))
        # 3. Compute raw outputs and apply softmax if needed (for probability distributions)
        if self.softmax_output:
            x = torch.softmax(self.fc3(x), dim=-1)
        else:
            x = self.fc3(x)
        return x


class REINFORCEAgent:
    def __init__(self, n_actions, n_states, alpha, gamma, hidden_dim, normalize):
        self.gamma = gamma  # Discount factor
        self.normalize = normalize  # Whether to normalize returns before update

        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Policy network: outputs a probability distribution over actions
        self.pi = NeuralNet(n_states, n_actions, self.device, hidden_dim, softmax_output=True)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=alpha)

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

    def update(self, trace_states, trace_actions, trace_rewards):
        # ------------------------- Step 1: Compute Discounted Returns -------------------------
        # Calculate the total discounted return for every timestep in the episode.
        trace_returns = deque()
        R_t = 0
        # Process rewards in reverse order for discounting
        for r in reversed(trace_rewards):
            R_t = r + self.gamma * R_t
            trace_returns.appendleft(R_t)
        # Convert the deque to a tensor
        trace_returns = torch.tensor(trace_returns, dtype=torch.float, device=self.device)

        # -------------------- Step 2: (Optional) Normalize the Returns --------------------
        if self.normalize:
            # Normalize to have mean 0 and unit variance.
            trace_returns = (trace_returns - trace_returns.mean()) / trace_returns.std()

        # -------------------- Step 3: Prepare States and Actions for Loss Computation --------------------
        # Convert the list of states and actions from the episode to tensors.
        trace_states = torch.tensor(trace_states, dtype=torch.float, device=self.device)
        trace_actions = torch.tensor(trace_actions, dtype=torch.int, device=self.device)

        # -------------------- Step 4: Compute Policy Loss --------------------
        # Get action probabilities from the policy network for all states in the episode.
        trace_action_probs = self.pi.forward(trace_states)
        # Create a categorical distribution to compute log probabilities.
        trace_m = Categorical(trace_action_probs)
        trace_log_probs = trace_m.log_prob(trace_actions)

        # Compute the policy loss, weighted by the (discounted and normalized) return.
        # A negative sign is used because we want to maximize expected return (i.e. perform gradient ascent).
        policy_loss = (-1 * trace_log_probs * trace_returns).sum()

        # -------------------- Step 5: Backpropagation and Policy Update --------------------
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


class ActorCriticAgent:
    def __init__(self, n_actions, n_states, alpha, gamma, hidden_dim, estim_depth, update_episodes, use_advantage):
        self.gamma = gamma  # Discount factor
        self.estim_depth = estim_depth  # Number of steps to use for multi-step bootstrapping
        self.update_episodes = update_episodes  # How many episodes to accumulate before an update
        self.use_advantage = use_advantage  # Flag to decide between advantage or full-return for policy loss

        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Policy network: outputs a probability distribution over actions
        self.pi = NeuralNet(n_states, n_actions, self.device, hidden_dim, softmax_output=True)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=alpha)
        # Value network: predicts a value (scalar) for a given state
        self.V = NeuralNet(n_states, 1, self.device, hidden_dim, softmax_output=False)
        self.optimizer_V = optim.Adam(self.V.parameters(), lr=alpha)

        # Initialize experience buffers for batch updating
        self.__reset_update_buffer()

    def __reset_update_buffer(self):
        # Reset accumulators for states, actions, and Q estimates.
        self.update_count = 0
        self.update_states = torch.empty(0, dtype=torch.float, device=self.device)
        self.update_actions = torch.empty(0, dtype=torch.int, device=self.device)
        self.Q_hat = torch.empty(0, dtype=torch.float, device=self.device)

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

    def update(self, trace_states, trace_actions, trace_rewards):
        # Convert the trace (episode) state and action data to tensors
        trace_states = torch.tensor(trace_states, dtype=torch.float, device=self.device)
        trace_actions = torch.tensor(trace_actions, dtype=torch.int, device=self.device)

        # -------------------- Step 1: n-Step Bootstrapped Return Calculation --------------------
        if self.estim_depth < len(trace_rewards):
            # Use the value network to get bootstrapped estimates for states beyond the estimation depth
            V_target_states = trace_states[self.estim_depth:]
            with torch.no_grad():
                V_target_pred = self.V.forward(V_target_states)
            # Prepare a list of n-step reward sequences
            n_step_rewards = [trace_rewards[i:i+self.estim_depth] for i in range(len(trace_rewards) - self.estim_depth)]
            n_step_rewards = torch.tensor(n_step_rewards, dtype=torch.float, device=self.device)
            # Initialize n-step returns using the bootstrapped values (squeezed to remove extra dimensions)
            n_step_returns = V_target_pred.squeeze(-1)
            # Apply discounting over the n-step rewards (from the end, reverse order)
            for step_rewards in reversed(n_step_rewards.T):
                n_step_returns *= self.gamma
                n_step_returns += step_rewards
            # Concatenate these n-step returns to the buffer Q_hat
            self.Q_hat = torch.cat((self.Q_hat, n_step_returns))

        # -------------------- Step 2: Terminal Monte-Carlo Return Calculation --------------------
        # For the final few steps (which can't form a full n-step segment), compute Monte-Carlo returns.
        def mc_discounted_returns(rewards):
            returns = deque()
            R_t = 0
            for r in reversed(rewards):
                R_t = r + self.gamma * R_t
                returns.appendleft(R_t)
            return returns

        terminal_rewards = trace_rewards[-self.estim_depth:]
        terminal_returns = torch.tensor(mc_discounted_returns(terminal_rewards), dtype=torch.float, device=self.device)
        # Append terminal Monte-Carlo returns to Q_hat.
        self.Q_hat = torch.cat((self.Q_hat, terminal_returns))

        # -------------------- Step 3: Accumulate Experiences for Batch Update --------------------
        self.update_states = torch.cat((self.update_states, trace_states))
        self.update_actions = torch.cat((self.update_actions, trace_actions))
        self.update_count += 1

        # Only update the networks after a specified number of episodes have been accumulated.
        if self.update_count < self.update_episodes:
            return

        # -------------------- Step 4: Critic (Value Network) Update --------------------
        # Compute current value estimates for the buffered states.
        V_current = self.V.forward(self.update_states).squeeze(-1)
        # Advantage estimation: difference between bootstrapped Q_hat and current value prediction.
        A_hat = self.Q_hat - V_current
        # Use sum squared error loss for the value network.
        V_loss = A_hat.pow(2).sum() / self.update_episodes

        self.optimizer_V.zero_grad()
        V_loss.backward()
        self.optimizer_V.step()

        # -------------------- Step 5: Actor (Policy Network) Update --------------------
        # Recompute action probabilities and log probabilities for the buffered states.
        pi_action_probs = self.pi.forward(self.update_states)
        pi_m = Categorical(pi_action_probs)
        pi_log_probs = pi_m.log_prob(self.update_actions)

        if self.use_advantage:
            # Use advantage to weight the policy loss; detach it so that gradients do not flow into the critic.
            pi_loss = (-1 * pi_log_probs * A_hat.detach()).sum() / self.update_episodes
        else:
            # Alternatively, use the full Q_hat (this corresponds to vanilla actor-critic).
            pi_loss = (-1 * pi_log_probs * self.Q_hat).sum() / self.update_episodes

        self.optimizer_pi.zero_grad()
        pi_loss.backward()
        self.optimizer_pi.step()

        # -------------------- Step 6: Reset Buffer for Next Batch --------------------
        self.__reset_update_buffer()
