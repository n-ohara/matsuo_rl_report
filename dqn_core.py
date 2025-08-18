import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class Dqn:
    def __init__(self, dim_state, num_action, memory_size=10000, target_update_freq=30, gamma=0.99, lr=1e-3):
        self.dim_state = dim_state
        self.num_action = num_action
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.replay_buffer = ReplayBuffer(memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(dim_state, num_action).to(self.device)
        self.target_net = QNetwork(dim_state, num_action).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.learn_step = 0

    def get_action(self, state, episode):
        epsilon = max(0.1, 1.0 - episode / 100)
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_action)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update_q(self):
        if len(self.replay_buffer.buffer) < 32:
            return

        batch = self.replay_buffer.sample(32)
        #states = torch.FloatTensor([t['state'] for t in batch]).squeeze(1).to(self.device)
        #next_states = torch.FloatTensor([t['next_state'] for t in batch]).squeeze(1).to(self.device)
        states = torch.FloatTensor([t['state'] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch]).to(self.device)

        rewards = torch.FloatTensor(np.array([t['reward'] for t in batch])).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array([t['done'] for t in batch])).unsqueeze(1).to(self.device)

        actions = torch.LongTensor([t['action'] for t in batch]).unsqueeze(1).to(self.device)
        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)  # [32, 1]

        # TDターゲット
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # 損失計算
        loss = F.mse_loss(q_values, target_q_values) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
