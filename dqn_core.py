import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, dim_state, num_action, hidden_size=16):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(dim_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_action)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))
        y = self.fc4(h)  # 最後は線形でOK
        return y

class ReplayBuffer:
    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.memory), size=batch_size)
        batch = [self.memory[i] for i in idxs]
        states      = np.array([b['state'] for b in batch])
        next_states = np.array([b['next_state'] for b in batch])
        rewards     = np.array([b['reward'] for b in batch])
        actions     = np.array([b['action'] for b in batch])
        dones       = np.array([b['done'] for b in batch])
        return {'states': states, 'next_states': next_states, 'rewards': rewards, 'actions': actions, 'dones': dones}

class Dqn:
    def __init__(self, dim_state, num_action, gamma=0.99, lr=0.001, batch_size=32, memory_size=50000, target_update_freq=30):
        self.dim_state = dim_state
        self.num_action = num_action
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.qnet = QNetwork(dim_state, num_action)
        self.target_qnet = QNetwork(dim_state, num_action)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)
        self._target_qnet_update_count = 0

    def update_q(self):
        if len(self.replay_buffer.memory) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        q = self.qnet(torch.tensor(batch["states"], dtype=torch.float)).gather(1, torch.tensor(batch["actions"]).reshape(-1, 1))

        with torch.no_grad():
            maxq = torch.max(self.target_qnet(torch.tensor(batch["next_states"], dtype=torch.float)), dim=1).values
            target = torch.tensor(batch["rewards"], dtype=torch.float) + self.gamma * maxq * torch.tensor(~batch["dones"], dtype=torch.bool)

        loss = nn.MSELoss()(q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._target_qnet_update_count += 1
        if self._target_qnet_update_count % self.target_update_freq == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).view(-1, self.dim_state)
        return torch.argmax(self.qnet(state_tensor)).item()

    def get_action(self, state, episode):
        epsilon = 0.7 * (1 / (episode + 1))
        if np.random.rand() > epsilon:
            return self.get_greedy_action(state)
        else:
            return np.random.choice(self.num_action)
