import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# QNetwork（Critic）
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.q_out(x)

# Actorネットワーク
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.logits(x)

# ReplayBuffer（簡易版）
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def append(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

# DiscreteSACAgent
class DiscreteSACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, alpha=0.2, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)

        self.gamma = gamma
        self.alpha = alpha
        self.replay_buffer = None
        self.episode = 0

    def set_buffer(self, buffer):
        self.replay_buffer = buffer
    '''
    def append(self, transition):
        if self.replay_buffer is not None:
            self.replay_buffer.append(transition)
    '''
    def append(self, transition):
        if self.replay_buffer is not None:
            self.replay_buffer.append((
                transition['state'],
                transition['action'],
                transition['reward'],
                transition['next_state'],
                transition['done']
            ))


    def set_episode(self, episode):
        self.episode = episode

    def get_action(self, state, eval=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_pi = dist.log_prob(action)
        return action.item(), log_pi.item()

    def learn(self, batch_size=64):
        if self.replay_buffer is None or len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Critic update
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs + 1e-8)

            q1_next = self.q1_target(next_states)
            q2_next = self.q2_target(next_states)
            min_q_next = torch.min(q1_next, q2_next)

            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + self.gamma * (1 - dones) * v_next

        q1 = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Actor update
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)

        q1_values = self.q1(states)
        actor_loss = (probs * (self.alpha * log_probs - q1_values)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        tau = 0.005
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# sac_core.py の末尾に追加してください

class SAC:
    def __init__(self, dim_state, num_action):
        self.agent = DiscreteSACAgent(state_dim=dim_state, action_dim=num_action)
        self.buffer = ReplayBuffer()
        self.agent.set_buffer(self.buffer)

    def get_action(self, state, episode=None):
        self.agent.set_episode(episode)
        action, log_pi = self.agent.get_action(state)
        return action, log_pi

    def append(self, transition):
        self.agent.append(transition)

    def learn(self):
        self.agent.learn()

    def set_episode(self, episode):
        self.agent.set_episode(episode)
