import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOActor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, states):
        return self.net(states)

    def sample(self, states):
        probs = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def evaluate_log_pi(self, states, actions):
        probs = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)

class PPOCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, states):
        return self.net(states).squeeze(-1)

class RolloutBuffer:
    def __init__(self):
        self.buffer = []

    def append(self, state, action, reward, done, log_pi, next_state):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done,
            'log_pi': log_pi,
            'next_state': next_state
        })

    def get(self):
        batch = self.buffer
        self.buffer = []
        return batch

class PPO:
    def __init__(self, dim_state, num_action, gamma=0.99, lambd=0.95, clip_eps=0.2, lr=3e-4, batch_size=64, num_epochs=10):
        self.gamma = gamma
        self.lambd = lambd
        self.clip_eps = clip_eps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = PPOActor(dim_state, num_action).to(self.device)
        self.critic = PPOCritic(dim_state).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def get_action(self, state, episode=None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state_tensor)
        return action.item(), log_pi.item()

    def append(self, transition):
        self.buffer.append(**transition)

    def set_episode(self, episode):
        self._episode_counter = episode

    def compute_gae(self, rewards, dones, values, next_values):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            advantages[t] = deltas[t] + self.gamma * self.lambd * (1 - dones[t]) * last_adv
            last_adv = advantages[t]
        return advantages

    def learn(self):
        batch = self.buffer.get()
        if len(batch) < self.batch_size:
            return

        states = torch.FloatTensor([b['state'] for b in batch]).to(self.device)
        actions = torch.LongTensor([b['action'] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b['reward'] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b['done'] for b in batch]).to(self.device)
        log_pis_old = torch.FloatTensor([b['log_pi'] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b['next_state'] for b in batch]).to(self.device)

        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        advantages = self.compute_gae(rewards, dones, values, next_values)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.num_epochs):
            indices = np.arange(len(batch))
            np.random.shuffle(indices)

            for start in range(0, len(batch), self.batch_size):
                idx = indices[start:start + self.batch_size]
                s_batch = states[idx]
                a_batch = actions[idx]
                r_batch = returns[idx]
                adv_batch = advantages[idx]
                log_pi_old_batch = log_pis_old[idx]

                value_pred = self.critic(s_batch)
                loss_critic = nn.MSELoss()(value_pred, r_batch)
                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                self.optimizer_critic.step()

                log_pi = self.actor.evaluate_log_pi(s_batch, a_batch)
                ratio = torch.exp(log_pi - log_pi_old_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
                loss_actor = -torch.min(surr1, surr2).mean()

                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()

        if hasattr(self, "_episode_counter") and self._episode_counter % 10 == 0:
            with torch.no_grad():
                log_pi_new = self.actor.evaluate_log_pi(states, actions)
                prob_ratio = torch.exp(log_pi_new - log_pis_old)
                #print(f"[Ep {self._episode_counter}] "
                #      f"ActorLoss: {loss_actor.item():.4f} | "
                #      f"ProbRatio μ={prob_ratio.mean().item():.3f} σ={prob_ratio.std().item():.3f} | "
                #      f"Advantage μ={advantages.mean().item():.4f} σ={advantages.std().item():.4f} | "
                #      f"Returns μ={returns.mean().item():.2f} | Values μ={values.mean().item():.2f}")
