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

    def evaluate_log_pi(self, states, actions):
        probs = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)

    def sample(self, states):
        probs = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

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
        return self.net(states)

class RolloutBuffer:
    def __init__(self):
        self.buffer = []

    def append(self, state, action, reward, done, log_pi, next_state):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done,
            'log_pi': log_pi.detach().cpu(),# ←ここをTensorとして保存log_pi,
            'next_state': next_state
        })

    def get(self):
        batch = self.buffer
        self.buffer = []
        return batch

class PPO:
    def __init__(self, dim_state, num_action, gamma=0.99, clip_eps=0.2, lr=3e-4, batch_size=64, num_epochs=4):
        self.gamma = gamma
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
        if episode is not None and episode < 20:  # 初期20エピソードはランダム行動
            action = np.random.randint(0, self.actor.net[-1].out_features)
            log_pi = torch.tensor(0.0)  # ダミー
            return action, log_pi
        with torch.no_grad():
            action, log_pi = self.actor.sample(state_tensor)
        return action.item(), log_pi.detach().cpu() if isinstance(log_pi, torch.Tensor) else torch.tensor(log_pi)

    def append(self, transition):
        self.buffer.append(**transition)
    
    def set_episode(self, episode):
        self._episode_counter = episode

    def learn(self):
        batch = self.buffer.get()
        if len(batch) < self.batch_size:
            return

        states = torch.FloatTensor([b['state'] for b in batch]).to(self.device)
        actions = torch.LongTensor([b['action'] for b in batch]).to(self.device)
        rewards = [b['reward'] for b in batch]
        dones = [b['done'] for b in batch]
        log_pis_old = torch.FloatTensor([b['log_pi'] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b['next_state'] for b in batch]).to(self.device)

        # Compute returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)

        # Compute advantage
        values = self.critic(states).squeeze()
        advantage = returns - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Training loop
        for _ in range(self.num_epochs):
            indices = np.arange(len(batch))
            np.random.shuffle(indices)

            for start in range(0, len(batch), self.batch_size):
                idx = indices[start:start + self.batch_size]
                s_batch = states[idx]
                a_batch = actions[idx]
                r_batch = returns[idx]
                adv_batch = advantage[idx]
                log_pi_old_batch = log_pis_old[idx]

                # Critic update
                value_pred = self.critic(s_batch).squeeze()
                loss_critic = nn.MSELoss()(value_pred, r_batch)
                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                self.optimizer_critic.step()

                # Actor update
                log_pi = self.actor.evaluate_log_pi(s_batch, a_batch)
                ratio = torch.exp(log_pi - log_pi_old_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch
                loss_actor = -torch.min(surr1, surr2).mean()

                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()
        # === 学習状況のログ出力（10エピソードごと） ===
        if hasattr(self, "_episode_counter") and self._episode_counter % 10 == 0:
            with torch.no_grad():
                log_pi_new = self.actor.evaluate_log_pi(states, actions)
                prob_ratio = torch.exp(log_pi_new - log_pis_old)
                print(f"[Ep {self._episode_counter}] "
                    f"ActorLoss: {loss_actor.item():.4f} | "
                    f"ProbRatio μ={prob_ratio.mean().item():.3f} σ={prob_ratio.std().item():.3f} | "
                    f"Advantage μ={advantage.mean().item():.4f} σ={advantage.std().item():.4f} | "
                    f"Returns μ={returns.mean().item():.2f} | Values μ={values.mean().item():.2f}")
