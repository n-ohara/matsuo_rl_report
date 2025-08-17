from market_env import MarketEnvironment
from dqn_trader import DQNTrader
import numpy as np
import matplotlib.pyplot as plt

env = MarketEnvironment()
agent = DQNTrader(env)

num_episode = 100
initial_memory_size = 500
episode_rewards = []

# 初期メモリ構築
state = env.reset()
for _ in range(initial_memory_size):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    agent.append({
        'state': state,
        'next_state': next_state,
        'reward': reward,
        'action': action,
        'done': int(done)
    })
    state = env.reset() if done else next_state

# 学習ループ
for episode in range(num_episode):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state, episode)
        next_state, reward, done, _ = env.step(action)
        agent.append({
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'done': int(done)
        })
        agent.learn()
        state = next_state
        total_reward += reward
    episode_rewards.append(total_reward)
    if episode % 10 == 0:
        print(f"Episode {episode}: Reward {total_reward:.2f}")

# 可視化
moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
plt.plot(moving_avg)
plt.title("DQN: Moving Average of Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
