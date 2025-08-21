from market_env import MarketEnvironment
import numpy as np
import matplotlib.pyplot as plt

from trader import Trader
from dqn_core import Dqn

num_traders = 1

env = MarketEnvironment(num_traders=num_traders)

agents = []
for i in range(num_traders):
    agent = Dqn(
        dim_state=env.observation_space.shape[0],
        num_action=env.action_space.n,
        memory_size=50000,
        target_update_freq=30
    )
    trader = Trader(env, agent=agent, trader_id=i)
    agents.append(trader)#ここまで

num_episode = 80000#100
initial_memory_size = 500
episode_rewards = [[] for _ in range(num_traders)]

# 初期メモリ構築
states = env.reset()
for _ in range(initial_memory_size):
    actions = [env.action_space.sample() for _ in range(num_traders)]
    next_states, rewards, done, _ = env.step(actions)

    for i in range(num_traders):
        agents[i].append({
            'state': states[i],
            'next_state': next_states[i],
            'reward': rewards[i],
            'action': actions[i],
            'done': int(done)
        })

    states = env.reset() if done else next_states

# 学習ループ
for episode in range(num_episode):
    states = env.reset()
    done = False
    total_rewards = [0 for _ in range(num_traders)]

    while not done:
        actions = [agents[i].act(states[i], episode) for i in range(num_traders)]
        next_states, rewards, done, _ = env.step(actions)

        for i in range(num_traders):
            agents[i].append({
                'state': states[i],
                'next_state': next_states[i],
                'reward': rewards[i],
                'action': actions[i],
                'done': int(done)
            })
            agents[i].learn()
            total_rewards[i] += rewards[i]  # ✅ rewards[i] は float 型

        states = next_states

    for i in range(num_traders):
        episode_rewards[i].append(total_rewards[i])
    if episode % 10 == 0:
        print(f"Episode {episode}: Rewards {[f'{r:.2f}' for r in total_rewards]}")

# 可視化
for i in range(num_traders):
    moving_avg = np.convolve(episode_rewards[i], np.ones(10)/10, mode='valid')
    plt.plot(moving_avg, label=f"Trader {i}")
plt.title("DQN: Moving Average of Rewards per Trader")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()