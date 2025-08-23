from market_env import MarketEnvironment
import numpy as np
import matplotlib.pyplot as plt

from trader import Trader
from dqn_core import Dqn
from ppo_core_no_gae import PPO as PPO_NoGAE
from ppo_core_gae import PPO as PPO_GAE

#example "dqn","ppo_no_gae","ppo_gae"
agent_types = ["ppo_no_gae","ppo_gae"]

num_traders = len(agent_types)
env = MarketEnvironment(num_traders=num_traders)

agents = []
for i in range(num_traders):
    agent_type = agent_types[i]
    if agent_type == "dqn":
        agent = Dqn(dim_state=env.observation_space.shape[0], num_action=env.action_space.n)
    elif agent_type == "ppo_no_gae":
        agent = PPO_NoGAE(dim_state=env.observation_space.shape[0], num_action=env.action_space.n)
    elif agent_type == "ppo_gae":
        agent = PPO_GAE(dim_state=env.observation_space.shape[0], num_action=env.action_space.n)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    trader = Trader(env, agent=agent, trader_id=i)
    agents.append(trader)

num_episode = 100
initial_memory_size = 500
episode_rewards = [[] for _ in range(num_traders)]

# 初期メモリ構築は DQN のみ
if "dqn" in agent_types:
    states = env.reset()
    for _ in range(initial_memory_size):
        actions = [env.action_space.sample() for _ in range(num_traders)]
        next_states, rewards, done, _ = env.step(actions)

        for i in range(num_traders):
            if isinstance(agents[i].agent, Dqn):  # DQN のみ append
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

    trajectory = [[] for _ in range(num_traders)]  #PPO用の軌跡保存
    
    while not done:
        #PPOとDQNを区別して actions と log_pis を構築
        actions = []
        log_pis = []
        for i in range(num_traders):
            if "ppo" in agent_types[i]:
                action, log_pi = agents[i].agent.get_action(states[i])
            else:
                action = agents[i].act(states[i], episode)
                log_pi = None
            actions.append(action)
            log_pis.append(log_pi)
        
        next_states, rewards, done, _ = env.step(actions)
        
        for i in range(num_traders):
            transition = {
                'state': states[i],
                'next_state': next_states[i],
                'reward': rewards[i],
                'action': actions[i],
                'done': int(done),
                'log_pi': log_pis[i]  # 🔧 修正②: PPO用 log_pi を保存
                }
            
            if isinstance(agents[i].agent, Dqn):
                agents[i].append(transition)
                agents[i].learn()  # DQNはステップごとに学習
            else:
                agents[i].agent.set_episode(episode)  # PPOインスタンスに現在のエピソード番号を渡す[i]
                trajectory[i].append(transition)  # PPOは後でまとめて学習
                
            total_rewards[i] += rewards[i]
        states = next_states

    # PPOはエピソード終了後にまとめて学習
    for i in range(num_traders):
        if "ppo" in agent_types[i]:
            for t in trajectory[i]:
                agents[i].append(t)
            agents[i].learn()
        episode_rewards[i].append(total_rewards[i])

    if episode % 10 == 0:
        print(f"Episode {episode}: Rewards {[f'{r:.2f}' for r in total_rewards]}")

# 可視化
for i in range(num_traders):
    moving_avg = np.convolve(episode_rewards[i], np.ones(10)/10, mode='valid')
    plt.plot(moving_avg, label=f"Trader {i}")
plt.title("Moving Average of Rewards per Trader")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()
