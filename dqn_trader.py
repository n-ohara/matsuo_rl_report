from dqn_core import Dqn

class DQNTrader:
    def __init__(self, env, trader_id=0):
        self.trader_id = trader_id
        self.agent = Dqn(
            dim_state=env.observation_space.shape[0],
            num_action=env.action_space.n,
            memory_size=50000,
            target_update_freq=30
        )

    def act(self, state, episode):
        return self.agent.get_action(state, episode)

    def learn(self):
        self.agent.update_q()

    def append(self, transition):
        self.agent.replay_buffer.append(transition)

