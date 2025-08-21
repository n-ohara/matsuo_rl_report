class Trader:
    def __init__(self, env, agent, trader_id=0):
        self.trader_id = trader_id
        self.env = env
        self.agent = agent

    def act(self, state, episode):
        return self.agent.get_action(state, episode)

    def learn(self):
        self.agent.learn()

    def append(self, transition):
        self.agent.append(transition)
