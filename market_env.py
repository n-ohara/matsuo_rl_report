import numpy as np

class MarketEnvironment:
    def __init__(self, num_traders=1):
        self.num_traders = num_traders
        self.last_price = 100.0
        self.step_count = 0
        self.max_steps = 1000
        self.positions = [0.0 for _ in range(num_traders)]
        self.avg_buy_price = [0.0 for _ in range(num_traders)]
        self.action_map = {
            0: -1.0,  # 売り
            1: 0.0,   # 待機
            2: 1.0    # 買い
        }

    def reset(self):
        self.last_price = 100.0
        self.step_count = 0
        self.positions = [0.0 for _ in range(self.num_traders)]
        self.avg_buy_price = [0.0 for _ in range(self.num_traders)]
        return [self._get_initial_state(i) for i in range(self.num_traders)]

    def step(self, actions):
        if isinstance(actions, int):
            actions = [actions]
        
        price_delta = sum(self.action_map[a] for a in actions)
        self.last_price += price_delta
        self.step_count += 1
        
        rewards = []
        for i, action in enumerate(actions):
            if action == 2:  # 買い
                self.positions[i] += 1
                self.avg_buy_price[i] = (
                    self.avg_buy_price[i] * (self.positions[i] - 1) + self.last_price
                    ) / self.positions[i]
                rewards.append(0.0)
            elif action == 0 and self.positions[i] > 0:  # 売り
                profit = (self.last_price - self.avg_buy_price[i]) * self.positions[i]
                rewards.append(profit)
                self.positions[i] = 0
                self.avg_buy_price[i] = 0.0
            else:
                rewards.append(0.0)
        next_states = [self._get_initial_state(i) for i in range(self.num_traders)]
        done = self.step_count >= self.max_steps
        
        return next_states, rewards, done, {}


    def _get_initial_state(self, trader_id):
        return np.array([self.last_price, self.positions[trader_id]], dtype=np.float32)

    @property
    def action_space(self):
        class Space:
            n = 3
            def sample(self):
                return np.random.randint(0, self.n)
        return Space()

    @property
    def observation_space(self):
        class Space:
            shape = (2,)
        return Space()
