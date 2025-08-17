import numpy as np
class MarketEnvironment:
    def __init__(self):
        self.last_price = 100.0
        self.step_count = 0
        self.max_steps = 1000
        self.action_map = {
            0: (-1.0, 1),  # 売り強め
            1: (-0.5, 1),
            2: (0.0, 1),   # 待機
            3: (0.5, 1),
            4: (1.0, 1)    # 買い強め
        }

    def reset(self):
        self.last_price = 100.0
        self.step_count = 0
        return np.array([self.last_price], dtype=np.float32)

    def step(self, action_id):
        price_delta, volume = self.action_map[action_id]
        price = self.last_price + price_delta
        self.last_price = price
        self.step_count += 1
        reward = volume * price_delta
        done = self.step_count >= self.max_steps
        return np.array([self.last_price], dtype=np.float32), reward, done, {}

    @property
    def action_space(self):
        class Space:
            n = 5
            def sample(self):
                return np.random.randint(0, self.n)
        return Space()


    @property
    def observation_space(self):
        class Space:
            shape = (1,)
        return Space()
