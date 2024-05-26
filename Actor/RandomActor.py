import numpy as np
from State.Signal import Signal

class RandomActor:
    def act(self, state):
        action_value = np.random.uniform(-1, 1)
        if action_value < -0.5:
            return Signal.SELL, (action_value + 1) / 0.5  # Scale to [0, 1]
        elif action_value < 0.5:
            return Signal.HOLD, 0.0
        else:
            return Signal.BUY, (action_value - 0.5) / 0.5  # Scale to [0, 1]
