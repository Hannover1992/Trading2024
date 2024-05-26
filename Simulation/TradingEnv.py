import gym
from gym import spaces
import numpy as np
from numpy._typing import _16Bit
from State.Signal import Signal
from Data.MockSyntheticSinusData import MockSyntheticSinusData
from State.State import State
from Reward.ValueBasedReward import ValueBasedReward
from Config import WINDOW_SIZE

class TradingEnv():
    def __init__(self):
        super(TradingEnv, self).__init__()
        self.initial_balance = 10000
        self.cash = self.initial_balance
        self.shares = 0
        self.data_generator = MockSyntheticSinusData()
        self.data = self.data_generator.get_data()
        self.current_step = 0
        self.state = State(self.data['Preis'].iloc[0])
        self.reward_calculator = ValueBasedReward()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Define observation space within the range of -10000 and +10000
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(WINDOW_SIZE,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.cash = self.initial_balance
        self.shares = 0
        self.current_step = 0
        initial_price = self.data['Preis'].iloc[0]
        self.state = State(initial_price)
        return self.state.get_state(), {}

    def step(self, action):
        previous_state = self.state.get_state()
        price = self.data['Preis'].iloc[self.current_step]
        
        volume = float(action[0].numpy())
        volume = (volume - 0.5)*2
        if volume > 0.0:
            amount_to_invest = self.cash * volume
            shares_to_buy = amount_to_invest / price
            self.shares += shares_to_buy
            self.cash -= amount_to_invest
        else:
            shares_to_sell = self.shares * volume
            amount_received = shares_to_sell * price
            self.shares -= shares_to_sell
            self.cash += amount_received

        new_balance_ratio = self.state.calculate_balance_ratio(self.cash, self.shares, price)
        self.state.push(price, new_balance_ratio)
        new_state = self.state.get_state()

        reward = self.reward_calculator.calculate_reward(previous_state, new_state)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        info = {}

        return new_state, reward, done,  info  # False is added for truncated

    def render(self, mode='human'):
        pass
