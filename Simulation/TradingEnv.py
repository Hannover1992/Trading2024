import gym
from gym import spaces
import numpy as np
from numpy._typing import _16Bit
from State.Signal import Signal
from Data.MockSyntheticSinusData import MockSyntheticSinusData
from State.State import State
from Reward.ValueBasedReward import ValueBasedReward
from Reward.StateBasedReward import StateBasedReward 

from Config import WINDOW_SIZE

class TradingEnv():
    def __init__(self):
        super(TradingEnv, self).__init__()
        self.initial_balance = 10000
        self.cash = self.initial_balance
        self.shares = 0
        self.data = MockSyntheticSinusData().get_data()
        self.current_step = 1
        self.state = State(self.data)
        self.reward_calculator = ValueBasedReward()
        # self.reward_calculator = StateBasedReward()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Define observation space within the range of -10000 and +10000
        self.observation_space = spaces.Box(low=self.data['Preis'].min(), high=self.data['Preis'].min(), shape=(WINDOW_SIZE,), dtype=np.float32)
        self.value_in_cash = self.cash

    def reset(self, seed=None, options=None):
        self.cash = self.initial_balance
        self.value_in_cash = self.cash
        self.shares = 0
        self.current_step = 1
        self.state = State(self.data)
        self.volume = 0.0
        return self.state.get_state(), {}

    def step(self, action):
        previous_state = self.state.get_state()

        price = self.data['Preis'].iloc[self.current_step]
        

        if action >= 0.5:
            if(self.cash >= 0.0):
                amount_to_invest = self.cash * action
                shares_to_buy = amount_to_invest / price
                self.cash -= amount_to_invest
                self.shares += shares_to_buy
        if action <= -0.5:
            if(self.shares > 0.0):
                shares_to_sell = self.shares * action
                amount_received = -shares_to_sell * price
                self.shares += shares_to_sell
                self.cash += amount_received

        new_balance_ratio = self.state.calculate_balance_ratio(self.cash, self.shares, price)
        self.state.push(price, new_balance_ratio)
        new_state = self.state.get_state()

        reward = self.reward_calculator.calculate_reward(previous_state, new_state, balance_ratio=new_balance_ratio)

        # self.current_amount_of_value_in_shares_and_cash = self.cash + self.shares * price
        # reward = self.previous_amount_of_value_in_shares_and_cash - self.current_amount_of_value_in_shares_and_cash

        self.combined_value_in_cash = self.cash + self.shares * price
        self.current_step += 1
        done = self.current_step >= len(self.data)
        info = {}
        # print(f"Action: {action[0].numpy():<10.2f} | Reward: {reward:<10.5f} | Cash: {self.cash:<15.2f} | Shares: {self.shares:<15.2f} | Price: {price:<10.2f} | Volume: {self.volume:<10.2f}")
        # besser formatiert mit fest definierten abstanden

        return new_state, reward, done,  info  # False is added for truncated

    def render(self, mode='human'):
        pass

    def normalize_state(self, state):
        # find the max and min in the data and normalize the state
        min_state = self.data['Preis'].min()
        max_state = self.data['Preis'].max()
        normalize = (state - min_state) / (max_state - min_state)
        # ich wrude geren alle werte ausser dem ersten in dem state normalisieren
        # normalize[0] = state[0]
        return normalize 

    def normalize_reward(self, reward):
        min_reward = -10000
        max_reward = 10000
        return (reward - min_reward) / (max_reward - min_reward)
