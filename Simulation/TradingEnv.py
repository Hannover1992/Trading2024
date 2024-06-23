import gym
from gym import spaces
import numpy as np
from numpy._typing import _16Bit
from State.Signal import Signal
from Data.MockSyntheticSinusData import MockSyntheticSinusData
from Data.BitcoinData import BitcoinData
from State.State import State
from Reward.ValueBasedReward import ValueBasedReward
from Reward.StateBasedReward import StateBasedReward 
from Reward.InvestmentCalculator import InvestmentCalculator

from Config import WINDOW_SIZE, TRANSACTION_PENELTY, CASH

class TradingEnv():
    def __init__(self):
        super(TradingEnv, self).__init__()

        self.initial_balance = CASH
        self.cash = self.initial_balance
        self.combined_value_in_cash = self.initial_balance
        self.previous_combined_value_in_cash = self.initial_balance

        self.shares = 0
        self.data = MockSyntheticSinusData().get_data()
        # self.data = BitcoinData("Data/BitcoinData.csv").get_data()
        self.current_step = 1
        self.state = State(self.data)
        self.reward_calculator = ValueBasedReward()
        # self.reward_calculator = StateBasedReward()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Define observation space within the range of -10000 and +10000
        self.observation_space = spaces.Box(low=self.data['Preis'].min(), high=self.data['Preis'].min(), shape=(WINDOW_SIZE,), dtype=np.float32)
        self.calculator = InvestmentCalculator(0, 0, 0, 0, 0, 0)



    def reset(self, seed=None, options=None):
        self.cash = self.initial_balance
        self.combined_value_in_cash = self.initial_balance
        self.previous_combined_value_in_cash = self.combined_value_in_cash
        self.value_in_cash = self.cash
        self.shares = 0
        self.current_step = 1
        self.state = State(self.data)
        self.calculator = InvestmentCalculator(0, 0, 0, 0, 0, 0)
        return self.state.get_state(), {}

    def step(self, action):
        current_price = self.data['Preis'].iloc[self.current_step]
        previous_price = self.data['Preis'].iloc[self.current_step - 1]

        previous_shares = self.shares
        previous_cash = self.cash



        if action >= 0.5:
            if(self.cash >= 0.0):
                amount_to_invest = self.cash * ((action - 0.5) * 1.98)
                shares_to_buy = amount_to_invest / previous_price

                amount_to_invest = min(amount_to_invest, self.cash)
                self.cash -= amount_to_invest
                if(self.cash < 0.0):
                    print("Cash is negative")
                self.shares += shares_to_buy * (1 - TRANSACTION_PENELTY)
        elif action <= -0.5:
           if(self.shares > 0.0):
                shares_to_sell = self.shares * abs((action + 0.5)*1.98)
                shares_to_sell = min(shares_to_sell, self.shares)
                self.shares -= shares_to_sell
                amount_received = shares_to_sell * previous_price
                if(self.shares < 0.0):
                    print("Shares are negative")
                self.cash += amount_received * (1 - TRANSACTION_PENELTY)


        new_balance_ratio = self.state.calculate_balance_ratio(self.cash, self.shares, current_price)
        #T + 1
        self.state.push(current_price, new_balance_ratio)

        self.calculator.initialize( previous_price, current_price, previous_shares, self.shares, previous_cash, self.cash)
        reward = self.calculator.calculate_reward()
        # reward = self.reward_calculator.new_calculate_reward(previous_price, current_price, previous_shares, previous_cash, self.combined_value_in_cash, self.previous_combined_value_in_cash)
        # print("Action: ", action, "Reward: ", reward, "Cash: ", self.cash, "Shares: ", self.shares, "Price: ", price)

        self.current_step += 1
        done = self.current_step >= len(self.data)
        info = {}
        # print(f"Action: {action[0].numpy():<10.2f} | Reward: {reward:<10.5f} | Cash: {self.cash:<15.2f} | Shares: {self.shares:<15.2f} | Price: {price:<10.2f} | Volume: {self.volume:<10.2f}")
        # besser formatiert mit fest definierten abstanden

        new_state = self.state.get_state()
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
