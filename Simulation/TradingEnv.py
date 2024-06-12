import gym
from gym import spaces
import numpy as np
from numpy._typing import _16Bit
from State.Signal import Signal
from Data.MockSyntheticSinusData import MockSyntheticSinusData
from Data.MockSyntheticSinusData import BitcoinData
from State.State import State
from Reward.ValueBasedReward import ValueBasedReward
from Reward.StateBasedReward import StateBasedReward 

from Config import WINDOW_SIZE, TRANSACTION_PENELTY

class TradingEnv():
    def __init__(self):
        super(TradingEnv, self).__init__()

        self.initial_balance = 10000
        self.cash = self.initial_balance
        self.combined_value_in_cash = self.initial_balance
        self.previous_combined_value_in_cash = self.initial_balance

        self.shares = 0
        self.data = MockSyntheticSinusData().get_data()
        # self.data = BitcoinData('./Data/bitcoin.csv').get_data()
        self.current_step = 1
        self.state = State(self.data)
        self.reward_calculator = ValueBasedReward()
        # self.reward_calculator = StateBasedReward()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Define observation space within the range of -10000 and +10000
        self.observation_space = spaces.Box(low=self.data['Preis'].min(), high=self.data['Preis'].min(), shape=(WINDOW_SIZE,), dtype=np.float32)



    def reset(self, seed=None, options=None):
        self.cash = self.initial_balance
        self.combined_value_in_cash = self.initial_balance
        self.previous_combined_value_in_cash = self.combined_value_in_cash
        self.value_in_cash = self.cash
        self.shares = 0
        self.current_step = 1
        self.state = State(self.data)
        return self.state.get_state(), {}

    def step(self, action):
        price = self.data['Preis'].iloc[self.current_step]
        previous_price = self.data['Preis'].iloc[self.current_step - 1]

        previous_shares = self.shares
        previous_cash = self.cash



        if action >= 0.0:
            if(self.cash >= 0.0):
                amount_to_invest = self.cash * action
                shares_to_buy = amount_to_invest / previous_price
                self.cash -= amount_to_invest
                self.shares += shares_to_buy * TRANSACTION_PENELTY
        else:
            if(self.shares > 0.0):
                shares_to_sell = self.shares * abs(action)
                amount_received = shares_to_sell * previous_price
                self.shares -= shares_to_sell
                if(self.shares < 0.0):
                    print("Shares are negative")
                self.cash += amount_received * TRANSACTION_PENELTY


        new_balance_ratio = self.state.calculate_balance_ratio(self.cash, self.shares, price)
        self.previous_combined_value_in_cash = self.combined_value_in_cash
        #T + 1
        self.state.push(price, new_balance_ratio)
        new_state = self.state.get_state()

        self.combined_value_in_cash = self.cash + self.shares * price * TRANSACTION_PENELTY

        reward = self.reward_calculator.new_calculate_reward(previous_price, price,  self.combined_value_in_cash, self.previous_combined_value_in_cash, previous_shares, previous_cash)

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


    def run_tests(self):
        test_no_price_change_new_reward()
        test_all_stock_gestiegen()
        test_price_decrease_new_reward()
        now_lets_test_right_decision_but_all_gain_eat_by_procent()


def test_no_price_change_new_reward():
    reward_calculator = ValueBasedReward()
    previous_state = [100, 100]
    current_state = [100, 100]
    balance_ratio = 0.5
    combined_value_in_cash = 100
    previous_combined_value_in_cash = 100
    reward = reward_calculator.new_calculate_reward(
            previous_state, current_state, balance_ratio, combined_value_in_cash, previous_combined_value_in_cash)
    print("No price change - New Reward should be 0:", reward)

def test_all_stock_gestiegen():
    reward_calculator = ValueBasedReward()
    previous_state = [100, 100]
    current_state = [100, 115]
    balance_ratio = 1.0
    combined_value_in_cash = 115
    previous_combined_value_in_cash = 100
    reward = reward_calculator.new_calculate_reward(
            previous_state, current_state, balance_ratio, combined_value_in_cash, previous_combined_value_in_cash)
    print("Price increase - New Reward:", reward)

def test_price_decrease_new_reward():
    reward_calculator = ValueBasedReward()
    previous_state = [100, 100]
    current_state = [100, 90]
    balance_ratio = 1.0
    combined_value_in_cash = 90
    previous_combined_value_in_cash = 100
    reward = reward_calculator.new_calculate_reward(
            previous_state, current_state, balance_ratio, combined_value_in_cash, previous_combined_value_in_cash)
    print("Price decrease - New Reward:", reward)

def now_lets_test_right_decision_but_all_gain_eat_by_procent():
    reward_calculator = ValueBasedReward()
    previous_state = [100, 100]
    current_state = [100, 101]
    balance_ratio = 1.0
    combined_value_in_cash = 100
    previous_combined_value_in_cash = 100
    reward = reward_calculator.new_calculate_reward(
            previous_state, current_state, balance_ratio, combined_value_in_cash, previous_combined_value_in_cash)
    print("High balance ratio - New Reward:", reward)
