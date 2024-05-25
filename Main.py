import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from collections import deque
from Data.MockSyntheticSinusData import MockSyntheticSinusData

class Signal(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class State:
    def __init__(self, size=10):
        self.size = size
        self.stack = deque([0.0] * size, maxlen=size)

    def push(self, price):
        self.stack.append(price)

    def get_state(self):
        return list(self.stack)

class RandomActor:
    def act(self, state):
        action_value = np.random.uniform(-1, 1)
        if action_value < -0.5:
            return Signal.SELL, (action_value + 1) / 0.5  # Scale to [0, 1]
        elif action_value < 0.5:
            return Signal.HOLD, 0.0
        else:
            return Signal.BUY, (action_value - 0.5) / 0.5  # Scale to [0, 1]

class TradingSimulation:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.shares = 0
        self.data = None
        self.history = []
        self.state = State()
        self.actor = RandomActor()

    def load_data(self):
        data_generator = MockSyntheticSinusData()
        self.data = data_generator.get_data()
        if self.data is None or self.data.empty:
            raise ValueError("Data could not be loaded. Please check the data source.")

    def execute_trade(self, signal, price, volume):
        if signal == Signal.BUY:
            amount_to_invest = self.balance * volume
            shares_to_buy = amount_to_invest / price
            self.shares += shares_to_buy
            self.balance -= amount_to_invest
        elif signal == Signal.SELL:
            shares_to_sell = self.shares * volume
            amount_received = shares_to_sell * price
            self.shares -= shares_to_sell
            self.balance += amount_received

    def calculate_reward(self, previous_state, current_state):
        previous_value = self.balance + self.shares * previous_state[-1]
        current_value = self.balance + self.shares * current_state[-1]
        reward = current_value - previous_value
        return reward

    def run_simulation(self):
        if self.data is not None and not self.data.empty:
            for price in self.data['Preis']:
                previous_state = self.state.get_state()
                self.state.push(price)
                current_state = self.state.get_state()
                signal, volume = self.actor.act(current_state)
                self.execute_trade(signal, price, volume)
                reward = self.calculate_reward(previous_state, current_state)
                self.history.append({'Action': signal.name, 'Price': price, 'Volume': volume, 'Shares': self.shares, 'Balance': self.balance, 'Reward': reward})

    def plot_results(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data before plotting.")

        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})

        # Plot the price data with buy/sell signals
        sns.lineplot(data=self.data, x='Tage', y='Preis', ax=ax1)
        buy_signals = [(i, h) for i, h in enumerate(self.history) if h['Action'] == 'BUY']
        sell_signals = [(i, h) for i, h in enumerate(self.history) if h['Action'] == 'SELL']
        ax1.scatter([i for i, _ in buy_signals], [h['Price'] for _, h in buy_signals], color='green', label='Buy', s=[h['Volume'] * 100 for _, h in buy_signals])
        ax1.scatter([i for i, _ in sell_signals], [h['Price'] for _, h in sell_signals], color='red', label='Sell', s=[h['Volume'] * 100 for _, h in sell_signals])
        ax1.set_title("Synthetische Preisdaten (Sinus) mit Kauf- und Verkaufsaktionen")
        ax1.set_xlabel("Tage")
        ax1.set_ylabel("Preis")
        ax1.legend()

        # Plot the portfolio value data
        portfolio_values = [{'Cash': h['Balance'], 'Shares': h['Shares'] * h['Price']} for h in self.history]
        cash_values = [pv['Cash'] for pv in portfolio_values]
        share_values = [pv['Shares'] for pv in portfolio_values]
        bar_width = 0.4
        r = np.arange(len(self.history))

        ax2.bar(r, cash_values, color='blue', edgecolor='grey', width=bar_width, label='Cash')
        ax2.bar(r, share_values, bottom=cash_values, color='orange', edgecolor='grey', width=bar_width, label='Shares')
        ax2.set_title("Portfolio Value (Cash and Shares)")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Value")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def print_history(self):
        history_df = pd.DataFrame(self.history)
        print(history_df)

# Simulation starten
simulation = TradingSimulation(initial_balance=10000)
simulation.load_data()
simulation.run_simulation()
simulation.plot_results()
simulation.print_history()

