from Data.MockSyntheticSinusData import MockSyntheticSinusData
from Actor.RandomActor import RandomActor
from State.Signal import Signal
from State.State import State
from Reward.ValueBasedReward import ValueBasedReward

class TradingSimulation:
    def __init__(self, initial_balance):
        self.cash = initial_balance
        self.shares = 0
        self.data = None
        self.history = []
        self.state = State(0.0)
        self.actor = RandomActor()
        self.reward_calculator = ValueBasedReward()

    def load_data(self):
        data_generator = MockSyntheticSinusData()
        self.data = data_generator.get_data()
        if self.data is None or self.data.empty:
            raise ValueError("Data could not be loaded. Please check the data source.")
        initial_price = self.data['Preis'].iloc[0]
        initial_balance_ratio = self.state.calculate_balance_ratio(self.cash, self.shares, initial_price)
        self.state.init_state_with_constant_price(initial_price)

    def execute_trade(self, signal, price, volume):
        if signal == Signal.BUY:
            amount_to_invest = self.cash * volume
            shares_to_buy = amount_to_invest / price
            self.shares += shares_to_buy
            self.cash -= amount_to_invest
        elif signal == Signal.SELL:
            shares_to_sell = self.shares * volume
            amount_received = shares_to_sell * price
            self.shares -= shares_to_sell
            self.cash += amount_received

    def run_simulation(self):
        if self.data is not None and not self.data.empty:
            previous_state = self.state.get_state()
            previous_price = self.data['Preis'].iloc[0]

            for price in self.data['Preis']:
                signal, volume = self.actor.act(previous_state)
                self.execute_trade(signal, previous_price, volume)
                
                new_balance_ratio = self.state.calculate_balance_ratio(self.cash, self.shares, price)
                cash_ratio, stock_ratio = new_balance_ratio
                self.state.push(price, new_balance_ratio)
                new_state = self.state.get_state()

                reward = self.reward_calculator.calculate_reward(previous_state, new_state, cash_ratio, stock_ratio, self.shares)

                self.history.append({
                    'State': previous_state,
                    'Action': signal.name,
                    'Action Volume': volume,
                    'Price': price,
                    'Shares': self.shares,
                    'Balance': self.cash,
                    'Reward': reward,
                    'Next State': new_state
                })

                previous_state = new_state
                previous_price = price
