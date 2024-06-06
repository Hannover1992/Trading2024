from collections import deque
from Config import WINDOW_SIZE

class State:
    def __init__(self,  data, size=WINDOW_SIZE,):
        price = data['Preis'].iloc[0]
        self.min = data['Preis'].min()
        self.max = data['Preis'].max()
        self.size = size
        price_normalized =  self.normalize_state(price)
        self.stack = deque([price_normalized] * size, maxlen=size)


    def push(self, price, balance):
        price = self.normalize_state(price)
        self.stack.append(price)
        # self.stack[0] = balance

    def get_state(self):
        return list(self.stack)
    
    def init_state_with_constant_price(self, price):
        self.stack = deque([price] * self.size, maxlen=self.size)

    def calculate_balance_ratio(self, cash, shares, current_price):
        total_value = cash + shares * current_price
        if total_value == 0:
            return 0  # If total value is zero, return a neutral ratio

        stock_value = shares * current_price
        ratio = (2 * stock_value / total_value) - 1

        return ratio

    def normalize_state(self, state):
        state = (state - self.min) / (self.max - self.min) 
        return state
