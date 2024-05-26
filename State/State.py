from collections import deque
from Config import WINDOW_SIZE

class State:
    def __init__(self,  price, size=WINDOW_SIZE,):
        self.size = size
        self.stack = deque([price] * size, maxlen=size)
        self.stack[0] = -1.0

    def push(self, price, balance):
        self.stack.append(price)
        self.stack[0] = balance

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
