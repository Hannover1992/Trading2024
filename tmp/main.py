TRANSACTION_PENALTY = 0.02  # Assuming a transaction penalty constant

class InvestmentCalculator:
    def __init__(self, previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash):
        self.previous_price = previous_price
        self.current_price = current_price
        self.previous_shares = previous_shares
        self.current_shares = current_shares
        self.previous_cash = previous_cash
        self.current_cash = current_cash

        # Calculate the combined values at previous and current states
        self.previous_combined_value = previous_cash + (previous_price * previous_shares)
        self.current_combined_value = current_cash + (current_price * current_shares)

    def calculate_reward(self):
        gain = self.calculate_real_gain()
        extrem_scenario_gain = self.calculate_extrem_scenario()

        difference_to_extrem_scenario = gain - extrem_scenario_gain
        # return self.normalize_reward(difference_to_extrem_scenario)
        return difference_to_extrem_scenario

    def calculate_real_gain(self):
        return self.current_combined_value - self.previous_combined_value

    def calculate_price_change(self):
        return self.current_price - self.previous_price

    def calculate_extrem_scenario(self):
        price_change = self.calculate_price_change()
        if price_change >= 0:
            extreme_value =  self.all_in_shares()
        else:
            extreme_value =  self.all_in_cash()
        return extreme_value - self.previous_combined_value

    def all_in_shares(self):
        previous_shares = self.previous_shares

        # Potential new shares bought with previous cash, considering the penalty
        potential_new_shares = self.previous_cash / (self.previous_price * (1 + TRANSACTION_PENALTY))
        return (potential_new_shares + previous_shares) * self.current_price

    def all_in_cash(self):
         return self.previous_cash + (self.previous_shares * (self.previous_price * (1 - TRANSACTION_PENALTY)))

    def normalize_reward(self, difference_to_extrem_scenario):
        # Normalize the reward using the current combined value
        return difference_to_extrem_scenario / self.current_combined_value if self.current_combined_value != 0 else 0

# #Info transaction Penelty is 2%
# # In the previous steps I have already bought only shares and the stocks came up. This means I will receive perhaps null reward because I am already in the extreme scenario. 
#
# initial_investment = 1000
#
# previous_price = 100
# current_price = 105  # 2% price increase
#
# # Scenario 1: All-In Stocks
# previous_shares = initial_investment / previous_price
# current_shares = initial_investment / previous_price
#
# previous_cash = 0
# current_cash = 0 
#
# calculator1 = InvestmentCalculator(previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash)
# print("Stock,Stock, Stock UP++")
# print("Reward", calculator1.calculate_reward())
#
# #Info transaction Penelty is 2%
# # In the previous steps I have already bought only shares and the stocks came up. This means I will receive perhaps null reward because I am already in the extreme scenario. 
#
# initial_investment = 1000
#
# previous_price = 100
# current_price = 102  # 2% price increase
#
# # Scenario 1: All-In Stocks
# previous_shares = initial_investment / previous_price
# current_shares = initial_investment / previous_price
#
# previous_cash = 0
# current_cash = 0 
#
# calculator1 = InvestmentCalculator(previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash)
# print("Stock,Stock, Stock UP")
# print("Reward", calculator1.calculate_reward())
#
# # 
#
#
# #Info transaction Penelty is 2%
# # In the previous steps I have already bought only shares and the stocks came up. This means I will receive perhaps null reward because I am already in the extreme scenario. 
#
# initial_investment = 1000
#
# previous_price = 100
# current_price = 102  # 2% price increase
#
# # Scenario 1: All-In Stocks
# previous_shares = (initial_investment/2) / previous_price
# current_shares = (initial_investment/2) / previous_price
#
# previous_cash = initial_investment/2
# current_cash = initial_investment/2
#
# calculator1 = InvestmentCalculator(previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash)
# print("0.5 Stock , 0.5 Cash, Stock UP = penelty")
# print("Reward", calculator1.calculate_reward())
#
# # 
#
#
#
# initial_investment = 1000
#
# previous_price = 100
# current_price = 105  # 2% price increase
#
# # Scenario 1: All-In Stocks
# previous_shares = (initial_investment/2) / previous_price
# current_shares = (initial_investment/2) / previous_price
#
# previous_cash = initial_investment/2
# current_cash = initial_investment/2
#
# calculator1 = InvestmentCalculator(previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash)
# print("0.5 Stock , 0.5 Cash, Stock UP++")
# print("Reward", calculator1.calculate_reward())
#
# # 
#
#
#
#
# initial_investment = 1000
#
# previous_price = 100
# current_price = 105  # 2% price increase
#
# # Scenario 1: All-In Stocks
# previous_shares = initial_investment / previous_price
# current_shares =  0
#
# previous_cash = 0 
# current_cash = 1000
#
# calculator1 = InvestmentCalculator(previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash)
# print("Stock, Cash,  Stock UP")
# print("Reward", calculator1.calculate_reward())
#
# # 
#
#
#
# initial_investment = 1000
#
# previous_price = 100
# current_price = 95
#
# # Scenario 1: All-In Stocks
# previous_shares = 0
# current_shares =  0
#
# previous_cash = 1000 
# current_cash = 1000
#
# calculator1 = InvestmentCalculator(previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash)
# print("Stock, Cash,  Stock UP")
# print("Reward", calculator1.calculate_reward())


calculator1 = InvestmentCalculator(394, 450, 0, 1, 0, 1)

print("Reward", calculator1.calculate_reward())
