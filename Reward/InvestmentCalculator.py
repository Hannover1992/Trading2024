from Config import TRANSACTION_PENELTY

class InvestmentCalculator:
    def __init__(self, previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash):
        self.initialize(previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash)

    def initialize(self, previous_price, current_price, previous_shares, current_shares, previous_cash, current_cash):
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
        current_value = self.current_combined_value
        maximum_value = self.calculate_extrem_positiv_scenario()
        minimal_value = self.calculate_extrem_negativ_scenario()
        if maximum_value == minimal_value:
            return 0  # Oder ein anderer Wert, der in diesem Fall Sinn macht

        # Normalisierung des realen Gewinns
        normalized_value = 2 * ((current_value - minimal_value) / (maximum_value - minimal_value)) - 1

        return normalized_value

    def calculate_real_gain(self):
        return self.current_combined_value - self.previous_combined_value

    def calculate_price_change(self):
        return self.current_price - self.previous_price

    def calculate_extrem_positiv_scenario(self):
        price_change = self.calculate_price_change()
        if price_change >= 0:
            return self.best_case_invested_already_in_shares()
        else:
            return self.previous_combined_value;

    def best_case_invested_already_in_shares(self):
        previous_combined_value = self.previous_combined_value

        maximal_value = previous_combined_value * (self.current_price / self.previous_price) 
        return maximal_value

    def calculate_extrem_negativ_scenario(self):
        price_change = self.calculate_price_change()
        if price_change >= 0:
            return self.hast_all_in_strocks_and_move_it_to_cash() 
        else:
            return self.hast_all_in_cash_and_move_it_to_stocks()

    def hast_all_in_cash_and_move_it_to_stocks(self):
         return self.previous_combined_value * ((self.current_price / self.previous_price) * (1 - TRANSACTION_PENELTY))

    def hast_all_in_strocks_and_move_it_to_cash(self):
         return self.previous_combined_value * (1 - TRANSACTION_PENELTY)

    def move_all_shares(self):
        previous_shares = self.previous_shares

        # Potential new shares bought with previous cash, considering the penalty
        potential_new_shares = self.previous_cash / (self.previous_price * (1 + TRANSACTION_PENELTY))
        return (potential_new_shares + previous_shares) * self.current_price

    def move_all_assets_in_cash(self):
         return self.previous_cash + (self.previous_shares * (self.previous_price * (1 - TRANSACTION_PENELTY)))

    def best_case_already_all_in_cash(self):
         return self.previous_combined_value

    def normalize_reward(self, difference_to_extrem_scenario):
        # Normalize the reward using the current combined value
        return difference_to_extrem_scenario / self.current_combined_value if self.current_combined_value != 0 else 0
