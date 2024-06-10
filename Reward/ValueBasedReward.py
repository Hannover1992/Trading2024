from Reward.IRewardCalculator import IRewardCalculator
import unittest

class ValueBasedReward(IRewardCalculator):
    def calculate_reward(self, previous_state, current_state,balance_ratio):

        previous_price = previous_state[-1]
        current_price = current_state[-1]

        # new_balance_ratio = current_state[0]
        new_balance_ratio = balance_ratio
        # Berechnung der Preisänderung
        price_change = current_price - previous_price


        # Reward-Berechnung
        reward = new_balance_ratio * price_change

        return reward


    def new_calculate_reward(self, previous_price, current_price, combined_value_in_cash, previous_combined_value_in_cash):

        # Zugriff auf den aktuellen und vorherigen kombinierten Barwert
        current_value = combined_value_in_cash
        previous_value = previous_combined_value_in_cash

        # real gain 
        gain =  current_value - previous_value

        # Preisänderung berechnen
        price_change = current_price - previous_price

        # Hypothetische Berechnungen:
        # 1. Wie viel Geld hätten Sie, wenn Sie alles beim vorherigen Preis verkauft hätten?
        cash_if_sold_all = previous_value

        # 2. Wie viel würden Sie jetzt verdienen, wenn Sie alles in Aktien investiert hätten?
        # Hier nehmen wir an, dass previous_value die Gesamtmenge an Cash war, die investiert werden könnte.
        stocks_value_if_invested_all = previous_value * (current_price / previous_price)

        # Die tatsächliche Portfolioänderung berechnen
        # actual_portfolio_change = current_value - previous_value

        # Belohnung basierend auf der Differenz zwischen dem aktuellen Portfoliowert und dem besten hypothetischen Szenario
        # maximum_profit = max(cash_if_sold_all, stocks_value_if_invested_all)

        if(price_change >= 0):
            # es sit gestigen, daswegen das beste wahre alles in stocks zu haben
            best_scenario_gain = stocks_value_if_invested_all - previous_value
        else:
            # es ist gefallen, daswegen das beste wahre alles in cash zu haben
            best_scenario_gain = cash_if_sold_all - previous_value

        differrence_to_best_scenario = gain - best_scenario_gain 

        
        # reward = balance_ratio * (actual_portfolio_change - best_hypothetical_scenario)
        
        
        return differrence_to_best_scenario/200

