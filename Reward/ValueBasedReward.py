from Reward.IRewardCalculator import IRewardCalculator
import unittest
from Config import TRANSACTION_PENELTY

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


    def new_calculate_reward(self, previous_price, current_price, combined_value_in_cash, previous_combined_value_in_cash, previous_shares, previous_cash):
        # Aktueller und vorheriger kombinierter Barwert
        current_value = combined_value_in_cash
        previous_value = previous_combined_value_in_cash

        # Realer Gewinn des Portfolios berechnen
        gain = current_value - previous_value

        # Preisänderung der Aktie berechnen
        price_change = current_price - previous_price
        best_scenario_gain = 0


        # Bestes hypothetisches Szenario basierend auf der Preisänderung
        if price_change >= 0:
            # Hypothetischer Aktienwert, wenn das gesamte Cash beim vorherigen Preis investiert worden wäre
            real_cash_value = previous_cash * TRANSACTION_PENELTY
            shares_buyed = real_cash_value / previous_price
            cumulative_shares = shares_buyed + previous_shares
            stocks_value_if_invested_all = cumulative_shares * current_price * TRANSACTION_PENELTY
            # Preis gestiegen, bestes Szenario ist vollständiges Investieren in Aktien
            best_scenario_gain = stocks_value_if_invested_all - previous_value

        if price_change < 0:
            # Hypothetischer Barwert, wenn alles beim vorherigen Preis verkauft worden wäre
            # best_scenario_gain = previous_cash + previous_shares * TRANSACTION_PENELTY * previous_price
            # Preis gefallen, bestes Szenario ist Halten des Cash
            best_scenario_gain = 0

        # Differenz zum besten hypothetischen Szenario berechnen
        difference_to_best_scenario = gain - best_scenario_gain 

        # Normalisierte Belohnung berechnen: Verhältnis der Differenz zum vorherigen Barwert
        reward = difference_to_best_scenario / previous_value

        return reward

