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
        # Aktueller und vorheriger kombinierter Barwert
        current_value = combined_value_in_cash
        previous_value = previous_combined_value_in_cash

        # Realer Gewinn des Portfolios berechnen
        gain = current_value - previous_value

        # Preisänderung der Aktie berechnen
        price_change = current_price - previous_price

        # Hypothetischer Barwert, wenn alles beim vorherigen Preis verkauft worden wäre
        cash_if_sold_all = previous_value

        # Hypothetischer Aktienwert, wenn das gesamte Cash beim vorherigen Preis investiert worden wäre
        stocks_value_if_invested_all = previous_value * (current_price / previous_price)

        # Bestes hypothetisches Szenario basierend auf der Preisänderung
        if price_change >= 0:
            # Preis gestiegen, bestes Szenario ist vollständiges Investieren in Aktien
            best_scenario_gain = stocks_value_if_invested_all - previous_value
        else:
            # Preis gefallen, bestes Szenario ist Halten des Cash
            best_scenario_gain = cash_if_sold_all - previous_value

        # Differenz zum besten hypothetischen Szenario berechnen
        difference_to_best_scenario = gain - best_scenario_gain 

        # Normalisierte Belohnung berechnen: Verhältnis der Differenz zum vorherigen Barwert
        reward = difference_to_best_scenario / previous_value

        return reward

