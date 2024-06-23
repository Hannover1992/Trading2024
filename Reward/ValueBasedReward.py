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


    def new_calculate_reward(self, 
                             previous_price, 
                             current_price, 
                             previous_shares, 
                             previous_cash, 
                             combined_value_in_cash, 
                             previous_combined_value_in_cash):

        # Aktueller und vorheriger kombinierter Barwert
        current_value = combined_value_in_cash
        previous_value = previous_combined_value_in_cash

        # Realer Gewinn des Portfolios berechnen
        gain = current_value - previous_value

        # Preisänderung der Aktie berechnen
        price_change = current_price - previous_price
        best_scenario_gain = 0


        #falsche benennun, das ist der scenario wenn es gestigen oder gefallen ist und ich bin voll inn gegangen
        if price_change >= 0:
            #Ich habe in letzten Schritt fur all mein Geld Aktien gekauft
            # Hypothetischer Barwert, wenn alles in Aktien investiert worden wäre

            previous_shares = previous_shares + (previous_cash/previous_price) * TRANSACTION_PENELTY
            Stock_value_if_invested_all_in_cash = (previous_shares * current_price) * TRANSACTION_PENELTY

            # Preis gestiegen, bestes Szenario ist vollständiges Investieren in Aktien
            best_scenario_gain = Stock_value_if_invested_all_in_cash - previous_value

        if price_change < 0:
            # Hypothetischer Barwert, wenn alles beim vorherigen Preis verkauft worden wäre
            # best_scenario_gain = previous_cash + previous_shares * TRANSACTION_PENELTY * previous_price
            # Preis gefallen, bestes Szenario ist Halten des Cash

            Stock_value_if_sold_all_in_cash = previous_cash + previous_shares * TRANSACTION_PENELTY * previous_price
            best_scenario_gain = Stock_value_if_sold_all_in_cash - previous_value

        # Differenz zum besten hypothetischen Szenario berechnen
        difference_to_best_scenario = gain - best_scenario_gain 

        # Normalisierte Belohnung berechnen: Verhältnis der Differenz zum vorherigen Barwert
        reward = difference_to_best_scenario / current_value

        #if preis < 0 and gain < 0 and reward > 0
            
        return reward

