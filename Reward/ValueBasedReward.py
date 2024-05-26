from Reward.IRewardCalculator import IRewardCalculator

class ValueBasedReward(IRewardCalculator):
    def calculate_reward(self, previous_state, current_state):

        previous_price = previous_state[-1]
        current_price = current_state[-1]

        scaling_factor = previous_state[0]

        # Berechnung der Preisänderung
        price_change = current_price - previous_price


        # Reward-Berechnung
        reward = scaling_factor * price_change

        return reward
