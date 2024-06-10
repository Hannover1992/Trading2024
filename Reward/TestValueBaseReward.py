import unittest
from ValueBasedReward import ValueBasedReward 

class TestValueBasedReward(unittest.TestCase):
    def setUp(self):
        self.reward_calculator = ValueBasedReward()

    def test_no_price_change(self):
        previous_state = [100, 100]
        current_state = [100, 100]
        balance_ratio = 0.5
        reward = self.reward_calculator.calculate_reward(previous_state, current_state, balance_ratio)
        self.assertEqual(reward, 0, "Reward should be zero if there's no price change")

    def test_price_increase(self):
        previous_state = [100, 100]
        current_state = [100, 110]
        balance_ratio = 0.5
        reward = self.reward_calculator.calculate_reward(previous_state, current_state, balance_ratio)
        self.assertEqual(reward, 5, "Reward calculation is incorrect for price increase")

    def test_price_decrease(self):
        previous_state = [100, 100]
        current_state = [100, 90]
        balance_ratio = 0.5
        reward = self.reward_calculator.calculate_reward(previous_state, current_state, balance_ratio)
        self.assertEqual(reward, -5, "Reward calculation is incorrect for price decrease")

    def test_high_balance_ratio(self):
        previous_state = [100, 100]
        current_state = [100, 110]
        balance_ratio = 1.0
        reward = self.reward_calculator.calculate_reward(previous_state, current_state, balance_ratio)
        self.assertEqual(reward, 10, "Reward should be higher with a higher balance ratio")

    def test_new_calculate_reward_scenario(self):
        previous_state = [100, 100]
        current_state = [100, 110]
        balance_ratio = 0.5
        combined_value_in_cash = 105
        previous_combined_value_in_cash = 100
        reward = self.reward_calculator.new_calculate_reward(
            previous_state, current_state, balance_ratio, combined_value_in_cash, previous_combined_value_in_cash)
        self.assertTrue(reward < 0, "Reward should be negative if the portfolio underperforms the best hypothetical scenario")

# Zum Ausführen der Tests
if __name__ == '__main__':
    unittest.main()
