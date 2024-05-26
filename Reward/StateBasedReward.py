class StateBasedReward():

    def calculate_reward(self, previous_state, current_state):
        previous_value = self.balance + self.shares * previous_state[-1]
        current_value = self.balance + self.shares * current_state[-1]
        reward = current_value - previous_value
        return reward
