from abc import ABC, abstractmethod

class IRewardCalculator(ABC):
    @abstractmethod
    def calculate_reward(self, previous_state, current_state):
        pass
