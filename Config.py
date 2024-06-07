import datetime
import random

ALPHA = 0.1
BETA = ALPHA * 2
TAU = ALPHA * 5
FC1 = 10
FC2 = 10
BATCH_SIZE = 10
EXPLOITAION = 0.1
ITERATION = 100
NOISE = 0.1
GAMMA = 0.99
WINDOW_SIZE = 1

ALPHA_MIN = 0.000001
ALPHA_MAX = 0.1

class Configuration:
    def __init__(self, learning_rate):
        ALPHA = learning_rate
        # ALPHA = random.uniform(ALPHA_MIN, ALPHA_MAX)
        self.unique_name = "Stefan"
        self.alpha = ALPHA
        self.beta = self.alpha * 2
        self.tau = self.alpha * 5
        self.fc1 = FC1
        self.fc2 = FC2
        self.batch_size = BATCH_SIZE
        self.noise = NOISE
        self.gamma = GAMMA
        self.iteration = ITERATION
        date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')  # %f gibt Mikrosekunden an
        rand_num = random.randint(1, 1000000)  # Generiert eine Zufallszahl zwischen 1 und 1000000

        self.unique_name = f"{self.unique_name}_{date_str}_{rand_num}"
