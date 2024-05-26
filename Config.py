import datetime
import random

ALPHA = 0.01039
BETA = ALPHA * 2
TAU = ALPHA * 5
FC1 = 30
FC2 = 30
BATCH_SIZE = 72
EXPLOITAION = 10
ITERATION = 100
NOISE = 0.6
GAMMA = 0.99
WINDOW_SIZE = 70

class Configuration:
    def __init__(self):
        self.unique_name = "Stefan"
        self.alpha = ALPHA
        self.beta = BETA
        self.tau = TAU
        self.fc1 = FC1
        self.fc2 = FC2
        self.batch_size = BATCH_SIZE
        self.noise = NOISE
        self.gamma = GAMMA
        self.iteration = ITERATION
        date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')  # %f gibt Mikrosekunden an
        rand_num = random.randint(1, 1000000)  # Generiert eine Zufallszahl zwischen 1 und 1000000

        self.unique_name = f"{self.unique_name}_{date_str}_{rand_num}"
