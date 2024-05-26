import datetime
import random

from faker import Faker
from faker.providers import BaseProvider

ALPHA = 0.001039
WORKER = 3
ITERATION = 1
STEPS = 2000
BATCH_SIZE = 72
EXPLOITAION = 10

LAYER_1 = 158
LAYER_2 = 52
NOISE = 0.3
GAMMA = 0.0


WINDOW_SIZE = 10

def hard_param():
    return ALPHA, BATCH_SIZE, LAYER_1, LAYER_2

# Define the global variables for the minimum and maximum boundaries of each parameter.
GLOBAL_ALPHA_MIN = 1e-7
GLOBAL_ALPHA_MAX = 0.2

GLOBAL_BATCH_MIN = 4
GLOBAL_BATCH_MAX = 128

GLOBAL_FC1_MIN = 1
GLOBAL_FC1_MAX = 1000

GLOBAL_FC2_MIN = 1
GLOBAL_FC2_MAX = 1000



class Configuration:
    def __init__(self, alpha, beta, tau, fc1, fc2, batch_size, noise, gamma, load_checkpoint, steps):
        self.unique_name = "Stefan"
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.fc1 = fc1
        self.fc2 = fc2
        self.batch_size = batch_size
        self.noise = noise
        self.gamma = gamma
        self.load_checkpoint = load_checkpoint
        self.steps = steps
        date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')  # %f gives microseconds which is the closest to nanoseconds that strftime can provide
        rand_num = random.randint(1, 1000000)  # Generates a random number between 1 and 10000

        self.unique_name = f"unique_name_{self.unique_name}_{date_str}_{rand_num}_alpha_{self.alpha}_beta_{self.beta}_tau_{self.tau}_fc1_{self.fc1}_fc2_{self.fc2}_batch_size_{self.batch_size}_noise_{self.noise}_gamma_{self.gamma}_load_checkpoint_{self.load_checkpoint}_steps_{self.steps}"
