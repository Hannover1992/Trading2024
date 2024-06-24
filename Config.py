import datetime
import random
import tensorflow as tf

ITERATION = 2000

ALPHA = 0.01
GAMMA = 0.99
BETA = ALPHA * 2
TAU = ALPHA * 50

BATCH_SIZE = 64
EXPLOITAION = 2

DATA_LENGTH = 128




NR_OF_LAYERS = 8

FC1 = 50
FC2 = 150

WINDOW_SIZE = 32
FC1_DIMS_LSTM = 8


FILTER_KERNEL = 1

TRANSACTION_PENELTY = 0.005
CASH = 200

NUMBER_OF_PROC = 18

NOISE = 0.7


NOISE_MIN = 0.3
NOISE_MAX = 0.7

ALPHA_MIN = 0.00001
ALPHA_MAX = 0.001

class Configuration:
    def __init__(self, learning_rate=ALPHA, noise=NOISE):
        ALPHA = learning_rate
        # ALPHA = random.uniform(ALPHA_MIN, ALPHA_MAX)
        self.unique_name = "Stefan"
        self.alpha = ALPHA
        self.beta = self.alpha * 2
        self.tau = self.alpha * 5
        self.fc1 = FC1
        self.fc2 = FC2
        self.batch_size = BATCH_SIZE
        self.noise = noise
        self.gamma = GAMMA
        self.iteration = ITERATION
        date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')  # %f gibt Mikrosekunden an
        rand_num = random.randint(1, 1000000)  # Generiert eine Zufallszahl zwischen 1 und 1000000
        self.setup_gpu()

        self.unique_name = f"{self.unique_name}_{date_str}_{rand_num}_ALPHA_{self.alpha}_NOISE_{self.noise}"



    def setup_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Set the GPU visible to TensorFlow and enable memory growth
                # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                # tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                print("RuntimeError:", e)

    def setup_cpu(self):
        # Set TensorFlow to only use the CPU
        try:
            # Get a list of all GPUs
            gpus = tf.config.experimental.list_physical_devices('GPU')
            # Set all GPUs as not visible
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_visible_devices([], 'GPU')
            print("Using CPU only.")
        except RuntimeError as e:
            print("Runtime error:", e)
