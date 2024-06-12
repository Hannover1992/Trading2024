import os

import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense


from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate
from tensorflow.keras.models import Model


from Config import WINDOW_SIZE, NR_OF_LAYERS    


class CriticNetwork(tf.keras.Model):
    def __init__(self, nr_of_layers=NR_OF_LAYERS, fc1_dims=512, fc2_dims=256,
                 name='critic', unique_name='hades', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.nr_of_layers = nr_of_layers

        self.lstm = LSTM(fc1_dims)  # Initial dimension for LSTM layer
        self.dense_layers = []
        for i in range(nr_of_layers):
            if i % 2 == 0:  # Even index layers
                self.dense_layers.append(Dense(fc1_dims, activation='relu'))
            else:           # Odd index layers
                self.dense_layers.append(Dense(fc2_dims, activation='relu'))
        self.q = Dense(1, activation=None)  # Output layer for Q value

        self.model_name = f"{name}_{unique_name}"
        self.checkpoint_dir = os.path.join(chkpt_dir, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.model_name}_ddpg.h5")

    def call(self, inputs):
        state, action = inputs
        state = self.lstm(state)
        combined_input = Concatenate()([state, action])
        for layer in self.dense_layers:
            combined_input = layer(combined_input)
        q = self.q(combined_input)
        return q

class ActorNetwork(tf.keras.Model):
    def __init__(self, n_actions, nr_of_layers=2, fc1_dims=512, fc2_dims=256,
                 name='actor', unique_name='hydra', chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.nr_of_layers = nr_of_layers

        self.lstm = LSTM(fc1_dims, input_shape=(WINDOW_SIZE, 1))  # Specifying input shape
        self.dense_layers = []
        for i in range(nr_of_layers):
            if i % 2 == 0:  # Even index layers
                self.dense_layers.append(Dense(fc1_dims, activation='relu'))
            else:           # Odd index layers
                self.dense_layers.append(Dense(fc2_dims, activation='relu'))
        self.mu = Dense(n_actions, activation='tanh')  # Output layer

        self.model_name = f"{name}_{unique_name}"
        self.checkpoint_dir = os.path.join(chkpt_dir, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.model_name}_ddpg.h5")

    def call(self, state):
        state = self.lstm(state)
        for layer in self.dense_layers:
            state = layer(state)
        return self.mu(state)
