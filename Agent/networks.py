import os

import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense


from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Conv1D
from tensorflow.keras.models import Model


from Config import WINDOW_SIZE, NR_OF_LAYERS, FC1_DIMS_LSTM


class CriticNetwork(Model):
    def __init__(self, nr_of_layers=4, fc1_dims=512, fc2_dims=256, window_size=100,
                 name='critic', unique_name='hades', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.nr_of_layers = nr_of_layers

        # Füge eine Conv1D Schicht hinzu
        self.conv1d = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(window_size, 1))

        # LSTM Schicht nach der Conv1D
        self.lstm = LSTM(fc1_dims)

        # Dense Schichten
        self.dense_layers = []
        for i in range(nr_of_layers):
            if i % 2 == 0:
                self.dense_layers.append(Dense(fc1_dims, activation='relu'))
            else:
                self.dense_layers.append(Dense(fc2_dims, activation='relu'))
        
        # Output Layer für Q-Werte
        self.q = Dense(1, activation=None)  # Keine Aktivierungsfunktion für Q-Wert Ausgabe

        # Setup für Checkpointing
        self.model_name = f"{name}_{unique_name}"
        self.checkpoint_dir = os.path.join(chkpt_dir, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.model_name}_ddpg.h5")

    def call(self, inputs):
        state, action = inputs
        state_processed = self.conv1d(state)
        state_processed = self.lstm(state_processed)
        combined_input = Concatenate()([state_processed, action])  # Kombiniere Zustand und Aktion
        for layer in self.dense_layers:
            combined_input = layer(combined_input)
        q = self.q(combined_input)
        return q

class ActorNetwork(Model):
    def __init__(self, n_actions, nr_of_layers=4, fc1_dims=512, fc2_dims=256, name='actor', unique_name='hydra', chkpt_dir='tmp/ddpg', window_size=100):
        super(ActorNetwork, self).__init__()
        self.nr_of_layers = nr_of_layers

        # Füge eine Conv1D Schicht hinzu
        self.conv1d = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(window_size, 1))

        # Update LSTM layer to follow Conv1D
        self.lstm = LSTM(512, return_sequences=False)

        # Dense layers
        self.dense_layers = []
        for i in range(nr_of_layers):
            if i % 2 == 0:
                self.dense_layers.append(Dense(fc1_dims, activation='relu'))
            else:
                self.dense_layers.append(Dense(fc2_dims, activation='relu'))
        
        self.mu = Dense(n_actions, activation='tanh')  # Output layer

        # Setup checkpointing
        self.model_name = f"{name}_{unique_name}"
        self.checkpoint_dir = os.path.join(chkpt_dir, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.model_name}_ddpg.h5")

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.lstm(x)
        for layer in self.dense_layers:
            x = layer(x)
        return self.mu(x)
