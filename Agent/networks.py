import os

import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense


from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Conv1D
from tensorflow.keras.models import Model


from Config import WINDOW_SIZE, NR_OF_LAYERS, FC1_DIMS_LSTM

FILTER_KERNEL = 1


class CriticNetwork(Model):
    def __init__(self, nr_of_layers=4, fc1_dims=512, fc2_dims=256, window_size=100,
                 name='critic', unique_name='hades', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.nr_of_layers = nr_of_layers

        # Füge eine Conv1D Schicht hinzu
        self.conv1d_1 = Conv1D(filters=FILTER_KERNEL, kernel_size=513, strides=1, padding='valid', activation='relu', input_shape=(window_size, 1))
        self.conv1d_2 = Conv1D(filters=FILTER_KERNEL, kernel_size=257, strides=1, padding='valid', activation='relu')
        self.conv1d_3 = Conv1D(filters=FILTER_KERNEL, kernel_size=129, strides=1, padding='valid', activation='relu')
        self.conv1d_4 = Conv1D(filters=FILTER_KERNEL, kernel_size=33, strides=1, padding='valid', activation='relu')
        self.conv1d_5 = Conv1D(filters=FILTER_KERNEL, kernel_size=17, strides=1, padding='valid', activation='relu')
        self.conv1d_6 = Conv1D(filters=FILTER_KERNEL, kernel_size=9, strides=1, padding='valid', activation='relu')
        self.conv1d_7 = Conv1D(filters=FILTER_KERNEL, kernel_size=5, strides=1, padding='valid', activation='relu')
        self.conv1d_8 = Conv1D(filters=FILTER_KERNEL, kernel_size=3, strides=1, padding='valid', activation='relu')
        self.conv1d_9 = Conv1D(filters=FILTER_KERNEL, kernel_size=2, strides=1, padding='valid', activation='relu')

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
        # state_processed = self.conv1d(state)

        x = self.conv1d_1(state)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        x = self.conv1d_4(x)
        x = self.conv1d_5(x)
        x = self.conv1d_6(x)
        x = self.conv1d_7(x)
        x = self.conv1d_8(x)
        state_processed = self.conv1d_9(x)
        print(x.shape)


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

        # Convolutional Layers
        self.conv1d_1 = Conv1D(filters=FILTER_KERNEL, kernel_size=513, strides=1, padding='valid', activation='relu', input_shape=(window_size, 1))
        self.conv1d_2 = Conv1D(filters=FILTER_KERNEL, kernel_size=257, strides=1, padding='valid', activation='relu')
        self.conv1d_3 = Conv1D(filters=FILTER_KERNEL, kernel_size=129, strides=1, padding='valid', activation='relu')
        self.conv1d_4 = Conv1D(filters=FILTER_KERNEL, kernel_size=33, strides=1, padding='valid', activation='relu')
        self.conv1d_5 = Conv1D(filters=FILTER_KERNEL, kernel_size=17, strides=1, padding='valid', activation='relu')
        self.conv1d_6 = Conv1D(filters=FILTER_KERNEL, kernel_size=9, strides=1, padding='valid', activation='relu')
        self.conv1d_7 = Conv1D(filters=FILTER_KERNEL, kernel_size=5, strides=1, padding='valid', activation='relu')
        self.conv1d_8 = Conv1D(filters=FILTER_KERNEL, kernel_size=3, strides=1, padding='valid', activation='relu')
        self.conv1d_9 = Conv1D(filters=FILTER_KERNEL, kernel_size=2, strides=1, padding='valid', activation='relu')

        # LSTM layer
        self.lstm = LSTM(512, return_sequences=False)

        # Dense layers
        self.dense_layers = [Dense(fc1_dims, activation='relu') if i % 2 == 0 else Dense(fc2_dims, activation='relu') for i in range(nr_of_layers)]
        
        self.mu = Dense(n_actions, activation='tanh')  # Output layer

        # Setup checkpointing
        self.model_name = f"{name}_{unique_name}"
        self.checkpoint_dir = os.path.join(chkpt_dir, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.model_name}_ddpg.h5")

    def call(self, inputs):
        x = self.conv1d_1(inputs)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        x = self.conv1d_4(x)
        x = self.conv1d_5(x)
        x = self.conv1d_6(x)
        x = self.conv1d_7(x)
        x = self.conv1d_8(x)
        x = self.conv1d_9(x)
        x = self.lstm(x)
        for layer in self.dense_layers:
            x = layer(x)
        return self.mu(x)
