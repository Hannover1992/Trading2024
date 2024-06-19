import os
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Concatenate, Conv1D
from tensorflow.keras.models import Model
import math as m

from Config import WINDOW_SIZE, NR_OF_LAYERS, FC1_DIMS_LSTM, FILTER_KERNEL

class ActorNetwork(Model):
    def __init__(self, n_actions, nr_of_layers=4, fc1_dims=512, fc2_dims=256, name='actor', unique_name='hydra', chkpt_dir='tmp/ddpg', window_size=WINDOW_SIZE):
        super(ActorNetwork, self).__init__()
        self.nr_of_layers = nr_of_layers

        # Generate convolutional layers dynamically based on window size
        self.conv_layers = self.create_conv_layers(1024, 64, 0)
        self.conv_layers2 = self.create_conv_layers(512, 64)
        self.conv_layers2 = self.create_conv_layers(256, 64)
        self.conv_layers2 = self.create_conv_layers(128, 64)
        self.conv_layers2 = self.create_conv_layers(64, 64)
        
        # LSTM layer
        self.lstm = LSTM(FC1_DIMS_LSTM, return_sequences=False)

        # Dense layers
        self.dense_layers = [Dense(fc1_dims, activation='relu') if i % 2 == 0 else Dense(fc2_dims, activation='relu') for i in range(nr_of_layers)]
        
        self.mu = Dense(n_actions, activation='tanh')  # Output layer

        # Setup checkpointing
        self.model_name = f"{name}_{unique_name}"
        self.checkpoint_dir = os.path.join(chkpt_dir, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.model_name}_ddpg.h5")

    def create_conv_layers(self, window_size, target_size, index):
        conv_nr_of_layers = m.log2(window_size) - m.log2(target_size)
        layers = []
        window_size = window_size // 2^index
        for i in range(int(conv_nr_of_layers)):
            kernel_size = window_size // 2 + 1
            layers.append(Conv1D(filters=FILTER_KERNEL, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', input_shape=(window_size, 1)))
            window_size = window_size // 2
        return layers

    def call(self, inputs):
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
        # print(x.shape)
        x = self.lstm(x)
        for layer in self.dense_layers:
            x = layer(x)
        return self.mu(x)

class CriticNetwork(Model):
    def __init__(self, nr_of_layers=4, fc1_dims=512, fc2_dims=256, window_size=WINDOW_SIZE,
                 name='critic', unique_name='hades', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.nr_of_layers = nr_of_layers

        # Generate convolutional layers dynamically based on window size
        self.conv_layers = self.create_conv_layers(window_size, FC1_DIMS_LSTM)
        
        # LSTM layer
        self.lstm = LSTM(FC1_DIMS_LSTM)

        # Dense layers
        self.dense_layers = [Dense(fc1_dims, activation='relu') if i % 2 == 0 else Dense(fc2_dims, activation='relu') for i in range(nr_of_layers)]
        
        # Output Layer for Q-values
        self.q = Dense(1, activation=None)  # No activation function for Q-value output

        # Setup checkpointing
        self.model_name = f"{name}_{unique_name}"
        self.checkpoint_dir = os.path.join(chkpt_dir, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.model_name}_ddpg.h5")

    def create_conv_layers(self, window_size, target_size):
        layers = []
        while window_size > target_size:
            kernel_size = window_size // 2 + 1
            layers.append(Conv1D(filters=FILTER_KERNEL, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', input_shape=(window_size, 1)))
            window_size = window_size // 2
        return layers

    def call(self, inputs):
        state, action = inputs
        x = state
        for conv in self.conv_layers:
            x = conv(x)
        x = self.lstm(x)
        combined_input = Concatenate()([x, action])  # Combine state and action
        for layer in self.dense_layers:
            combined_input = layer(combined_input)
        q = self.q(combined_input)
        return q
