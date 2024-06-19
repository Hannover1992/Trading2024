import os
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Concatenate, Conv1D, Flatten
from tensorflow.keras.models import Model
import math as m

from Config import WINDOW_SIZE, NR_OF_LAYERS, FC1_DIMS_LSTM, FILTER_KERNEL

import os
import math as m
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Concatenate, Conv1D, Flatten, Reshape
from tensorflow.keras.models import Model


class ActorNetwork(Model):
    def __init__(self, n_actions, nr_of_layers=NR_OF_LAYERS, fc1_dims=512, fc2_dims=256, name='actor', unique_name='hydra', chkpt_dir='tmp/ddpg', window_size=WINDOW_SIZE):
        super(ActorNetwork, self).__init__()
        self.nr_of_layers = nr_of_layers

        # Calculate the number of layers for each convolutional set
        self.conv_nr_of_layers = int(m.log2(window_size) - m.log2(FC1_DIMS_LSTM))

        # Generate convolutional layers dynamically based on window size
        self.conv_layers_sets = []
        for i in range(self.conv_nr_of_layers + 1):
            current_window_size = window_size // (2 ** i)
            kernels = self.calculate_dynamic_kernels(current_window_size, FC1_DIMS_LSTM)
            self.conv_layers_sets.append(self.create_conv_layers(current_window_size, kernels))

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

    def calculate_dynamic_kernels(self, input_size, target_size):
        kernels = []
        while input_size > target_size:
            kernel_size = input_size - target_size + 1
            if kernel_size >= target_size:
                kernels.append(kernel_size)
                input_size = target_size
                target_size = max(target_size // 2, 1)
            else:
                break

        # Ensure the last kernel achieves the exact target size
        if input_size > target_size:
            kernels.append(input_size - target_size + 1)
        
        return kernels

    def create_conv_layers(self, window_size, kernels):
        layers = []
        for kernel_size in kernels:
            padding = 'valid'
            if window_size - kernel_size + 1 < 64:
                padding = 'same'
                kernel_size = 3  # Using smaller kernel size with padding
            layers.append(Conv1D(filters=FILTER_KERNEL, kernel_size=kernel_size, strides=1, padding=padding, activation='relu', input_shape=(window_size, 1)))
            window_size = window_size - kernel_size + 1 if padding == 'valid' else window_size
        return layers

    def call(self, inputs):
        # Process each segment with its corresponding convolutional layers
        outputs = []
        for i, conv_layers in enumerate(self.conv_layers_sets):
            segment = inputs[:, - (WINDOW_SIZE // (2 ** i)):, :]
            x = segment
            for conv in conv_layers:
                x = conv(x)
            # if x.shape[1] != 64:
            #     x = tf.image.resize(x, [64, x.shape[2]])
            x = Flatten()(x)
            outputs.append(tf.expand_dims(x, axis=-1))

        # Stack the results along a new axis
        x = tf.concat(outputs, axis=-1)

        x = self.lstm(x)
        for layer in self.dense_layers:
            x = layer(x)
        return self.mu(x)


class CriticNetwork(Model):
    def __init__(self, nr_of_layers=NR_OF_LAYERS, fc1_dims=512, fc2_dims=256, window_size=WINDOW_SIZE, name='critic', unique_name='hades', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.nr_of_layers = nr_of_layers

        # Calculate the number of layers for each convolutional set
        self.conv_nr_of_layers = int(m.log2(window_size) - m.log2(FC1_DIMS_LSTM))

        # Generate convolutional layers dynamically based on window size
        self.conv_layers_sets = []
        for i in range(self.conv_nr_of_layers + 1):
            current_window_size = window_size // (2 ** i)
            kernels = self.calculate_dynamic_kernels(current_window_size, FC1_DIMS_LSTM)
            self.conv_layers_sets.append(self.create_conv_layers(current_window_size, kernels))

        # LSTM layer
        self.lstm = LSTM(FC1_DIMS_LSTM, return_sequences=False)

        # Dense layers
        self.dense_layers = [Dense(fc1_dims, activation='relu') if i % 2 == 0 else Dense(fc2_dims, activation='relu') for i in range(nr_of_layers)]
        
        # Output Layer for Q-values
        self.q = Dense(1, activation=None)  # No activation function for Q-value output

        # Setup checkpointing
        self.model_name = f"{name}_{unique_name}"
        self.checkpoint_dir = os.path.join(chkpt_dir, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.model_name}_ddpg.h5")

    def calculate_dynamic_kernels(self, input_size, target_size):
        kernels = []
        while input_size > target_size:
            kernel_size = input_size - target_size + 1
            if kernel_size >= target_size:
                kernels.append(kernel_size)
                input_size = target_size
                target_size = max(target_size // 2, 1)
            else:
                break

        # Ensure the last kernel achieves the exact target size
        if input_size > target_size:
            kernels.append(input_size - target_size + 1)
        
        return kernels

    def create_conv_layers(self, window_size, kernels):
        layers = []
        for kernel_size in kernels:
            padding = 'valid'
            if window_size - kernel_size + 1 < 64:
                padding = 'same'
                kernel_size = 3  # Using smaller kernel size with padding
            layers.append(Conv1D(filters=FILTER_KERNEL, kernel_size=kernel_size, strides=1, padding=padding, activation='relu', input_shape=(window_size, 1)))
            window_size = window_size - kernel_size + 1 if padding == 'valid' else window_size
        return layers

    def call(self, inputs):
        state, action = inputs
        outputs = []
        for i, conv_layers in enumerate(self.conv_layers_sets):
            segment = state[:, - (WINDOW_SIZE // (2 ** i)):, :]
            x = segment
            for conv in conv_layers:
                x = conv(x)
            x = Flatten()(x)
            outputs.append(tf.expand_dims(x, axis=-1))

        # Stack the results along a new axis
        x = tf.concat(outputs, axis=-1)

        # Pass the concatenated output through the LSTM
        x = self.lstm(x)

        # Combine state and action
        combined_input = Concatenate()([x, action])

        # Pass through Dense layers
        for layer in self.dense_layers:
            combined_input = layer(combined_input)

        # Q-value output
        q = self.q(combined_input)
        return q
