import os

import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense


from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate
from tensorflow.keras.models import Model

import os
import keras
from tensorflow.keras.layers import Dense, LSTM, Concatenate
import tensorflow as tf

from Config import WINDOW_SIZE

class CriticNetwork(keras.Model):
    def __init__(self,  fc1_dims=512, fc2_dims=512,
                 name='critic', unique_name='hades', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        # state_dim = (WINDOW_SIZE, 1)  # Hardcoded state dimension
        
        self.lstm = LSTM(self.fc1_dims)  # Specify input_shape for LSTM
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

        self.model_name = name + '_' + unique_name
        self.checkpoint_dir = chkpt_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.h5')

    def call(self, inputs):
        state, action = inputs
        state = self.lstm(state)
        combined_input = Concatenate()([state, action])
        action_value = self.fc1(combined_input)
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=512, 
                 name='actor', unique_name='hydra', chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        state_dim = (WINDOW_SIZE, 1)  # Hardcoded state dimension
        
        self.lstm = LSTM(64, input_shape=state_dim)  # Specify input_shape for LSTM
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(n_actions, activation='tanh')

        self.model_name = name + '_' + unique_name
        self.checkpoint_dir = chkpt_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name + '_ddpg.h5')

    def call(self, state):
        state = self.lstm(state)
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        return mu
