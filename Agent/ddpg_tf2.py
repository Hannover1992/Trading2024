import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

from Agent.buffer import ReplayBuffer
from Agent.networks import ActorNetwork, CriticNetwork
from Config import EXPLOITAION


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=2, max_size=1000, tau=0.005,
                 fc1=400, fc2=300, batch_size=64, noise=0.1, unique_name='hades', load_checkpoint=True, steps=0):
        self.gamma = gamma
        self.tau = tau

        self.max_size = int(max_size)
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.noise_initial = noise
        self.noise_final = 0.0
        self.noise_decay_episodes = 1000

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(n_actions=n_actions, name='actor',
                                  unique_name=unique_name,
                                    fc1_dims=fc1, fc2_dims=fc2
                                  )
        self.critic = CriticNetwork(name='critic',
                                    unique_name=unique_name,
                                    fc1_dims=fc1, fc2_dims=fc2
                                    )
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         fc1_dims=fc1, fc2_dims=fc2,
                                         name='target_actor',
                                         unique_name=unique_name,
                                         )
        self.target_critic = CriticNetwork(name='target_critic',
                                           unique_name=unique_name,
                                           fc1_dims=fc1, fc2_dims=fc2)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))
        self.update_network_parameters(tau=1)

    def __del__(self):
        del self.memory

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        if reward == 1.0 or reward == -1.0:
            self.memory = ReplayBuffer(self.max_size, self.input_dims, self.n_actions)
        else:
            self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action_evaluate(self, observation):
        actions = self.get_action_given_state(observation)
        return actions[0]

    def choose_action_train(self, observation, episode):
        actions = self.get_action_given_state(observation)
        current_noise = self.calculate_noise(episode)
        if episode % EXPLOITAION != 0:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=current_noise)

        return actions

    def calculate_noise(self, episode):
        noise = max(self.noise_final, self.noise_initial - (self.noise_initial - self.noise_final) * (episode / self.noise_decay_episodes))
        return noise

    def get_action_given_state(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        return actions

    def has_negative_value(self, tensor):
        return bool(tf.reduce_any(tensor < 0).numpy())

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma * critic_value_ * (1 - done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
