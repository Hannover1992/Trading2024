import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from Agent.buffer import ReplayBuffer
from Agent.networks import ActorNetwork, CriticNetwork
from State.Signal import Signal

class Agent:
    def __init__(self, input_dims, env, config):
        self.gamma = config.gamma
        self.tau = config.tau
        self.max_size = config.iteration
        self.input_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        self.noise_initial = config.noise
        self.noise_final = 0.0
        self.noise_decay_episodes = config.iteration

        self.current_noise = self.noise_initial
        self.max_iterations = config.iteration

        self.memory = ReplayBuffer(config.iteration, input_dims, self.n_actions)
        self.batch_size = config.batch_size
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(n_actions=self.n_actions, name='actor',
                                  unique_name=config.unique_name,
                                  fc1_dims=config.fc1, fc2_dims=config.fc2)
        self.critic = CriticNetwork(name='critic',
                                    unique_name=config.unique_name,
                                    fc1_dims=config.fc1, fc2_dims=config.fc2)
        self.target_actor = ActorNetwork(n_actions=self.n_actions,
                                         fc1_dims=config.fc1, fc2_dims=config.fc2,
                                         name='target_actor',
                                         unique_name=config.unique_name)
        self.target_critic = CriticNetwork(name='target_critic',
                                           unique_name=config.unique_name,
                                           fc1_dims=config.fc1, fc2_dims=config.fc2)

        self.actor.compile(optimizer=Adam(learning_rate=config.alpha))
        self.critic.compile(optimizer=Adam(learning_rate=config.beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=config.alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=config.beta))
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
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
        print("Observation Length: ", len(observation))
        actions = self.get_action_given_state(observation)
        return actions

    def choose_action_train(self, observation, episode):
        actions = self.get_action_given_state(observation)
        actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.current_noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions

    def decay_noise(self, current_iteration):
        decay_ratio = current_iteration / self.max_iterations
        self.current_noise = max(self.noise_final, self.noise_initial * (1 - decay_ratio))

    def get_action_given_state(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        dones = tf.convert_to_tensor(done, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma * critic_value_ * (1 - dones)
            critic_loss = keras.losses.MSE(target, critic_value)
            print("Critic Loss: ", critic_loss)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def get_signal_from_action(self, action):
        action_value = float(action[0])
        if action_value < -0.5:
            return Signal.SELL, (action_value + 1) / 0.5  # Scale to [0, 1]
        elif action_value < 0.5:
            return Signal.HOLD, 0.0
        else:
            return Signal.BUY, (action_value - 0.5) / 0.5  # Scale to [0, 1]
