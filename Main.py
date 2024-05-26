from Simulation.TradingEnv import TradingEnv  # Ensure this path is correct

from Agent.ddpg_tf2 import Agent
from Config import Configuration

def train_ddpg(env, agent, num_episodes):


    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action_train(state, episode)
            new_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            episode_reward += reward

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

if __name__ == "__main__":
    # Environment setup
    env = TradingEnv()

    # DDPG Agent setup
    config = Configuration(alpha=0.001, beta=0.002, tau=0.005, fc1=400, fc2=300, batch_size=64, noise=0.1, gamma=0.99, load_checkpoint=False, steps=10000)
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.shape[0], env=env, **vars(config))

    # Train the agent
    num_episodes = 1000
    train_ddpg(env, agent, num_episodes)
