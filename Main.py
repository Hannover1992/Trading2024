from Simulation.TradingEnv import TradingEnv  # Ensure this path is correct
from Agent.ddpg_tf2 import Agent
from Config import EXPLOITAION, Configuration
from Simulation.TradingEnv import TradingEnv  # Ensure this path is correct
from Agent.ddpg_tf2 import Agent
from Config import Configuration

def train_ddpg(env, agent, num_episodes, instance_id):
    # Open the file in append mode
    with open(f"ergebnis_{instance_id}.txt", "a") as file:
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                if episode % EXPLOITAION == 0:
                    action = agent.choose_action_evaluate(state)
                else:
                    action = agent.choose_action_train(state, episode)
                new_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, new_state, done)

                agent.learn()
                state = new_state
                episode_reward += reward

            agent.decay_noise(episode)
            if episode % EXPLOITAION == 0:
                file.write(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}\n")
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

def run_training_process(instance_id):
    # Environment setup
    env = TradingEnv()

    # DDPG Agent setup
    config = Configuration()
    agent = Agent(input_dims=env.observation_space.shape, env=env, config=config)

    # Train the agent
    train_ddpg(env, agent, config.iteration, instance_id)
    print(f"Training instance {instance_id} completed")

def main():
    import multiprocessing

    num_processes = 1
    processes = []

    for i in range(num_processes):
        process = multiprocessing.Process(target=run_training_process, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()
