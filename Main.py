from Simulation.TradingEnv import TradingEnv  # Ensure this path is correct
from Agent.ddpg_tf2 import Agent
from Config import EXPLOITAION, Configuration
from Simulation.TradingEnv import TradingEnv  # Ensure this path is correct
from Agent.ddpg_tf2 import Agent
from Config import Configuration, ALPHA_MIN, ALPHA_MAX
import random
import multiprocessing

def train_ddpg(env, agent, num_episodes, instance_id):
    # Open the file in append mode
    with open(f"./Ergebnis/ergebnis_{instance_id}.txt", "a") as file:
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                if episode % EXPLOITAION == 0 and episode > 0:
                    action = agent.choose_action_evaluate(state)
                    file.write(f"Evaluate:")
                    print(f"Evaluate:")
                else:
                    action = agent.choose_action_train(state, episode)
                new_state, reward, done, _ = env.step(action)
                if episode % EXPLOITAION != 0 and episode > 0:
                    agent.remember(state, action, reward, new_state, done)
                    agent.learn()
                state = new_state
                episode_reward += reward

            agent.decay_noise(episode)

            file.write(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}\n")
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

def run_training_process(instance_id, learning_rate):
    # Environment setup
    env = TradingEnv()

    # DDPG Agent setup
    config = Configuration(learning_rate)
    agent = Agent(input_dims=env.observation_space.shape, env=env, config=config)

    # Train the agent
    train_ddpg(env, agent, config.iteration, instance_id)
    print(f"Training instance {instance_id} completed")

def main():

    # run_training_process(0, 0.0001)

    num_processes = 22
    processes = []

    for i in range(num_processes):
        #abtasten geliche abstand zwiche nalpah mi nund alph max 
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) / num_processes * i
        # learning_rate = random.uniform(ALPHA_MIN, ALPHA_MAX)
        learning_rate = alpha
        process = multiprocessing.Process(target=run_training_process, args=(i, learning_rate))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()
