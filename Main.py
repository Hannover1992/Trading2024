from Simulation.TradingEnv import TradingEnv  # Ensure this path is correct
from Agent.ddpg_tf2 import Agent
from Config import EXPLOITAION, NOISE_MIN, Configuration
from Simulation.TradingEnv import TradingEnv  # Ensure this path is correct
from Agent.ddpg_tf2 import Agent
from Config import Configuration, ALPHA_MIN, ALPHA_MAX, NOISE_MIN, NOISE_MAX
import random
import multiprocessing

def train_ddpg(env, agent, num_episodes, instance_id):
    # Open the file in append mode
    with open(f"./Ergebnis/ergebnis_{agent.name}.txt", "a") as file:
        for episode in range(1, num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                if episode % EXPLOITAION == 0:
                    action = agent.choose_action_evaluate(state)
                else:
                    action = agent.choose_action_train(state)
                new_state, reward, done, _ = env.step(action)
                if episode % EXPLOITAION == 0:
                    agent.remember(state, action, reward, new_state, done)
                    agent.learn()
                state = new_state
                episode_reward += reward
                #not a number
            cash = env.combined_value_in_cash
            #ich mochte hier verlgiechen ob das cash -57.98... ist es hat mehrer float ich verlgiehc nur die ersten 2 stellen
            if round(cash, 2) == -57.98:
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward} Cash: {cash}")

            agent.decay_noise(episode)

            if episode % EXPLOITAION == 0:
                print(f"Evaluation")
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward} Cash: {cash}")
                file.write(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward} Cash: {cash} \n")

def run_training_process(instance_id, learning_rate, noise):
    # Environment setup
    env = TradingEnv()

    # DDPG Agent setup
    config = Configuration(learning_rate, noise)
    agent = Agent(input_dims=env.observation_space.shape, env=env, config=config)

    # Train the agent
    train_ddpg(env, agent, config.iteration, instance_id)

    print(f"Training instance {instance_id} completed")

def trainMultiDDPG():
    num_processes = 22
    processes = []

    for i in range(num_processes):
        #abtasten geliche abstand zwiche nalpah mi nund alph max 
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) / num_processes * i
        noise = NOISE_MIN + (NOISE_MAX - NOISE_MIN) / num_processes * i
        # learning_rate = random.uniform(ALPHA_MIN, ALPHA_MAX)
        learning_rate = alpha
        print(learning_rate)

        process = multiprocessing.Process(target=run_training_process, args=(i, learning_rate, noise))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()



if __name__ == "__main__":
    run_training_process(0, 0.01, 0.7)
    # trainMultiDDPG()
