from Simulation.TradingEnv import TradingEnv  # Ensure this path is correct
from Agent.ddpg_tf2 import Agent
from Config import EXPLOITAION, Configuration
from Simulation.TradingSimulation import TradingSimulation  # Ensure this path is correct
from Printer.SimulationPrinter import SimulationPrinter

def train_ddpg(env, agent, num_episodes):
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        simulation_history = []

        normalized_state = env.normalize_state(state)

        first = True
        while not done:
            if episode % EXPLOITAION == 0:
                action = agent.choose_action_evaluate(state)
            else:
                action = agent.choose_action_train(state, episode)
            new_state, reward, done, _ = env.step(action)


            signal, volume = agent.get_signal_from_action(action)
            volume = env.volume
            # Save the history for plotting
            simulation_history.append({
                'State': state,
                'Action': signal.name,                
                'Action Volume': volume,
                'Price': env.data['Preis'].iloc[env.current_step],
                'Shares': env.shares,
                'Balance': env.cash,
                'Reward': reward,
                'Next State': new_state
            })


            # normalized_reward = env.normalize_reward(reward)
            normalized_reward = reward
            
            if(first):
                first = False
            else:
                agent.remember(normalized_state, action, normalized_reward, new_state, done)
                agent.learn()

            state = new_state
            episode_reward += reward

        agent.decay_noise(episode)
        
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
        if episode % EXPLOITAION == 0:
            if(episode != 0):
                print(f"Evaluation: {episode_reward}")
                # Plot the results using SimulationPrinter
                simulation = TradingSimulation(10000)
                simulation.data = env.data
                simulation.history = simulation_history
                SimulationPrinter.plot_results(simulation)

if __name__ == "__main__":
    # Environment setup
    env = TradingEnv()

    # DDPG Agent setup
    config = Configuration()
    agent = Agent(input_dims=env.observation_space.shape, env=env, config=config)

    # Train the agent
    train_ddpg(env, agent, config.iteration)
