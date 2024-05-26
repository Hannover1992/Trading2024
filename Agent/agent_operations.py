import ctypes

from modules.Optimization.DDPG.agent.ddpg_tf2 import Agent
from modules.Optimization.DDPG.config import ITERATION
from modules.Optimization.DDPG.enviroment.EnviromentDDPG import EnviromentDDPG
from modules.Optimization.DDPG.train import train

def run_simulation(config):
    best_eval_value = -10e10

    env = EnviromentDDPG()
    agent = Agent(**vars(config),input_dims=env.observation_space.shape, n_actions=env.action_space.shape[0], env=env )

    for x in range(ITERATION):
        best_eval_value = train(agent=agent,env=env, config=config, train_iter=x)

        if x == 0:  # Update the load_checkpoint after the first iteration
            config.load_checkpoint = True

    clean_up(env, agent)
    return best_eval_value


def induce_segfault():
    ctypes.string_at(0)

def clean_up(env, agent):
    env.close()                # Close the environment
    del env                    # Explicitly delete the environment
    del agent                  # Explicitly delete the agent
