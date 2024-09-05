from settings import *
from environment import RacingEnv
from SAC import Agent

if __name__ == '__main__':
    # To change initial configuration change INDEX parameter in settings.py

    env = RacingEnv(multi_agent=False)
    agent = Agent(input_dim=env.observation_space.shape[0] * N_STATES, env=env, n_actions=env.action_space.shape[0])
    agent.load_models()
    env.render(agent=agent, time_limit=False)