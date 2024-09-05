from environment import RacingEnv
from sac_torch import Agent
from settings import *

env = RacingEnv(SIM=True)
agent = Agent(input_dim=env.observation_space.shape[0] * N_STATES, env=env, n_actions=env.action_space.shape[0])

observation = env.reset()
action = agent.choose_action(observation)

observation_, reward, done = env.step(action)
print(observation_.shape, reward.shape, done.shape)