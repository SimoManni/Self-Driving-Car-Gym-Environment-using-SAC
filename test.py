from environment import RacingEnv
from sac_torch import Agent
from settings import *

env = RacingEnv(SIM=False)
state = env.reset()
observation, reward, done = env.step(np.array([0,0,0]))
print(observation.shape)
agent = Agent(input_dim=env.observation_space.shape[0] * N_STATES, env=env, n_actions=env.action_space.shape[0])
