import pygame
import numpy as np
import matplotlib.pyplot as plt

from settings import *
from environment import RacingEnv
from sac_torch import Agent


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


MAX_TIME_SECONDS = 10



if __name__ == '__main__':

    env = RacingEnv(config='normal')
    agent = Agent(input_dim=env.observation_space.shape[0] * N_STATES, env=env, n_actions=env.action_space.shape[0])
    agent.load_models()

    filename = 'SAC_car.png'
    figure_file = 'plots/' + filename

    score_history = []
    best_score = 0

    env.render(agent=agent, time_limit=False)