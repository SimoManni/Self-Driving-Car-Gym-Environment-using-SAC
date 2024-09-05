import matplotlib.pyplot as plt

from settings import *
from environment import RacingEnv
from sac_torch import Agent


MAX_TIME_SECONDS = 10

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def learn_straight(n_episodes=50):
    env = RacingEnv(config='normal')
    agent = Agent(input_dim=env.observation_space.shape[0] * N_STATES, env=env, n_actions=env.action_space.shape[0])
    agent.load_models()
    filename = 'straight.png'
    figure_file = 'plots/' + filename

    score_history = []
    best_score = 0

    for e in range(n_episodes):

        if e % 10 == 0:
            env.render(agent=agent)

        observation = env.reset()
        done = False
        score = 0
        counter = 0

        max_steps = 200
        step = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)

            if reward == 0:
                counter += 1
            if counter > 100:
                done = True
            step += 1
            if step > max_steps:
                done = True

            score += reward
            agent.rememeber(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('Episode:', e, ', Score: %.2f' % score, ', Avg score %.2f' % avg_score)

    agent.save_models()

    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)


def learn_right(n_episodes=100):
    env = RacingEnv(config='right1')
    agent = Agent(input_dim=env.observation_space.shape[0] * N_STATES, env=env, n_actions=env.action_space.shape[0])
    agent.load_models()
    filename = 'right.png'
    figure_file = 'plots/' + filename

    score_history = []
    best_score = 0

    for e in range(n_episodes):

        if e % 10 == 0:
            env.render(agent=agent)

        observation = env.reset()
        done = False
        score = 0
        counter = 0

        max_steps = 200
        step = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)

            if reward == 0:
                counter += 1
            if counter > 100:
                done = True
            step += 1
            if step > max_steps:
                done = True

            score += reward
            agent.rememeber(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('Episode:', e, ', Score: %.2f' % score, ', Avg score %.2f' % avg_score)

    agent.save_models()

    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)


def learn_left(n_episodes=50):
    env = RacingEnv(config='normal')
    agent = Agent(input_dim=env.observation_space.shape[0] * N_STATES, env=env, n_actions=env.action_space.shape[0])
    agent.load_models()
    filename = 'left.png'
    figure_file = 'plots/' + filename

    score_history = []
    best_score = 0

    for e in range(n_episodes):

        if e % 10 == 0:
            env.render(agent=agent)

        observation = env.reset()
        done = False
        score = 0
        counter = 0

        max_steps = 200
        step = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)

            if reward == 0:
                counter += 1
            if counter > 100:
                done = True
            step += 1
            if step > max_steps:
                done = True

            score += reward
            agent.rememeber(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('Episode:', e, ', Score: %.2f' % score, ', Avg score %.2f' % avg_score)

    agent.save_models()

    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)




if __name__ == '__main__':
    learn_left()

