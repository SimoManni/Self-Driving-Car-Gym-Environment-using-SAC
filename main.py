import matplotlib.pyplot as plt

from settings import *
from environment import RacingEnv
from SAC import Agent



def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def train(n_episodes=200):
    env = RacingEnv(multi_agent=True)
    agent = Agent(input_dim=env.observation_space.shape[0] * N_STATES, env=env, n_actions=env.action_space.shape[0])

    filename = 'learning.png'
    figure_file = 'plots/' + filename

    mean_score_history = []
    best_score = 0

    for e in range(n_episodes):

        # if e % 10 == 0:
        #     env.render(agent=agent)

        observation = env.reset()
        done_array = np.zeros(len(env.cars)).astype(np.bool_)
        score = np.zeros(len(env.cars))

        max_steps = 200
        step = 0
        while not np.all(done_array):
            if step > max_steps:
                break
            step += 1
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)

            for i, r in enumerate(reward):
                if r is not np.nan:
                    score[i] += r
            reward = reward[~np.isnan(reward)]
            if reward.size == 0:
                break
            agent.rememeber(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
        mean_score = np.mean(score)
        mean_score_history.append(mean_score)
        avg_score = np.mean(mean_score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('Episode:', e,
              ', Mean Score: %.2f' % mean_score,
              ', Max Score: %.2f' % np.max(score),
              ', Min Score: %.2f' % np.min(score),
              ', Average score (last 100) %.2f' % avg_score)

    agent.save_models()

    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, mean_score_history, figure_file)



if __name__ == '__main__':
    train()

