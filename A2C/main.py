import gym
import matplotlib.pyplot as plt
import numpy as np
from Agent import Agent


def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Average of previos 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make('e')
    agent = Agent(gamma=0.99, lr=5e-6, input_dims=[8], n_actions=4,
                  fc1_dims=2048, fc2_dims=1536)
    n_games = 3000

    fname = 'ACTOR_CRITIC_' + 'lunar_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score+=reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode', i, 'scores %.2f' % score,
              'average score %.2f' % avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)