import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v0', is_slippery=False)

class Qlearning:
    def __init__(self, Q, total_episodes, max_steps, epsilon = 0.9, alpha = 0.81, gamma = 0.96):
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.Q = Q
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self,state):
        action=0
        if np.random.uniform(0, 1) < self.epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def learn(self, state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + gamma * np.max(Q[state2, :])
        Q[state, action] = Q[state, action] + lr_rate * (target - predict)

    def train(self):
        for episode in range(self.total_episodes):
            print('Episodes', episode)
            state = env.reset()
            t = 0

            while t < self.max_steps:
                # env.render()

                action = self.choose_action(state)

                state2, reward, done, info = env.step(action)

                self.learn(state, state2, reward, action)

                state = state2

                t += 1

                if done:
                    break

                time.sleep(0.1)


if __name__=="__main__":
    epsilon = 0.9
    total_episodes = 100
    max_steps = 100

    lr_rate = 0.81
    gamma = 0.96

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    agent = Qlearning(Q, total_episodes, max_steps)
    agent.train()
    print(Q)

#with open(path + "frozenLake_qTable.pkl", 'wb') as f:
#    pickle.dump(Q, f)
