import gym
import numpy as np
import time, pickle, os
from utils import env
import random
import cv2

env = env.Map()


class Qlearning:
    def __init__(self, Q, total_episodes, max_steps, epsilon=0.9, alpha=0.81, gamma=0.96):
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.Q = Q
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, state):
        action = 0
        valid_action_set = []
        for action in env.action_set:
            if env.is_valid_action(action):
                valid_action_set.append(action)

        if np.random.uniform(0, 1) < self.epsilon:
            action = random.sample(valid_action_set, 1)[0]
        else:
            return np.argmax(Q[state,:])
            #action_set = self.Q[state, :]
            #action_set.sort()
            #for index in range(len(action_set)):
            #    if env.is_valid_action(index):
            #        return index
        return action

    def learn(self, state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.alpha * (target - predict)

    def train(self):
        for episode in range(self.total_episodes):
            print('Episodes', episode)
            env.reset()
            state = self.get_current_state()
            t = 0

            while t < self.max_steps:
                env.render()

                action = self.choose_action(state)

                state2, reward, done = env.move_robot(action)

                self.learn(state, state2, reward, action)

                state = state2

                t += 1

                if done:
                    env.render()
                    break


    def get_current_state(self):
        height, width = env.map.shape
        state = env.curr_loc[0]*width + env.curr_loc[1]
        return state

    #def choose_action(self, state):
    #    action = np.argmax(self.Q[state, :])
    #    return action

    def planPath(self):
        env.reset()
        state = self.get_current_state()
        t = 0
        done = False
        waypoints = []
        while True:
            env.render()

            action = self.choose_action(state)
            waypoints.append((state,action))

            state2, reward, done = env.move_robot(action)

            state = state2

            if reward == 1:
                env.render()
                print("Success!!!!!!")
                cv2.waitKey(0)
                time.sleep(3)
                break
            elif reward == -1:
                env.render()
                print("Fell into the Hole:(")
                cv2.waitKey(0)
                time.sleep(3)
                break

            time.sleep(0.5)
            os.system('clear')
        return waypoints


if __name__=="__main__":
    total_episodes = 100
    max_steps = 10000

    #with open('../Qtable/frozenLake_qTable_final.pkl', 'rb') as f:
    #    Q = pickle.load(f)
    #Q = np.zeros((100, 4))
    Q = np.load('../Qtable/Qtable.npy')
    agent = Qlearning(Q, total_episodes, max_steps, epsilon=0)
    #agent.train()
    waypoints = agent.planPath()
    np.save('../waypoints.npy', waypoints)
    #print(Q)
