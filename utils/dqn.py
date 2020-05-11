import gym
import numpy as np
import time, pickle, os
from utils import env
import random
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

env = env.Map()


class DQN:
    def __init__(self, state_size, action_size, total_episodes=100, max_steps=100):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.total_episodes = total_episodes
        self.max_steps = max_steps

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size,  activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        valid_action_set = []
        for action in env.action_set:
          if env.is_valid_action(action):
              valid_action_set.append(action)

        if np.random.rand() <= self.epsilon:
            return random.sample(valid_action_set, 1)[0]
        state_input = np.zeros((1, 100))
        state_input[0][state] = 1
        act_values = self.model.predict(state_input)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_input = np.zeros((1, 100))
                next_state_input[0][next_state] = 1
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state_input)[0]))
            state_input = np.zeros((1, 100))
            state_input[0][state] = 1
            target_f = self.model.predict(state_input)
            target_f[0][action] = target
            self.model.fit(state_input, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, batch_size = 32):
        for episode in range(self.total_episodes):
            print('Episodes', episode)
            env.reset()
            state = self.get_current_state()
            t = 0
            while t < self.max_steps:
                #env.render()

                action = self.choose_action(state)

                state2, reward, done = env.move_robot(action)
                if state2 is None:
                    continue

                self.memorize(state, action, reward, state2, done)

                state = state2

                t += 1

                if done:
                    env.render()
                    break
                if len(self.memory) > batch_size:
                    self.replay(batch_size)

            if episode % 10 == 0:
                self.save('./model_default.ckpt')

    def get_current_state(self):
        height, width = env.map.shape
        state = env.curr_loc[0]*width + env.curr_loc[1]
        return state

    def planPath(self):
        #self.model.load_weights('../Qtable/model_default.ckpt')
        self.model.load_weights('./model_default.ckpt')
        self.epsilon = 0
        env.reset()
        state = self.get_current_state()
        t = 0
        done = False
        waypoints = []
        while True:
            env.render()

            action = self.choose_action(state)
            waypoints.append((state, action))

            state2, reward, done = env.move_robot(action)
            if state2 is None:
                print('Invalid Action')
                continue

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
    n_states = 100
    n_actions = 4
    agent = DQN(state_size=n_states, action_size=n_actions, total_episodes=11, max_steps=10000)
    #agent.train()
    #agent.planPath()
    #cv2.destroyAllwindows()
