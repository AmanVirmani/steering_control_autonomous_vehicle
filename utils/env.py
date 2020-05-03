import numpy as np
import cv2


class Map:
    def __init__(self, width=10, height=10, scale=40):
        self.width = width
        self.height = height
        self.scale = scale
        self.start = (0,0)
        self.goal = (9,9)
        self.action_set = [0, 1, 2, 4]
        self.map = self.create_map()
        self.curr_loc = self.get_current_loc()

    def create_map(self):
        #map = np.zeros(self.width, self.height)
        return np.load('world.npy')

    def render(self):
        # create curr_loc point
        world = cv2.rectangle(self.map, (self.curr_loc[0]*self.scale, self.curr_loc[1]*self.scale), (self.curr_loc[0]+1 * self.scale, self.curr_loc[1]+1 * self.scale), (0, 0, 255), -1)

        # display world
        world = cv2.flip(world, -1)
        cv2.imshow('world', world)
        cv2.waitKey(0)

    def move_robot(self, action):
        if not self.is_valid_action(action):
            return None
        # free space is 1
        curr_loc = self.curr_loc
        #self.map[self.curr_loc[0]][self.curr_loc[1]] = 1
        # left
        if action == 0:
            curr_loc[1] -= 1
        # down
        elif action == 1:
            curr_loc[0] += 1
        # right
        elif action == 2:
            curr_loc[1] += 1
        # up
        elif action == 3:
            curr_loc[0] -= 1

        # robot location is 2
        if curr_loc == self.goal:
            reward = 1
        elif self.map[curr_loc[0]][curr_loc[1]] == 0:
            reward = -1
            curr_loc = self.curr_loc
        else:
            reward = 0

        self.curr_loc = curr_loc
        state = self.curr_loc[0] * self.height + self.curr_loc[1]
        return state, reward, abs(reward)



    def get_current_loc(self):
        #loc = np.where(self.map == np.amax(self.map))
        loc = np.unravel_index(self.map.argmax(),self.map.shape)
        return loc

    def is_valid_action(self, action):
        valid = True
        # left
        if action == 0 and (self.curr_loc[1] - 1 < 0 or not self.map[self.curr_loc[0]][self.curr_loc[1]-1]):
            valid = False
        # down
        elif action == 1 and (self.curr_loc[0] + 1 >= self.height or not self.map[self.curr_loc[0]+1][self.curr_loc[1]]):
            valid = False
        # right
        elif action == 2 and (self.curr_loc[1] + 1 >= self.width or not self.map[self.curr_loc[0]][self.curr_loc[1]+1]):
            valid = False
        # up
        elif action == 3 and (self.curr_loc[0] - 1 < 0 or not self.map[self.curr_loc[0]-1][self.curr_loc[1]]):
            valid = False
        return valid

    def reset(self):
        self.map = self.create_map()