import numpy as np
import cv2


class Map:
    def __init__(self, width=10, height=10, scale=40):
        self.width = width
        self.height = height
        self.scale = scale
        self.start = (0,0)
        self.goat = (9,9)
        self.curr_loc = self.get_current_loc()
        self.action_set = [0, 1, 2, 4]
        self.map = self.create_map()

    def create_map(self):
        #map = np.zeros(self.width, self.height)
        return np.load('world.npy')

    def render(self, curr_loc):
        # create curr_loc point
        world = cv2.rectangle(self.map, (curr_loc[0]*self.scale, curr_loc[1]*self.scale), ((curr_loc[0]+1) * self.scale, (curr_loc[1]+1) * self.scale), (0, 0, 255), -1)

        # display world
        world = cv2.flip(world, -1)
        cv2.imshow('world', world)
        cv2.waitKey(0)

    def move_robot(self, action):
        if not self.is_valid_action(action):
            return None
        # free space is 1
        self.map[self.curr_loc[0]][self.curr_loc[1]] == 1
        # left
        if action == 0:
            self.curr_loc[1] -= 1
        # down
        elif action == 1:
            self.curr_loc[0] += 1
        # right
        elif action == 2:
            self.curr_loc[1] += 1
        # up
        elif action == 3:
            self.curr_loc[0] -= 1

        # robot location is 2
        self.map[self.curr_loc[0]][self.curr_loc[1]] == 2

    def get_current_loc(self):
        loc = np.where(self.map == np.amax(self.map))
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