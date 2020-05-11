import numpy as np
import cv2


class Map:
    def __init__(self, width=10, height=10, scale=40):
        self.width = width
        self.height = height
        self.scale = scale
        self.start = (0,0)
        self.goal = (9,9)
        self.action_set = [0, 1, 2, 3]
        self.curr_loc = None
        self.map = self.create_map()
        self.curr_loc = self.get_current_loc()

    #def current_map(self):
    #    if self.curr_loc is not None:
    #        state = self.curr_loc[0] * self.width + self.curr_loc[1]
    #


    def create_map(self):
        world = np.load('world.npy')
        # start location
        if self.curr_loc is not None:
            world[0][0] = 1
            world[self.curr_loc[0]][self.curr_loc[1]] = 2
        # goal location
        world[-1][-1] = 1
        return world

    def render(self):
        # scaling up world
        height = self.map.shape[0]*self.scale
        width = self.map.shape[1]*self.scale
        world = np.full((height, width, 3), fill_value=255, dtype=np.uint8)  # cv2.resize(self.map, (width, height))

        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        black = (0, 0, 0)
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if self.map[row][col] == 0:
                    world = cv2.rectangle(world, (col * self.scale, row * self.scale), ((col + 1 )* self.scale, (row + 1) * self.scale), black, -1)

        # create goal point
        world = cv2.rectangle(world, (self.goal[1]*self.scale, self.goal[0]*self.scale), ((self.goal[1]+1) * self.scale, (self.goal[0]+1) * self.scale), red, -1)

        # create curr_loc point
        world = cv2.rectangle(world, (self.curr_loc[1]*self.scale, self.curr_loc[0]*self.scale), ((self.curr_loc[1]+1) * self.scale, (self.curr_loc[0]+1) * self.scale), green, -1)

        # display world
        #world = cv2.flip(world, -1)
        cv2.imshow('world', world)
        cv2.waitKey(1)

    def move_robot(self, action):
        if not self.is_valid_action(action):
            return None, 0, 0
        # free space is 1
        curr_loc = np.copy(self.curr_loc)
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
        if (curr_loc == self.goal).all():
            reward = 1
        elif self.map[curr_loc[0]][curr_loc[1]] == 0:
            reward = -1
            curr_loc = self.curr_loc
        else:
             reward = -0.01

        self.curr_loc = curr_loc
        self.map = self.create_map()
        state = self.curr_loc[0] * self.width + self.curr_loc[1]
        return state, reward, abs(reward)==1

    def get_current_loc(self):
        loc = np.unravel_index(self.map.argmax(), self.map.shape)
        return list(loc)

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
        self.curr_loc = [0,0]
        self.map = self.create_map()
