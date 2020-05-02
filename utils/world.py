import cv2
import numpy as np


# function to generate map
def createMap(world, goal=(9, 9), scale=40):

    # scaling up world
    height = world.shape[0]*scale
    width = world.shape[1]*scale
    world = cv2.resize(world, (width, height))

    # get random obstacles
    for _ in range(10):
        obs_coord = np.random.randint(10, size=2)
        world = cv2.rectangle(world, (obs_coord[0]*scale, obs_coord[1]*scale),
                              ((obs_coord[0]+1) * scale, (obs_coord[1]+1) * scale), (0, 0, 0), -1)

    # create goal point
    world = cv2.rectangle(world, (goal[0]*scale, goal[1]*scale), ((goal[0]+1) * scale, (goal[1]+1) * scale), (0, 255, 0), -1)
    #
    # # create start point
    # world = cv2.rectangle(world, (start[0]*scale, start[1]*scale), ((start[0]+1) * scale, (start[1]+1) * scale), (0, 0, 255), -1)

    # display world
    # world = cv2.flip(world, -1)

    return world


# function to render world
def render(world, start=(0, 0), scale=40):

    # create start point
    world = cv2.rectangle(world, (start[0]*scale, start[1]*scale), ((start[0]+1) * scale, (start[1]+1) * scale), (0, 0, 255), -1)

    # display world
    world = cv2.flip(world, -1)
    cv2.imshow('world', world)


# main function
if __name__ == '__main__':

    # creating empty world
    w = np.ones((10, 10, 3))

    start_point = (0, 0)

    w = createMap(w, scale=40)

    render(w, start_point)

    # cv2.imshow('World', w)
    cv2.waitKey(0)
