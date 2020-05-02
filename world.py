import cv2
import numpy as np


# function to generate map
def createMap(world, scale=40):

    # scaling up world
    height = world.shape[0]*scale
    width = world.shape[1]*scale
    world = cv2.resize(world, (width, height))

    # get random obstacles
    for _ in range(10):
        obs_coord = np.random.randint(10, size=2)
        world = cv2.rectangle(world, (obs_coord[0]*scale, obs_coord[1]*scale),
                              (obs_coord[0]*scale + scale, obs_coord[1]*scale + scale), (0, 0, 0), -1)

    # create goal point and start point
    world = cv2.rectangle(world, (10*scale, 10*scale), (9*scale, 9*scale), (0, 255, 0), -1)
    world = cv2.rectangle(world, (0*scale, 0*scale), (1*scale, 1*scale), (0, 0, 255), -1)

    # display world
    world = cv2.flip(world, -1)

    return world


# main function
if __name__ == '__main__':

    # creating empty world
    w = np.ones((10, 10, 3))

    w = createMap(w, scale=40)

    # display world
    cv2.imshow('World', w)
    cv2.waitKey(0)
