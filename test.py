import cv2
import numpy as np


# function to generate map
def createMap():

    # creating empty world
    world = np.ones((400, 400, 3))

    # scaling up world
    height = world.shape[0]/40
    width = world.shape[1]/40
    world.reshape((int(width), int(height)))

    cv2.imshow('world', world)

    # get random obstacles
    for _ in range(10):
        a = np.random.randint(10, size=2)
        world[a[0], a[1]] = (0, 0, 0)

    # set start point and goal point
    world[0, 0] = (0, 255, 0)
    world[9, 9] = (0, 0, 255)

    # display world
    cv2.imshow('world', world)
    cv2.waitKey(0)


# main function
if __name__ == '__main__':
    createMap()
