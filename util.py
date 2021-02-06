import numpy as np


def normalize(angle):
    if abs(angle) > np.pi:
        return angle % (2 * np.pi)

    return angle
