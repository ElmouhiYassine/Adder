import numpy as np


def Max_values(noisy_image):
    return np.where(noisy_image == 0, 1, noisy_image)


def Min_values(noisy_image):
    return np.where(noisy_image == 0, -1, noisy_image)
