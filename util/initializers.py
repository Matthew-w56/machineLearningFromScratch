import numpy as np


def zero_initializer(*size):
    return np.zeros([s for s in size])


# def random_initializer(size, value_range=2.0, value_min=-1.0):
def random_initializer(*size):
    return np.random.rand(*size) * 2.0 - 1.0
