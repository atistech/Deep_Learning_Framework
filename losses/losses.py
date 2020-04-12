import numpy as np

def mean_squared_error(actual, current):
    diff = np.subtract(actual, current)
    return np.mean(np.power(diff,2))

def CrossEntropy(yHat, y):
    if y == 1:
        return -np.log(yHat)
    else:
        return -np.log(1-yHat)