import numpy as np

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

def mean_squared_error(actual, current):
    diff = np.subtract(actual, current)
    return np.mean(np.power(diff,2))

def cross_entropy(yHat, y):
    if y == 1:
        return -np.log(yHat)
    else:
        return -np.log(1-yHat)