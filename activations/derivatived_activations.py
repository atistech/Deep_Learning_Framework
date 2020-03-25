import activations as a
import numpy as np

def logsigmoid(x):
    return a.logsigmoid(x)*(np.subtract(1,a.logsigmoid(x)))