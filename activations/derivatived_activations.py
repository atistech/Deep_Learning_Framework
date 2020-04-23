import activations.activations as a
import numpy as np

def sigmoid(x):
    return a.sigmoid(x)*(np.subtract(1,a.sigmoid(x)))