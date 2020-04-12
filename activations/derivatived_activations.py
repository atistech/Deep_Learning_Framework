import activations as a
import numpy as np

def logsigmoid(x):
    func = a.Activation(x, 'logsigmoid')
    return func.callFunc()*(np.subtract(1,func.callFunc()))