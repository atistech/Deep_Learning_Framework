import numpy as np

def hardlimit(x):
    if x < 0:
        return 0
    elif x >= 0:
        return 1

def symmetricalhardlimit(x):
    if x < 0:
        return -1
    elif x >= 0:
        return 1

def linear(x):
    return x

def saturatinglinear(x):
    if x < 0:
        return 0
    elif 0 <= x <= 1:
        return x
    elif x > 1:
        return 1

def symmetricsaturatinglinear(x):
    if x < -1:
        return -1
    elif -1 <= x <= 1:
        return x
    elif x > 1:
        return 1

def logsigmoid(x):
    return 1/(1+np.exp(-x))

def hyperbolictangentsigmoid(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

#Positive Linear
def relu(x):
    if np.all(x < 0):
        return 0
    elif np.all(0 <= x):
        return x

def softmax(x):
    return  np.exp(x-max(x))/sum(np.exp(x-max(x)))