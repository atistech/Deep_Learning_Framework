import math

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
    return 1/(1+math.exp(-x))

def hyperbolictangentsigmoid(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

#Positive Linear
def relu(x):
    if x < 0:
        return 0
    elif 0 <= x:
        return x

def softmax(x):
    y = []
    for i in x:
        y.append(math.exp(i)/sum(x, math.exp(i)))
    return y