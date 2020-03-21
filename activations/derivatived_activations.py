import activations as a

def logsigmoid(x):
    return a.logsigmoid(x)*(1-a.logsigmoid(x))