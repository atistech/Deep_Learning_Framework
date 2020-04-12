import numpy as np

class Activation:
    def __init__(self, inputs, name):
        self.input = inputs
        self.name = name

    def callFunc(self):
        if(self.name=="hardlimit"):
            return self.hardlimit()
        elif(self.name=="symmetricalhardlimit"):
            return self.symmetricalhardlimit()
        elif(self.name=="linear"):
            return self.linear()
        elif(self.name=="saturatinglinear"):
            return self.saturatinglinear()
        elif(self.name=="symmetricsaturatinglinear"):
            return self.symmetricsaturatinglinear()
        elif(self.name=="logsigmoid"):
            return self.logsigmoid()
        elif(self.name=="hyperbolictangentsigmoid"):
            return self.hyperbolictangentsigmoid()
        elif(self.name=="relu"):
            return self.relu()
        elif(self.name=="softmax"):
            return self.softmax()

    def hardlimit(self):
        if self.input < 0:
            return 0
        elif self.input >= 0:
            return 1

    def symmetricalhardlimit(self):
        if self.input < 0:
            return -1
        elif self.input >= 0:
            return 1

    def linear(self):
        return self.input

    def saturatinglinear(self):
        if self.input < 0:
            return 0
        elif 0 <= self.input <= 1:
            return self.input
        elif self.input > 1:
            return 1

    def symmetricsaturatinglinear(self):
        if self.input < -1:
            return -1
        elif -1 <= self.input <= 1:
            return self.input
        elif self.input > 1:
            return 1

    def logsigmoid(self):
        return 1/(1+np.exp(-self.input))

    def hyperbolictangentsigmoid(self):
        return (np.exp(self.input)-np.exp(-self.input))/(np.exp(self.input)+np.exp(-self.input))

    def relu(self):
        return np.maximum(0, self.input)

    def softmax(self):
        return np.exp(self.input-max(self.input))/sum(np.exp(self.input-max(self.input)))