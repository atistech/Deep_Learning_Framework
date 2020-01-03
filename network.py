import layer

class Network:
    def __init__(self, input):
        self.input = input
        self.last_output = input

    def addLayer(self, size, activation):
        newLayer = layer.Layer(self.last_output, size, activation)
        self.last_output = newLayer.getOutput()

    def getResult(self):
        return self.last_output