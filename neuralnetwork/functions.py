import math
import numpy as np

class Functions():

    def dotproduct(self, inputs, weights):
        return np.dot(inputs, weights)

    def sigmoid(self, x):
        return 1/(1 + math.exp(-x))

    def tanh(self, x):
        return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
    
    # o is actual output (actual output)
    # t is expected output (target) 
    def derSigmoid(self, x):
        return x * (1 - x)

    def localError(self, o, t):
        return o - t

    def getIncdices(self, id):
        if math.floor(id / 700) == 0:
            return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 1:
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 2:
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 3:
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 4:
            return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif math.floor(id / 700) == 5:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif math.floor(id / 700) == 6:
            return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif math.floor(id / 700) == 7:
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif math.floor(id / 700) == 8:
            return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif math.floor(id / 700) == 9:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

