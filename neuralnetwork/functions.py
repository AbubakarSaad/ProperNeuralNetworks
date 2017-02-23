import math
import numpy as np

class Functions():

    def dotproduct(self, inputs, weights):
        return np.dot(inputs, weights)

    def sigmoid(self, x, derv=False):
        if derv == True:
            return x * (1 - x)
        return 1/(1 + np.exp(-(x)))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

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
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 5:
            return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif math.floor(id / 700) == 6:
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif math.floor(id / 700) == 7:
            return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif math.floor(id / 700) == 8:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif math.floor(id / 700) == 9:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def getErrorResult(self, num):
        if  num == 0:
            return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif num == 1:
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif num == 2:
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif num == 3:
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif num == 4:
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif num == 5:
            return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif num == 6:
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif num == 7:
            return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif num == 8:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif num == 9:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])