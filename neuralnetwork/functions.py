import math
import numpy as np

class Functions():

    def dotproduct(self, inputs, weights):
        return np.dot(inputs, weights)

    def sigmoid(self, x):
        return 1/(1 + math.exp(-x))

    def tanh(self, x):
        return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
    