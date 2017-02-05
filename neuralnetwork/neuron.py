"""
    This Class is neuron
"""
import abc
import numpy as np
"""
    This Class is AbsNeuron
"""
class AbsNeuron(object):
    __metaclass__ = abc.ABCMeta
    print("neuron class")

    def __init__(self, inputx, size):
        self._input = inputx
        self._weights = np.random.uniform(-0.5, 0.5, size)
    """
        This Class is neuron
    """
    def getInput(self):
        return self._input
    """
        This Class is neuron
    """
    def getWeights(self):
        return self._weights

