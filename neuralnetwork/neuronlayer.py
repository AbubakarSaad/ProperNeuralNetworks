"""
    This Class is neuron layer
"""
import numpy as np

class NeuronLayer(object):

    def __init__(self, sizeoflayer, nameoflayer):
        self._sizeoflayer = sizeoflayer
        self._nameoflayer = nameoflayer

    def createlayer(self):
        if self._nameoflayer == 'input':
            pass
        elif self._nameoflayer == 'hidden':
            return np.random.uniform(-0.5, 0.5, size=(self._sizeoflayer, self._sizeoflayer))
        elif self._nameoflayer == 'output':
            return np.random.uniform(-0.5, 0.5, size=(self._sizeoflayer, 10))

    # random bais for each neuron in a layer between 0 to 1 (but not including 1)
    def biasforeach(self, size):
        return  np.random.uniform(-0.5, 0.5, size=size)
