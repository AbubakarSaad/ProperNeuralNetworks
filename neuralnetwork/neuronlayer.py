"""
    This Class is neuron layer
"""
import numpy as np

class NeuronLayer(object):

    def __init__(self, nameoflayer):
        # self._connectiontoIn = connectiontoIn
        # self._sizeofH = sizeofH
        # self._sizeofO = sizeofO
        self._nameoflayer = nameoflayer

    # def createlayer(self):
    #     if self._nameoflayer == 'input':
    #         pass
    #     elif self._nameoflayer == 'hidden':
    #         self.weightsforh = np.random.uniform(-0.5, 0.5, size=(self._connectiontoIn, self._sizeofH))
    #         return self.weightsforh
    #     elif self._nameoflayer == 'output':
    #         self.weightsforo = np.random.uniform(-0.5, 0.5, size=(self._sizeofO, self._sizeofO))
    #         return self.weightsforo

    # random bais for each neuron in a layer between 0 to 1 (but not including 1)
    def biasforeach(self, size):
        return  np.random.uniform(-0.5, 0.5, size=size)

    # def getWeightsHidden(self):
    #     return self.weightsforh

    # def getWeightsOutput(self):
    #     return self.weightsforo
