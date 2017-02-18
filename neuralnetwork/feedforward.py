from neuronlayer import NeuronLayer
from functions import Functions
import numpy as np

class FeedForward(object):

    def __init__(self, connectiontoH, sizeofH, sizeofO):
        self.hiddenlayer = NeuronLayer(connectiontoH, sizeofH, 0, 'hidden')
        self.weightconnectionh = self.hiddenlayer.createlayer()
        self.outputlayer = NeuronLayer(0, 0, sizeofO, 'output')
        self.weightconnectiono = self.outputlayer.createlayer()

    def feedforward(self, inputs):
        # np.array([[4, 3], [2, 4]])
        inputlayer = inputs
    
        print('inputlayer: \n', inputlayer)
        print('weightconnections for hiddenlayer: \n', self.weightconnectionh)

        # dotproduct for summtation of input times the weight (inputlayer to hiddenlayer)
        dotproducth = Functions().dotproduct(inputlayer, self.weightconnectionh)
        print('dotproduct for hiddenlayer: ', dotproducth)

        # bais to add to the summation of input and weights
        baish = self.hiddenlayer.biasforeach(10)
        print('bais: \n', baish)

        # add bais and dotproduct sum
        sumh = np.add(dotproducth, baish)
        print('sum of hiddenlayer dotproduct+bais: \n', sumh)

        # feed it to the activationfunction and list is used to print out the values
        self.outputofhs = list(map(Functions().sigmoid, sumh))
        print('activationfunction of hiddenlayer: \n', self.outputofhs)
        # outputofhs ==== hidden layer outputs
        print('weightconnections for outputlayer: \n', self.weightconnectiono)

        # dotproduct for summation of the hiddenlayer and weights (hiddenlayer to outputlayer)
        dotproducto = Functions().dotproduct(self.outputofhs, self.weightconnectiono)
        print('dotproduct for outputlayer: ', dotproducto)

        # bais for output layer
        baiso = self.outputlayer.biasforeach(10)
        print('bais for output: \n', baiso)

        # add bais and dotproduct sum (outputlayer)
        sumo = np.add(dotproducto, baiso)
        print('sum of outputlayer dotproduct+bais: \n', sumo)

        # feed it to the activationfunction and list is used to print out the values (outputlayer)
        self.outputofos = list(map(Functions().sigmoid, sumo))
        print('activationfunction of outputlayer y: \n',  self.outputofos)
        # outputofos ==== output at the outputlayer
        return self.outputofos

    def getWeightsForHiddenLayer(self):
        return self.hiddenlayer.getWeightsHidden()

    def getWeightsForOutputLayer(self):
        return self.outputlayer.getWeightsOutput()

    def getOutputofHiddenLayer(self):
        return self.outputofhs
    
    def getOutputofOutputLayer(self):
        return self.outputofos