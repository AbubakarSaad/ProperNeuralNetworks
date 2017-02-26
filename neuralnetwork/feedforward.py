from neuronlayer import NeuronLayer
from functions import Functions
import numpy as np

class FeedForward(object):

    def __init__(self, weightsforh, weightsforo, baish, baiso):
        # self.hiddenlayer = NeuronLayer('hidden')
        self.weightconnectionh = weightsforh # weights for hidden layer
        # self.outputlayer = NeuronLayer('output')
        self.weightconnectiono = weightsforo # weights for output layer
        # bais to add to the summation of input and weights
        self.baish = baish
        # bais for output layer
        self.baiso = baiso

    def feedforward(self, inputs):
        inputlayer = inputs
        # inputlayer = np.resize(inputlayer, (1, len(inputs)))
        # dotproduct for summtation of input times the weight (inputlayer to hiddenlayer)
        dotproducth = np.dot(inputlayer, self.weightconnectionh)
        sumh = dotproducth + self.baish
        
        # feed it to the activationfunction and list is used to print out the values
        self.outputofhs = Functions().sigmoid(sumh) 

        # dotproduct for summation of the hiddenlayer and weights (hiddenlayer to outputlayer)
        dotproducto = np.dot(self.outputofhs, self.weightconnectiono)
        sumo = dotproducto + self.baiso
        
        # feed it to the activationfunction and list is used to print out the values (outputlayer)
        self.outputofos = Functions().sigmoid(sumo)
        

    def getOutputofHiddenLayer(self):
        return self.outputofhs
    
    def getOutputofOutputLayer(self):
        return self.outputofos