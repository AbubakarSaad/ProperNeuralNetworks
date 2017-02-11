from neuronlayer import NeuronLayer
from functions import Functions
import numpy as np

class FeedForward(object):

    def __init__(self, sizeoflayer, inputs):
        self._inputs = inputs
        self._sizeoflayer = sizeoflayer

    def feedforward(self):
        # np.array([[4, 3], [2, 4]])
        inputlayer = self._inputs
        hiddenlayer = NeuronLayer(self._sizeoflayer, 'hidden')
        outputlayer = NeuronLayer(self._sizeoflayer, 'output')

        weightconnectionh = hiddenlayer.createlayer()
        weightconnectiono = outputlayer.createlayer()
        print('inputlayer: \n', inputlayer)
        print('weightconnections: \n', weightconnectionh)

        # dotproduct for summtation of input times the weight (inputlayer to hiddenlayer)
        dotproducth = Functions().dotproduct(inputlayer, weightconnectionh)
        print('dotproduct for hiddenlayer: ', dotproducth)

        # bais to add to the summation of input and weights
        baish = hiddenlayer.biasforeach(self._sizeoflayer)
        print('bais: \n', baish)

        # add bais and dotproduct sum
        sumh = np.add(dotproducth, baish)
        print('sum of hiddenlayer dotproduct+bais: \n', sumh)

        # feed it to the activationfunction and list is used to print out the values
        activationfunctionofh = list(map(Functions().sigmoid, sumh))
        print('activationfunction of hiddenlayer: \n', activationfunctionofh)

        print(weightconnectiono)

        # dotproduct for summation of the hiddenlayer and weights (hiddenlayer to outputlayer)
        dotproducto = Functions().dotproduct(activationfunctionofh, weightconnectiono)
        print('dotproduct for outputlayer: ', dotproducto)

        # bais for output layer
        baiso = outputlayer.biasforeach(10)
        print('bais for output: \n', baiso)

        # add bais and dotproduct sum (outputlayer)
        sumo = np.add(dotproducto, baiso)
        print('sum of outputlayer dotproduct+bais: \n', sumo)

        # feed it to the activationfunction and list is used to print out the values (outputlayer)
        activationfunctionofo = list(map(Functions().sigmoid, sumo))
        print('activationfunction of outputlayer y: \n', activationfunctionofo)


        # global error = add local error of output layer





