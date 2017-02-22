from functions import Functions
import numpy as np

class Backprop(object):


    def __init__(self, biash, biaso, momentum):
        self.biash = biash
        self.biaso = biaso
        self.momentum = momentum

    def backpropagation(self, outputsofo, outputofh, error, weightsofHtoO, learningRate, sample, weightsofItoH):
        # local error at output 
        self.weightsOfHtoO = weightsofHtoO
        self.weightsOfItoH = weightsofItoH
        outo1neto1 = list(map(Functions().derSigmoid, outputsofo))
        # rate of change at output layer
        erroratoutputlayer = np.multiply(outo1neto1, error)
        self.deltaArrayOutput = np.zeros((len(weightsofHtoO), len(weightsofHtoO[0])))
        for i in range(len(weightsofHtoO[0])):
            for j in range(len(weightsofHtoO)):
                delta = erroratoutputlayer[i]*outputofh[j]*learningRate
                self.weightsOfHtoO[j][i] -= (delta + self.deltaArrayOutput[j][i] * self.momentum)
                self.deltaArrayOutput[j][i] = delta
        # updating the bias
        self.deltabiaso = np.zeros(len(self.biaso))
        for i in range(len(self.biaso)):
            delta = learningRate * erroratoutputlayer[i]
            self.biaso[i] -= delta
            self.deltabiaso = delta

        # calucate the error at the hidden layer
        errorContr = np.dot(erroratoutputlayer, np.matrix.transpose(weightsofHtoO))
        bi = list(map(Functions().derSigmoid, outputofh))
        errorHiddenLayer = np.multiply(errorContr, bi)
        # print('error ', errorHiddenLayer)
        self.deltaArrayHidden = np.zeros((len(weightsofItoH), len(weightsofItoH[0])))
        # print(len(weightsofItoH), len(weightsofItoH[0]))
        for i in range(len(weightsofItoH[0])):
            for j in range(len(weightsofItoH)):
                delta = errorHiddenLayer[i] * sample[j] * learningRate
                self.weightsOfItoH[j][i] -= (delta + self.deltaArrayHidden[j][i] * self.momentum)
                self.deltaArrayHidden[j][i] = delta
        
        # updating the bias
        self.deltabiash = np.zeros(len(self.biash))
        for i in range(len(self.biash)):
            delta = learningRate * errorHiddenLayer[i]
            self.biash[i] -= delta
            self.deltabiash = delta
    
    def getOutputLayerError(self):
        return self.weightsOfHtoO
    
    def getHiddenLayerError(self):
        return self.weightsOfItoH

    def getHiddenLayerBias(self):
        return self.deltabiash
    
    def getOutputLayerBias(self):
        return self.deltabiaso