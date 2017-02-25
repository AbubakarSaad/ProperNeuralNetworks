from functions import Functions
import numpy as np

class Backprop(object):


    def __init__(self, biash, biaso, momentum, weightsofHtoO, weightsofItoH):
        self.biash = biash
        self.biaso = biaso
        self.momentum = momentum
        self.deltaArrayOutput = np.zeros((len(weightsofHtoO), len(weightsofHtoO[0])))
        self.deltabiaso = np.zeros(len(self.biaso))
        self.weightsOfHtoO = weightsofHtoO
        self.deltabiash = np.zeros(len(self.biash))
        self.deltaArrayHidden = np.zeros((len(weightsofItoH), len(weightsofItoH[0])))
        self.weightsOfItoH = weightsofItoH        

    def backpropagation(self, outputsofo, outputofh, error, weightsofHtoO, learningRate, sample, weightsofItoH):
        # local error at output 
        
        outo1neto1 = Functions().sigmoid(outputsofo, True)
        erroratoutputlayer = np.multiply(outo1neto1, error)
        
        
        erroutputlayer = np.reshape(erroratoutputlayer, (-1, 1))
        tempOutputofh = np.resize(outputofh, (1,len(outputofh)))
        deltao = learningRate * np.dot(erroutputlayer, tempOutputofh).T
        deltaMomentum = self.momentum * self.deltaArrayOutput
        self.weightsOfHtoO -= (deltao + deltaMomentum)
        self.deltaArrayOutput = deltao
        
        # print('deltaAarrayoutput', self.deltaArrayOutput)
        # for i in range(len(weightsofHtoO[0])):
        #     for j in range(len(weightsofHtoO)):
        #         delta = erroratoutputlayer[i]*outputofh[j]*learningRate
        #         self.weightsOfHtoO[j][i] -= (delta + self.deltaArrayOutput[j][i] * self.momentum)
        #         self.deltaArrayOutput[j][i] = delta
        

        deltaob = learningRate * erroratoutputlayer
        self.biaso -= deltaob
        self.deltabiaso = deltaob
        # updating the bias
        # for i in range(len(self.biaso)):
        #     delta = learningRate * erroratoutputlayer[i]
        #     self.biaso[i] -= delta
        #     self.deltabiaso = delta

        # calucate the error at the hidden layer
        errorContr = np.dot(erroratoutputlayer, np.matrix.transpose(weightsofHtoO))
        bi = Functions().sigmoid(outputofh, True)
        errorHiddenLayer = np.multiply(errorContr, bi)

        
        errHiddenLayer = np.reshape(errorHiddenLayer, (-1, 1))
        tempSample = np.resize(sample, (1, len(sample)))
        deltah = learningRate * np.dot(errHiddenLayer, tempSample).T
        deltaMomentumh = self.momentum * self.deltaArrayHidden
        self.weightsOfItoH -= (deltah + deltaMomentumh)
        self.deltaArrayHidden = deltah
        # print(len(weightsofItoH), len(weightsofItoH[0]))
        # for i in range(len(weightsofItoH[0])):
        #     for j in range(len(weightsofItoH)):
        #         delta = errorHiddenLayer[i] * sample[j] * learningRate
        #         self.weightsOfItoH[j][i] -= (delta + self.deltaArrayHidden[j][i] * self.momentum)
        #         self.deltaArrayHidden[j][i] = delta
        
        # updating the bias
        deltaoh = learningRate * errorHiddenLayer
        self.biash -= deltaoh
        self.deltabiash = deltaoh
        # for i in range(len(self.biash)):
        #     delta = learningRate * errorHiddenLayer[i]
        #     self.biash[i] -= delta
        #     self.deltabiash = delta
    
    def getOutputLayerError(self):
        return self.weightsOfHtoO
    
    def getHiddenLayerError(self):
        return self.weightsOfItoH

    def getHiddenLayerBias(self):
        return self.deltabiash
    
    def getOutputLayerBias(self):
        return self.deltabiaso