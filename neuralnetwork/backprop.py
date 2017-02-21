from functions import Functions
import numpy as np

class Backprop(object):


    def __init__(self):
        pass

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
                self.weightsOfHtoO[j][i] -= delta
                self.deltaArrayOutput[j][i] = delta
        
        # calucate the error at the hidden layer
        errorContr = np.dot(np.matrix.transpose(weightsofHtoO), erroratoutputlayer)
        bi = list(map(Functions().derSigmoid, outputofh))
        errorHiddenLayer = np.multiply(errorContr, bi)
        # print('error ', errorHiddenLayer)
        self.deltaArrayHidden = np.zeros((len(weightsofItoH), len(weightsofItoH[0])))
        # print(len(weightsofItoH), len(weightsofItoH[0]))
        for i in range(len(weightsofItoH[0])):
            for j in range(len(weightsofItoH)):
                delta = errorHiddenLayer[i] * sample[j] * learningRate
                self.weightsOfItoH[j][i] -= delta
                self.deltaArrayHidden[j][i] = delta

    def getOutputLayerError(self):
        return self.weightsOfHtoO
    
    def getHiddenLayerError(self):
        return self.weightsOfItoH
