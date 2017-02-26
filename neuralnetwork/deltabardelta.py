from functions import Functions
import numpy as np

class Deltabar(object):


    def __init__(self, gradientofHtoO, gradientofItoH, learningRateofItoH, learningRateofHtoO):
        # self.biash = biash
        # self.biaso = biaso
        # self.deltabiaso = np.zeros(len(self.biaso))
        # self.weightsOfHtoO = weightsofHtoO
        # self.deltabiash = np.zeros(len(self.biash))
        # self.weightsOfItoH = weightsofItoH
        self.currgradientOfHtoO = gradientofHtoO
        self.currgradientOfItoH = gradientofItoH
        self.learingRateofItoH = learningRateofItoH
        self.learingRateofHtoO = learningRateofHtoO
        
        
    def Deltabardelta(self, outputsofo, outputofh, error, weightsofHtoO, sample):
        # local error at output 
        
        outo1neto1 = Functions().sigmoid(outputsofo, True)
        erroratoutputlayer = outo1neto1 * error
        
        # print("hello",outputofh)
        # print(outputofh)
        erroutputlayer = np.reshape(erroratoutputlayer, (-1, 1))
        tempOutputofh = np.resize(outputofh, (1,len(outputofh)))
        # current gradient at Hidden Layer to Output Layer
        # print(erroratoutputlayer.T)
        # print(self.currgradientOfHtoO)
        # gradientOfHtoO =  np.dot(erroratoutputlayer.T, outputofh)
        gradientOfHtoO = np.dot(erroutputlayer, tempOutputofh).T
        self.currgradientOfHtoO = gradientOfHtoO
        

        # calucate the error at the hidden layer
        errorContr = np.dot(erroratoutputlayer, np.matrix.transpose(weightsofHtoO))
        bi = Functions().sigmoid(outputofh, True)
        errorHiddenLayer = errorContr * bi

       
        errHiddenLayer = np.reshape(errorHiddenLayer, (-1, 1))
        tempSample = np.resize(sample, (1, len(sample)))
        # current gradient of Input layer to Hidden Layer
        # gradientOfItoH =  np.dot(errHiddenLayer, tempSample).T
        # gradientOfItoH = np.dot(errorHiddenLayer.T, tempSample)
        gradientOfItoH = np.dot(errHiddenLayer, tempSample).T
        self.currgradientOfItoH = gradientOfItoH

        
        # updating the bias
        # delta = learningRate * errorHiddenLayer
        # self.biash -= delta
        # self.deltabiash = delta


    def getGradientHtoO(self):
        return self.currgradientOfHtoO
    
    def getGradientItoH(self):
        return self.currgradientOfItoH