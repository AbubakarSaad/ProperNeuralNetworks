from functions import Functions
import numpy as np

class Rprop(object):


    def __init__(self, gradientofHtoO, gradientofItoH):
        self.currgradientOfHtoO = gradientofHtoO
        self.currgradientOfItoH = gradientofItoH
        
        
    def Respropagation(self, outputsofo, outputofh, error, weightsofHtoO, sample):
        # local error at output 
        
        outo1neto1 = Functions().sigmoid(outputsofo, True)
        erroratoutputlayer = outo1neto1 * error
        
       
        erroutputlayer = np.reshape(erroratoutputlayer, (-1, 1))
        tempOutputofh = np.resize(outputofh, (1,len(outputofh)))
        
        gradientOfHtoO =  np.dot(erroratoutputlayer.T, outputofh)
        self.currgradientOfHtoO = np.matrix.transpose(gradientOfHtoO)
        

        # calucate the error at the hidden layer
        errorContr = np.dot(erroratoutputlayer, np.matrix.transpose(weightsofHtoO))
        bi = Functions().sigmoid(outputofh, True)
        errorHiddenLayer = errorContr * bi

       
        errHiddenLayer = np.reshape(errorHiddenLayer, (-1, 1))
        tempSample = np.resize(sample, (1, len(sample)))
        gradientOfItoH = np.dot(errorHiddenLayer.T, tempSample)
        self.currgradientOfItoH = np.matrix.transpose(gradientOfItoH)


    def getGradientHtoO(self):
        return self.currgradientOfHtoO
    
    def getGradientItoH(self):
        return self.currgradientOfItoH