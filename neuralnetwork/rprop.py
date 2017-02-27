from functions import Functions
import numpy as np

class Rprop(object):


    def __init__(self, gradientofHtoO, gradientofItoH):
        self.currgradientOfHtoO = gradientofHtoO
        self.currgradientOfItoH = gradientofItoH
        
        
    def Respropagation(self, outputsofo, outputofh, error, weightsofHtoO, sample):
        
        # error at output 
        outo1neto1 = Functions().sigmoid(outputsofo, True)
        erroratoutputlayer = outo1neto1 * error
        
        # resizing and reshaping for dot product
        erroutputlayer = np.reshape(erroratoutputlayer, (-1, 1))
        tempOutputofh = np.resize(outputofh, (1,len(outputofh)))
        
        # Instead of applying double for loops, using matrix and dot product to calculate gradient 
        # This inceases the efficieny of the program by a factor of 10
        # gradient at Hidden Layer to Output Layer
        gradientOfHtoO = np.dot(erroutputlayer, tempOutputofh).T
        self.currgradientOfHtoO = gradientOfHtoO
        

        # calucate the error at the hidden layer
        errorContr = np.dot(erroratoutputlayer, np.matrix.transpose(weightsofHtoO))
        bi = Functions().sigmoid(outputofh, True)
        errorHiddenLayer = errorContr * bi

        # same concept as from hidden to output layer 
        errHiddenLayer = np.reshape(errorHiddenLayer, (-1, 1))
        tempSample = np.resize(sample, (1, len(sample)))
        
        #gradient of Input layer to Hidden Layer
        gradientOfItoH = np.dot(errHiddenLayer, tempSample).T
        self.currgradientOfItoH = gradientOfItoH

    
    # returns the gradient for hidden to output layer
    def getGradientHtoO(self):
        return self.currgradientOfHtoO
    
    # returns the gradient from input to hidden layer
    def getGradientItoH(self):
        return self.currgradientOfItoH