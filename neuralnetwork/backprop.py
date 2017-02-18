from functions import Functions
import numpy as np

class Backprop(object):


    def __init__(self, outputsofo, outputofh, error, weightsofHtoO):
        self.outputsfeed = outputsofo
        self.outputofh = outputofh
        self.error = error
        self.weightofHtoO = weightsofHtoO


    def backpropagation(self):
        # local error at output 
        localerrorofO = list(map(Functions().derSigmoid, self.outputsfeed))
        print("local error at the output layer", localerrorofO)
        print("self_error", self.error, localerrorofO)

        # dj
        erroratoutputlayer = np.multiply(localerrorofO, self.error)
        print("error at the output layer", erroratoutputlayer)

        # calucate the error at the hidden layer
        errorathiddenlayer = Functions().dotproduct(self.weightofHtoO, np.matrix.transpose(erroratoutputlayer))
        # print(np.reshape(erroratoutputlayer, (-1, 1)))
        print('error at hidden layer', errorathiddenlayer)

        print('output of the hidden layer', self.outputofh)
        # ei = bi(1 - bi)sum(wij*dj)
        errorHiddenLayer = np.multiply(list(map(Functions().derSigmoid, self.outputofh)), errorathiddenlayer)
        print('ei of the hidden layer', errorHiddenLayer)