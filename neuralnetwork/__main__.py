import math
import numpy as np
import os as os
from neuronlayer import NeuronLayer
from feedforward import FeedForward
from backprop import Backprop
from functions import Functions


# momentum
# validation
# k-fold
# update bais

def main():

    learningRate = 0.2
    weightconnectionstoH = 64
    numberofHiddenNeuron = 10
    numberofOutputNeuron = 10 # applys to same number of weights connection from hidden to output
    sizeofBaish = 10
    sizeofBaiso = 10
    
    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\assignment1\\neuralnetwork\\a1digits\\"
    dirlisting = os.listdir(filepath)
    # print(dirlisting)

    # 0 - 9 are the test files and 10 - 19 are the training file
    # np.random.seed(100)

    # path = filepath + dirlisting[random_file_num]
    # inputs = np.loadtxt(path, delimiter=',', dtype='float')
    inputs = []
    # print(inputs[0])
    for i in range(10, len(dirlisting)): 
        path = filepath + dirlisting[i]
        inp = np.loadtxt(path, delimiter=',', dtype='float')
        for j in range(len(inp)):
            inputs.append(inp[j])

    # k-fold cross validation

    data = inputs
    track = [i for i in range(0, len(inputs))]
    input_data = list(zip(data, track))

     

    # epochs
    for fe in range(40):
        print('\nEpoch number: ', fe)
        np.random.shuffle(input_data)
        data, track = zip(*input_data)
        # number of weights connections from inputs, number of hidden neurons, output weight connections and number of output neuron
        weightsforh = np.random.uniform(-0.5, 0.5, size=(weightconnectionstoH, numberofHiddenNeuron))
        weightsforo = np.random.uniform(-0.5, 0.5, size=(numberofHiddenNeuron, numberofOutputNeuron))
        feedfor = FeedForward(weightsforh, weightsforo, sizeofBaish, sizeofBaiso)
        backprop = Backprop()
        for e in range(len(data)):
            trackId = track[e]
            feedfor.feedforward(data[e])
            feedforwardoutput = feedfor.getOutputofOutputLayer()
            # simplyling calcuation for backprop keep track of output of hidden layer
            outputofh = feedfor.getOutputofHiddenLayer()
            # calculate the error right here because based on the track number error will change then send it to the backprop
            indices = Functions().getIncdices(trackId)
            errorofoutputofo = feedforwardoutput - indices
            backprop.backpropagation(feedforwardoutput, outputofh, errorofoutputofo, weightsforo, learningRate, data[e], weightsforh)
            # adjust the weights here 
            weightsforh = backprop.getHiddenLayerError()
            weightsforo = backprop.getOutputLayerError()
            if e == len(data) - 1:
                lastInput = list(Functions().getIncdices(trackId)).index(1)
                # print(list(Functions().getIncdices(trackId)).index(1))
        # print(len(data))
        # for u in range(numberofOutputNeuron):
        #     err = np.subtract(Functions().getErrorResult(u), feedforwardoutput)
        #     globalError = 0.5*(np.sum((err) ** 2))
        #     print("Number: ", u)
        #     print("Error in the system: ", globalError)
        
        print('Output layer: \n', feedforwardoutput)
        print('Digit: ', lastInput)
        err = np.subtract(Functions().getErrorResult(lastInput), feedforwardoutput)
        print("Error in the system: ", 0.5*(np.sum((err) ** 2)))
        


if __name__ == "__main__":
    main()
