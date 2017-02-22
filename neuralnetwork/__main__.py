import math
import numpy as np
import os as os
from neuronlayer import NeuronLayer
from feedforward import FeedForward
from backprop import Backprop
from functions import Functions


# validation
# k-fold
# hold out (70, 30)

def main():

    learningRate = 0.05
    momentum = 0.01
    epoch = 200
    weightconnectionstoH = 64
    numberofHiddenNeuron = 25
    numberofOutputNeuron = 10 
    sizeofBaish = numberofHiddenNeuron
    sizeofBaiso = numberofOutputNeuron

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

    
    weightsforh = np.random.uniform(-0.5, 0.5, size=(weightconnectionstoH, numberofHiddenNeuron))
    weightsforo = np.random.uniform(-0.5, 0.5, size=(numberofHiddenNeuron, numberofOutputNeuron))

    biash =  np.random.uniform(-0.5, 0.5, size=sizeofBaish)
    biaso = np.random.uniform(-0.5, 0.5, size=sizeofBaiso)

    feedfor = FeedForward(weightsforh, weightsforo, biash, biaso)
    backprop = Backprop(biash, biaso, momentum)
    # epochs
    for fe in range(epoch):
        print('\nEpoch number: ', fe)
        np.random.shuffle(input_data)
        data, track = zip(*input_data)
        # number of weights connections from inputs, number of hidden neurons, output weight connections and number of output neuron
        accuracy = 0
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
            biash = backprop.getHiddenLayerBias()
            weightsforo = backprop.getOutputLayerError()
            biaso = backprop.getOutputLayerBias()

            maxValue = np.amax(feedforwardoutput)
            digit = list(feedforwardoutput).index(maxValue)
            excepted = list(Functions().getIncdices(trackId)).index(1)
            if maxValue > 0.5000000000000: 
                if digit == excepted:
                    accuracy += 1

            # if e == len(data) - 1:
            #     lastInput = list(Functions().getIncdices(trackId)).index(1)
            #     print('Max value for each output:', np.amax(feedforwardoutput))
            #     print('index of Max Value', list(feedforwardoutput).index(np.amax(feedforwardoutput)))
            
                # print(list(Functions().getIncdices(trackId)).index(1))
        # print(len(data))
        # for u in range(numberofOutputNeuron):
        #     err = np.subtract(Functions().getErrorResult(u), feedforwardoutput)
        #     globalError = 0.5*(np.sum((err) ** 2))
        #     print("Number: ", u)
        #     print("Error in the system: ", globalError)
        
        # print('Output layer: \n', feedforwardoutput)
        # print('Digit: ', lastInput)
        # err = np.subtract(Functions().getErrorResult(lastInput), feedforwardoutput)
        # print("Error in the system: ", 0.5*(np.sum((err) ** 2)))
        print("accuracy of the system: ", (accuracy/len(data)), "%", accuracy)
        


if __name__ == "__main__":
    main()
