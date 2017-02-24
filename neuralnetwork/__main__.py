import math
import numpy as np
import os as os
from neuronlayer import NeuronLayer
from feedforward import FeedForward
from backprop import Backprop
from functions import Functions


# validation
# k-fold




def main():

    learningRate = 0.3
    momentum = 0.1
    epoch = 1
    weightconnectionstoH = 64
    numberofHiddenNeuron = 40
    numberofOutputNeuron = 10 
    sizeofBaish = numberofHiddenNeuron
    sizeofBaiso = numberofOutputNeuron
    k = 10

    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "/Users/abubakarsaad/Documents/cosc4p80/ProperNeuralNetworks/neuralnetwork/a1digits/"
    dirlistingtraining = os.listdir(filepath)
    # print(dirlistingtraining)

    # 0 - 9 are the test files and 10 - 19 are the training file
    # np.random.seed(100)

    # path = filepath + dirlistingtraining[random_file_num]
    # inputs = np.loadtxt(path, delimiter=',', dtype='float')
    inputsTraining = []
    
    # print(inputsTraining[0])
    for i in range(10, len(dirlistingtraining)): 
        path = filepath + dirlistingtraining[i]
        inp = np.loadtxt(path, delimiter=',', dtype='float')
        for j in range(len(inp)):
            inputsTraining.append(inp[j])

    # k-fold cross validation

    # hold out
    inputsTesting = []
    for i in range(0, len(dirlistingtraining) - 10): 
        path = filepath + dirlistingtraining[i]
        inp = np.loadtxt(path, delimiter=',', dtype='float')
        for j in range(len(inp)):
            inputsTesting.append(inp[j])
    
    trackTesting = [i for i in range(0, len(inputsTesting))]
    
    data = inputsTraining
    track = [i for i in range(0, len(inputsTraining))]
    input_data = list(zip(data, track))
    input_testing = list(zip(inputsTraining, trackTesting))
    
    # number of weights connections from inputs, number of hidden neurons, output weight connections and number of output neuron
    weightsforh = np.random.uniform(-0.5, 0.5, size=(weightconnectionstoH, numberofHiddenNeuron))
    weightsforo = np.random.uniform(-0.5, 0.5, size=(numberofHiddenNeuron, numberofOutputNeuron))

    biash =  np.random.uniform(-0.5, 0.5, size=sizeofBaish)
    biaso = np.random.uniform(-0.5, 0.5, size=sizeofBaiso)

    feedfor = FeedForward(weightsforh, weightsforo, biash, biaso)
    backprop = Backprop(biash, biaso, momentum, weightsforo, weightsforh)
    # epochs
    for fe in range(epoch):
        print('\nEpoch number: ', fe)
        np.random.shuffle(input_data)
        data, track = zip(*input_data)
        
        test_data =np.split(np.asarray(data), k)
        test_track = np.split(np.asarray(track), k)
        print('Chucks: ', len(test_track), len(test_track))

        # for i in range(k):
        #     for j in range(k-1):
        #     # go through training
        #     for z in range(1):
        #     # collect the aveage 


        accuracy = 0
        for e in range(len(data) - 6999):
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
        print("Training accuracy of the system: ", (accuracy/len(data)), "%", accuracy)

        if fe == epoch-1:
            np.random.shuffle(input_testing)
            inputsTesting, trackTesting = zip(*input_testing)
            accuracyTesting = 0
            for t in range(len(inputsTesting)):
                feedfor.feedforward(inputsTesting[t])
                feedforwardoutput = feedfor.getOutputofOutputLayer()
                maxValue = np.amax(feedforwardoutput)
                digit = list(feedforwardoutput).index(maxValue)
                excepted = list(Functions().getIncdices(trackTesting[t])).index(1)
                if maxValue > 0.5000000000000: 
                    if digit == excepted:
                        accuracyTesting += 1
            print("Testing accuracy of the system: ", (accuracyTesting/len(inputsTesting)), "%", accuracyTesting)
        


if __name__ == "__main__":
    main()
