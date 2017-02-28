import math
import numpy as np
import os as os
import glob
from feedforward import FeedForward
from backprop import Backprop
from functions import Functions


def main(runs):

    # NN parameters
    learningRate = 0.05
    momentum = 0.01
    epoch = 1
    weightconnectionstoH = 64
    numberofHiddenNeuron = 40
    numberofOutputNeuron = 10 
    sizeofBaish = numberofHiddenNeuron
    sizeofBaiso = numberofOutputNeuron
    

    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\assignment1\\neuralnetwork\\a1digits\\"
    # print(os.path.dirname(__file__))
    dirlistingtraining = os.listdir(filepath)
    # print(dirlistingtraining)
    # storing all the training samples
    inputsTraining = []
    
    # print(inputsTraining[0])
    for i in range(10, len(dirlistingtraining)): 
        path = filepath + dirlistingtraining[i]
        inp = np.loadtxt(path, delimiter=',', dtype='float')
        for j in range(len(inp)):
            inputsTraining.append(inp[j])

    dataList = []
    Timesruns = 'Run: ' + str(runs)
    dataList.append(Timesruns)

    
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
        np.seterr(all='print')
        print('\nEpoch number: ', fe)
        np.random.shuffle(input_data)
        data, track = zip(*input_data)
        # collecting data to store in file
        accuaryList = []
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
        print("Training accuracy of the system: ", (accuracy/len(data)), "%", accuracy)

        accuaryList.append(fe)
        accuaryList.append(accuracy/len(data))
        dataList.append(accuaryList)
        if fe == epoch-1:
            np.random.shuffle(input_testing)
            inputsTesting, trackTesting = zip(*input_testing)
            accuracyTesting = 0
            accuaryListTesting = []
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
            accuaryListTesting.append('Testing accuracy')
            accuaryListTesting.append((accuracyTesting/len(inputsTesting)))
            dataList.append(accuaryListTesting)

    return dataList



if __name__ == "__main__":
   
    storelist = []
    filename = 'hold-out.csv'
    for i in range(2):
        finallist = main(i)
        storelist.append(finallist)
    Functions().storeInFile(storelist, filename)
