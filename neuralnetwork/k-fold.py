import math
import numpy as np
import os as os
from neuronlayer import NeuronLayer
from feedforward import FeedForward
from backprop import Backprop
from functions import Functions


# validation
# k-fold




def main(runs):

    learningRate = 0.05
    momentum = 0.01
    epoch = 5
    weightconnectionstoH = 64
    numberofHiddenNeuron = 25
    numberofOutputNeuron = 10 
    sizeofBaish = numberofHiddenNeuron
    sizeofBaiso = numberofOutputNeuron
    k = 10

    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\assignment1\\neuralnetwork\\a1digits\\"
    dirlistingtraining = os.listdir(filepath)
    
    # Storing data for collection
    dataList = []
    Timesruns = 'Run: ' + str(runs)
    dataList.append(Timesruns)


    inputsTraining = []
    
    # print(inputsTraining[0])
    for i in range(10, len(dirlistingtraining)): 
        path = filepath + dirlistingtraining[i]
        inp = np.loadtxt(path, delimiter=',', dtype='float')
        for j in range(len(inp)):
            inputsTraining.append(inp[j])


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
    avgAccuarcyofEpoch = []
    # epochs
    for fe in range(epoch):
        print('\nEpoch number: ', fe)
        np.random.shuffle(input_data)
        data, track = zip(*input_data)
        
        test_data =np.split(np.asarray(data), k)
        test_track = np.split(np.asarray(track), k)
        # Collection of data
        accuaryList = []
        accuracyArr = []

        for i in range(k):
            accuracy = 0
            for j in range(k - 1):
                for e in range(len(test_data[j])):
                    trackId = test_track[j][e]
                    feedfor.feedforward(test_data[j][e])
                    feedforwardoutput = feedfor.getOutputofOutputLayer()
                    # simplyling calcuation for backprop keep track of output of hidden layer
                    outputofh = feedfor.getOutputofHiddenLayer()
                    # calculate the error right here because based on the track number error will change then send it to the backprop
                    indices = Functions().getIncdices(trackId)
                    errorofoutputofo = feedforwardoutput - indices
                    backprop.backpropagation(feedforwardoutput, outputofh, errorofoutputofo, weightsforo, learningRate, test_data[j][e], weightsforh)
                    # adjust the weights here 
                    weightsforh = backprop.getHiddenLayerError()
                    biash = backprop.getHiddenLayerBias()
                    weightsforo = backprop.getOutputLayerError()
                    biaso = backprop.getOutputLayerBias()
                if j == k-2:
                    for t in range(len(test_data[j])):
                        feedfor.feedforward(test_data[j+1][t])
                        feedforwardoutput = feedfor.getOutputofOutputLayer()
                        maxValue = np.amax(feedforwardoutput)
                        digit = list(feedforwardoutput).index(maxValue)
                        excepted = list(Functions().getIncdices(test_track[j+1][t])).index(1)
                        if maxValue > 0.5000000000000: 
                            if digit == excepted:
                                accuracy += 1
            accuracyArr.append(accuracy/len(test_data[i]))
            print("Training accuracy of the system: ", accuracy)
        print('accuracy array', accuracyArr)
        
        accuaryList.append(fe)
        accuaryList.append(accuracyArr)
        avgAccuacry = np.sum(accuracyArr)/k
        avgAccuarcyofEpoch.append(avgAccuacry)
        accuaryList.append('avgAccuarcyofEpoch')
        
        accuaryList.append(avgAccuarcyofEpoch)
        print("avgAccuacryofEpoch: ", avgAccuarcyofEpoch)
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
    filename = 'k-fold.csv'
    for i in range(2):
        finallist = main(i)
        storelist.append(finallist)
    Functions().storeInFile(storelist, filename)
