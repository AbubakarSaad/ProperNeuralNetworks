import numpy as np
import os as os
import math
from feedforward import FeedForward
from deltabardelta import Deltabar
from functions import Functions


def main(runs):

    # learningRate = 0.1
    # momentum = 0.01
    epoch = 200
    weightconnectionstoH = 64
    numberofHiddenNeuron = 25
    numberofOutputNeuron = 10 
    sizeofBaish = numberofHiddenNeuron
    sizeofBaiso = numberofOutputNeuron
    # k = 10
    npos = 1.2
    nneg = 0.5
    D = 0.20 # decay
    K = 0.0001 # growth

    # initialLR = 0.0005
    # maxLearningRate = .001
    # weight decay = 0.2
    # weigthgrwoth = 0.0001

    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\assignment1\\neuralnetwork\\a1digits\\"
    dirlistingtraining = os.listdir(filepath)
    # print(dirlistingtraining)

    dataList = []
    Timesruns = 'Run: ' + str(runs)
    dataList.append(Timesruns)

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
    weightsforitoh = np.random.uniform(-0.5, 0.5, size=(weightconnectionstoH, numberofHiddenNeuron))
    weightsforhtoO = np.random.uniform(-0.5, 0.5, size=(numberofHiddenNeuron, numberofOutputNeuron))

    biash =  np.random.uniform(-0.5, 0.5, size=sizeofBaish)
    biaso = np.random.uniform(-0.5, 0.5, size=sizeofBaiso)

    feedfor = FeedForward(weightsforitoh, weightsforhtoO, biash, biaso)    

    # Input to hidden Layer
    currGradientItoH = np.zeros((len(weightsforitoh), len(weightsforitoh[0])))
    prevGradientItoH = np.zeros((len(weightsforitoh), len(weightsforitoh[0])))
    learingRateItoH = np.ones((len(weightsforitoh), len(weightsforitoh[0]))) * 0.0005

    # Hidden to Output Layers
    currGradientHtoO = np.zeros((len(weightsforhtoO), len(weightsforhtoO[0])))
    prevGradientHtoO = np.zeros((len(weightsforhtoO), len(weightsforhtoO[0])))
    learingRateHtoO = np.ones((len(weightsforhtoO), len(weightsforhtoO[0]))) * 0.0005


    deltabar_delta = Deltabar(currGradientHtoO, currGradientItoH, learingRateItoH, learingRateHtoO)
    # epochs
    for fe in range(epoch):
        print('\nEpoch number: ', fe)
        np.random.shuffle(input_data)
        data, track = zip(*input_data)
        
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
            
            deltabar_delta.Deltabardelta(feedforwardoutput, outputofh, errorofoutputofo, weightsforhtoO, data[e])
            
            currGradientHtoO += deltabar_delta.getGradientHtoO()
            currGradientItoH += deltabar_delta.getGradientItoH()

            maxValue = np.amax(feedforwardoutput)
            digit = list(feedforwardoutput).index(maxValue)
            excepted = list(Functions().getIncdices(trackId)).index(1)
            if maxValue > 0.5000000000000: 
                if digit == excepted:
                    accuracy += 1
        
        # Deltabar Delta starts here
        for n in range(len(weightsforitoh)):
            for m in range(len(weightsforitoh[n])):
                # step 1: you check the direction of your gradient and prev gradient
                # if they in the same direction
                # multiple delta by 1.2
                if((currGradientItoH[n][m] < 0 and prevGradientItoH[n][m] < 0) or (currGradientItoH[n][m] > 0 and prevGradientItoH[n][m] > 0)):
                    learingRateItoH[n][m] = learingRateItoH[n][m] + K
                elif((currGradientItoH[n][m] < 0 and prevGradientItoH[n][m] > 0) or (currGradientItoH[n][m] > 0 and prevGradientItoH[n][m] < 0)):
                    learingRateItoH[n][m] = learingRateItoH[n][m] * (1 - D)
                    
                currGradientItoH[n][m] = currGradientItoH[n][m] * learingRateItoH[n][m]
                weightsforitoh[n][m] = weightsforitoh[n][m] - currGradientItoH[n][m]
                
        
        
        for o in range(len(weightsforhtoO)):
            for p in range(len(weightsforhtoO[o])):

                if((currGradientHtoO[o][p] < 0 and prevGradientHtoO[o][p] < 0) or (currGradientHtoO[o][p] > 0 and prevGradientHtoO[o][p] > 0)):
                    learingRateHtoO[o][p] = learingRateHtoO[o][p] + K
                elif((currGradientHtoO[o][p] < 0 and prevGradientHtoO[o][p] > 0) or (currGradientHtoO[o][p] > 0 and prevGradientHtoO[o][p] < 0)):
                    learingRateHtoO[o][p] = learingRateHtoO[o][p] * (1 - D)
                   
                currGradientHtoO[o][p] = currGradientHtoO[o][p] * learingRateHtoO[o][p]
                weightsforhtoO[o][p] = weightsforhtoO[o][p] - currGradientHtoO[o][p]
                

        prevGradientItoH = currGradientItoH
        currGradientItoH = np.zeros((len(weightsforitoh), len(weightsforitoh[0])))
        prevGradientHtoO = currGradientHtoO
        currGradientHtoO = np.zeros((len(weightsforhtoO), len(weightsforhtoO[0])))
        np.clip(learingRateItoH, 0.0001, 0.001)
        np.clip(learingRateHtoO, 0.0001, 0.001)
    

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
    filename = 'deltabardelta.csv'
    for i in range(2):
        finallist = main(i)
        storelist.append(finallist)
    Functions().storeInFile(storelist, filename)