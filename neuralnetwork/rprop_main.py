import numpy as np
import os as os
import math
from feedforward import FeedForward
from rprop import Rprop
from functions import Functions


def main(runs):

    # Parameters for NN
    # learningRate = 0.1
    # momentum = 0.01
    epoch = 100
    weightconnectionstoH = 64
    numberofHiddenNeuron = 25
    numberofOutputNeuron = 10 
    sizeofBaish = numberofHiddenNeuron
    sizeofBaiso = numberofOutputNeuron
    # k = 10
    # npositive and negative
    npos = 1.2
    nneg = 0.5

    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\assignment1\\neuralnetwork\\a1digits\\"
    dirlistingtraining = os.listdir(filepath)
    # print(dirlistingtraining)

    # Stores the result to printit out in csv file
    dataList = []
    Timesruns = 'Run: ' + str(runs)
    dataList.append(Timesruns)

    # store all training samples into an array
    inputsTraining = []
    for i in range(10, len(dirlistingtraining)): 
        path = filepath + dirlistingtraining[i]
        inp = np.loadtxt(path, delimiter=',', dtype='float')
        for j in range(len(inp)):
            inputsTraining.append(inp[j])

    
    
    # Collecting the test samples into array
    inputsTesting = []
    for i in range(0, len(dirlistingtraining) - 10): 
        path = filepath + dirlistingtraining[i]
        inp = np.loadtxt(path, delimiter=',', dtype='float')
        for j in range(len(inp)):
            inputsTesting.append(inp[j])
    # tracks the test samples
    trackTesting = [i for i in range(0, len(inputsTesting))]
    
    # make copy of training sample array 
    data = inputsTraining
    # tracks indices of training sample array
    track = [i for i in range(0, len(inputsTraining))]
    
    # pack them into one array
    input_data = list(zip(data, track))
    input_testing = list(zip(inputsTraining, trackTesting))
    
    # number of weights connections from inputs, number of hidden neurons, output weight connections and number of output neuron
    weightsforitoh = np.random.uniform(-0.5, 0.5, size=(weightconnectionstoH, numberofHiddenNeuron))
    weightsforhtoO = np.random.uniform(-0.5, 0.5, size=(numberofHiddenNeuron, numberofOutputNeuron))

    # generate random bias
    biash =  np.random.uniform(-0.5, 0.5, size=sizeofBaish)
    biaso = np.random.uniform(-0.5, 0.5, size=sizeofBaiso)

    # iniatiate feedforward 
    feedfor = FeedForward(weightsforitoh, weightsforhtoO, biash, biaso)    

    # Input to hidden Layer (keep track of current graident and previous gradiate)
    currGradientItoH = np.zeros((len(weightsforitoh), len(weightsforitoh[0])))
    prevGradientItoH = np.zeros((len(weightsforitoh), len(weightsforitoh[0])))
    deltaWeightItoH = np.ones((len(weightsforitoh), len(weightsforitoh[0]))) * 0.1

    # Hidden to Output Layers
    currGradientHtoO = np.zeros((len(weightsforhtoO), len(weightsforhtoO[0])))
    prevGradientHtoO = np.zeros((len(weightsforhtoO), len(weightsforhtoO[0])))
    deltaWeightHtoO = np.ones((len(weightsforhtoO), len(weightsforhtoO[0]))) * 0.1

    # inialize Rprop
    rprop = Rprop(currGradientHtoO, currGradientItoH)
    # epochs
    for fe in range(epoch):

        print('\nEpoch number: ', fe)
        # shuffle the training samples and its tracking indices
        np.random.shuffle(input_data)
        # returns the shuffled arrays
        data, track = zip(*input_data)
        
        # store the result for csv file
        accuaryList = []
        # keep track of accuracy
        accuracy = 0
        # loops over the training samples
        for e in range(len(data)):
            # get the indices of sample
            trackId = track[e]
            feedfor.feedforward(data[e])
            feedforwardoutput = feedfor.getOutputofOutputLayer()
            
            # output of hidden layer 
            outputofh = feedfor.getOutputofHiddenLayer()
            # error (output - traget)
            indices = Functions().getIncdices(trackId)
            errorofoutputofo = feedforwardoutput - indices
            
            rprop.Respropagation(feedforwardoutput, outputofh, errorofoutputofo, weightsforhtoO, data[e])
            
            # accumlate gradient
            currGradientHtoO += rprop.getGradientHtoO()
            currGradientItoH += rprop.getGradientItoH()

            # keep calculate the accuary
            maxValue = np.amax(feedforwardoutput)
            digit = list(feedforwardoutput).index(maxValue)
            excepted = list(Functions().getIncdices(trackId)).index(1)
            if maxValue > 0.5000000000000: 
                if digit == excepted:
                    accuracy += 1
        
        # RProp starts here
        # Rprop scenoirs, for input to hidden layer
        for n in range(len(weightsforitoh)):
            for m in range(len(weightsforitoh[n])):
                # step 1: you check the direction of your gradient and prev gradient
                # if they in the same direction
                # multiple delta by 1.2
                if((currGradientItoH[n][m] < 0 and prevGradientItoH[n][m] < 0) or (currGradientItoH[n][m] > 0 and prevGradientItoH[n][m] > 0)):
                    deltaWeightItoH[n][m] = deltaWeightItoH[n][m] * npos
                elif((currGradientItoH[n][m] < 0 and prevGradientItoH[n][m] > 0) or (currGradientItoH[n][m] > 0 and prevGradientItoH[n][m] < 0)):
                    deltaWeightItoH[n][m] = deltaWeightItoH[n][m] * nneg
                    # currGradientItoH[n][m] = 0
                else:
                    weightsforitoh[n][m] = weightsforitoh[n][m]

                # step 2: if current gradient is positive 
                # decrease the weight
                # if current graident is negative 
                # increase the weight
                if(currGradientItoH[n][m] > 0):
                    weightsforitoh[n][m] = weightsforitoh[n][m] - deltaWeightItoH[n][m]
                elif(currGradientItoH[n][m] < 0):
                    weightsforitoh[n][m] = weightsforitoh[n][m] + deltaWeightItoH[n][m]
                # else:
                #     weightsforitoh[n][m] = weightsforitoh[n][m]
        
        # Rprop scenoirs, for hidden to output layer
        for o in range(len(weightsforhtoO)):
            for p in range(len(weightsforhtoO[o])):

                if((currGradientHtoO[o][p] < 0 and prevGradientHtoO[o][p] < 0) or (currGradientHtoO[o][p] > 0 and prevGradientHtoO[o][p] > 0)):
                    deltaWeightHtoO[o][p] = deltaWeightHtoO[o][p] * npos
                elif((currGradientHtoO[o][p] < 0 and prevGradientHtoO[o][p] > 0) or (currGradientHtoO[o][p] > 0 and prevGradientHtoO[o][p] < 0)):
                    deltaWeightHtoO[o][p] = deltaWeightHtoO[o][p] * nneg
                    # currGradientHtoO[o][p] = 0
                else:
                    weightsforhtoO[o][p] = weightsforhtoO[o][p]

                if(currGradientHtoO[o][p] > 0):
                    weightsforhtoO[o][p] = weightsforhtoO[o][p] - deltaWeightHtoO[o][p]
                elif(currGradientHtoO[o][p] < 0):
                    weightsforhtoO[o][p] = weightsforhtoO[o][p] + deltaWeightHtoO[o][p]
                # else:
                #     weightsforhtoO[o][p] = weightsforhtoO[o][p]

        
        
        prevGradientItoH = currGradientItoH
        currGradientItoH = np.zeros((len(weightsforitoh), len(weightsforitoh[0])))
        prevGradientHtoO = currGradientHtoO
        currGradientHtoO = np.zeros((len(weightsforhtoO), len(weightsforhtoO[0])))
        
        # only allows max of 50 and min of 0.00001 weights
        np.clip(deltaWeightItoH, 0.000001, 50)
        np.clip(deltaWeightHtoO, 0.000001, 50)
    

        print("Training accuracy of the system: ", (accuracy/len(data)), "%", accuracy)
        accuaryList.append(fe)
        accuaryList.append(accuracy/len(data))
        dataList.append(accuaryList)

        # Running the testing sample to find the accuray of system
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
    # Can run multiple runs and store the result in csv file
    storelist = []
    filename = 'rprop.csv'
    for i in range(2):
        finallist = main(i)
        storelist.append(finallist)
    Functions().storeInFile(storelist, filename)
