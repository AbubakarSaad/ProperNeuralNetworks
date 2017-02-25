import numpy as np
import os as os
import math
from feedforward import FeedForward
from rprop import Rprop
from functions import Functions


def main():

    # learningRate = 0.1
    # momentum = 0.01
    epoch = 100
    weightconnectionstoH = 64
    numberofHiddenNeuron = 30
    numberofOutputNeuron = 10 
    sizeofBaish = numberofHiddenNeuron
    sizeofBaiso = numberofOutputNeuron
    # k = 10
    npos = 1.2
    nneg = 0.5

    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = "c:\\Users\\Abu\\Documents\\ANN\\assignment1\\neuralnetwork\\a1digits\\"
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
    weightsforitoh = np.random.uniform(-0.5, 0.5, size=(weightconnectionstoH, numberofHiddenNeuron))
    weightsforhtoO = np.random.uniform(-0.5, 0.5, size=(numberofHiddenNeuron, numberofOutputNeuron))

    biash =  np.random.uniform(-0.5, 0.5, size=sizeofBaish)
    biaso = np.random.uniform(-0.5, 0.5, size=sizeofBaiso)

    feedfor = FeedForward(weightsforitoh, weightsforhtoO, biash, biaso)    

    # Input to hidden Layer
    currGradientItoH = np.zeros((len(weightsforitoh), len(weightsforitoh[0])))
    prevGradientItoH = np.zeros((len(weightsforitoh), len(weightsforitoh[0])))
    initWeightItoH = np.ones((len(weightsforitoh), len(weightsforitoh[0]))) * 0.1

    # Hidden to Output Layers
    currGradientHtoO = np.zeros((len(weightsforhtoO), len(weightsforhtoO[0])))
    prevGradientHtoO = np.zeros((len(weightsforhtoO), len(weightsforhtoO[0])))
    initWeightHtoO = np.ones((len(weightsforhtoO), len(weightsforhtoO[0]))) * 0.1

    rprop = Rprop(currGradientHtoO, currGradientItoH)
    # epochs
    for fe in range(epoch):
        print('\nEpoch number: ', fe)
        np.random.shuffle(input_data)
        data, track = zip(*input_data)
        
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
            
            rprop.Respropagation(feedforwardoutput, outputofh, errorofoutputofo, weightsforhtoO, data[e])
            
            currGradientHtoO += rprop.getGradientHtoO()
            currGradientItoH += rprop.getGradientItoH()

            maxValue = np.amax(feedforwardoutput)
            digit = list(feedforwardoutput[0]).index(maxValue)
            excepted = list(Functions().getIncdices(trackId)).index(1)
            if maxValue > 0.5000000000000: 
                if digit == excepted:
                    accuracy += 1
        
        # RProp starts here
        # currGradientHtoO = rprop.getGradientHtoO()
        # print(currGradientItoH)
        # currGradientItoH = rprop.getGradientItoH()
        for n in range(len(weightsforitoh)):
            for m in range(len(weightsforitoh[n])):
                # step 1: you check the direction of your gradient and prev gradient
                # if they in the same direction
                # multiple delta by 1.2
                if((currGradientItoH[n][m] < 0 and prevGradientItoH[n][m] < 0) or (currGradientItoH[n][m] > 0 and prevGradientItoH[n][m] > 0)):
                    initWeightItoH[n][m] = initWeightItoH[n][m] * npos
                elif((currGradientItoH[n][m] < 0 and prevGradientItoH[n][m] > 0) or (currGradientItoH[n][m] > 0 and prevGradientItoH[n][m] < 0)):
                    initWeightItoH[n][m] = initWeightItoH[n][m] * nneg
                    # currGradientItoH[n][m] = 0
                else:
                    weightsforitoh[n][m] = weightsforitoh[n][m]

                if(currGradientItoH[n][m] > 0):
                    weightsforitoh[n][m] = weightsforitoh[n][m] - initWeightItoH[n][m]
                elif(currGradientItoH[n][m] < 0):
                    weightsforitoh[n][m] = weightsforitoh[n][m] + initWeightItoH[n][m]
                # else:
                #     weightsforitoh[n][m] = weightsforitoh[n][m]
        
        
        for o in range(len(weightsforhtoO)):
            for p in range(len(weightsforhtoO[o])):

                if((currGradientHtoO[o][p] < 0 and prevGradientHtoO[o][p] < 0) or (currGradientHtoO[o][p] > 0 and prevGradientHtoO[o][p] > 0)):
                    initWeightHtoO[o][p] = initWeightHtoO[o][p] * npos
                elif((currGradientHtoO[o][p] < 0 and prevGradientHtoO[o][p] > 0) or (currGradientHtoO[o][p] > 0 and prevGradientHtoO[o][p] < 0)):
                    initWeightHtoO[o][p] = initWeightHtoO[o][p] * nneg
                    # currGradientHtoO[o][p] = 0
                else:
                    weightsforhtoO[o][p] = weightsforhtoO[o][p]

                if(currGradientHtoO[o][p] > 0):
                    weightsforhtoO[o][p] = weightsforhtoO[o][p] - initWeightHtoO[o][p]
                elif(currGradientHtoO[o][p] < 0):
                    weightsforhtoO[o][p] = weightsforhtoO[o][p] + initWeightHtoO[o][p]
                # else:
                #     weightsforhtoO[o][p] = weightsforhtoO[o][p]

        
    
        prevGradientItoH = currGradientItoH
        currGradientItoH = np.zeros((len(weightsforitoh), len(weightsforitoh[0])))
        prevGradientHtoO = currGradientHtoO
        currGradientHtoO = np.zeros((len(weightsforhtoO), len(weightsforhtoO[0])))
        np.clip(initWeightItoH, 0.000001, 50)
        np.clip(initWeightHtoO, 0.000001, 50)
    

        print("Training accuracy of the system: ", (accuracy/len(data)), "%", accuracy)
        


        # if fe == epoch-1:
        #     np.random.shuffle(input_testing)
        #     inputsTesting, trackTesting = zip(*input_testing)
        #     accuracyTesting = 0
        #     for t in range(len(inputsTesting)):
        #         feedfor.feedforward(inputsTesting[t])
        #         feedforwardoutput = feedfor.getOutputofOutputLayer()
        #         maxValue = np.amax(feedforwardoutput)
        #         digit = list(feedforwardoutput).index(maxValue)
        #         excepted = list(Functions().getIncdices(trackTesting[t])).index(1)
        #         if maxValue > 0.5000000000000: 
        #             if digit == excepted:
        #                 accuracyTesting += 1
        #     print("Testing accuracy of the system: ", (accuracyTesting/len(inputsTesting)), "%", accuracyTesting)
        


main()
