import math
import numpy as np
import os as os
from neuronlayer import NeuronLayer
from feedforward import FeedForward
from backprop import Backprop
from functions import Functions


# learning rate
# momentum
# bias 
# validation
# k-fold

def main():
    # opening a file
    # os.path.dirname gets the dir path for this file
    filepath = os.path.dirname(__file__) + "\\a1digits\\"
    dirlisting = os.listdir(filepath)
    print(dirlisting)

    # 0 - 9 are the test files and 10 - 19 are the training file
    random_file_num = np.random.randint(10, 20)
    np.random.seed(100)
    print(random_file_num)

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
    print(inputs[0])
    track = [i for i in range(0, len(inputs))]
    input_data = list(zip(data, track))

    np.random.shuffle(input_data)
    data, track = zip(*input_data)

    print(data[0])
    print("tracking index of ", track[0])
    trackId = track[0]

    # number of weights connections from inputs, number of hidden neurons, output weight connections and number of output neuron
    feedfor = FeedForward(64, 10, 10)
    feedforwardoutput = feedfor.feedforward(data[0])
    # simplyling calcuation for backprop keep track of output of hidden layer
    outputofh = feedfor.getOutputofHiddenLayer()


    # calculate the error right here because based on the track number error will change then send it to the backprop
    indices = Functions().getIncdices(trackId)
    print(indices, trackId)

    errorofoutputofo = np.subtract(feedforwardoutput, indices)
    print("error of output at outputlayer", errorofoutputofo)

    backprop = Backprop(feedforwardoutput, outputofh, errorofoutputofo, feedfor.getWeightsForOutputLayer())
    backprop.backpropagation()

    # adjust the weights here 


    # test if the weights are the same with the one generated
    # print(feedfor.getWeightsForHiddenLayer())

if __name__ == "__main__":
    main()
