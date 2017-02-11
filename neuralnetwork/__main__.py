import numpy as np
import os as os
from neuronlayer import NeuronLayer
from feedforward import FeedForward

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
    print(track[0])


    # feedfor = FeedForward(64, inputs[0])
    # feedfor.feedforward()

if __name__ == "__main__":
    main()
