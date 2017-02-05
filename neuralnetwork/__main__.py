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

    path = filepath + dirlisting[random_file_num]
    inputs = np.loadtxt(path, delimiter=',', dtype='float')
    # print(inputs[0])

    feedfor = FeedForward(64, inputs[0])
    feedfor.feedforward()

if __name__ == "__main__":
    main()
