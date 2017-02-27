import csv
import math
import numpy as np

class Functions():

    # sigmoid activation function
    def sigmoid(self, x, derv=False):
        if derv == True:
            return x * (1 - x)
        return 1/(1 + np.exp(-(x)))

    # Tanh activation function
    def tanh(self, x, derv=False):
        if derv == True:
            return 1 - (np.power(x, 2))


    # This function helps calculate the error by tracking index of input training array
    def getIncdices(self, id):
        if math.floor(id / 700) == 0:
            return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 1:
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 2:
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 3:
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 4:
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif math.floor(id / 700) == 5:
            return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif math.floor(id / 700) == 6:
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif math.floor(id / 700) == 7:
            return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif math.floor(id / 700) == 8:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif math.floor(id / 700) == 9:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    # This function helps with mean squared errors
    def getErrorResult(self, num):
        if  num == 0:
            return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif num == 1:
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif num == 2:
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif num == 3:
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif num == 4:
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif num == 5:
            return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif num == 6:
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif num == 7:
            return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif num == 8:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif num == 9:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # Stores Results in a csv file
    def storeInFile(self, data, filename):
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for line in data:
                for singleline in line:
                    writer.writerow(singleline)