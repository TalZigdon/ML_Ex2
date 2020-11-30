import random
import numpy as np
import math
import sys

# import validationFile
import validationFile
EPOCHS_OF_PER = 5

def getStdAndAvgOfTrain(array):
    std = np.zeros(len(array[0]))
    sum = np.zeros(len(array[0]))
    avg = np.zeros(len(array[0]))
    for i in range(len(array)):
        for j in range(len(array[0])):
            sum[j] += array[i][j]
    for i in range(len(array[0])):
        avg[i] = sum[i] / len(array)
    for i in range(len(array)):
        for j in range(len(array[0])):
            std[j] = std[j] + math.pow((array[i][j] - avg[j]), 2)
    for i in range(len(std)):
        std[i] = std[i] / len(array)
        std[i] = math.sqrt(std[i])
    return std, avg


def normalization(array, std, avg):
    for i in range(len(array)):
        for j in range(1, len(array[0])):
            array[i][j] = (array[i][j] - avg[j]) / std[j]
    return array


def convertWAndRTo0And1AndReturnNpArray(file):
    fin = open(file, "r")
    lines = fin.readlines()
    npArray = np.zeros((len(lines), 13))
    counter = 0
    for line in lines:
        list = [1]
        if (line[-1] == '\n'):
            line = line[:-1]
        values = line.split(',')
        for value in values:
            if value == "W":
                value = 0
            if value == "R":
                value = 1
            list.append(float(value))
        npArray[counter] = np.array(list)
        counter += 1
    return npArray


train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
train_x = convertWAndRTo0And1AndReturnNpArray(train_x)
std, avg = getStdAndAvgOfTrain(train_x)
train_x = normalization(train_x, std, avg)
train_y = np.loadtxt(train_y)
test_x = convertWAndRTo0And1AndReturnNpArray(test_x)
test_x = normalization(test_x, std, avg)


def scoreOfKNN(ans, realAns):
    counter = 0
    size = len(ans)
    for i in range(len(ans)):
        if (ans[i] != realAns[i]):
            counter += 1
    return (1 - counter / size)


def KNN(k, trainX, trainY, test_x):
    ans = np.arange(len(test_x))
    for j in range(len(test_x)):
        listOfDiffs = np.zeros(len(trainX))
        for i in range(len(trainX)):
            diff = np.linalg.norm(test_x[j] - trainX[i])
            listOfDiffs[i] = diff
        listOfMinimum = np.zeros(k)
        for i in range(k):
            minIndex = np.argmin(listOfDiffs)
            listOfMinimum[i] = trainY[minIndex]
            listOfDiffs[minIndex] = 100000000
        counter = np.zeros(3)
        for i in range(k):
            counter[int(listOfMinimum[i])] += 1
        ans[j] = int(np.argmax(counter))
    return ans


def perceptronTestResult(validationX, validationY, w):
    counter = 0.0
    size = len(validationX)
    for x, y in zip(validationX, validationY):
        y_hat = np.argmax(np.dot(w, x))
        if y == y_hat:
            counter += 1
    return counter / size


def perceptron(test_x, train_x, train_y):
    eta = 1
    w = np.zeros((3, 13))
    w[0] = 0.6
    epoch_count = 25
    for epoch in range(epoch_count):
        random.shuffle(list(zip(train_x, train_y)))
        c = zip(train_x, train_y)
        for x, y in c:
            y_hat = np.argmax(np.dot(w, x))
            if int(y) != y_hat:
                w[int(y), :] = w[int(y), :] + eta * x
                w[y_hat, :] = w[int(y_hat), :] - eta * x
    res = np.zeros(len(test_x), dtype=int)
    for i in range(len(test_x)):
        res[i] = np.argmax(np.dot(w, test_x[i]))
    return res , w


def passiveAgressive(test_x, train_x, train_y):
    w = np.zeros((3, 13))
    w[0] = 0.8
    epoch_count = 15
    for epoch in range(epoch_count):
        random.shuffle(list(zip(train_x, train_y)))
        c = zip(train_x, train_y)
        for x, y in c:
            y_hat = np.argmax(np.dot(w, x))
            if round(y) != y_hat:
                # loss
                l = max(0, 1 - np.dot(w[int(y), :], x) + np.dot(w[int(y_hat), :], x))
                # tau
                T = l / (2 * np.dot(x, x))
                w[int(y), :] = w[int(y), :] + T * x
                w[y_hat, :] = w[y_hat, :] - T * x
    res = np.zeros(len(test_x), dtype=int)
    for i in range(len(test_x)):
        res[i] = int(np.argmax(np.dot(w, test_x[i])))
    return res, w


if __name__ == '__main__':
    resPassiveAgressive, w = passiveAgressive(test_x, train_x, train_y)
    resKnn = KNN(5, train_x, train_y, test_x)
    resPerceptron, w = perceptron(test_x, train_x, train_y)
    counter = 0
    for i in range(len(resKnn)):
        if (resKnn[i] == resPerceptron[i] and resPerceptron[i] == resPassiveAgressive[i]):
            counter += 1
        print(f"knn: {resKnn[i]}, perceptron: {resPerceptron[i]}, pa: {resPassiveAgressive[i]}")

