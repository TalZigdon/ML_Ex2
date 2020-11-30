from sklearn.model_selection import KFold
import numpy as np
from ex_2 import KNN
from ex_2 import perceptron
from ex_2 import perceptronTestResult
from ex_2 import passiveAgressive
from ex_2 import scoreOfKNN
import random

NUMOFSPLITS = 10
NUMOFROUNDS = 11


def checkPerceptron(array_x, array_y):
    avgOfAvgPA = 0.0
    avgOfAvgPer = 0.0
    avgOfAvgKNN = 0.0
    bias = 0.8
    for i in range(NUMOFROUNDS):
        cv = KFold(n_splits=NUMOFSPLITS, random_state=42, shuffle=True)
        avgPer = 0.0
        avgPa = 0.0
        avgKNN = 0.0
        for train_index, test_index in cv.split(array_x, array_y):
            trainX = array_x[train_index]
            trainY = array_y[train_index]
            validationX = array_x[test_index]
            validationY = array_y[test_index]
            res, w = perceptron(validationX,trainX, trainY)
            avgPer = avgPer + scoreOfKNN(res, validationY)
            #print("perceptron:" + str(scoreOfKNN(res, validationY)))
            res, w = passiveAgressive(validationX,trainX, trainY)
            avgPa = avgPa + scoreOfKNN(res, validationY)
            #print("PA:" + str(scoreOfKNN(res, validationY)))
            avgKNN += scoreOfKNN(validationY, KNN(11, trainX,trainY, validationX))  # 3 = 8588 #966
            #print("score of knn: " + str(scoreOfKNN(validationY, KNN(11,trainX,trainY, validationX))))
        bias += 0.2
        print("AvgPerceptron:" + str(avgPer / NUMOFSPLITS))
        print(str(avgPa / NUMOFSPLITS))
        print("avgKNN:" + str(avgKNN / NUMOFSPLITS))
        avgOfAvgPA += (avgPa/NUMOFSPLITS)
        avgOfAvgPer += (avgPer / NUMOFSPLITS)
        avgOfAvgKNN += (avgKNN / NUMOFSPLITS)
    print("PA")
    print(avgOfAvgPA / NUMOFROUNDS)
    print("KNN")
    print(avgOfAvgKNN / NUMOFROUNDS)
    print("PER")
    print(avgOfAvgPer / NUMOFROUNDS)


