import pickle
import random
from PIL import Image
import numpy as np
import os

# Citation: [5]
def main():
    makePickle()
    callPickle()
    
def callPickle():
    with open('pickled_trainDict', 'rb') as myPickle:
      trainDict = pickle.load(myPickle)

    with open('pickled_testDict', 'rb') as myPickle:
      testDict = pickle.load(myPickle)
    
    for i in range(20):
        print(testDict[i])
    print("TEEHEEHEEEEEEEEPICKLE")
    for j in range(20):
        print(trainDict[j])

def makeMiniPickle():
    trainDict = {}
    testDict = {}
    i = 0
    for file in os.listdir('./LULC-pngs/train/imageSubset'):
        name = file[:-8]
        trainDict[i] = name
        i += 1
    i = 0
    for file in os.listdir('./LULC-pngs/test/imageSubset'):
        name = file[:-8]
        testDict[i] = name
        i += 1
    
    with open('pickled_miniTrainDict', 'wb') as myPickle:
        pickle.dump(trainDict, myPickle)
    
    with open('pickled_miniTestDict', 'wb') as myPickle:
        pickle.dump(testDict, myPickle)

def makePickle():
    trainDict={}
    testDict={}
    x = 0
    for file in os.listdir('./LULC-pngs/train/imageTiles/'):
        trainDict[x] = str(file[:-7])
        x+=1
    y = 0
    for file in os.listdir('./LULC-pngs/test/imageTiles/'):
        testDict[y] = str(file[:-7])
        y+=1

    with open('pickled_trainDict', 'wb') as myPickle:
        pickle.dump(trainDict, myPickle)

    with open('pickled_testDict', 'wb') as myPickle:
        pickle.dump(testDict, myPickle)

main()