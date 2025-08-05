from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from math import log
import operator
import pickle
def createDataSet():
    dataSet = [
        [0,0,0,1,'no'],
        [0,0,0,1,'no'],
        [0,0,0,1,'no'],
        [1,0,0,1,'no'],
        [0,0,1,1,'no'],
        [1,1,0,0,'no'],
        [0,1,0,0,'no'],
        [0,0,1,0,'no'],
        [0,1,1,0,'no'],
        [1,0,1,0,'no'],
        [2, 0, 2, 1, 'yes'],
        [2, 0, 2, 1, 'yes'],
        [2, 2, 2, 1, 'yes'],
        [2, 2, 0, 1, 'yes'],
        [0, 2, 2, 1, 'yes'],
        [1, 2, 2, 0, 'yes'],
        [0, 2, 2, 1, 'yes'],
        [0, 0, 2, 1, 'yes'],
        [0, 1, 2, 1, 'yes'],
        [1, 0, 2, 1, 'yes']
    ]
    labels = ['AGE','WORK','HOME','LOAN']
    return dataSet, labels

def createTree(dataset,labels,featurelabels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classlist)
    bestFeature = chooseBestFeatureToSplit(dataset)
    bestFeatureLabel = labels[bestFeature]
    featurelabels.append(bestFeatureLabel)
    myTree = {bestFeature: {}}
    del labels[bestFeature]
    feavalue =[example[bestFeature] for example in dataset]
    uniquevals = set(feavalue)
    for value in uniquevals:
        sublabels = labels[:]
        myTree[bestFeature][value] = createTree(splitDataSet(dataset,bestFeature,value),sublabels,featurelabels)
    return myTree

def majorityCnt(classlist):
    classCount = {}
    for vote in classlist:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return classCount[0][0]

def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0])-1
    baseEntropy = calShang(dataset)
    bestFeature = -1
    bestInfogain = 0
    for i in range(numFeatures):
        featlist = [example[i] for example in dataset]
        uniquevals = set(featlist)
        newEntropy = 0
        for value in uniquevals:
            subDataSet = splitDataSet(dataset,i,value)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob * calShang(subDataSet)
        infogain = baseEntropy - newEntropy
        if (infogain > bestInfogain):
            bestInfogain = infogain
            bestFeature = i
    return bestFeature

def splitDataSet(dataset,axis,val):
    retDataSet = []
    for featvec in dataset:
        if featvec[axis] == val:
            reducedFeatvec = featvec[:axis]
            reducedFeatvec.extend(featvec[axis+1:])
            retDataSet.append(reducedFeatvec)
    return retDataSet


def calShang(dataset):
    numexamples = len(dataset)
    labelCount = {}
    for featvec in dataset:
        currentlabels = featvec[-1]
        if currentlabels not in labelCount.keys():
            labelCount[currentlabels] = 0
        labelCount[currentlabels] += 1
    shangEnt = 0
    for key in labelCount:
        prob = float(labelCount[key]) / numexamples
        shangEnt -= prob * log(prob,2)
    return shangEnt

if __name__ == "__main__":
    dataSet, labels = createDataSet()
    featurelabels = []
    myTree = createTree(dataSet,labels,featurelabels)