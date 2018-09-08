# -*- coding: utf-8 -*-

import numpy as np

infoBoundPath = './util/infoBound.json'
infoIndexPath = './util/infoIndex.json'
infoConf = './util/infoConf.npy'

def getTrainFeature(trainSetPath):
    data = np.load(trainSetPath)
    dataLen = len(data)
    print(data.shape)
    #for index in rang e(lastNDays,dataLen):
     #   train = data[index-lastNDays:index]
     
def getTestFeature(testPath):
    pass


if __name__ == '__main__':
    trainSetPath = '../transform_data/trainingset/station_90001.npy'
    getTrainFeature(trainSetPath)






