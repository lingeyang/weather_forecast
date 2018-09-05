# -*- coding: utf-8 -*-

import numpy as np

obs_index = 29 # this index is 'psur_obs'

def getFeature(trainSetPath,lastNDays):
    data = np.load(trainSetPath)
    dataLen = len(data)
    for index in range(lastNDays,dataLen):
        train = data[index-lastNDays:index]
        
    

if __name__ == '__main__':
    trainSetPath = '../transform_data/trainingset/station_90001.npy'
    lastNDays = 1
    getFeature(trainSetPath,lastNDays)






