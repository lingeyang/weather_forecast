# -*- coding: utf-8 -*-

import numpy as np

obs_index = 29 # this index is 'psur_obs'

def getFeature(trainSetPath):
    data = np.load(trainSetPath)
    dataLen = len(data)
    print(data[1][0])
    #for index in rang e(lastNDays,dataLen):
     #   train = data[index-lastNDays:index]

if __name__ == '__main__':
    trainSetPath = '../transform_data/validation/station_90001.npy'
    getFeature(trainSetPath)






