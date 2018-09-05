# -*- coding: utf-8 -*-

import numpy as np

def getFeature(trainSetPath):
    data = np.load(trainSetPath)
    
    print(data.shape)
    

if __name__ == '__main__':
    trainSetPath = '../transform_data/trainingset/station_90001.npy'
    getFeature(trainSetPath)







