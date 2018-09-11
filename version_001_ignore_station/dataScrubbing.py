# -*- coding: utf-8 -*-

import numpy as np
import json
import os
from sklearn.preprocessing import Imputer

infoIndexPath = '../util/infoIndex.json'

def getTrainFeature(trainSetDir, nTimes):
    imp = Imputer(missing_values=-9999.)
    foretimes = 37
    with open(infoIndexPath, 'r') as f:
        infoIndex = json.load(f)
    
    files = os.listdir(trainSetDir)
    X = np.array([])
    y = np.array([])
    for file in files:
        trainSetPath = trainSetDir + os.sep + file
        data = np.load(trainSetPath)
        dataLen = len(data)
        for index in range(1,dataLen):
            #x = data[index,:foretimes-nTimes][:,29:]
            x = data[index,foretimes-nTimes:][:,[infoIndex['t2m_obs'], infoIndex['rh2m_obs'], infoIndex['w10m_obs']]]
            #x2 = data[index][foretimes-nTimes:,:29]
            feature = x.flatten()[np.newaxis,:]
            #x2 = x2.flatten()
            #feature = np.concatenate((x1,x2))[np.newaxis,:]
            label = data[index,foretimes-nTimes:] \
                        [:,[infoIndex['t2m_obs'], infoIndex['rh2m_obs'], infoIndex['w10m_obs']]]

            label = label.flatten()[np.newaxis,:]
            
            if -9999. in label or np.any(np.isnan(feature)):
                continue
            X = np.concatenate((X, feature)) if len(X) != 0 else feature
            y = np.concatenate((y, label)) if len(y) != 0 else label
    X = imp.fit_transform(X)
    np.save('X.npy',X)
    np.save('y.npy',y)
    return X, y


if __name__ == '__main__':
    trainSetPath = '../transform_data/trainingset'
    getTrainFeature(trainSetPath,24)



