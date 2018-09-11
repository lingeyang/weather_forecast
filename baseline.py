# -*- coding: utf-8 -*-

#use the last day's info to predict
from sklearn.preprocessing import Imputer
import numpy as np
import os
import json

infoIndexPath = './util/infoIndex.json'

def getScore(dirPath, nTimes):
    cost = 0.0
    t2m_score = 0.0
    rh2m_score = 0.0
    w10m_score = 0.0
    
    with open(infoIndexPath,'r') as f:
        infoIndex = json.load(f)
    files = os.listdir(dirPath)
    stations = len(files)
    
    imp = Imputer(missing_values=-9999., strategy='mean', axis=0)
    
    for file in files:
        filePath = dirPath + os.sep + file
        data = np.load(filePath)[-1]
        data = imp.fit_transform(data)
        #print(data[data[:,:] == -9999.0])
        t2m = data[:,infoIndex['t2m_obs']][37-nTimes:] - data[:,infoIndex['t2m_M']][37-nTimes:]
        rh2m = data[:,infoIndex['rh2m_obs']][37-nTimes:] - data[:,infoIndex['rh2m_M']][37-nTimes:]
        w10m = data[:,infoIndex['w10m_obs']][37-nTimes:] - data[:,infoIndex['w10m_M']][37-nTimes:]
 
        #print(data[:,infoIndex['rh2m_obs']][37-nTimes:])
        #print(data[:,infoIndex['rh2m_M']][37-nTimes:])
        #print(rh2m)
        t2m_score += sum(t2m**2)
        rh2m_score += sum(rh2m**2)
        w10m_score += sum(w10m**2)
        #print(t2m_score, rh2m_score, w10m_score)
        
    cost = (t2m_score/(nTimes*stations)) ** 0.5 + \
            (rh2m_score/(nTimes*stations)) ** 0.5 + \
            (w10m_score/(nTimes*stations)) ** 0.5
    cost = cost / 3
    #print(score)
    return cost
    
if __name__ == '__main__':
    dirPath = './transform_data/validation'
    getScore(dirPath, 24)




