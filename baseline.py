# -*- coding: utf-8 -*-

#use the last day's info to predict
from sklearn.preprocessing import Imputer
from sklearn import metrics
import numpy as np
import os
import json

infoIndexPath = './util/infoIndex.json'

def metricsGetCost(dirPath, nTimes):
    with open(infoIndexPath,'r') as f:
        infoIndex = json.load(f)
    
    cost = 0.0
    imp = Imputer(missing_values=-9999., strategy='mean', axis=0)
    
    files = os.listdir(dirPath)
    for file in files:
        filePath = dirPath + os.sep + file
        data = np.load(filePath)[-1][37-nTimes:]
        data = imp.fit_transform(data)
        y = data[:,[infoIndex['t2m_obs'], infoIndex['rh2m_obs'], infoIndex['w10m_obs']]]
        y_pred = data[:,[infoIndex['t2m_M'], infoIndex['rh2m_M'], infoIndex['w10m_M']]]
        tmp_cost = metrics.mean_squared_error(y, y_pred) * nTimes
        #bcost = metrics.mean_squared_error(y.flatten(), y_pred.flatten())
        cost += tmp_cost
        #print(y, y_pred)
        #print('baseline cost : %.4f' % (bcost))
    cost = (cost/(nTimes*len(files))) ** 0.5
    print('baseline cost : %.4f' % (cost))

if __name__ == '__main__':
    dirPath = './transform_data/validation'
    metricsGetCost(dirPath, 24)

