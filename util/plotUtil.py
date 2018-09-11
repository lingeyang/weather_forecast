# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import Imputer

infoIndexPath = 'infoIndex.json'

def plotWeatherInfo(filePath, weatherInfo):
    with open(infoIndexPath, 'r') as f:
        infoIndex = json.load(f)
    imp = Imputer(missing_values=-9999.)
    
    data = np.load(filePath)
    dayDim, timeDim, infoDim = data.shape
    info = data.reshape((dayDim*timeDim, infoDim))[:,infoIndex[weatherInfo]]
    info = info[np.newaxis,:]
    info = imp.fit_transform(info)
    plt.hist(info[0], 50, density=1)

if __name__ == '__main__':
    trainSetPath = '../transform_data/trainingset/station_90004.npy'
    weatherInfo = 'rh2m_obs'
    plotWeatherInfo(trainSetPath, weatherInfo)
