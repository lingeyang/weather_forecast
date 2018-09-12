# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import json
import random

def visualization(filePath, weatherInfo, infoIndexPath):
    with open(infoIndexPath, 'r') as f:
        infoIndex = json.load(f)
    imp = Imputer(missing_values=-9999.0)
    t2m_obs, rh2m_obs, w10m_obs = 't2m_obs', 'rh2m_obs', 'w10m_obs'
    
    data = np.load(filePath)
    (dayDim, timeDim, featureDim) = data.shape
    day = random.randint(0,dayDim)
    
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(weatherInfo)
    
    plt.subplot(221)
    plt.title('variety')
    ax_00_data = data[day]
    ax_00_lenth = len(ax_00_data)
    ax_00_t2m_obs = ax_00_data[:,infoIndex[t2m_obs]]
    ax_00_rh2m_obs = ax_00_data[:,infoIndex[rh2m_obs]]
    ax_00_w10m_obs = ax_00_data[:,infoIndex[w10m_obs]]
    plt.plot(range(ax_00_lenth),ax_00_t2m_obs)
    plt.plot(range(ax_00_lenth),ax_00_rh2m_obs)
    plt.plot(range(ax_00_lenth),ax_00_w10m_obs)
    #plt.plot(range(ax_00_lenth),ax_00_data[:,infoIndex[weatherInfo]])

    plt.show()
    '''
    data = data.reshape((dayDim*timeDim, featureDim))
    info = data[:, infoIndex[weatherInfo]][np.newaxis,:]
    info = imp.fit_transform(info)
    
    
    plt.hist(info[0],50,density=1,histtype='stepfilled',alpha=0.75)
    '''

if __name__ == '__main__':
    trainSetPath = '../transform_data/trainingset/station_90009.npy'
    infoIndexPath = 'infoIndex.json'
    weatherInfo = 'psur_obs'
    visualization(trainSetPath, weatherInfo, infoIndexPath)
    


