# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import preprocessing

def visualization(filePath, weatherInfo, infoIndexPath, day):
    with open(infoIndexPath, 'r') as f:
        infoIndex = json.load(f)

    t2m_obs, rh2m_obs, w10m_obs = 't2m_obs', 'rh2m_obs', 'w10m_obs'
    
    data = np.load(filePath)
    (dayDim, timeDim, featureDim) = data.shape
    data = preprocessing.fillWithDefaultValue(data[day-1])
    if data.shape[1] != featureDim:
        print('too many default values')
        return
    dataLenth = len(data)
    
    fig = plt.figure(figsize=(10,5))
    fig.suptitle(str(day)+' day')
    
    plt.subplot(121)
    plt.title(weatherInfo+':plot')
    plt.plot(range(dataLenth),data[:,infoIndex[weatherInfo]])
    
    plt.subplot(122)
    plt.title('variety')
    ax_00_t2m_obs = data[:,infoIndex[t2m_obs]]
    ax_00_rh2m_obs = data[:,infoIndex[rh2m_obs]]
    ax_00_w10m_obs = data[:,infoIndex[w10m_obs]]
    pt, =plt.plot(range(dataLenth),ax_00_t2m_obs)
    pr, =plt.plot(range(dataLenth),ax_00_rh2m_obs)
    pw, =plt.plot(range(dataLenth),ax_00_w10m_obs)
    plt.legend([pt, pr, pw], [t2m_obs, rh2m_obs, w10m_obs], loc='upper right')

    plt.show()

    
if __name__ == '__main__':
    trainSetPath = '../transform_data/validation/station_90008.npy'
    infoIndexPath = 'infoIndex.json'
    weatherInfo = 'rh2m_obs'
    day = 4
    visualization(trainSetPath, weatherInfo, infoIndexPath, day)
    


