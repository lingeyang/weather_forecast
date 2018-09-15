# -*- coding: utf-8 -*-

import numpy as np
import json
import os
#from sklearn.preprocessing import Imputer
#from util.preprocessing import fillWithDefaultValue

'''
this function is duplicate
'''
def __getTrainFeature(dataDir, infoIndexPath, nTimes=24): 
    #imp = Imputer(missing_values=-9999.)
    foretimes = 37
    with open(infoIndexPath, 'r') as f:
        infoIndex = json.load(f)
    
    files = os.listdir(dataDir)
    X = np.array([])
    y = np.array([])
    for file in files:
        trainSetPath = dataDir + os.sep + file
        allDaysData = np.load(trainSetPath)
        where_are_nan = np.isnan(allDaysData)
        allDaysData[where_are_nan] = -9999.
        dataLen = len(allDaysData)
        
        for index in range(1,dataLen):
            data = allDaysData[index]
            label = data[foretimes-nTimes:] \
                        [:,[infoIndex['t2m_obs'], infoIndex['rh2m_obs'], infoIndex['w10m_obs']]]
            label = label.T.flatten()
            if -9999. in label:
                continue
            label = label[np.newaxis,:]
            '''
            x1 = data[:foretimes-nTimes]
            x1 = x1.flatten()
            x2 = data[foretimes-nTimes:][:,:infoIndex['psur_obs']]
            x2 = x2.flatten()
            x = np.concatenate((x1, x2))
            '''
            x = data[foretimes-nTimes:] \
            [:,infoIndex['rh2m_M']]
            x = x.flatten()
            feature = x[np.newaxis,:]
            
            X = np.concatenate((X, feature)) if len(X) != 0 else feature
            y = np.concatenate((y, label)) if len(y) != 0 else label
        
    #print(np.any(np.isnan(X)))
    #print(X.shape)
    #print(y.shape)
    #np.save('X.npy',X)
    #np.save('y.npy',y)
    return X, y

def dataScrub():
   
    TrainingPath=r"../transform_data/trainingset/station_900"
    WindowSize=0
    f = open(r"../util/infoIndex.json")
    foreTimes=37
    repeatTimes=13
    infoIndex = json.load(f)
    datax = []
    datay = []
    for name in range(1,11):
        filePath=""
        if name<10:
            filePath=TrainingPath+"0"+str(name)+".npy"
        else:
            filePath=TrainingPath+str(name)+".npy"
        context = np.load(filePath)
        for loop in range(1,len(context)-WindowSize):
            l=[]
            m=[]
            first=True
            if WindowSize==0:
                for i in range(repeatTimes):
                    for j in context[loop][i]:
                        l.append(j)
                datax.append(l)
                for i in range(repeatTimes,foreTimes):
                    m.append(context[loop ][i][infoIndex['t2m_obs']])
                    m.append(context[loop ][i][infoIndex['rh2m_obs']])
                    m.append(context[loop ][i][infoIndex['w10m_obs']])
                datay.append(m)
            else:
                for day in range(loop,loop+WindowSize):
                    if first:
                        for i in range(foreTimes):
                            for j in context[day][i]:
                                l.append(j)
                        first=False
                    else:
                        for i in range(repeatTimes,foreTimes):
                            for j in context[day][i]:
                                l.append(j)
                datax.append(l)
                if (loop+WindowSize)<len(context):
                    for i in  range(repeatTimes,foreTimes):
                        m.append(context[loop+WindowSize][i][infoIndex['t2m_obs']])
                        m.append(context[loop + WindowSize][i][infoIndex['rh2m_obs']])
                        m.append(context[loop + WindowSize][i][infoIndex['w10m_obs']])
                    datay.append(m)
    np.save("X.npy",datax)
    np.save("y.npy",datay)

if __name__ == '__main__':
    trainSetPath = '../transform_data/trainingset'
    infoIndexPath = '../util/infoIndex.json'
    __getTrainFeature(trainSetPath,infoIndexPath)
    #dataScrub()


