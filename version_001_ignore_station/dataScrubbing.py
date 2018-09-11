# -*- coding: utf-8 -*-

import numpy as np
import  json

TrainingPath=r"../transform_data/trainingset/station_900"
WindowSize=0
f = open(r"../util/infoIndex.json")
foreTimes=37
repeatTimes=13
infoIndex = json.load(f)

def dataScrub():
    for i in range(1,11):
        filePath=""
        if i<10:
            filePath=TrainingPath+"0"+str(i)+".npy"
        else:
            filePath=TrainingPath+str(i)+".npy"
        context = np.load(filePath)
        datax = []
        datay=[]
        for loop in range(1,len(context)-WindowSize):
            l=[]
            m=[]
            first=True
            if WindowSize==0:
                for i in range(repeatTimes):
                    for j in context[loop][i]:
                        l.append(j)
                datax.append(l)
                if  loop<len(context):
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
    np.save("datax",datax)
    np.save("datay",datay)

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
    #trainSetPath = '../transform_data/trainingset'
    #getTrainFeature(trainSetPath,24)
    #dataScrub()
    x = np.load('datax.npy')
    y = np.load('datay.npy')
    print(x.shape)
    print(y.shape)



