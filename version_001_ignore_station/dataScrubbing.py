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
dataScrub()



