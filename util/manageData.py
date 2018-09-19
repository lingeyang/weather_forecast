# -*- coding: utf-8 -*-

import sys
import os
from netCDF4 import Dataset
import warnings
import numpy as np

warnings.filterwarnings('ignore')

def getNCdata(filePath):
    '''
    filePath : the data's path, '*.nc'
    '''
    fileisExists = os.path.exists(filePath)
    if not fileisExists:
        print('file not found')
        sys.exit(0)
        
    data = Dataset(filePath)
    return data

def transformData(filePath,dumpDir,infoConfPath):
    '''
    filePath : the data's path, '*.nc'
    dumpPath : the processed data's path, '{stationID}.npy'
    '''
    fileisExists = os.path.exists(filePath)
    infoisExists = os.path.exists(infoConfPath)
    if not fileisExists:
        print('file not found')
        sys.exit(0)
    elif not infoisExists:
        print('configure not found')
        sys.exit(0)
    dirisExists = os.path.exists(dumpDir)
    if not dirisExists:
        os.makedirs(dumpDir)
    
    # weather info 
    info = np.load(infoConfPath)
    data = Dataset(filePath)
    variables = data.variables
    (dayDim, timeDim, stationDim) = variables[info[0]].shape
    
    featureTuple = []
    for cmd in info:
        dataArray = np.array(variables[cmd])
        feature = dataArray.reshape((dayDim*timeDim, 10)).T
        featureTuple.append(feature)
    res = np.stack(featureTuple,axis=1)
    
    for station in range(stationDim):
        sdata = res[station].T
        print(sdata[1,:])
        sdata = sdata.reshape((dayDim, timeDim, len(info)))
        print(sdata[0][1,:])
        dumpPath = '%s/station_%d.npy' % (dumpDir, 90001+station)
        np.save(dumpPath,sdata)
        
if __name__ == "__main__":
    #pass
    filePath = '../data/ai_challenger_wf2018_trainingset_20150301-20180531.nc'
    dumpDir = '../transform_data/trainingset'
    
    #filePath = '../data/ai_challenger_wf2018_validation_20180601-20180828_20180905.nc'
    #dumpDir = '../transform_data/validation'

    infoConf = 'infoConf.npy'
    # run this function
    transformData(filePath, dumpDir, infoConf)
    

