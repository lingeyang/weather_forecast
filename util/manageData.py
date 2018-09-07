# -*- coding: utf-8 -*-

from netCDF4 import Dataset
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

def transformData(filePath,dumpDir):
    '''
    filePath : the data's path, '*.nc'
    dumpPath : the processed data's path, '{stationID}.npy'
    '''
    fileisExists = os.path.exists(filePath)
    if not fileisExists:
        print('file not found')
        return
    dumpDirisExists=os.path.exists(dumpDir)
    if not dumpDirisExists:
        os.makedirs(dumpDir)
    
    # weather info 
    info = np.load('infoConf.npy')
    data = Dataset(filePath)
    variables = data.variables
    (dayDim, timeDim, stationDim) = variables[info[0]].shape
    
    featureTuple = []
    for cmd in info:
        dataArray = np.array(variables[cmd])
        
        feature = dataArray.reshape((dayDim*timeDim, 10)).T
        featureTuple.append(feature)

    res = np.stack(featureTuple,axis=1)
    
    for station in range(len(stationDim)):
        sdata = res[station].T
        
        sdata = sdata.reshape((dayDim, timeDim, len(info)))
        dumpPath = '%s/station_%d.npy' % (dumpDir, 90001+station)
        np.save(dumpPath,sdata)
        
if __name__ == "__main__":
    filePath = '../data/ai_challenger_wf2018_trainingset_20150301-20180531.nc'
    dumpDir = '../transform_data/trainingset'
    # run this function
    transformData(filePath, dumpDir)





