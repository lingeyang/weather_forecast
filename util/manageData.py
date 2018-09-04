# -*- coding: utf-8 -*-

from netCDF4 import Dataset
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# weather info 
cmds = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', \
       'u10m_M', 'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', \
       'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M', \
       'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M', \
       'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', \
       'Q975_M', 'Q925_M', 'Q850_M', 'Q700_M', 'Q500_M', \
       'psur_obs', 't2m_obs', 'q2m_obs', 'w10m_obs', 'd10m_obs', \
       'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']


def transformData(filePath,dumpDir):
    '''
    filePath : the data's path, '*.nc'
    dumpPath : the processed data's path, '{stationID}.npy'
    '''
    data = Dataset(filePath)
    variables = data.variables
    featureTuple = []
    for cmd in cmds:
        dataArray = np.array(variables[cmd])
        (dayDim, timeDim, stationDim) = dataArray.shape
        feature = dataArray.reshape((dayDim*timeDim, 10)).T
        featureTuple.append(feature)

    res = np.stack(featureTuple,axis=1)
    
    for station in range(10):
        sdata = res[station].T
        dumpPath = '%s/station_%d.npy' % (dumpDir, 90001+station)
        
        np.save(dumpPath,sdata)
        
if __name__ == "__main__":
    filePath = '../data/wf2018_trainingset_20150301-20180531.nc'
    dumpDir = '../transform_data'
    # run this function
    #transformData(filePath, dumpDir)





