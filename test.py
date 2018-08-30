# -*- coding: utf-8 -*-

import os
from netCDF4 import Dataset

date = 'date'
foretimes = 'foretimes'
stations = 'station'

columns = ['station','date', 'foretimes', 'psfc_M', \
       't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', \
       'u10m_M', 'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', \
       'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M', \
       'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M', \
       'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', \
       'Q975_M', 'Q925_M', 'Q850_M', 'Q700_M', 'Q500_M', \
       'psur_obs', 't2m_obs', 'q2m_obs', 'w10m_obs', 'd10m_obs', \
       'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']

def readFile(filePath):
    res = []
    files = os.listdir(filePath)
    file = filePath + os.sep + files[0]
    data = Dataset(file)
    #dimensions = data.dimensions

    variables = data.variables
    print(variables[stations][:])
    for station in variables[stations]:
        pass
    
    
if __name__ == "__main__":
    filePath = 'data'
    readFile(filePath)






