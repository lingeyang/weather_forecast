# -*- coding: utf-8 -*-

import csv
from netCDF4 import Dataset
import warnings

warnings.filterwarnings('ignore')

columns = ['date','foretimes', 'station', 'psfc_M', \
       't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', \
       'u10m_M', 'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', \
       'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M', \
       'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M', \
       'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', \
       'Q975_M', 'Q925_M', 'Q850_M', 'Q700_M', 'Q500_M', \
       'psur_obs', 't2m_obs', 'q2m_obs', 'w10m_obs', 'd10m_obs', \
       'rh2m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']

def readFile(filePath, dateName, foretimeName, stationName):
    res = []
    data = Dataset(filePath)
    #dimensions = data.dimensions
    variables = data.variables
    date = variables[dateName]
    foretimes = variables[foretimeName]
    stations = variables[stationName]
    
    with open("./transform_data/trainingset.csv","w") as f:
        writer = csv.writer(f)
        writer.writerows(a)
        for dateIndex in range(len(date)):
            for foretimeIndex in range(len(foretimes)):
                for stationIndex in range(len(stations)):
                    day = float(date[dateIndex])
                    foretime = float(foretimes[foretimeIndex])
                    station = float(stations[stationIndex])
                    tmp = []
                    for fea in columns[3:]:
                        feature = variables[fea][:]
                        f = feature[dateIndex][foretimeIndex][stationIndex]
                        if str(f) != '--':
                            f = float(f)
                        else:
                            f = str(f)
                        tmp.append(f)
                    res.append([day, foretime, station] + tmp)
        
    
    
if __name__ == "__main__":
    dateName = 'date'
    foretimeName = 'foretimes'
    stationName = 'station'
    filePath = 'data/ai_challenger_wf2018_trainingset_20150301-20180531.nc'
    readFile(filePath, dateName, foretimeName, stationName)






