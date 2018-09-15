# -*- coding: utf-8 -*-

from util import manageData as md
from version_001_ignore_station import dataScrubbing as ds

def run():
    #filePath = './data/ai_challenger_wf2018_trainingset_20150301-20180531.nc'
    #dumpDir = './transform_data/trainingset'
    #filePath = './data/ai_challenger_wf2018_validation_20180601-20180828_20180905.nc'
    #dumpDir = './transform_data/validation'
    #infoConf = './util/infoConf.npy'
    #md.transformData(filePath, dumpDir, infoConf)
    
    infoIndexPath = './util/infoIndex.json'
    trainSetPath = './transform_data/trainingset'
    ds.__getTrainFeature(trainSetPath,infoIndexPath)

if __name__ == '__main__':
    run()














