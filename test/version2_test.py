# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import sys
from scipy import stats
from xgboost import XGBRegressor, XGBModel
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import svm
import matplotlib.pyplot as plt

foretimes = 37
sepTime = 29

def getStatistics(data):
    res = []
    data = data[data != -9999.]
    
    maxValue = np.max(data)
    minValue = np.min(data)
    mean = np.mean(data)
    median = np.median(data)
    
    percentile = np.percentile(data, (25, 50, 75), interpolation='midpoint')
    up_percentile = percentile[0] #'上四分位数'
    mid_percentile = percentile[1] #'中四分位数'
    down_percentile =  percentile[2] #'下四分位数'
    diff_percentile = down_percentile - up_percentile #'四分位差'
    
    ptp = maxValue - minValue
    var = np.var(data)
    std = np.std(data)
    skew = stats.skew(data) #'偏度'
    kurtosis = stats.kurtosis(data) #'峰度'
    
    res = [maxValue, minValue , mean, median, up_percentile, \
           mid_percentile, down_percentile, diff_percentile, ptp, var, \
           std, skew, kurtosis]
    
    return res
    
def descriptiveStatistics(data, nDays, nTimes):
    feature = []
    
    dayDim, timeDim, infoDim = data.shape
    if nDays == 0:
        print('not suppport 0 days yet !!')
        sys.exit(0)
    else:
        x1 = data.reshape((dayDim*timeDim,infoDim))[:,:sepTime]
        for ruitu in x1.T:
            f_tmp = getStatistics(ruitu)
            feature.extend(f_tmp)
        
        if dayDim > 2:
            x21 = data[0][:,sepTime:]
            x22 = data[1:-2][foretimes-nTimes:,sepTime:]
            x23 = data[-2][foretimes-nTimes:,sepTime:]
            (i1, i2) = x21.shape
            (j1, j2, j3) = x22.shape
            x22 = x22.reshape((j1*j2,i2))
            x2 = np.concatenate((x21, x22, x23),axis=0)
        else:
            x2 = data[0][:,sepTime:]
        for obs in x2.T:
            f_tmp = getStatistics(obs)
            feature.extend(f_tmp)
    return feature

def preprocess(data):
    where_are_nan = np.isnan(data)
    data[where_are_nan] = -9999.
    index = []
    for i in range(len(data)):
        value_count = np.where(data[i] == -9999.)
        if len(value_count[0]) >= foretimes:
            index.append(i)
    res = np.delete(data,index,axis=0)
    return res
        
def getFeature(dataDir, infoIndexPath, nDays, nTimes=24):
    with open(infoIndexPath, 'r') as f:
        infoIndex = json.load(f)
    X = []
    y = []
    files = os.listdir(dataDir)
    for file in files:
        filePath = dataDir + os.sep + file
        data = np.load(filePath)
        data = preprocess(data)
        dayDim, timeDim, infoDim = data.shape
        for day in range(nDays, dayDim):
            label = data[day][foretimes-nTimes:] \
            [:,[infoIndex['t2m_obs'], infoIndex['rh2m_obs'], infoIndex['w10m_obs']]]
            label = label.T.flatten()[np.newaxis,:]
            featureData = data[day-nDays:day+1]
            feature = descriptiveStatistics(featureData, nDays, nTimes)
            X.append(feature)
            y = np.concatenate((y, label)) if len(y) != 0 else label
            
    X = np.array(X)
    
    np.save('X.npy',X)
    np.save('y.npy',y)
    return X, y

def train(testDataDir, infoIndexPath, nDays, nTimes = 24):
    x_test, y_test = getFeature(testDataDir, infoIndexPath, nDays)
    
    x_train = np.load('X.npy')
    y_train = np.load('y.npy')
    
    index = 30
    y_test = y_test[:,index]
    y_train = y_train[:,index]
    
    # normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(x_train)
    x_test_minmax = min_max_scaler.transform(x_test)
    #print(x_train_minmax.shape)
    '''
    # svr
    svr =  svm.SVR()
    svr.fit(x_train_minmax, y_train)
    y_pred = svr.predict(x_test_minmax)
    
    '''
    # linearRegression
    lr = linear_model.LinearRegression(normalize=True)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    
    #print(y_test[0])
    #print(y_pred[0])
    '''
    # xgboost
    param_dist = {'objective':'reg:linear', \
                  'max_depth':9, \
                  'eta':0.1, \
                  'colsample_bytree':1, \
                  'n_estimators':10, \
                  'lambda':100
                  }
    clf = XGBModel(**param_dist)
    clf.fit(x_train, y_train,
        eval_set=[(x_train, y_train),(x_test, y_test)],
        eval_metric='rmse',
        verbose=True)
    
    index = 0
    y_pred = clf.predict(x_test[index][np.newaxis,:])
    print(y_test[index])
    print(y_pred)
    '''
    cost = mean_squared_error(y_test, y_pred)

    fig = plt.figure(figsize=(10,10))
    ptest, = plt.plot(range(len(y_test)), y_test)
    ppred, = plt.plot(range(len(y_pred)), y_pred)
    plt.legend([ptest, ppred], ['test', 'pred'], loc='upper right')
    plt.show()
    print(cost)

if __name__ == '__main__':
    trainDataDir = '../transform_data/trainingset'
    testDataDir = '../transform_data/validation'
    infoIndexPath = '../util/infoIndex.json'
    nDays = 1
    #getFeature(trainDataDir, infoIndexPath, nDays)
    train(testDataDir, infoIndexPath, nDays)











