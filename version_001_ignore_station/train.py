# -*- coding: utf-8 -*-

import dataScrubbing as ds
#from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from xgboost import XGBModel
from sklearn import metrics
from sklearn.preprocessing import Imputer
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn import linear_model

def modelXGBoost(max_depth=100, learning_rate=0.3, n_estimators=50):
    infoBoundPath = '../util/infoBound.json'
    with open(infoBoundPath,'r') as f:
        infoBound = json.load(f)
    index = 30
    lr = linear_model.LinearRegression()
    infoIndexPath = '../util/infoIndex.json'
    validationPath = '../transform_data/validation'
    trainSetPath = '../transform_data/trainingset'
    x_train, y_train = ds.__getTrainFeature(trainSetPath, infoIndexPath)
    y_train = y_train[:,index]
    x_test, y_test = ds.__getTrainFeature(validationPath, infoIndexPath)
    y_test = y_test[:,index]
    #x_test = x_test[-1][np.newaxis,:]
    #y_test = [y_test[-1]]
    #print(x_test.shape,y_test.shape)
    '''
    #max_depth = int(infoBound['t2m_obs'][1] ** 0.5)
    xlf = XGBRegressor(max_depth=max_depth, 
                        learning_rate=learning_rate, 
                        n_estimators=n_estimators, 
                        objective='reg:linear', 
                        subsample=1, 
                        colsample_bytree=1, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1,
                        missing=-9999.)
    
    xlf.fit(x_train, y_train,
        eval_set=[(x_train, y_train),(x_test, y_test)],
        eval_metric='rmse',
        verbose=True)
    y_pred = xlf.predict(x_test)
    '''
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    pt, = plt.plot(range(len(y_test)),y_test)
    pp, = plt.plot(range(len(y_pred)),y_pred)
    plt.legend([pt,pp],['y_test','y_pred'],loc='upper right')
    plt.show()
    #print(y_pred[:10])
    #print(y_test[:10])
    #return xlf
    

if __name__ == '__main__':
    modelXGBoost()

