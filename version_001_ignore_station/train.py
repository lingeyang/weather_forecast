# -*- coding: utf-8 -*-

import dataScrubbing as ds
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from xgboost import XGBModel
from sklearn import metrics
from sklearn.preprocessing import Imputer
import numpy as np
import json


def multiXGBoost(max_depth=4, learning_rate=0.1, n_estimators=100):
    X_train = np.load('X.npy')
    y_train = np.load('y.npy')
    y_train = y_train[:,5]
    
    xlf = XGBRegressor(max_depth=max_depth, 
                        learning_rate=learning_rate, 
                        n_estimators=n_estimators, 
                        objective='reg:linear', 
                        n_jobs=64,
                        subsample=0.85, 
                        colsample_bytree=0.85, 
                        colsample_bylevel=1.0, 
                        reg_alpha=0, 
                        reg_lambda=100, 
                        scale_pos_weight=1, 
                        random_state=1000)
    
    #clf = MultiOutputRegressor(xlf)
    #clf.fit(X, y)
    xlf.fit(X_train, y_train)
    return xlf
    
def train(X_test, y_test):
    
    X_train = np.load('X.npy')
    y_train = np.load('y.npy')
    y_train = y_train[:,24]
    param_dist = {'objective':'reg:linear', \
                  'max_depth':4, \
                  'eta':0.1, \
                  'colsample_bytree':0.8, \
                  'n_estimators':1, \
                  'lambda':100
                  }
    clf = XGBModel(**param_dist)
    clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        eval_metric='rmse',
        verbose=True)
    return clf

if __name__ == '__main__':
    infoIndexPath = '../util/infoIndex.json'
    validationPath = '../transform_data/validation'
    x_test, y_test = ds.__getTrainFeature(validationPath,infoIndexPath)

    '''
    with open(infoIndexPath, 'r') as f:
        infoIndex = json.load(f)
    imp = Imputer(missing_values=-9999.)
    validationPath = '../transform_data/validation/station_90001.npy'
    
    vdata = np.load(validationPath)[-1]
    vdata = imp.fit_transform(vdata)
    X = vdata[13:][:,:29] \
    [:,[infoIndex['t2m_M'], infoIndex['rh2m_M'], infoIndex['w10m_M']]].flatten()[np.newaxis,:]
    
    #X = np.concatenate((x1,x2))[np.newaxis,:]
    y = vdata[13:, \
               [infoIndex['t2m_obs'], infoIndex['rh2m_obs'], infoIndex['w10m_obs']]]
    y_train = np.load('y.npy')
    y = vdata[13,infoIndex['t2m_obs']]
    y = y.flatten()
    #print(X.shape, y.shape)
    '''
    x_test = x_test[-1][np.newaxis,:]
    y_test = [y_test[:,23][-1]]
    clf = train(x_test, y_test)
    #print(clf.evals_result())
    
    #y_pred = clf.predict(x_test)
    #print(y_test)
    #print(y_pred)
    #cost = metrics.mean_squared_error(y, y_pred)
    #print('model cost : %.4f' % cost)
    

