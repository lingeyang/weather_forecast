# -*- coding: utf-8 -*-

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
import datetime
import random

random.seed(datetime.datetime.now())

def loadData(filePath):
    pass

def multiXGBoost(x_train, y_train, \
                 max_depth=3, learning_rate=0.01, n_estimators=10, gamma=0):
    
    xlf = XGBRegressor(max_depth=max_depth, 
                        learning_rate=learning_rate, 
                        n_estimators=n_estimators, 
                        silent=True, 
                        objective='reg:linear', 
                        gamma=gamma,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.8, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        seed=random.randint(0,10000), 
                        missing=-9999.)
    
    clf = MultiOutputRegressor(xlf)
    clf.fit(x_train, y_train)
    return clf

def predict(multiOutputRe, x_test, y_test):
    y_pred = multiOutputRe.predict(x_test)
    
    print(y_pred[0])
    cost = metrics.mean_squared_error(y_pred, y_test)
    cost = cost ** 0.5
    print(cost)

if __name__ == '__main__':
    #multiXGBoost(X, y)
    cost = metrics.mean_squared_error([[-1,-1],[2,2],[3,3]], [[3,3],[4,4],[5,5]])
    print(cost)
