# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor as mor
from sklearn import metrics

def multiXGBoost():
    X, y = make_regression(n_samples=5000, n_features=100, \
                           n_targets=33,random_state=1)
    x_train, y_train = X[:4500], y[:4500]
    x_test, y_test = X[4500:], y[4500:]
    
    xlf = XGBRegressor(max_depth=3, 
                        learning_rate=0.01, 
                        n_estimators=2, 
                        silent=True, 
                        objective='reg:linear', 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.8, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)
    
    clf = mor(xlf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    print(y_pred[0])
    cost = metrics.mean_squared_error(y_pred, y_test)
    print(cost)
    
if __name__ == '__main__':
    multiXGBoost()


