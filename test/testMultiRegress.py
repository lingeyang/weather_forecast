# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics

def multiXGBoost():
    X, y = make_regression(n_samples=1000, n_features=20, \
                           n_targets=20,random_state=1)
    
    x_train, y_train = X[:800], y[:800]
    x_test, y_test = X[200:], y[200:]
    
    xlf = XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=36, 
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
    
    clf = MultiOutputRegressor(xlf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    est = clf.estimators_[0]
    cost = metrics.mean_squared_error(y_pred, y_test)
    print(cost)
    
if __name__ == '__main__':
    multiXGBoost()


