# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import ShuffleSplit
import xgboost as xgb
import pandas as pd

data = pd.read_csv('ex1data1.txt')
data = np.array(data)
X = data[:,0].reshape(len(data),1)
y = data[:,1]

rs=ShuffleSplit(n_splits=1,test_size=.25,random_state=0)
rs.get_n_splits(X)

X_train,X_test,y_train,y_test = None,None,None,None

for train_index,test_index in rs.split(X,y):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]

xlf = xgb.XGBRegressor(max_depth=3,
                        learning_rate=0.1,
                        n_estimators=30,
                        silent=True,
                        objective='reg:linear',
                        booster='gbtree',
                        gamma=0,
                        subsample=0.85,
                        colsample_bytree=1,
                        colsample_bylevel=1,
                        reg_lambda=1,
                        missing=None)

xlf.fit(X_train, y_train, eval_metric='rmse', \
        verbose = True, eval_set = [(X_test, y_test)])
preds = xlf.predict(X_test)
print(sum((preds - y_test)**2))
