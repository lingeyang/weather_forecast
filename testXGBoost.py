# -*- coding: utf-8 -*-

import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('')
dtest = xgb.DMatrix('')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)