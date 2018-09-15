# -*- coding: utf-8 -*-

import numpy as np
import json
import os
from sklearn.preprocessing import Imputer

def fillWithDefaultValue(data, axis=0):
    where_are_nan = np.isnan(data)
    data[where_are_nan] = -9999.
    imp = Imputer(missing_values=-9999., axis=axis)
    res = imp.fit_transform(data)
    return res

if __name__ == '__main__':
    pass






