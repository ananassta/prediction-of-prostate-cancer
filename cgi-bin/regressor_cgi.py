import numpy as np
import re
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
from sklearn.preprocessing import OneHotEncoder
import class_to_int as cl

models_glison_nm = [sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_features ='sqrt')]
models_t = [sklearn.linear_model.LogisticRegression()]

def count_regression(Xtrn, Xtest, Ytrn, Ytest, number_predicition, enc):
    if number_predicition == 0:
        models = models_glison_nm
        lj = 7
    elif number_predicition == 1:
        models = models_t
        lj = 5
    else:
        models = models_glison_nm
        lj = 1
    # Ytrn = enc.inverse_transform(Ytrn)
    # Ytest = enc.inverse_transform(Ytest)
    for model in models:
        m = str(model)
        m = m[:m.index('(')]
        if number_predicition == 1:
            ktrn = enc.inverse_transform(Ytrn)
            model.fit(Xtrn, ktrn)
            prediction = np.ravel(model.predict(Xtest))
        else:
            model.fit(Xtrn, Ytrn)
            prediction = np.ravel(enc.inverse_transform(model.predict(Xtest)))
    l = 0
    j = 0
    if number_predicition == 0:
        prediction = cl.g_to_it(prediction)
    elif number_predicition == 1:
        prediction = cl.t_to_int(prediction)
        # Ytest = cl.t_to_int(Ytest)
    else:
        prediction = cl.nm_to_int(prediction)
        # Ytest = cl.nm_to_int(Ytest)
    # for i in range(len(Ytest)):
    #     if prediction[i] == Ytest[i]:
    #         if prediction[i] == lj:
    #             l = l + 1
    #         else:
    #             j = j + 1
    return prediction, l, j

def t_to_int(t):
    if t == '1':
        res = 1
    if t == '2':
        res = 2
    if t == '2b':
        res = 3
    if t == '2c':
        res = 4
    if t == '3':
        res = 5
    if t == '3a':
        res = 6
    if t == '3b':
        res = 7
    if t == '4':
        res = 8
    return res

def nm_to_int(n):
    res = int(n)
    return res

def g_to_it(g):
    return round(g)