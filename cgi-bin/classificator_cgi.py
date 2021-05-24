from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import class_to_int as cl

models_glison = [LinearDiscriminantAnalysis(),
	             KNeighborsClassifier(),
	             GaussianNB(),
                 RandomForestClassifier()]

models_t = [LinearDiscriminantAnalysis()]

models_n = [GaussianNB(),
            RandomForestClassifier()]

models_m = [GaussianNB()]

def count_classification(X_train, X_test, y_train, y_test, number_predicition, enc):
    if number_predicition == 0:
        models = models_glison
        lj = 7
    if number_predicition == 1:
        models = models_t
        lj = 5
    if number_predicition == 2:
        models = models_n
        lj = 1
    if number_predicition == 3:
        models = models_m
        lj = 1

    # y_test = enc.inverse_transform(y_test)
    y_train = np.ravel(enc.inverse_transform(y_train))
    sum = np.zeros(len(y_test)) #(y_test.size)
    vote = []
    for i in range(len(y_test)): #(y_test.size):
        vote.append([])
        for j in range(10):
            vote[i].append(0)
    sum_vote = []
    leny = (len(y_test))
    for model in models:
        m = str(model)
        m = m[:m.index('(')]
        model.fit(X_train, y_train)
        prediction = np.ravel(model.predict(X_test))
        if number_predicition == 1:
            prediction = cl.t_to_int(prediction)
            # y_test = cl.t_to_int(y_test)
        elif number_predicition == 0:
            prediction = cl.g_to_it(prediction)
        else:
            prediction = cl.nm_to_int(prediction)
            # y_test = cl.nm_to_int(y_test)
        for j in range(leny):
            sum[j] = prediction[j] + sum[j]
            if number_predicition == 2 or number_predicition == 3:
                vote[j][prediction[j]] = vote[j][prediction[j]] + 1
            else:
                vote[j][prediction[j] - 1] = vote[j][prediction[j] - 1] + 1
    for j in range(leny):
        sum[j] = round(sum[j] / len(models))
        max = 0
        index = 0
        for ij in range(10):
            if max < vote[j][ij]:
                max = vote[j][ij]
                if number_predicition == 2 or number_predicition == 3:
                    index = ij
                else:
                    index = ij + 1
        sum_vote.append(round(((sum[j] + index) / 2)+0.01))
    l = 0
    j = 0
    # for i in range(leny):
    #     if sum_vote[i] == y_test[i]:
    #         if sum_vote[i] == lj:
    #             l = l + 1
    #         else:
    #             j = j + 1
    return sum_vote, l, j, y_test

