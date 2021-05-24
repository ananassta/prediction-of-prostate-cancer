import pandas as pd
import sqlite3 as sl
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras_cgi as kr
import classificator_cgi as cl
import regressor_cgi as rg

# def start_count(age, bmi, volume, psa):
con = sl.connect("prostate_cancer.db")
df = pd.read_sql("SELECT * FROM PATIENT", con)
X = df[['age', 'bmi', 'prostate_volume', 'psa']]

y = [[0], [0], [0], [0]]
Y = [[0], [0], [0], [0]]
Y_train = [[0], [0], [0], [0]]
Y_test = [[0], [0], [0], [0]]
X_train = [[0], [0], [0], [0]]
X_test = [[0], [0], [0], [0]]
enc = [[0], [0], [0], [0]]
y[0] = df['G']
y[1] = df['T']
y[2] = df['N']
y[3] = df['M']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_features = X.shape[1]
n_classes = []

for i in range(4):
    enc[i] = OneHotEncoder()
    Y[i] = enc[i].fit_transform(y[i][:, np.newaxis]).toarray()
    X_train[i], X_test[i], Y_train[i], Y_test[i] = train_test_split(X_scaled, Y[i], test_size=0.4, random_state=27)
    n_classes.append(Y[i].shape[1])

# for future res on web
# res = [[0,0,0], [0,0,0], [0,0,0]]

for i in range(4):
    res_r, l_r, k_r = rg.count_regression(X_train[i], X_test[i], Y_train[i], Y_test[i], i, enc[i])
    res_c, l_c, k_c, y_check = cl.count_classification(X_train[i], X_test[i], Y_train[i], Y_test[i], i, enc[i])
    res_k, l_k, k_k = kr.count_keras(X_train[i], X_test[i], Y_train[i], Y_test[i], i, n_features, n_classes[i], enc[i])
    print(l_r, "/////", k_r)
    print(l_c, "/////", k_c)
    print(l_k, "/////", k_k)
    sum = []
    vote = []
    for k in range(len(res_r)): #(res_r.size):
        vote.append([])
        for j in range(10):
            vote[k].append(0)
    sum_vote = []
    if i == 0:
        lj = 7
    if i == 1:
        lj = 5
    if i == 2:
        lj = 1
    if i == 3:
        lj = 1
    for j in range(len(res_r)): #(res_r.size):
        sum.append(res_r[j] + res_c[j] + res_k[j])
        sum[j] = round(sum[j]/3)
        if i == 2 or i == 3:
            vote[j][res_r[j]] = vote[j][res_r[j]] + 1
            vote[j][res_c[j]] = vote[j][res_c[j]] + 1
            vote[j][res_k[j]] = vote[j][res_k[j]] + 1
        else:
            vote[j][res_r[j] - 1] = vote[j][res_r[j] - 1] + 1
            vote[j][res_c[j] - 1] = vote[j][res_c[j] - 1] + 1
            vote[j][res_k[j] - 1] = vote[j][res_k[j] - 1] + 1
        max = 0
        index = 0
        for ij in range(10):
            if max < vote[j][ij]:
                max = vote[j][ij]
                if i == 2 or i == 3:
                    index = ij
                else:
                    index = ij + 1
        sum_vote.append(round(((sum[j]+index)/2)+0.01)) # +0.01???
    # print(res_r)
    # print(res_c)
    # print(res_k)
    # print(y_check)
    # print(sum_vote) # check them?
    l = 0
    j = 0
    for k in range(len(y_check)): #(y_check.size):
        if sum_vote[k] == y_check[k]:
            if sum_vote[k] == lj:
                l = l + 1
            else:
                j = j + 1
    print(l)
    print(j)
    print("-------------------------------------------------------------")