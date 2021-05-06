from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import sqlite3 as sl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Get required data
con = sl.connect("prostate_cancer.db")
df = pd.read_sql("SELECT * FROM PATIENT", con)
df.drop('index', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)
df.drop('weight', axis=1, inplace=True)
df.drop('height', axis=1, inplace=True)
df.drop('density_psa', axis=1, inplace=True)
df.drop('T', axis=1, inplace=True)
df.drop('N', axis=1, inplace=True)
df.drop('M', axis=1, inplace=True)
x = df[['age', 'bmi', 'prostate_volume', 'psa']]
y = df['G']

# Create training and test samples
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=27)

# Choose classificators
models = [LogisticRegression(),
	          LinearDiscriminantAnalysis(),
	          KNeighborsClassifier(),
	          GaussianNB(),
	          DecisionTreeClassifier(),
              SVC(),
              RandomForestClassifier()
	          ]

# Start classification
for model in models:
    m = str(model)
    m = m[:m.index('(')]
    model.fit(X_train, y_train) # download our train sample
    prediction = model.predict(X_test) # get classification for test sample
    # show results for each method
    print(m)
    print(accuracy_score(prediction, y_test))
    print(confusion_matrix(prediction, y_test))

# following three classifiers performed best
# choose final classification based on their results using mean and voting
golosovanie = [KNeighborsClassifier(),
	          DecisionTreeClassifier(),
              RandomForestClassifier()]
golosovanie_prediction = np.zeros(y_test.size)
votes = np.zeros([y_test.size,3])
k = 0
for model in golosovanie:
    m = str(model)
    m = m[:m.index('(')]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(m)
    print(accuracy_score(prediction, y_test))
    print(confusion_matrix(prediction, y_test))
    tester = ''
    l = 0
    j = 0
    print(y_test.size)
    for i in range(y_test.size):
        golosovanie_prediction[i] = golosovanie_prediction[i] + np.array(prediction)[i]
        votes[i][k] = np.array(prediction)[i]
        if np.array(y_test)[i] == np.array(prediction)[i]:
            tester = 'True'
            if np.array(y_test)[i] == 7.0:
                j = j + 1
            else:
                l = l + 1
        else:
            tester = ''
        print('expected: ', np.array(y_test)[i], '; ', 'get: ', np.array(prediction)[i], '  ', tester)
    print(j) # number of true classified gleason score '7'
    print(l) # number of true classified other gleason scores
    k = k + 1
    print()

# Start voting and find mean
final_votes = np.zeros(y_test.size)
final_votes_counter = np.zeros(10)
for i in range(y_test.size):
    golosovanie_prediction[i] = round(golosovanie_prediction[i]/3) # first step was made in previous cycle
    for j in range(3):
        final_votes_counter[int(votes[i][j])-1] = final_votes_counter[int(votes[i][j])-1] + 1
    max_i = 0
    index = 0
    for j in range(10):
        if final_votes_counter[j] > max_i:
            max_i = final_votes_counter[j]
            index = j
        final_votes_counter[j] = 0
    final_votes[i] = float(index+1)
    final_votes[i] = round((golosovanie_prediction[i]+final_votes[i]+0.01)/2)

# Show final results
l = 0
j = 0
for i in range(y_test.size):
    if np.array(y_test)[i] == final_votes[i]:
        tester = 'True'
        if np.array(y_test)[i] == 7.0:
            j = j + 1
        else:
            l = l + 1
    else:
        tester = ''
    print('ожидалось: ', np.array(y_test)[i], '; ', 'получилось: ', final_votes[i], '  ', tester)
print(j)
print(l)
print()
