import numpy as np
import pandas as pd
import sqlite3 as sl
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn import preprocessing
import tensorflowjs as tfjs

nb_classes = 11

con = sl.connect("prostate_cancer.db")
df = pd.read_sql("SELECT * FROM PATIENT", con)
x = df[['age', 'bmi', 'prostate_volume', 'psa']] #[age, bmi, prostate_volume, psa] #df[['age', 'bmi', 'prostate_volume', 'psa']]
x = preprocessing.normalize(x)
y = df['G']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=27)

# the data, shuffled and split between tran and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()


#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)
#X_train = X_train.astype("float32")
#X_test = X_test.astype("float32")
#X_train /= 255
#X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
#model1 = Sequential()
model.add(Dense(nb_classes, input_dim=4, activation='softmax'))
#model.add(Dense(100, input_dim=4, activation='relu'))
#model.add(Dense(nb_classes,input_dim=100, activation='softmax'))
#model1.add(Dense(nb_classes, input_dim=4, activation='softmax'))

model.summary() #Print model info

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=130, verbose=2, validation_data=(X_test, Y_test)) #для одного слоя 130
#model1.fit(X_train, Y_train, batch_size=128, epochs=500, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test,verbose=0)
#print('Test score:', score[0])
print('Test accuracy:', score[1])

res = model.predict(X_test)
#res1 = model.predict(X_test)
sum = []
#sum1 = []
for i in range(res.shape[0]):
    s = res[i][0]
    #s1 = res1[i][0]
    k = 0
    #k1 = 0
    for j in range(res[i].size):
        if s < res[i][j]:
            s = res[i][j]
            k = j
        #if s1 < res1[i][j]:
        #    s1 = res1[i][j]
        #    k1 = j
    sum.append(k)
    #sum1.append(k1)
print(sum)
#print(sum1)
l = 0
j = 0
#l1 = 0
#j1 = 0
for i in range(len(sum)):
    if int(np.array(y_test)[i]) == sum[i]:
        if np.array(y_test)[i] == 7.0:
            j = j + 1
        else:
            l = l + 1
    #if int(np.array(y_test)[i]) == sum1[i]:
    #    if np.array(y_test)[i] == 7.0:
    #        j1 = j1 + 1
    #    else:
    #        l1 = l1 + 1
print(j)
print(l)
#print(j1)
#print(l1)
print(X_test)
#model.save("my_model.h5")
#tfjs.converters.save_keras_model(model, "tfjs_target_dir")