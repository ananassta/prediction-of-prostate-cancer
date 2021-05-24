import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import tensorflowjs as tfjs
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import class_to_int as cl

def count_keras(X_train, X_test, Y_train, Y_test, number_predicition, n_features, n_classes, enc):
    def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
        def create_model():
            model = Sequential(name=name)
            k = 0
            l = input_dim
            for i in range(n):
                model.add(Dense(nodes + k, input_dim=l, activation='relu'))
                l = nodes + k
                k = k + 2
            model.add(Dense(output_dim, activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            return model

        return create_model

    models = [create_custom_model(n_features, n_classes, 8, 3, 'model')]

    history_dict = {}
    cb = TensorBoard()
    # y_check = np.ravel(enc.inverse_transform(Y_test))
    # Y_test = np.array(np.array(Y_test))
    for create_model in models:
        model = create_model()
        # print('Model name:', model.name)
        history_callback = model.fit(X_train, Y_train,
                                     batch_size=5,
                                     epochs=50,
                                     verbose=0,
                                     # validation_data=(X_test, Y_test),
                                     callbacks=[cb])
        # score = model.evaluate(X_test, Y_test, verbose=0)
        history_dict[model.name] = [history_callback, model]
        l = 0
        j = 0
        # if number_predicition == 0:
        #     lj = 7
        # elif number_predicition == 1:
        #     lj = '3'
        # else:
        #     lj = '1'
        prediction = np.ravel(enc.inverse_transform(model.predict(X_test)))
        # for i in range(prediction.size):
        #     if prediction[i] == y_check[i]:
        #         if prediction[i] == lj:
        #             l = l + 1
        #         else:
        #             j = j + 1
        if number_predicition == 0:
            prediction = cl.g_to_it(prediction)
        elif number_predicition == 1:
            prediction = cl.t_to_int(prediction)
        else:
            prediction = cl.nm_to_int(prediction)
    return prediction, l, j



