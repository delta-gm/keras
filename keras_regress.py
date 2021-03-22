

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import statistics

concrete_data = pd.read_csv('concrete_data.csv')

# link for dataset: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv

concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

n_cols = predictors.shape[1]
predictors_norm = (predictors - predictors.mean())/predictors.std()
predictors_norm.head()


def regression_model():
    model = Sequential()
    model.add(Dense(10, activation = 'relu', input_shape = (n_cols,)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train)
    return model

scores = []
for i in range(0,50):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=4)
    model = regression_model()
    model.fit(X_train, y_train, epochs=50, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)
    scores.append(score)
    if i == 49:
        print(scores)
        avg = statistics.mean(scores)
        std = statistics.stdev(scores)
        print("Average of MSEs: ", avg)
        print("Standard Deviation of MSEs: ", std)


