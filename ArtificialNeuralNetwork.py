#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:19:13 2018

@author: amal
"""

"""###################################### ANN ######################################"""


################################# Libraries #####################################

#1 Theano
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#2 Tensorflow requires 64 bit CPU
#https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html

#3 Keras
#pip install --upgrade Keras


"""####################################### Data Preprocessing #########################################"""

#1 Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2 importing the data set
dataset = pd.read_csv('Churn_Modelling.csv')

#3 Classify the dependent and independent variable sets
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

#4 Deal with missing data

#5 Deal with categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoded_X_1 = LabelEncoder()
X[:,1] = label_encoded_X_1.fit_transform(X[:,1])
label_encoded_X_2 = LabelEncoder()
X[:,2] = label_encoded_X_2.fit_transform(X[:,2])
ohe_x1 = OneHotEncoder(categorical_features = [1])
X = ohe_x1.fit_transform(X).toarray()

#6 Avoiding dummy variable trap by removing one dummy variable
X = X[:,1:]

#7 splitting into training and testing sets

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train,  Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#8 Feature Scaling

from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()
X_train = std_sc.fit_transform(X_train)
X_test = std_sc.transform(X_test)


"""####################################### Artificial Neural Network #################################################"""

#1 import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense


#2 initialising the ANN classifier
classifier = Sequential() 

#3 creating the Input-layer and the first hidden layer
classifier.add(Dense(units = 7, activation='relu', kernel_initializer='uniform', input_dim = 11))

#4 creating the second hidden layer
classifier.add(Dense(units = 7, activation='relu', kernel_initializer='uniform'))


#5 creating the second hidden layer
classifier.add(Dense(units = 1, activation='sigmoid', kernel_initializer='uniform'))


#6 Compiling the ANN classifier
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#7 Fitting the classifier with the training set

classifier.fit(X_train,Y_train,batch_size=1, epochs=20)


#8 predicting the result for test set

prob = classifier.predict(X_test)
Y_pred = (prob > 0.5)

#9 confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)