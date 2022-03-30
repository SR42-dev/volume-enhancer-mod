'''
Notes :

    - for loop to change polynomial degree values
    - plot all accuracies
    - check for peak
    - set for 80% of peak
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# getting dataset
dataset_orig = pd.read_csv('data.csv')
dataset = dataset_orig
# dataset = dataset_orig.iloc[1:,:]

# getting variables
y = dataset['user_volume'] # dependent variable
x1 = dataset['artist'] # independent variable, catagorical
x2 = dataset['genre'] # independent variable, catagorical
x3 = dataset['zero_crossing'] # independent variable, numerical
x4 = dataset['spectral_centroid_mean'] # independent variable, numerical
x5 = dataset['spectral_rolloff_mean'] # independent variable, numerical

# encoding catagorical data
labelencoder_x1 = LabelEncoder() # converts string data to numerical data
x1 = labelencoder_x1.fit_transform(x1) # encoding all features of the artist column into numbers 0, 1, 2 for priority 2 > 1 > 0, which is a nonsensical relation atm
ct1 = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough') # replaces column variables with 1s and 0s
x1 = ct1.fit_transform(x1)

labelencoder_x2 = LabelEncoder() # converts string data to numerical data
x2 = labelencoder_x1.fit_transform(x2) # encoding all features of the artist column into numbers 0, 1, 2 for priority 2 > 1 > 0, which is a nonsensical relation atm
ct2 = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough') # replaces column variables with 1s and 0s
x2 = ct2.fit_transform(x2)

# spliting dataset into training set and testing test
x1_train,x1_test,y1_train,y1_test=train_test_split (x1,y,test_size=0.25,random_state=0)
x2_train,x2_test,y2_train,y2_test=train_test_split (x2,y,test_size=0.25,random_state=0)
x3_train,x3_test,y3_train,y3_test=train_test_split (x3,y,test_size=0.25,random_state=0)
x4_train,x4_test,y4_train,y4_test=train_test_split (x4,y,test_size=0.25,random_state=0)
x5_train,x5_test,y5_train,y5_test=train_test_split (x5,y,test_size=0.25,random_state=0)

# fitting polynomial linear regression to the data set
poly_reg1 = PolynomialFeatures(degree = 4) # experiment by changing values
x1_poly = poly_reg1.fit_transform(x1_train)
lin_reg1 = LinearRegression()
lin_reg1.fit(x1_poly,y1_train)

poly_reg2 = PolynomialFeatures(degree = 4) # experiment by changing values
x2_poly = poly_reg2.fit_transform(x2_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x2_poly,y2_train)

poly_reg3 = PolynomialFeatures(degree = 4) # experiment by changing values
x3_poly = poly_reg3.fit_transform(x3_train)
lin_reg3 = LinearRegression()
lin_reg3.fit(x3_poly,y3_train)

poly_reg4 = PolynomialFeatures(degree = 4) # experiment by changing values
x4_poly = poly_reg4.fit_transform(x4_train)
lin_reg4 = LinearRegression()
lin_reg4.fit(x4_poly,y4_train)

poly_reg5 = PolynomialFeatures(degree = 4) # experiment by changing values
x5_poly = poly_reg5.fit_transform(x5_train)
lin_reg5 = LinearRegression()
lin_reg5.fit(x5_poly,y5_train)

# predicting the test set results
y1_pred = lin_reg1.predict(x1_test)
y2_pred = lin_reg2.predict(x2_test)
y3_pred = lin_reg3.predict(x3_test)
y4_pred = lin_reg4.predict(x4_test)
y5_pred = lin_reg5.predict(x5_test)

# making the confusion matrices
cm1 = confusion_matrix(y1_test, y1_pred)
cm2 = confusion_matrix(y2_test, y2_pred)
cm3 = confusion_matrix(y3_test, y3_pred)
cm4 = confusion_matrix(y4_test, y4_pred)
cm5 = confusion_matrix(y5_test, y5_pred)

# calculating final prediction
data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
savetxt('data.csv', data, delimiter=',')






