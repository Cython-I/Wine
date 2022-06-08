

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:40:45 2022

@author: christopherbenjaminscottii
"""

# Wine Quality with Multiple Linear Regression (With Normalization)

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics

# Import data
wine = pd.read_csv('/home/christopherbenjaminscottii/Documents/Machine Learning/winequality-red.csv', sep = ';')

#splitting the features from the target data
x = wine.drop('quality',axis=1)
y = wine['quality']

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#fit scaler on the training data
norm = MinMaxScaler().fit(x_train)

# transform training data
X_train_norm = norm.transform(x_train)

# transform testing data
X_test_norm = norm.transform(x_test)

#creates linear regressor
regressor = LinearRegression()

#fit regressor on the normalized data
regressor.fit(X_train_norm,y_train)

#dict for all the features
featuredict = {0:'fixed acidity',
               1:'volatile acidity',
               2:'citric acid',
               3:'residual sugar',
               4:'chlorides',
               5:'free sulfur dioxide',
               6:'total sulfur dioxide',
               7:'density',
               8:'pH',
               9:'sulphates',
               10:'alcohol'          
               }
#gather the importance of each feature of the data to determine which is most
#important towards the quality of a wine, due to the model fitting the hyperplane
#such that error is minimal
importance = regressor.coef_
highestimportancefeature = -999
vip = ''

#temporary storage for data so can put into dataframe,doing so rather than doing
#with dataframe reduces the algorithmic complexity and helps with runtime
data = []

#loop through the regressors coefs and find the most important, in addition, add 
#all to list for dataframe
for i,v in enumerate(importance):
    if(v > highestimportancefeature):
        highestimportancefeature = v
        vip = featuredict.get(i)
    data.append([featuredict.get(i), v])
    
#create dataframe for the feature importance
df = pd.DataFrame(data, columns = ['Feature','Importance'])
#some plotting details
df.set_index("Feature",inplace=True)
ax = df["Importance"].plot(kind = 'bar',rot=90)
ax.set_title("Feature Importance Towards Quality(MLR)")
ax.set_ylabel("Relative Importance")
ax.set_xlabel("Features")
plt.show()

print(df)

#Some data about the regressions results, since it is set to randstate=0 should remain constant
y_pred = regressor.predict(X_test_norm)
score=r2_score(y_test,y_pred)
print('\nr2 socre is',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_pred))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_pred)))
print('The highest important feature is: ', vip)