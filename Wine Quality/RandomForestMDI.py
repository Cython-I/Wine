#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:40:45 2022

@author: christopherbenjaminscottii
"""

# Wine Quality with Random Forest

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
from sklearn.tree import export_graphviz
from subprocess import call
import time

# Import data
wine = pd.read_csv('/home/christopherbenjaminscottii/Documents/Machine Learning/winequality-red.csv', sep = ';')

#print(wine.head())


X = wine.iloc[:,0:11].values
y = wine.iloc[:, 11].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(n_estimators=75, random_state=0)
#fitting of data
regressor.fit(X_train,y_train)

#Unlike linear regression models the Random Forest does not fit on the hyper plane

y_pred = regressor.predict(X_test)

feature_names = [f"feature {i}" for i in range(X.shape[1])]

start_time = time.time()
importances = regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")



forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI (Random Forest)")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()


print('Mean Absolute Error:', sklearn.metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', sklearn.metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)))