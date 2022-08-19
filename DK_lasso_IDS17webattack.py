# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:39:28 2021

@author: desdina.kof
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as sm
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import RFE
# from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# from sklearn.linear_model import LogisticRegresion

# Getting the dataset

lst = set(range(82))
lst.remove(0)
lst.remove(1)
lst.remove(3)
lst.remove(6)
#lst.remove(84)


df = pd.read_csv("C:/Users/desdina.kof/Desktop/calismalarim/saga/IDS2017/TrafficLabelling/Thursday-WebAttacks_2.csv", usecols=lst)
#df = pd.read_csv("C:/Users/desdina.kof/Desktop/calismalarim/saga/IDS2017/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos", usecols=lst, encoding='cp1252')
df[' Label'].unique()

df[' Label'] = np.where(df[' Label'] == 'Web Attack – Brute Force', 'ATTACK', df[' Label'])
df[' Label'] = np.where(df[' Label'] == 'Web Attack – XSS', 'ATTACK', df[' Label'])
df[' Label'] = np.where(df[' Label'] == 'Web Attack – Sql Injection', 'ATTACK', df[' Label'])

df[' Label'] = df[' Label'].map({"BENIGN": 0,
                               "ATTACK": 1}).astype(float)
                               
'''
OR
df['Label'] = df['Label'].map({"BENIGN": 0,
                                "Web Attack � Brute Force": 1,
                                "Web Attack � XSS": 1,
                                "Web Attack � Sql Injection": 1}).astype(float)
'''

# Removing Inf Values & Deleting Null Values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df = df[np.isfinite(df).all(1)]
# df = df.dropna()

# Data Visualization

print(df.shape)
print(list(df.columns))
df.dtypes
df.astype("int64")

df.head()

# Correlation Heat Map
cor = df.corr()
plt.figure(figsize=(70,42))

sns.heatmap(cor, annot=True)

columns = np.full((cor.shape[0],), True, dtype=bool)

# Removal of Variables Regarding Correlation Heat Map
for i in range(cor.shape[0]):
    for j in range(i+1, cor.shape[0]):
        if cor.iloc[i, j] >= 0.75:
            if columns[i]:
                columns[j] = False
                
selected_columns = df.columns[columns]
df = df[selected_columns]

# Data Division
y = df.iloc[:, -1]
X_norm = df.iloc[:, 0:len(df.columns)-1]
print(y)
b = (y==0)
# Backward Elimination
"""
selected_columns = selected_columns[1:].values

def backwardElimination(x, Y, sl, cols):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_ols = sm.ols(Y, x).fit()
        maxVar = max(regressor_ols.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if regressor_ols.pvalues[j].astype(float) == maxVar:
                    x = np.delete(x, j, 1)
                    cols = np.delete(cols, j)
                    
    regressor_ols.summary()
    return x, cols

SL = 0.05
# data_modeled, selected_columns = backwardElimination(X_norm.values, y.values, SL, selected_columns)
data_modeled, selected_columns = backwardElimination(df.iloc[:, 1:].values, df.iloc[:, 0].values, SL, selected_columns)
"""
# Recursive Feature Elimination

#df['Fwd.PSH.Flags'].mean()

num_feats = 20
  
logreg = LogisticRegression()

rfe_selector = RFE(logreg, n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X_norm.loc[:, rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

# Lasso

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X_norm.loc[:, embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')







