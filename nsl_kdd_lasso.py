# bismillahirrahmanirrahim

import pandas as pd
import numpy as np
import seaborn as sns
# import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from numpy import loadtxt
import keras.models as km
import keras.layers as kl
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Getting the dataset
# IDS2017-DDoS
lst = set(range(43))
lst.remove(0)
lst.remove(2)
lst.remove(3)
lst.remove(4)
# df = pd.read_csv("C:/Users/desdina.kof/Desktop/calismalarim/saga/DDoS2019/03-11/Syn.csv",
#                 usecols=lst, encoding='cp1252')

# df = pd.read_csv("C:/Users/desdina.kof/Desktop/calismalarim/milas/DDoS2019/01-12/Syn.csv",
# usecols = lst, encoding = 'cp1252')

df = pd.read_csv("C:/Users/desdina.kof/Desktop/calismalarim/milas/NSL-KDD/csv_result-KDDTrain.csv",
                 usecols=lst, encoding='cp1252')
df2 = pd.read_csv("C:/Users/desdina.kof/Desktop/calismalarim/milas/NSL-KDD/csv_KDDTest-21.csv",
                  usecols =lst, encoding='cp1252')

x = df['class'].unique()

np.size(df)
"""
df['Label'] = np.where(df[' Label']=='Web Attack – Brute Force', 'ATTACK', df[' Label'])
df['Label'] = np.where(df[' Label']=='Web Attack – XSS', 'ATTACK', df[' Label'])
df['Label'] = np.where(df[' Label']=='Web Attack – Sql Injection', 'ATTACK', df[' Label'])
"""
df['class'] = df['class'].map({"normal": 0,
                               "anomaly": 1}).astype(float)

# Removing Inf Values & Deleting Null Values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df[np.isfinite(df).all(1)]

# Data Visualization
print(df.shape)
print(list(df.columns))
# ty = df.dtypes
# df.astype("int64")
df.dtypes
df.astype("int64")
df.head()
# a = np.all(np.isfinite(X_norm))
# b = np.any(np.isnan(X_norm))

# Correlation Heat Map
cor = df.corr()
plt.figure(figsize=(70, 42))

sns.heatmap(cor, annot=True)

columns = np.full((cor.shape[0],), True, dtype=bool)

# Removal of Variables Regarding Correlation Heat Map
for i in range(cor.shape[0]):
    for j in range(i + 1, cor.shape[0]):
        if cor.iloc[i, j] >= 0.75:
            if columns[i]:
                columns[j] = False

selected_columns = df.columns[columns]
df = df[selected_columns]

# Data Division

y = df.iloc[:, -1]
X_norm = df.iloc[:, 0:len(df.columns) - 1]
# X_norm = X_norm.dropna(axis =1)
# y = y.dropna()
print(y)

# Backward Elimination
selected_columns = selected_columns[1:].values


def backward_elimination(x, y, sl, cols):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_ols = sm.OLS(y, x).fit()
        maxVar = max(regressor_ols.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if regressor_ols.pvalues[j].astype(float) == maxVar:
                    x = np.delete(x, j, 1)
                    cols = np.delete(cols, j)

    regressor_ols.summary()
    return x, cols


SL = 0.05
data_modeled, selected_columns = backward_elimination(X_norm.values, y.values, SL, selected_columns)
# data_modeled, selected_columns = backward_elimination(df.iloc[:, 1:].values, df.iloc[:, 0].values, SL, selected_columns)
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
"""
# Lasso
num_feats = 20
print(X_norm)

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2", max_iter=50), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X_norm.loc[:, embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')
print(str(embeded_lr_feature))
X = X_norm[embeded_lr_feature]

# Logistic Regression

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
x_no_train, x_no_test, y_no_train, y_no_test = train_test_split(X_norm, y, test_size=0.25, random_state=0)

logisticregr = LogisticRegression()
logisticregr.fit(x_train, y_train)
logisticregr.predict(x_test)

logisticregr_no = LogisticRegression()
logisticregr_no.fit(x_no_train, y_no_train)
logisticregr_no.predict(x_no_test)

# Accuracy
score = logisticregr.score(x_test, y_test)
score_no = logisticregr_no.score(x_no_test, y_no_test)
print(score, score_no)

# ANN
b = len(X_norm[embeded_lr_feature])

model = km.Sequential()
model.add(kl.Dense(9, input_dim=29, activation='relu'))
model.add(kl.Dense(3, activation='relu'))
model.add(kl.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_norm, y, epochs=15)
# evaluate the keras model
_, accuracy = model.evaluate(X_norm, y)
print('Accuracy: %.2f' % (accuracy * 100))

model_2 = km.Sequential()
model_2.add(kl.Dense(9, input_dim=9, activation='relu'))
model_2.add(kl.Dense(3, activation='relu'))
model_2.add(kl.Dense(1, activation='sigmoid'))

model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model_2.fit(X, y, epochs=15)
# evaluate the keras model
_, accuracy_2 = model_2.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy_2 * 100))

# kNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_nr_train, X_nr_test, y_nr_train, y_nr_test = train_test_split(X_norm, y, test_size=0.25, random_state=0)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
# SVM
X_red = X.iloc[1:20000, :]
X_n_red = X_norm.iloc[1:20000, :]
y_red = y.iloc[1:20000]
model_svm = SVC(kernel='linear', C=1E3)
model_svm.fit(X_n_red, y_red)
_, accuracy_svm = model_svm.evaluate(X_n_red, y_red)
print('Accuracy: %.2f' % (accuracy_svm*100))

model_svm_2 = SVC(kernel='linear', C=1E3)
model_svm_2.fit(X_red, y_red)
_, accuracy_svm_2 = model_svm_2.evaluate(X_red, y_red)
print('Accuracy: %.2f' % (accuracy_svm_2*100))
"""


