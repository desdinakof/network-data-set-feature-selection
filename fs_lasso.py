
import pandas as pd
import numpy as np
import seaborn as sns
#import statsmodels.formula.api as sm
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import time

# Getting the dataset
# IDS2017-DDoS

lst = set(range(88))
lst.remove(0)
lst.remove(1)
lst.remove(2)
lst.remove(3)
lst.remove(4)
lst.remove(7)
#df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/DDoS2019/01-12/Syn.csv", usecols=lst, nrows = 500001, encoding='cp1252')

df = pd.read_csv("C:/Users/desdina.kof/Desktop/calismalarim/milas/DDoS2019/01-12/Syn.csv",
                 usecols = lst, encoding ='cp1252')

# Correlation Heat Map
cor = df2.corr()
plt.figure(figsize=(70, 42))

sns.heatmap(cor, annot=True)

columns = np.full((cor.shape[0],), True, dtype=bool)

# Removal of Variables Regarding Correlation Heat Map
for i in range(cor.shape[0]):
    for j in range(i + 1, cor.shape[0]):
        if cor.iloc[i, j] >= 0.75:
            if columns[i]:
                columns[j] = False

selected_columns = df2.columns[columns]
df2 = df2[selected_columns]

# Data Division
y2 = df2.iloc[:, -1]
X2 = df2.iloc[:, 0:len(df2.columns) - 1]
# X_norm = X_norm.dropna(axis =1)
# y = y.dropna()

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
data_modeled, selected_columns = backward_elimination(X2.values, y2.values, SL, selected_columns)
# data_modeled, selected_columns = backward_elimination(df.iloc[:, 1:].values, df.iloc[:, 0].values, SL, selected_columns)

# Lasso
num_feats = 20
print(X2)

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2", max_iter=50), max_features=num_feats)
embeded_lr_selector.fit(X2, y2)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X2.loc[:, embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')
print(str(embeded_lr_feature))
X = X2[embeded_lr_feature]

#Data Splitting

y_no_fs = df.iloc[:, -1]
X_no_fs = df.iloc[:, 0:len(df.columns)-1]

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.25, random_state= 0)
X_nr_train, X_nr_test, y_nr_train, y_nr_test =train_test_split(X_no_fs, y_no_fs, test_size = 0.25, random_state= 0)

# kNN

#Feature Selected Model
start = time.time()
classifier = KNeighborsClassifier(n_neighbors= 5)
classifier.fit(X_train, y_train)
end = time.time()
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average= 'macro'))
print(end-start)

#NO Feature Selection
end2 = time.time()
classifier = KNeighborsClassifier(n_neighbors= 5)
classifier.fit(X_nr_train, y_nr_train)
end3 = time.time()

y_nr_pred = classifier.predict(X_nr_test)
print(confusion_matrix(y_nr_test, y_nr_pred))
print(classification_report(y_nr_test,y_nr_pred))
print(accuracy_score(y_nr_test, y_nr_pred))
print(precision_score(y_nr_test, y_nr_pred, average ='macro'))
print(recall_score(y_nr_test, y_nr_pred, average ='macro'))

print(end3-end2)

# Naive Bayes

from sklearn.naive_bayes import GaussianNB, BernoulliNB

classifier = GaussianNB()
start = time.time()
#classifier = BernoulliNB()
y_pred = classifier.fit(X_train, y_train).predict(X_test)
end1 = time.time()
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average = 'macro'))
print(recall_score(y_test, y_pred, average = 'macro'))
print(end1-start)

classifier2 = GaussianNB()
start1 = time.time()
#classifier2 = BernoulliNB()
y_nr_pred = classifier2.fit(X_nr_train, y_nr_train).predict(X_nr_test)
end2 = time.time()
print(confusion_matrix(y_nr_test, y_nr_pred))
print(classification_report(y_nr_test, y_nr_pred))
print(accuracy_score(y_nr_test, y_nr_pred))
print(precision_score(y_nr_test, y_nr_pred, average='macro'))
print(recall_score(y_nr_test, y_nr_pred, average='macro'))
print(end2-start1)

# Logistic Regression
sta_lg =time.time()
logisticregr = LogisticRegression()
logisticregr.fit(X_train, y_train)
end_lg =time.time()
y_pred = logisticregr.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(end_lg-sta_lg)

sta_lg_2 = time.time()
logisticregr_no = LogisticRegression()
logisticregr_no.fit(X_nr_train, y_nr_train)
end_lg_2 =time.time()
y_pred_no = logisticregr_no.predict(X_nr_test)
print(confusion_matrix(y_nr_test, y_nr_pred))
print(classification_report(y_nr_test, y_pred_no))
print(accuracy_score(y_nr_test, y_nr_pred))
print(precision_score(y_nr_test, y_pred_no, average='macro'))
print(recall_score(y_nr_test, y_nr_pred, average='macro'))
print(end_lg_2-sta_lg_2)

# Accuracy
score = logisticregr.score(X_test, y_test)
score_no = logisticregr_no.score(X_nr_test, y_nr_test)
print(score, end_lg-sta_lg, score_no, end_lg_2-sta_lg_2)

# ANN

model = km.Sequential()
model.add(kl.Dense(40, input_dim=79, activation='relu'))
model.add(kl.Dense(9, activation='relu'))
model.add(kl.Dense(1, activation='sigmoid'))

start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_nr_train, y_nr_train, epochs= 15)
end1 = time.time()

# evaluate the keras model
_, accuracy_nr = model.evaluate(X_nr_train, y_nr_train)

y_nr_hat = model.predict_classes(X_nr_test, verbose = 0)
precision_nr = precision_score(y_nr_test, y_nr_hat)
recall_nr = recall_score(y_nr_test, y_nr_hat)
print('Accuracy: %.2f' % (accuracy_nr*100))
print('Precision: %.2f' % (precision_nr*100))
print('Recall: %.2f' % (recall_nr*100))
print(end1-start)

model_2 = km.Sequential()
model_2.add(kl.Dense(6, input_dim=6, activation='relu'))
model_2.add(kl.Dense(3, activation='relu'))
model_2.add(kl.Dense(1, activation='sigmoid'))

end2 = time.time()
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model_2.fit(X_train, y_train, epochs= 15)
end3 = time.time()
# evaluate the keras model
_, accuracy_1 = model_2.evaluate(X_train, y_train)

yhat = model_2.predict_classes(X_test, verbose = 0)
precision_1 = precision_score(y_test, yhat)
recall_1 = recall_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy_1*100))
print('Precision: %.2f' % (precision_1 *100))
print('Recall: %.2f' % (recall_1 *100))
print(end3-end2)

"""
y_pred1 = model.predict(X_test)
y_pred = np.argmax(y_pred1, axis=1)

# Print f1, precision, and recall scores
print(precision_score(y_test, y_pred , average="macro"))
print(recall_score(y_test, y_pred , average="macro"))
print(f1_score(y_test, y_pred , average="macro"))
"""

