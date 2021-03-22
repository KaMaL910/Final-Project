# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:55:56 2021

@author: kamal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



data = pd.read_csv('data.csv')
data = data.drop('Unnamed: 32',1) # 1 represent column base

""" Encoding the dependent variable """
diagnosis = LabelEncoder()
data['diagnosis'] = diagnosis.fit_transform(data['diagnosis'])

""" Spliting the data into training data and test data """

X = data.iloc[:,2:].values  
y = data.iloc[:,1].values   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)

""" Check if there more missing values yet, In Case it founded ! """

data.isnull().values.any()
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X[:,:])
X[:,:] = imp.transform(X[:,:])

""" Scaling to guarantee better accuracy """

scaler = MinMaxScaler(feature_range = (0,1)) 
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.fit_transform(X_test)

""" Apply SVM to do classification """

# training
classifier_1 = SVC(kernel = 'linear', random_state=None)
classifier_1.fit(X_train_sc,y_train)

# testing
y_pred_1 = classifier_1.predict(X_test_sc)
print(y_test)
print(y_pred_1)

""""Apply KNeighborsClassifier """

# training
classifier_2 = KNeighborsClassifier(n_neighbors=5)                                                          
classifier_2.fit(X_train_sc,y_train)

# testing
y_pred_2 = classifier_2.predict(X_test_sc)
print(y_test)
print(y_pred_2)

""" Apply GaussianNB """

# training
classifier_3= GaussianNB()
classifier_3.fit(X_train_sc, y_train)

# testing
y_pred_3 = classifier_3.predict(X_test_sc)
print(y_test)
print(y_pred_3)

""" Models Accuracy """

acc_1 = accuracy_score(y_test, y_pred_1)
print("Model_1 Accuracy =",acc_1*100,"%")

acc_2 = accuracy_score(y_test, y_pred_2)
print("Model_2 Accuracy =",acc_2*100,"%")

acc_3 = accuracy_score(y_test, y_pred_3)
print("Model_3 Accuracy =",acc_3*100,"%")



""" Models Evaluation """

from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(y_test, y_pred_1)

class_names = ['M','B']
df_cm = pd.DataFrame(cm_1, index = [i for i in class_names], columns = [i for i in class_names])
sn.heatmap(df_cm, annot = True)
cmap = plt.cm.Blues
plt.imshow(cm_1, interpolation = 'nearest', cmap = cmap)


"""cm_2 = confusion_matrix(y_test, y_pred_2)

class_names = ['M','B']
df_cm = pd.DataFrame(cm_2, index = [i for i in class_names], columns = [i for i in class_names])
sn.heatmap(df_cm, annot = True)
cmap = plt.cm.Blues
plt.imshow(cm_2, interpolation = 'nearest', cmap = cmap)

cm_3 = confusion_matrix(y_test, y_pred_3)

class_names = ['M','B']
df_cm = pd.DataFrame(cm_3, index = [i for i in class_names], columns = [i for i in class_names])
sn.heatmap(df_cm, annot = True)
cmap = plt.cm.Blues
plt.imshow(cm_3, interpolation = 'nearest', cmap = cmap)"""
