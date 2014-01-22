# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:08:49 2014

@author: predicador
"""

import numpy as np
from sklearn import preprocessing as pp
from sklearn import cross_validation as cv
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def printMetrics(estimator, X_train, y_train, y_test, y_pred):
    
    scores = cross_validation.cross_val_score(estimator, X_train,y_train, cv=5)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() / 2))
    print "EVS: %.4f" % explained_variance_score(y_test, y_pred)
    print "MAE: %.4f" % mean_absolute_error(y_test, y_pred)
    print "MSE: %.4f" % mean_squared_error(y_test, y_pred)
    print "R2: %.4f" % r2_score(y_test, y_pred)

X=[] #train dataset
data=[] #kaggle test dataset
y=[] #train labels

# Read data
X = np.genfromtxt(open('./csv/train.csv','rb'), delimiter=',')
y = np.genfromtxt(open('./csv/trainLabels.csv','rb'), delimiter=',')
data = np.genfromtxt(open('./csv/test.csv','rb'), delimiter=',')

# Scale data
#X = pp.scale(X)
#data = pp.scale(data)

# Select features
#selector = ExtraTreesClassifier(compute_importances=True, random_state=0)
#train = selector.fit_transform(train, target)
#data = selector.transform(test)

X_train, X_test, y_train, y_test = cv.train_test_split(X,y, test_size=0.2, random_state=0)


#Hyperparameter selection
#param_grid = {'C': np.logspace(0, 2, 50), 'gamma':np.logspace(-2,1,50)}
param_grid = {'kernel':['rbf'], 'C': np.linspace(0.1, 10, 10), 'gamma':np.linspace(0.01,1,50)}
#param_grid = {'C': [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,1000], 'gamma':[0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.17] }

# Define crossvalidation folds
n_cv_folds = 5

# Hyperparameter search: get optimal C and gamma
cvk = cv.StratifiedKFold(y_train, n_folds=20)
clf = GridSearchCV(SVC(), param_grid,scoring='accuracy',cv = n_cv_folds)

print("The model is trained on the full development set.")
clf.fit(X_train, y_train)

# Print log and best parameter

print("Grid scores on development set:")
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.04f) for %r" % (mean_score, scores.std() / 2, params))
print("The scores are computed on the full evaluation set.")
y_true, y_pred = np.ravel(y_test), clf.predict(X_test)
print("Detailed classification report:")
print(classification_report(y_true, y_pred))
print("Best parameters set found on development set:")
print(clf.best_params_)

printMetrics(SVC(kernel='rbf',C=clf.best_params_['C'],gamma=clf.best_params_['gamma']), X_train, y_train, y_test, y_pred)


# Final estimation with best param

clf = SVC(kernel='rbf',C=clf.best_params_['C'],gamma=clf.best_params_['gamma'])
clf.fit(X,np.ravel(y))
y_test = clf.predict(data)
idColumn = np.arange(1,y_test.size+1,1).astype(np.integer)

# Generate CSV with Kaggle competition format
np.savetxt('./csv/svc.csv',np.c_[idColumn, y_test], fmt='%d', delimiter=',', header='id,Solution',comments='')

