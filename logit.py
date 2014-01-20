# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def printMetrics(estimator, X_train, y_train, y_test, y_pred):
    
    scores = cross_validation.cross_val_score(estimator, X_train,y_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
    print "EVS: %.4f" % explained_variance_score(y_test, y_pred)
    print "MAE: %.4f" % mean_absolute_error(y_test, y_pred)
    print "MSE: %.4f" % mean_squared_error(y_test, y_pred)
    print "R2: %.4f" % r2_score(y_test, y_pred)

X=[] #train dataset
data=[] #kaggle test dataset
y=[] #train labels

X = np.genfromtxt(open('./csv/train.csv','rb'), delimiter=',')
y = np.genfromtxt(open('./csv/trainLabels.csv','rb'), delimiter=',')
data = np.genfromtxt(open('./csv/test.csv','rb'), delimiter=',')

# Data scaling: showed no improvement at all
'''
X= pp.scale(X)
data = pp.scale(data)
'''
# Feature selection: showed no improvement at all
'''
selector = ExtraTreesClassifier(compute_importances=True, random_state=0)
selector.fit_transform(X, y)
X=selector.transform(X)
data = selector.transform(data)
'''

#Dimensionality reduction
# Find PCA components which variances are greater than 0.01
'''
pcaFind = PCA()
pcaFind.fit(X).transform(X)

nComponents=pcaFind.explained_variance_ratio_[pcaFind.explained_variance_ratio_>0.01].size
'''
# Find PCA components that explain 99% of the variance
'''
total = 0;
nComponents=0;
for component in pcaFind.explained_variance_ratio_:
    print component
    print np.sum(pcaFind.explained_variance_ratio_)
    total = total + component/np.sum(pcaFind.explained_variance_ratio_)
    nComponents+=1
    if (total>= 0.8):
        break
'''
# Apply PCA and dimensionality reduction: showed no improvement at all
'''
pca=PCA(n_components=nComponents)
pca.fit(X).transform(X)
pca.transform(data)
'''

# Generate train and test data from original train dataset
X_train, X_test, y_train, y_test = cv.train_test_split(X,np.ravel(y), test_size=0.2, random_state=0)

# Define parameter space

param_grid = {'C': np.logspace(-2, 0, 100)}

# Define crossvalidation folds
n_cv_folds = 20

# Hyperparameter search: get optimal C
clf = GridSearchCV(LogisticRegression(penalty='l1', tol=1e-6), param_grid, scoring='accuracy',cv = n_cv_folds)

print("The model is trained on the full development set.")
clf.fit(X_train, y_train)

# Print log and best parameter

print("Grid scores on development set:")
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
print("The scores are computed on the full evaluation set.")
y_true, y_pred = np.ravel(y_test), clf.predict(X_test)
print("Detailed classification report:")
print(classification_report(y_true, y_pred))
print("Best parameters set found on development set:")
print(clf.best_params_)

printMetrics(LogisticRegression(C=clf.best_params_['C'], penalty='l1', tol=1e-6), X_train, y_train, y_test, y_pred)

# Training, test and validation error graphical representation
# Error computing 
mse = metrics.mean_squared_error
training_error = []
test_error = []
grid_error= []

for c in param_grid['C']:
    model = LogisticRegression(C=c, penalty='l1', tol=1e-6).fit(X_train, y_train)
    training_error.append(mse(model.predict(X_train), y_train))
    test_error.append(mse(model.predict(X_test), y_test))
cv_error =[1- g[1] for g in clf.grid_scores_]
    
# Graphical representation: errors vs parameter C
plt.semilogx(param_grid['C'], training_error, label='training')
plt.semilogx(param_grid['C'], test_error, label='test')
plt.semilogx(param_grid['C'], cv_error, label='cv')
plt.legend()
plt.xlabel('Regularization parameter C')
plt.ylabel('MSE') 
plt.axvline(clf.best_params_['C'], color='black')
plt.draw()
plt.show() 
 
# Final estimation with best param

clf = LogisticRegression(C=clf.best_params_['C'], penalty='l1', tol=1e-6)
clf.fit(X,np.ravel(y))
y_test = clf.predict(data)
idColumn = np.arange(1,y_test.size+1,1).astype(np.integer)

# Generate CSV with Kaggle competition format
np.savetxt('./csv/result.csv',np.c_[idColumn, y_test], fmt='%d', delimiter=',', header='id,Solution',comments='')

