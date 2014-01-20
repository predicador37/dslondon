# -*- coding: utf-8 -*-
import csv as csv 
import numpy as np
import pylab as pl
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn import metrics
from sklearn import preprocessing as pp
from sklearn.ensemble import ExtraTreesClassifier
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression


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
pcaFind = PCA()
pcaFind.fit(X).transform(X)
#PCA components which variances are greater than 0.01
nComponents=pcaFind.explained_variance_ratio_[pcaFind.explained_variance_ratio_>0.01].size

#PCA components that explain 99% of the variance
#total = 0;
#nComponents=0;
#for component in pcaFind.explained_variance_ratio_:
#    print component
#    print np.sum(pcaFind.explained_variance_ratio_)
#    total = total + component/np.sum(pcaFind.explained_variance_ratio_)
#    ncomponent+=1
#    if (total>= 0.8):
#        break
#print nComponents

pca=PCA(n_components=nComponents)
pca.fit(X).transform(X)
pca.transform(data)

X_train, X_test, y_train, y_test = cv.train_test_split(X,np.ravel(y), test_size=0.2, random_state=0)

#param_grid = {'C': [0.0005,0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1] }
#param_grid = {'C': stats.expon(scale=100)}
#param_grid = {'C': np.logspace(-3, 2, 100)} #0.81 precision
param_grid = {'C': np.logspace(-2, 0, 100)}
#param_grid = {'C':  np.arange(0.01,1,0.01)}
n_cv_folds = 20
n_iter_search = 100

clf = GridSearchCV(LogisticRegression(C=1.0,penalty='l1', tol=1e-6), param_grid, scoring='accuracy',cv = n_cv_folds)
#clf = RandomizedSearchCV(LogisticRegression(C=1.0,penalty='l1', tol=1e-6), param_grid, n_iter = n_iter_search)

clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
print("Detailed classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
y_true, y_pred = np.ravel(y_test), clf.predict(X_test)
print(classification_report(y_true, y_pred))


rates = np.array([1.0 - x[1] for x in clf.grid_scores_])
stds   = [np.std(1.0 - x[2]) / math.sqrt(n_cv_folds) for x in clf.grid_scores_]
 
mse = metrics.mean_squared_error
training_error = []
test_error = []
grid_error= []

for c in param_grid['C']:
    model = LogisticRegression(C=c, penalty='l1', tol=1e-6).fit(X_train, y_train)
    training_error.append(mse(model.predict(X_train), y_train))
    test_error.append(mse(model.predict(X_test), y_test))

cv_error =[1- g[1] for g in clf.grid_scores_]
    
# note that the test error can also be computed via cross-validation
plt.semilogx(param_grid['C'], training_error, label='training')
plt.semilogx(param_grid['C'], test_error, label='test')
plt.semilogx(param_grid['C'], cv_error, label='cv')
#plt.plot([(1-c.mean_validation_score) for c in clf.grid_scores_], label="validation error")
plt.legend()
plt.xlabel('Regularization parameter C')
plt.ylabel('MSE') 
plt.axvline(clf.best_params_['C'], color='black')
plt.draw()
plt.show() 
 
#plt.figsize(12,6) 
#plt.plot([c.mean_validation_score for c in clf.grid_scores_], label="validation error")
#plt.plot([c.mean_training_score for c in clf.grid_scores_], label="training error")
#plt.xticks(np.arange(6), param_grid['C']);plt.xlabel("C");
#plt.ylabel("Accuracy");plt.legend(loc='best');
#plt.gca().grid()
#plt.draw()
#plt.show() 
'''
plt.plot([0.0005,0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1], rates, 'o-k', label = 'Avg. error rate across folds')
plt.fill_between([0.0005,0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1], rates - stds, rates + stds, color = 'steelblue', alpha = .4)
plt.xlabel('C (regularization parameter)')
plt.ylabel('Avg. error rate (and +/- 1 s.e.)')
plt.legend(loc = 'best')
plt.gca().grid()
plt.draw()
plt.show()
'''
#Estimate with best param

clf = LogisticRegression(C=clf.best_params_['C'], penalty='l1', tol=1e-6)
clf.fit(X,np.ravel(y))
y_test = clf.predict(data)
idColumn = np.arange(1,y_test.size+1,1).astype(np.integer)
np.savetxt('./csv/result.csv',np.c_[idColumn, y_test], fmt='%d', delimiter=',', header='id,Solution',comments='')

'''
scores = cv.cross_val_score(clf, X_test, y_test, cv=30)
print('Results for best parameter C: ')
print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

print "Accuracy score: %.4f" % accuracy_score(np.ravel(y_test),y_pred)
print "Score (mean accuracy): %4f" % clf.score(X_test, y_test)
print "Confusion matrix: " 
cm= confusion_matrix(np.ravel(y_test),y_pred)
print cm
print classification_report(y_test,y_pred)
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.draw()
'''