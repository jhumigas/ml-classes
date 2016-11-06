# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:50:08 2015

@author: geist_mat
matthieu.geist@centralesupelec.fr

strongly inspired by the example "face_recognition.py" of scikit-learn:
http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html

WARNING : runing the whole script takes some time (normally, less then 15 min
wihtout the commented part, more than 1 hour with the grid search)
"""

from __future__ import print_function

import numpy as np

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA, NMF, FastICA, FactorAnalysis
from sklearn.grid_search import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

import sys

plt.close("all")

###############################################################################
# Some useful functions
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


###############################################################################
# Define the classifiers
nb = 500 # number of base learners
nds = 12 # max number of nodes (for boosted trees)

name_clfs = ['Tree (deep)', 'Tree (shallow)', 'Bagging', 'Random Forest', 'ExtraTree', 'AdaBoost']

classifiers = [
    DecisionTreeClassifier(
        class_weight="auto",
        min_samples_leaf=5),
    DecisionTreeClassifier(
        max_leaf_nodes  = nds,
        class_weight = 'auto'),
    BaggingClassifier(
        base_estimator = DecisionTreeClassifier(class_weight = 'auto', min_samples_leaf=5),
        n_estimators = nb),
    RandomForestClassifier(
        n_estimators = nb,
        class_weight ='subsample',
        min_samples_leaf=5),
    ExtraTreesClassifier(
        n_estimators = nb,
        class_weight ='subsample',
        min_samples_leaf=5),
    AdaBoostClassifier(
        DecisionTreeClassifier(max_leaf_nodes  = nds, splitter = 'random', class_weight="auto"),
        n_estimators = 2*nb,
        algorithm = 'SAMME')
    ]
  
print("======================================================================")
print("                          No preprocessing                            ")
print("======================================================================")
    
for name, clf in zip(name_clfs, classifiers):
    print(" ")
    print("=====================")
    print(name)
    print("=====================")
    # train
    print("Fitting the classifier to the training set")
    t0 = time()
    clf.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    # quantitative result
    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test)
    print("done in %0.3fs" % (time() - t0))
    print("Risk on test set:"+str(1./len(y_test)*np.sum(1-(y_test==y_pred))))
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    sys.stdout.flush()


print("======================================================================")
print("                          preproc: eigenfaces                         ")
print("======================================================================")
###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten = True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

for name, clf in zip(name_clfs, classifiers):
    print(" ")
    print("=====================")
    print(name)
    print("=====================")
    # train
    print("Fitting the classifier to the training set")
    t0 = time()
    clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    # quantitative result
    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))
    print("Risk on test set:"+str(1./len(y_test)*np.sum(1-(y_test==y_pred))))
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    sys.stdout.flush()


"""
print("======================================================================")
print("               preproc: eigenfaces, xTree CV GS                       ")
print("======================================================================")
# grid search cross validation for finding a good set of parameters for extra-trees
# warning: quite slow
###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 50

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'criterion': ['gini', 'entropy'],
              'bootstrap': [True, False],
              'class_weight': ['auto', 'subsample'],
            'min_samples_leaf': [1, 5, 10],
            'n_estimators': [100, 300, 500]}
clf = GridSearchCV(RandomForestClassifier(), param_grid, verbose=2)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
"""

"""
print("======================================================================")
print("               preproc: eigenfaces, AdaB CV GS                       ")
print("======================================================================")
# grid search cross validation for finding a good set of parameters for boosting
# warning: quite slow
###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 50

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = RandomizedPCA(n_components=n_components).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'algorithm': ['SAMME', 'SAMME.R'],
              'learning_rate': [.5, 1., 1.5],
              'n_estimators': [100, 500, 1000],
              'base_estimator__max_leaf_nodes': [6, 12, 18],
              'base_estimator__class_weight': ['auto', None]}
clf = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier()), param_grid, verbose=2)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
"""