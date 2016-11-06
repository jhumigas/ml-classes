# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:22:18 2015

@author: geist_mat
matthieu.geist@centralesupelec.fr

used for be_ensembles_part1.py
"""

import numpy as np

# import ploting stuffs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rc

import sys 

# import sklearn stuffs
from sklearn.datasets import make_moons

# define the color maps
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
# for TeX in labels
rc('text', usetex = True)
# step size in the mesh
h = .02  

def dataSole(X, y, name):
    # plot the data in X, y, save the fig as name
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0], X[:, 1], c = y, cmap=cm_bright)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    fig.savefig(name)
    
def computeRisk(X_test, y_test, clf):
    # compute the empirical risk of clf on X_test, y_test
    y_pred = clf.predict(X_test)
    return 1./len(y_test)*np.sum(1 - (y_pred == y_test))
    
def dataBound(X, y, clf, name):
    # plot the decision boundaries of clf, as well as training points X,y, save as name
    xx1, xx2 = np.meshgrid(np.arange(-1.5, 2.5, h),  np.arange(-1, 1.5, h))
    Z = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(xx1, xx2, Z, cmap=cm, alpha=.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    fig.savefig(name)
    
def statsRisk(clf, step, num, nstats, ntest, nm, name):
    # compute statistic on risk evolution of clf (resp to ntest ind. sampled points), 
    # for step:step:num*step training points
    # stats done on n_stats
    # nm is the noise for the moon problem
    # saved on name
    n_try = np.arange(nstats)
    n_train = np.arange(step,num*step,step)
    stats = np.zeros((nstats,len(n_train)))
    for t in n_try:
        print "\rStats %i " % t,
        sys.stdout.flush()
        for n in n_train:
            [X_train, y_train] = make_moons(n_samples = n, noise = nm)
            [X_test, y_test] = make_moons(n_samples = ntest, noise = nm)
            clf.fit(X_train, y_train)
            risk = computeRisk(X_test, y_test, clf)
            stats[t, n/step-1] = risk
    print(' ')
    means = np.mean(stats,0)
    stds = np.std(stats,0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(n_train, means)
    ax.fill_between(n_train, means-stds, means+stds, alpha=0.5)
    ax.set_xlabel('number of learning points')
    ax.set_ylabel('estimated risk')
    fig.savefig(name)
    return means, stds
    
def compRisks(step, num, means1, stds1, means2, stds2, name):
    # plot two risks (mean +/- std as a function of n) on the same fig
    n_train = np.arange(step,num*step,step)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(n_train, means1, color ='b')
    ax.fill_between(n_train, means1-stds1, means1+stds1, facecolor = 'b', alpha=0.5)
    ax.plot(n_train, means2, color ='r')
    ax.fill_between(n_train, means2-stds2, means2+stds2, facecolor = 'r', alpha=0.5)
    ax.set_xlabel('number of learning points')
    ax.set_ylabel('estimated risk')
    fig.savefig(name)