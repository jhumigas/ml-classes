# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:20:40 2015

@author: geist_mat
matthieu.geist@centralesupelec.fr
"""

# import my stuf
import myTools as mt
reload(mt)

# import ploting stuf
import matplotlib.pyplot as plt

# import numpy
# import numpy as np

# import scikit-learn stuffs
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

plt.close("all")

nm = 0.25 # noise moons

# for statistics (risk)
step = 50 #from step datapoint to num*step datapoints, with a step of...step !
num = 21
nstats = 50 # number of time to repeat the exp for stats

m = 50 # number of base learners in the ensembles

# -----------------------------------------------------------------------------
# ----------------------------------------------------------DATASET (TWO MOONS)
# -----------------------------------------------------------------------------
print('-----====---- Datasets')

# first shows the dataset for a small number of samples
print("shows the dataset for a small number of samples (fig 1)")
[X, y] = make_moons(n_samples = 100, noise = nm)
mt.dataSole(X,y,'01_dataset_small.pdf')

# then shows the dataset for a large number of samples
print("shows the dataset for a large number of samples (fig 2)")
[X, y] = make_moons(n_samples = 3000, noise = nm)
mt.dataSole(X,y,'02_dataset_large.pdf')


# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------BAGGING
# -----------------------------------------------------------------------------
print('-----====---- Bagging')

# dpth = 15 # depth of the base learner

# approximate the ideal classifier with a deep tree trained with a large number of samples
n_train = 10000
n_test = 10000
n_plot = 1000
print("Train an approximate minimizer f0, with a fully grown tree trained on "+ str(n_train) + " samples (fig 3)")
[X_train, y_train] = make_moons(n_samples = n_train, noise = nm)
clf_0 = DecisionTreeClassifier()
clf_0.fit(X_train, y_train)
# approximate the risk
[X_test, y_test] = make_moons(n_samples = n_test, noise = nm)
risk_0 = mt.computeRisk(X_test, y_test, clf_0)
print("risk R0 :" + str(risk_0*100) + '%')
# show the decision boundaries
mt.dataBound(X_train[:n_plot,:], y_train[:n_plot], clf_0, '03_bagg_base_f_0.pdf')


# now, train with 100 datapoints only
n_train = 100
n_test = 10000
print("Train an approximate minimizer f_n, with a fully grown tree trained on "+ str(n_train) + " samples (fig 4)")
[X_train, y_train] = make_moons(n_samples = n_train, noise = nm)
clf_n = DecisionTreeClassifier()
clf_n.fit(X_train, y_train)
# approximate the risk
[X_test, y_test] = make_moons(n_samples = n_test, noise = nm)
risk_n = mt.computeRisk(X_test, y_test, clf_n)
print("risk Rn :" + str(risk_n*100) + '%')
# show the decision boundaries
mt.dataBound(X_train, y_train, clf_n, '04_bagg_base_f_n.pdf')

# shows how the risk evolve (mean and std)
print("Shows how the risk evolves (fig 5)")
clf = DecisionTreeClassifier()
[means1, stds1] = mt.statsRisk(clf, step, num, nstats, n_test, nm, '05_bagg_base_risk_evol.pdf')

# do the same thing for a bagged ensemble
print("Now train a bagging ensemble of fully grown trees trained on "+str(n_train)+" data points (fig 6)")
tree = DecisionTreeClassifier()
clf_b = BaggingClassifier(tree, n_estimators = m)
[X_train, y_train] = make_moons(n_samples = n_train, noise = nm)
clf_b.fit(X_train, y_train)
# approximate the risk
risk_b = mt.computeRisk(X_test, y_test, clf_b)
print("risk Rb :" + str(risk_b*100) + '%')
# show the decision boundaries
mt.dataBound(X_train, y_train, clf_b, '06_bagg_f_n.pdf')

# shows how the risk evolve

print("Shows how the risk evolves for bagging (fig 7a and b)")
[means2, stds2] = mt.statsRisk(clf_b, step, num, nstats, n_test, nm, '07a_bagg_risk_evol.pdf')
mt.compRisks(step, num, means1, stds1, means2, stds2, '07b_bagg_risk_comp.pdf')



# -----------------------------------------------------------------------------
# ---------------------------------------------------------------------BOOSTING
# -----------------------------------------------------------------------------
# Roughly the same story
print('-----====---- Boosting')

dpth = 1 # depth of the base learner

# approximate the ideal classifier with a shalow tree trained with a large number of samples
n_train = 10000
n_test = 10000
n_plot = 1000
print("Train an approximate minimizer f0, with a tree of depth "+str(dpth)+" trained on "+ str(n_train) + " samples (fig 8)")
[X_train, y_train] = make_moons(n_samples = n_train, noise = nm)
clf_0 = DecisionTreeClassifier(max_depth = dpth)
clf_0.fit(X_train, y_train)
# approximate the risk
[X_test, y_test] = make_moons(n_samples = n_test, noise = nm)
risk_0 = mt.computeRisk(X_test, y_test, clf_0)
print("risk R0 :" + str(risk_0*100) + '%')
# show the decision boundaries
mt.dataBound(X_train[:n_plot,:], y_train[:n_plot], clf_0, '08_boost_base_f_0.pdf')

# now, train with 100 datapoints only
n_train = 100
n_test = 10000
print("Train an approximate minimizer f_n, with a tree of depth "+str(dpth)+" trained on "+ str(n_train) + " samples (fig 9)")
[X_train, y_train] = make_moons(n_samples = n_train, noise = nm)
clf_n = DecisionTreeClassifier(max_depth = dpth)
clf_n.fit(X_train, y_train)
# approximate the risk
[X_test, y_test] = make_moons(n_samples = n_test, noise = nm)
risk_n = mt.computeRisk(X_test, y_test, clf_n)
print("risk Rn :" + str(risk_n*100) + '%')
# show the decision boundaries
mt.dataBound(X_train, y_train, clf_n, '09_boost_base_f_n.pdf')

# shows how the risk evolves (mean and std)
print("Shows how the risk evolves (fig 10)")
clf = DecisionTreeClassifier(max_depth = dpth)
[means1, stds1] = mt.statsRisk(clf, step, num, nstats,n_test, nm, '10_boost_base_risk_evol.pdf')

# do the same thing for a bagged ensemble
print("Now train a boosting ensemble of "+str(m)+" same trees trained on "+str(n_train)+" data points (fig 11)")
tree = DecisionTreeClassifier(max_depth = dpth)
clf_bb = AdaBoostClassifier(tree, n_estimators = m, algorithm = 'SAMME')
[X_train, y_train] = make_moons(n_samples = n_train, noise = nm)
clf_bb.fit(X_train, y_train)
# approximate the risk
risk_b = mt.computeRisk(X_test, y_test, clf_bb)
print("risk Rb :" + str(risk_b*100) + '%')
# show the decision boundaries
mt.dataBound(X_train, y_train, clf_bb, '11_boost_f_n.pdf')

# shows how the risk evolve

print("Shows how the risk evolves for boosting (fig 12a and b)")
[means2, stds2] = mt.statsRisk(clf_bb, step, num, nstats, n_test, nm, '12a_boost_base_risk_evol.pdf')
mt.compRisks(step, num, means1, stds1, means2, stds2, '12b_boost_risk_comp.pdf')