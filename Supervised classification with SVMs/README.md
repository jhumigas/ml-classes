# Supervised Learning with SVMs

Initial subject can be found [here](http://www.metz.supelec.fr//metz/personnel/frezza/ApprentissageNumerique/TP-MachineLearning/SupervisePy.html).
The goal is to classify handwritten digits thanks to a SVM method. We use the scikit-learn library which implements those supervised learning methods, briefly explained on [scikit-learn website](http://scikit-learn.org/stable/modules/svm.html).
Perhaps the biggest advantage of SVMs is their efficiency in high dimensional spaces, theoritically explained by the margins that we try to maximize when we are building a classifier.

## Overview

Before talking about what we did in the lab, one note about python generators.
The generator-examples.py is a set of examples about lazy generators. They are needed because usual problems involves massive datasets. Lazy generators provide memory management efficiency, as their generate data on the fly.
To see how to load and visualize the dataset, simple refer to scikit-digits.py.
scikitsvm.py is a complete example of a linear SVM  applied to the dataset.

In few steps, here are the steps : 
* Load dataset : We extract only a subset of the dataset, s.t the datasize is less than 10000. Otherwise the computation might take too much time.
* Design the predictor : Choice of what type of SVM (linear, nu-svm, etc) and the parameter
* Learn the predictor 
* Analyse the efficiency of the predictor : compute the scores and deduce the risk
* Cross-validate
* Plot a sample (with original and predicted labels)

To get more insight about the classifier efficiency, one can generate a classificaton report and computing the confusion matrix.

Since the SVM parameters can be numerous, one can perform a gridsearch i.e testing a couple of parameters placed in a grid and selecting the values of the parameters s.t the overall score is maximized.

## Requirements

* matplotlib : To visualize the samples
* numpy : Tool to perform scientific operations
* pickle : To save the predictors
* sklearn : Machine Learning library