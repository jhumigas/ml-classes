# Naive Bayes 

Here we implement Naive Bayes from scratch. 
Naive Bayes is often in classification problems. It is a method used to model a classification problem probalistically.
In every supervised learning problem, we are provided with inputs X and outputs y. In fact, we consider that there is an underlying processing label each input xi by an output yi.
Such a process is called oracle, and is unknown. Machine Learning has at its core the objective of connecting a model of the oracle to a set of data (X,y) provided. We usually consider a hypothesis space where lives hypothesis functions s.t f(x) = y.

Contrary to frequentist approach where the so called hypothesis function are parametrized, in bayesian learning we keep updating a distribution P(theta|x,y) conditionnaly to the set of available samples (xi,yi).
As the samples are received, some hypothesis get more likely than others. Naive Bayes consider samples to be independently distributed. This rather strong assumption explains the name 'naive' but result in simple factorization of the generic distribution.
tl; dr: “Naive Bayes” assumes the feature variables are independent of each other given the target class Y : Xi ind Xj | Y.

## Dataset 

https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

> This problem is comprised of 768 observations of medical details for Pima indians patents. The records describe instantaneous measurements taken from the patient such as their age, the number of times pregnant and blood workup. All patients are women aged 21 or older. All attributes are numeric, > and their units vary from attribute to attribute.

> Each record has a class value that indicates whether the patient suffered an onset of diabetes within 5 years of when the measurements were taken (1) or not (0).

## Pipeline

Following are the steps followed to apply Naive Bayes on Pima indians patents data : 

* Handle data : Python packages like csv or even pandas work very well. We used them in a numpy array form for computation purposes 
* Split set into training and test samples 
* Estimate  distributions
* Compute predictions on test set 
* Evaluate accuracy (or risk) on test set

The overall objective is to maximize the likelihood P(y|X) this using the Maximum Likelihood Estimator (MLE) principle.
To do so, first using Bayes rules we know that P(y|X) = P(X|y) * P(y) / P(X) at each step.
We assume X to be continious variable and y to be a discrete one. Here we suppose that P(X|y) is a normal distribution and P(y) a uniform distribution.
With these assumptions, maximising the likelihood P(y|X) is indeed equivalent to taking P(X|y) ~ N(mean(Xj), var(Xj)), with Xj the set of attribute for feature j.
One just estimates the distribution P(X|y), for each attribute per class. 
In the prediction step, we keep the biggest class probability.

One can try different distributions, as we used here Normal distribution, on can use Bernouilli distribution, Poisson distribution, but this would need further assumptions about the data, and adapting the model.
For a stronger bayesian version, one can consider the parameter modeled by a non uniform distribution, which would required the MAP estimator.
As a reminder, the MLE is directly derived from the Maximum A Priori Estimator, where we suppose the priori distribution to be uniform.

## Sources

I mostly looked on the following link to see the actual work pipe. 
http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

For the theory, I was applying my ML courses.