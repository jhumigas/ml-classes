# ml-classes

Few examples taken from ML-courses from my major.
Most of them are coded in python.

Current content is about:
* Unsupervised learning
    * Vector Quantization (introduction)
    * Bayesian Learning with EM
    * Bayesian Linear Regression
    * Clustering on  Images
* Ensemble Methods
* Bayesian Learning
    * Naive Bayes   
* Reinforcement Learning
* SVMs

## Illustrations

* Clustering on Images (from [here](Clustering%20on%20Images/README.md))

![Eye Original](Clustering%20on%20Images/eye.png)
![Eye Full](Clustering%20on%20Images/images/eye_gmm_full_out_.png)
![Eye Diag](Clustering%20on%20Images/images/eye_gmm_diag_out_.png)
![Eye Tied](Clustering%20on%20Images/images/eye_gmm_tied_out_.png)
![Eye Tied](Clustering%20on%20Images/images/eye_kmeans.png)

>Here are a few illustrations on a small portion of an image found around the internet, with 3 clusters. We consider each pixel characterised by a vector(R,G,B) to be a sample. We want to learn a Gaussian Mixture Model on the set of all those samples. Read more [here](Clustering%20on%20Images/README.md)


* Reinforcement Learning  (from [here](Reinforcement%20Learning/code/README.md))

![fittedQ_val_policy](Reinforcement%20Learning/code/batch/figs/fittedQ/val_pol_iter_150.gif)

> On the left side is plotted the value function. Red is for high value whereas blue one is for lower one. Notice that a the beginning almost the value function assignes the same value to the whole state space. But as the iterations go one The value function learns which states will cost more to reach. Namely, as the higher the position and velocity are, the more value is associated. We can even somehow foresee what is the optimized trajectory (position and velocity) to follow: we just can start from any position and get to a nearby state that allows more value (i.e where the value function is high) On the right side is plotted the policy. Red corresponds to going right, blue to left and green to none. Basically, our learnt policy tells us to push left when the velocity is negativity and right on the contrary, this helps to acquire more momentum. We can notice however that when we are too far on the left, we'll likely throttle right, to avoid hitting the wall. (Read more [here](Reinforcement%20Learning/code/README.md))