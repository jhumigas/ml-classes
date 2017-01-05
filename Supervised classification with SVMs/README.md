# Supervised Learning with SVMs

Initial subject can be found [here](http://www.metz.supelec.fr//metz/personnel/frezza/ApprentissageNumerique/TP-MachineLearning/SupervisePy.html).
The goal is to classify handwritten digits thanks to a SVM method. We use the scikit-learn library which those supervised learning methods, briefly explained on [scikit-learn website](http://scikit-learn.org/stable/modules/svm.html).
Perhaps the biggest advantages of SVMs is their efficiency in high dimensional spaces, theoritically explained by the margins that we try to maximize. As a matter of fact, SVMs are about maximizing margins around the to-be-learnt classifier.

The generator-examples.py is a set of examples on python3 lazy generators. They are needed because usual problems involves massive datasets. Lazy generators provide memory management efficiency.
To see how to load and visualize the dataset, simple refer to scikit-digits.py.
scikitsvm.py is a complete example of a linear SVM  applied to the dataset.
Since the parameters can be numerous, one idea is to perform a gridsearch i.e testing a couples of parameters place in 2D-graph and selecting the couple s.t the overall score is the biggest