import matplotlib.pyplot as plt # for plotting.
import itertools as it          # for smart lazy iterations.
import numpy as np              # for fast array manipulations.
import pickle                   # for dumping python object instances.

# This is scikit modules
import sklearn.datasets 
import sklearn.utils
import sklearn.svm

# Let us load the dataset.
iris           = sklearn.datasets.load_iris()
inputs, labels = sklearn.utils.shuffle(iris.data, iris.target)

# Let us apply a linear C-SVM (see the documentation for parameters)
# ovo means "one vs one"
print()
print('learning...')
c_ovo = sklearn.svm.SVC(C=10, kernel='rbf', gamma=.1, tol=1e-6, decision_function_shape='ovo')
p_ovo = c_ovo.fit(inputs, labels)
print('done')

# Let us now retrieve the inner classifiers. As the coefficients are
# organized in a tricky way, let us define utilities for that purpose.

def ovo_name(class1, class2):
    return '{} vs {}'.format(class1,class2)

def ovo_names(nb_classes):
    classes = range(nb_classes)
    for (idx,c1) in enumerate(classes):
        for c2 in classes[idx+1:] :
            yield ovo_name(c1,c2)
    raise StopIteration()

def ovo_other_names(the_class, nb_classes):
    for c in range(nb_classes) :
        if c < the_class:
            yield ovo_name(c, the_class)
        elif c > the_class:
            yield ovo_name(the_class, c)
    raise StopIteration()

print()
print('4-class ovo names : {}'.format(   [s for s in ovo_names      (   4)] ))
print('class 2 classifiers : {}'.format( [s for s in ovo_other_names(2, 4)] ))

# Lets us see which vectors in the input is actually used as a support.
print()
print('Support vectors')
for (idx,support) in zip(p_ovo.support_,p_ovo.support_vectors_) :
    print('  input #{} : {} == {}'.format(idx,support,inputs[idx]))

# Let us collect (support_idx, alpha) for each classifiers in a
# dictionary, indexed by the classifier name. This is quite
# intricated...
# Each dictionary entry is :
#   key : classifier name (e.g '1 vs 2')
#   value : a pair :
#     first : the offset (intersect)
#     second : a list of pairs (support_idx, coef)
def ovo_separators(classifier):
    nb_classes  = len(classifier.classes_)
    seps        = dict()                               # This is the result
    offsets     = (offset for offset in classifier.intercept_)
    for name in ovo_names(nb_classes) :                # we init the dictionary content as empty lists.
        seps[name] = [next(offsets),[]]
    nb_supports = (n for n in classifier.n_support_)   # nb_sup_class1, nb_sup_class2, ...
    support_idx = (idx for idx in classifier.support_) # 1,4,18,2,.... the ranks of the supports in the input set
    alphas_y_s  = (coefs for coefs in classifier.dual_coef_.transpose()) # read the docs, it is intricated.
    for the_class in range(nb_classes) :                  # for each class
        for sup_rank in range(next(nb_supports)) :        # for each support for that class
            sup_idx = next(support_idx)                   # The idx of the support vector concerned by the coefs
            coefs   = (coef for coef in next(alphas_y_s)) # the coefs for each classifier for that support.
            for name in ovo_other_names(the_class, nb_classes) :
                coef = next(coefs)
                if coef != 0 : seps[name][1].append((sup_idx,coef))         
    return seps

print()
print('Separators')
print()
seps = ovo_separators(p_ovo)
print(seps)
print()
for sep_name in ovo_names(len(p_ovo.classes_)):
    print('Separator "{}" has {} support vectors.'.format(sep_name, len(seps[sep_name][1])))
print()


        
    





