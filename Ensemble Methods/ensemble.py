import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def popo(ntraining, ntest, noise=0.25):
    dataset = make_moons(n_samples=ntraining, noise=noise)
    X=dataset[0]
    y=dataset[1]

    plt.subplot(1,2,1)
    plt.scatter(dataset[0][:,0], dataset[0][:,1], c=dataset[1])

    plt.subplot(1,2,2)  

    #DT Classifier
    classifier= DecisionTreeClassifier().fit(dataset[0], dataset[1])


    # Parameters
    n_classes = 2
    plot_step = 0.02

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z)

    # Plot the training points
    #for i in range(n_classes)):
     #   idx = np.where(y == i)
     #   plt.scatter(X[:, 0], X[:, 1], c=dataset[1])
    #plt.scatter(X[:, 0], X[:, 1], c=dataset[1])

    #plt.show()

    testdataset = make_moons(n_samples=ntest ,noise=noise)

    y_test = classifier.predict(testdataset[0])


    estim=np.sum(testdataset[1]==y_test)
    #print(estim)
    #print(float(estim)/ntest)
    
    plt.scatter(testdataset[0][:,0], testdataset[0][:,1], c=testdataset[1])
    #plt.show()
    return float(estim)/ntest

def bagging(nmin, nmax, ntest):

    risks = []
    for i in range(nmin, nmax, 50):
        risks.append([i,popo(i, ntest)])

    return risks
def ensemble(nbase, nlearnsize, ntest):

    risks = []
    for i in range(1,nbase):
        risks.append(popo(nlearnsize, ntest))

    return risks


def pupu(ntraining, ntest, noise=0.25, depth=1):
    dataset = make_moons(n_samples=ntraining, noise=noise)
    X=dataset[0]
    y=dataset[1]

    plt.subplot(1,2,1)
    plt.scatter(dataset[0][:,0], dataset[0][:,1], c=dataset[1])

    plt.subplot(1,2,2)  

    #DT Classifier
    classifier= DecisionTreeClassifier(max_depth=depth).fit(dataset[0], dataset[1])


    # Parameters
    n_classes = 2
    plot_step = 0.02

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z)

    # Plot the training points
    #for i in range(n_classes)):
     #   idx = np.where(y == i)
     #   plt.scatter(X[:, 0], X[:, 1], c=dataset[1])
    plt.scatter(X[:, 0], X[:, 1], c=dataset[1])

    #plt.show()

    testdataset = make_moons(n_samples=ntest ,noise=noise)

    y_test = classifier.predict(testdataset[0])


    estim=np.sum(testdataset[1]==y_test)
    #print(estim)
    #print(float(estim)/ntest)
    
    plt.scatter(testdataset[0][:,0], testdataset[0][:,1], c=testdataset[1])
    #plt.show()
    return float(estim)/ntest

def baselearn(ntrain,ntest,niter):
    #pupu(ntrain,ntest)
    #plt.show()

    risks = []
    for i in range(1,niter):
        risks.append(pupu(ntrain, ntest))

   # X=[x[0] for x in risks]
   # Y=[x[1] for x in risks]
   # plt.scatter(X,Y)
    # plt.clf()
    plt.close()
    plt.plot(risks)
    plt.show()
   


