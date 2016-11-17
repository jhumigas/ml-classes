import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

####################
# Fonctions utiles #
####################

def drawInputsInSquare(n):
    """
    Generate n points in the square [0,1] X [0,1]
    as a matrix of size (n,2)
    """
    return np.random.random((n,2))

def addConstantInput(X):
    """
    This function returns a copy of X with an extra column full of ones
    """
    n,m = X.shape
    X2 = np.zeros((n,m+1))
    X2[:,:-1] = X
    X2[:,-1] = np.ones(n)
    return np.asmatrix(X2)
    
def drawOutput(X, theta, sigmae):
    """
    Generate outputs for inputs in matrix X
    using model provided by theta = [theta0, theta1, theta2]
    and a normal white noise of standard deviation sigmae
    """
    n = X.shape[0]
    theta = np.asmatrix(theta).reshape((3,1))
    Y = X * theta + np.random.normal(0,sigmae, size = (n,1))
    return Y

def drawModel(ax, theta, c):
    """
    Draw the plane of predictions for parameters theta
    """
    theta = np.asarray(theta)
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = theta[0]*X+theta[1]*Y+theta[2]
    return ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color = c)

def quadraticRisk(Y1,Y2):
    """
    Compute the quadratic risk between two outputvectors of same dimension
    """
    assert(Y1.shape == Y2.shape)
    n = Y1.shape[0]
    D = Y1 - Y2
    return 1./n * (D.T * D)[0,0]
    
def compareModels(realTheta, estTheta, X, realY):
    """
    Draw the estimated and real models (i.e. plane)
    - realTheta and estTheta are 3-arrays of type [theta0, theta1, theta2]
    - If inputs X and outputs realY are provided, real points and estimated points (for the same inputs) are also drawn
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    drawModel(ax, realTheta, 'blue')
    drawModel(ax, estTheta, 'red')
    if(not(X is None)):
       ax.scatter(np.asarray(X[:,0]), np.asarray(X[:,1]), np.asarray(realY), c = 'blue', marker = 'o')
       estY = X * np.asmatrix(np.reshape(estTheta, (3,1)))
       plt.suptitle("Quadratic Risk J = {:.4f}".format(quadraticRisk(realY, estY), ))
       ax.scatter(np.asarray(X[:,0]), np.asarray(X[:,1]), np.asarray(estY), c = 'red', marker = '+')

       X = np.asarray(X)
       n = X.shape[0]
       realY = np.asarray(realY).reshape(n)
       estY = np.asarray(estY).reshape(n) 
       for (x,yr,yp) in zip(X,realY,estY):
          ax.plot([x[0], x[0]], [x[1], x[1]], [yr, yp], label = 'red')
    
    blue_patch = mpatches.Patch(color='blue', label='Real')
    red_patch = mpatches.Patch(color='red', label='Estimation')
    plt.legend(handles=[red_patch, blue_patch])
    plt.draw()

    
def drawTrainAndTestEmpiricalRisk(nSamples, sigmas, Jtrain, Jtest):
    """
    Draws colormaps of square roots of two empirical risks, one computed on a train dataset, the other on a test dataset
    Risks must be computed for each couple of (number of samples, value of error standard deviation
    Vector nSamples contains the set of number of samples
    Vector sigmas contains the set of values of error standard deviation
    Sizes of matrices Jtrain and Jtest must be compatible with vectors nSamples and sigmas
      - Number of rows of Jtrain and Jtest must be equal to the length of sigmas
      - Number of columns of Jtrain and Jtest must be equal to the length of nSamples
    """
    XX,YY = np.meshgrid(nSamples,sigmas)
    fig, axarr = plt.subplots(2, sharex=True)
    maxJ = np.sqrt(np.mean(Jtrain)) * 10.
    plt.subplot(2, 1, 1)
    plt.pcolor(XX,YY,np.sqrt(Jtrain),vmin=0, vmax=maxJ)
    plt.colorbar()    
    plt.title('Square root of empirical risk of train data set')
    plt.subplot(2, 1, 2)
    plt.pcolor(XX,YY,np.sqrt(Jtest),vmin=0, vmax=maxJ)
    plt.colorbar()    
    plt.title('Square root of empirical risk of test data set')
    plt.draw()
    
def drawInputsAlmostAligned(n, alignmentFactor = 0.1):
    """
    Draws n points almost aligned on the first diagonal
    alignmentFactor is an alignment factor. alignmentFactor = 0 gives perfectly aligned points.
    """
    alpha = np.pi/4
    ca = np.cos(alpha); sa = np.sin(alpha)
    A = (np.random.random((n,2)) - 0.5)
    B = A * np.matrix([[1., 0.], [0., alignmentFactor]])
    X = B * np.matrix([[ca, sa], [-sa, ca]]) + np.array([0.5,0.5])
    return X

def plotFilterState(theta, sigmas):
    '''
    Plot the three estimated theta coefficients as function of time.
    - theta must be a matrix of size (3,n) where i-th column contains estimated parameters at i-th iteration.
    - sigmas must be a matrix of size (3,n) where i-th column contains estimated variances of each parameter at i-th iteration.
    Covariances contained in sigmas are used to draw confidence intervals of 95% around expected parameter values mu, 
    equal to [ mu - 2*sigma, mu + 2*sigma].
    '''
    fig = plt.figure()
    n = theta.shape[1]
    for i in range(3):
      plt.subplot(3,1,i+1)
      plt.plot(theta[i,:],'r')
      intervalTop = theta[i,:] + 2 * sigmas[i,:]
      intervalBottom = theta[i,:] - 2 * sigmas[i,:]
      plt.fill_between(range(n), intervalBottom, intervalTop, facecolor = 'lightgray')
      
      plt.plot(intervalTop,'--')
      plt.plot(intervalBottom,'--')
      plt.xlabel('$\\theta_{}$'.format(i))
      ymean = np.mean(theta[i,:])
      sigmaMean = np.median(sigmas[i,:])
      axes = plt.gca()
      axes.set_ylim([ymean - 4 * sigmaMean, ymean + 4 * sigmaMean])

####################
# Code à compléter #
####################

# Question 1.2

def evalLinearRegressor(X, y):
    """
    Computes OLS given inputs and outputs
    X the row vector in inputs 
    y the row vector of ouputs
    """
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def testSimpleOLS(n, sigmae):
    theta = np.arange(1,4,1)
    X = drawInputsInSquare(n)
    # Insert an offset 
    X = np.insert(X, 2, values=1, axis=1)
    y = drawOutput(X, theta, sigmae)
    estTheta = evalLinearRegressor(X,Y)
    #print(estTheta)
    compareModels(theta, estTheta, X, y)
    print('Question 1.2: Done')
    
# Question 1.3


def evaluateSimpleOLS():
    print('Question 1.3: Evaluate Simple OLS')
    realTheta = np.matrix([1,2,3]).T
    sigmaes = np.arange(2, 30, 0.5)
    nSamples = np.arange(3, 50, 1)
    # This is the matrix of the estimated empirical error
    JTrain = np.zeros((len(sigmaes), len(nSamples)))
    # This is the matrix of the real empirical error
    JTest = np.zeros((len(sigmaes), len(nSamples)))
    K = 20

    for k in range(K):
        XTest = drawInputsInSquare(100)
        XTest = addConstantInput(XTest)
        for i, sigmae in enumerate(sigmaes):
            YTest = drawOutput(XTest, realTheta, sigmae)
            for j, n in enumerate(nSamples):
                XTrain = drawInputsInSquare(n)
                XTrain = addConstantInput(XTrain)
                yTrain = drawOutput(XTrain, realTheta, sigmaes[k])
                theta = evalLinearRegressor(XTrain, yTrain)
                JTrain[i,j] = quadraticRisk(yTrain, XTrain*theta) 
                JTest[i,j] = quadraticRisk(YTest, XTest* theta)
    JTrain /= K
    JTest /= K
    drawTrainAndTestEmpiricalRisk(nSamples, sigmaes, JTrain, JTest)
    
# Question 1.5           

def testSimpleOLSWithAlignedPoints(n, sigmae, alignmentFactor=0.1):
    print('Question 1.5: Simple OLS with almost aligned input points')
    theta = np.arange(1,4,1)
    X = drawInputsAlmostAligned(n, alignmentFactor)
    # Insert an offset 
    X = addConstantInput(X)
    y = drawOutput(X, theta, sigmae)
    estTheta = evalLinearRegressor(X,y)
    #print(estTheta)
    compareModels(theta, estTheta, X, y)


# Question 2.1
def evalRegularizedLinearRegressor(lambdaFactor, X, y):
    n = X.shape[1]
    theta = (X.T * X + lambdaFactor * np.eye(n)).I * X.T * y
    return theta
 
def testRidgeRegression(n, sigmae, alignmentFactor, lambdaFactor):
    theta = np.arange(1,4,1)
    X = drawInputsAlmostAligned(n, alignmentFactor = 0.1)
    X = addConstantInput(X)
    y = drawOutput(X, theta, sigmae)
    estTheta = evalRegularizedLinearRegressor(lambdaFactor, X, y)
    compareModels(theta, estTheta, X, y)

    print('Ridge regression with almost aligned input points')

# Question 3.14

class Filter:
    def __init__(self, sigma0):
       '''
       Constructor of a Kalman fitler for recursive least square
       - self is a reference to the current object ("self" in Python is like "this" in Java)
       - Each state component is supposed to be known with an uncertainty given by the same initial standard deviation sigma0.
       '''
       self.state = 0
       self.P = 0
       
    def update(self, y, x, sigmaY):
       '''
       Function called to update the filter state when a new observation (x,y) is received.
       - self is a reference to the current object ("self" in Python is like "this" in Java)
       - sigmaY is the standard deviation of the observation noise.
       '''
       self.state = 0
       self.P = 0

    def integrate(self, Q):
       '''
       Function called to integrate the filter state every second.
       - self is a reference to the current object ("self" in Python is like "this" in Java)
       - Q is a square matrix containing the state noise integrated over one second
       '''
       self.state = 0
       self.P = 0
    
def testRecursiveLeastSquare(n, sigmae):
    print('Recursive Least Square')
    
    realTheta = np.matrix([1,2,3]).T

# Question 3.15

def testKalman(n, sigmae, frequency):
    print('Kalman filter with integration')
    
    realTheta = np.matrix([1,2,3]).T
  

########
# Main #
########

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default = 1,
	            help="procedure to run")
    parser.add_argument("--n", type=int, default = 100,
	            help="number of generated samples")
    parser.add_argument("--sig", type=float, default = 1.,
	            help="standard deviation of error")
    parser.add_argument("--alignment", type=float, default = 0.1,
	            help="alignment factor")
    parser.add_argument("--lambdaFactor", type=float, default = 1.,
	            help="lambda factor")
    parser.add_argument("--frequency", type=float, default = 1.,
	            help="frequencyof observations")
    args = parser.parse_args()
    
    if(args.test == 1):
        testSimpleOLS(args.n, args.sig)
    elif(args.test == 2):
        evaluateSimpleOLS()
    elif(args.test == 3):
        testSimpleOLSWithAlignedPoints(args.n, args.sig, args.alignment)
    elif(args.test == 4):
        testRidgeRegression(args.n, args.sig, args.alignment, args.lambdaFactor)
    elif(args.test == 5):
        testRecursiveLeastSquare(args.n, args.sig)
    elif(args.test == 6):
        testKalman(args.n, args.sig, args.frequency)
    plt.show()
if __name__ == "__main__":
    main()
    
