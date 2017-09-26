"""Perform Clustering on Image using Mixture Model

We use Edward to perform inference
"""
import cv2
import argparse
import os.path
import edward as ed
import tensorflow as tf
import numpy as np

def init_centroids(n=3):
    """
    Initializes random centroids 
    
    Args:
        n(int): Number of centroids to initialize
    Returns:
        An Array with n centroids, each row being one centroid
        
    """
    return np.random.randint(256, size =(n,3))

def read_image(path):
    """
    Reads an image and output a 2D array

    Args: 
        path(str): Path to the image to read
    returns:
        An array, resulting of the reshaping of the initial image array
        Shape of the image, expect 3D
    """
    im = cv2.imread(path)
    original_shape = im.shape
    im2 = np.reshape(im, (original_shape[0]*original_shape[1], original_shape[2]))
    return im2, original_shape

# Consider independence between V1, V2, V3
# Consider same covariance matric between Vs
# Consider diagonal covariance matrice for Vs
# Consider single variance matrices
def em(data, nc=3, nIt=4, epsilon=100):
    """
    EM algorithm
    We added the following constraint: The covariance matrix are diagonal

    Args:
        data: Data on which to fit the model
        nc: Number of clusters
        nIt : Number of iterations
        epsilon : Convergence parameter (not activated here)
    
    Returns:
        clusters means and covariances matrices, parameters distribution
    """
    # Dimensionnality of the data
    D = data.shape[1]

    #
    pass

def codesamples(centroids_means, centroids_variances, samples2D, fshape):
    """
    Recode samples based on the model fitted on them
    Adapted to ouptut of function em. 

    Args:
        centroids_means: 3D vectors
        centroids_variances: 3*3 matrices
        samples2D: N*3 array of samples2D
        fshape : Original shapes of the samples array 
    
    Returns:
        coded: Image resulting from the clustering
        result: Samples clustered (in a N*3 shape)
    """
    distributions = []
    nClusters, m = centroids_means.shape
    for c in range(0, nClusters):
        distributions.append(multivariate_normal(centroids_means[c,:], centroids_variances[c*m:(c+1)*m,c*m:(c+1)*m]))
    
    result = np.zeros(samples2D.shape)
    for k in range(samples2D.shape[0]):
        c_label = 0
        for c in range(0, len(distributions)):
            if distributions[c_label].pdf(samples2D[k,:]) < distributions[c].pdf(samples2D[k,:]):
                result[k,:] = centroids_means[c,:]
                c_label = c
    
    coded = np.reshape(result, fshape)
    return coded

def codesamples2(nClusters, samples2D, fshape):
    """
    Recode samples based on the model fitted on them
    Uses sklearn GMM.

    Args:
        centroids_means: 3D vectors
        centroids_variances: 3*3 matrices
        samples2D: N*3 array of samples2D
        fshape : Original shapes of the samples array 
    
    Returns:
        coded: Image resulting from the clustering
        result: Samples clustered (in a N*3 shape)
    """
    # estimator = GaussianMixture(nClusters, covariance_type='diag')
    # estimator.fit(samples2D)
    # result = np.zeros(samples2D.shape)
    # for k in range(samples2D.shape[0]):
    #     cluster = estimator.predict(samples2D[k,:].reshape(1,3))
    #     result[k,:] = estimator.means_[cluster,:]
    # coded = np.reshape(result, fshape)
    # return coded




ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
    help="path to the image")

ap.add_argument("-n", "--nClusters", required=True,
    help="path to the image")

args = vars(ap.parse_args())

im2, shape = read_image(args['path'])
# print(shape)
# centroids = init_centroids(int(args["nClusters"]))
# (mix_coef, model_mean, model_var) = em(im2[:,:], nc=int(args['nClusters']))
# print("Mixture coefficients =\n {}\n  Model variances = \n{}\n  Model means=\n{}\n".format(mix_coef, model_var, model_mean))
# coded= codesamples(model_mean, model_var, im2, shape)
coded = codesamples2(int(args["nClusters"]), im2, shape)
# print("Mixture coefficients =\n {}\n  Model means = \n{}\n  Model variances=\n{}\n".format(estimator.weights_, estimator.means_ ,estimator.covariances_))
# print("Total distorsion: {}".format(compute_distorsion(im2, np.reshape(coded, (coded.shape[0]*coded.shape[1], coded.shape[2])))))
extension = os.path.splitext(args["path"])[1]
cv2.imwrite(args["path"].replace(extension, "_gmm_tied_out_"+extension), coded)


