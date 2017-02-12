import numpy as np
import cv2
import argparse
import os.path
from sklearn.cluster import KMeans

def init_centroids(n=5):
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

def closest_centroid(sample, centroids):
    """
    Finds the closest centroid to a sample i.e one pixel.
    We use the classic euclidian distance here.
    This is the actual E-step of kmeans

    Args:
        sample(np.ndarray): Array representing one pixel
        centroids(np.ndarray): Array of centroids

    Returns:
        Index of the centroid closest to the sample
    """
    best_label = 0
    smallest_distance = 10000000000000
    centroids_shape = centroids.shape
    for l in range(0, centroids_shape[0]):
        current_distance = np.linalg.norm(sample - centroids[l,:])
        if current_distance < smallest_distance:
            smallest_distance = current_distance
            best_label = l
    return int(best_label)

def learn(sample, label, centroids, alpha):
    """
    Updates centroids. It will chose the centroid whose index is label and move it closer to the 
    sample. Actual M step of kmeans

    Args:
        sample(np.ndarray): An Array representing a pixel to get closer too
        label(int): Index of the centroid to update
        centroids(np.ndarray): Array of all the centroids
        alpha(float): Learning rate
    Returns:
        Updated centroids
    """
    centroids[label, :] = alpha*sample + (1-alpha)*centroids[label, :]
    return centroids

def kmeans(samples, centroids, alpha=.1, nIt=100):
    """
    Perform kmeans on an array of samples i.e pixel

    Args:
        samples: Reshaped array of an image to learn 
        centroids: Array of centroids
        alpha: Learning rate to keep usually closer to 0
        nIt: Number of iteration to perform kmeans 
    Returns: 
        labels associated to each pixel and final centroids
    """
    labels = np.zeros(samples.shape[0])
    for t in range(0, nIt):
        for k in range(0, samples.shape[0]):
            labels[k] = closest_centroid(samples[k,:], centroids)
            learn(samples[k,:], labels[k], centroids, alpha)
        print("{} percent done".format(int(100*t/nIt)))
    return labels, centroids

def compute_distorsion(original, result):
    """
    Compute distorsion i.e error being made each time a centroid is chosen
    instead of the real pixel
    """
    return np.linalg.norm(original - result)/(original.shape[0]*original.shape[1])

def codesamples(centroids, labels, fshape):
    """
    Output result of replacing each sample by its labelled centroid 
    Returns: 
        3D and 2D array of an image being compressed thanks to kmeans
    """
    result = np.zeros((labels.shape[0], 3))
    for k in range(labels.shape[0]):
        result[k,:] = centroids[labels[k], :]
    coded = np.reshape(result, fshape)
    return coded, result

def codesamples2(nClusters,samples2D, fshape):
    """
    Output result of replacing each sample by its labelled centroid 
    Returns: 
        3D and 2D array of an image being compressed thanks to kmeans
    """
    estimator = KMeans(n_clusters=nClusters).fit(im2)
    result = np.zeros(samples2D.shape)
    for k in range(samples2D.shape[0]):
        cluster = estimator.predict(samples2D[k,:].reshape(1,3))
        result[k,:] = estimator.cluster_centers_[cluster,:]
    coded = np.reshape(result, fshape)
    return coded, result

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
    help="path to the image")

ap.add_argument("-n", "--nClusters", required=True,
    help="path to the image")

args = vars(ap.parse_args())

im2, shape = read_image(args['path'])
centroids = init_centroids(int(args["nClusters"]))
# labels, _ = kmeans(im2, centroids)
# coded, result = codesamples(centroids, labels, shape)
coded, result = codesamples2(nClusters, im2, shape)
print("Total distorsion: {}".format(compute_distorsion(im2, result)))
extension = os.path.splitext(args["path"])[1]
cv2.imwrite(args["path"].replace(extension, "_kmeans"+extension), coded)
