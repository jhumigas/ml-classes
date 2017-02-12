import numpy as np
import cv2
import argparse
import os.path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

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
def em(data, nc=3, nIt=10, epsilon=3000):
    """
    EM algorithm
    """
    m, n = data.shape
    # q s.t qi = q[i,:] ~ P(Hi|Vi, theta)
    qLog = np.ones((nc, m))/nc
    q = np.random.randint(9, size=((nc, m))).astype(float)
    for k in range(m):
        q[:, k] = q[:, k]/np.sum(q[:, k])

    # ph = P(H), mixture coefficients
    # ph = np.ones((nc,1))/nc
    ph = np.random.randint(10, size=(nc,1)).astype(float)
    ph = ph/ np.sum(ph, axis=0)

    # pvh = P(V|H,theta), modelled by Gaussian distributions
    means_k = KMeans(n_clusters=nc).fit(data).cluster_centers_
    centroids_means = means_k
    centroids_variances = np.random.randint(45*45, high=85*85,  size=(nc*n, nc*n)).astype(float)


    # Threshold
    delta_ = epsilon*10
    while(delta_ > epsilon):
    # for iteration in range(nIt):
        # Expectation step, we keep pvh parameter and distribution ph constant
        # Minimization of the entropy
        # q ~ P(H|V) = K*P(V|H) * P(H)
        for s in range(m):
            for c1 in range(nc):
                delta_x = data[s,:] - centroids_means[c1,:]
                delta_x = delta_x.reshape(n,1)
                det_variance = np.linalg.det(centroids_variances[c1*n:(c1+1)*n, c1*n:(c1+1)*n])
                if det_variance < 0.1:
                    centroids_variances[c1*n:(c1+1)*n, c1*n:(c1+1)*n] += np.random.randint(10,54,size=(n,n))
                    det_variance = np.linalg.det(centroids_variances[c1*n:(c1+1)*n, c1*n:(c1+1)*n])
                inv_variance = np.linalg.inv(centroids_variances[c1*n:(c1+1)*n, c1*n:(c1+1)*n])
                exp_arg = delta_x.T.dot(inv_variance).dot(delta_x)
                qLog[c1, s] = exp_arg - 0.5* np.log(det_variance) + np.log(ph[c1,0])

            # Normalize q 
            # print(np.sum(q[:,s]))
            # qLog[:, s] -= np.log(np.sum(q[:,s]))
            q[:,s] = np.exp(qLog[:,s])
            q[np.isnan(q[:,s])] = np.random.rand(1)[0]
            q[q[:,s]>1] = np.random.rand(1)[0]
        
        q = q/np.sum(q, axis=0)
              
        
        # Maximisation step, we keep q constant
        # Very similar to 
        delta_ = 0
        for c in range(nc):
            # Update hidden variable distribution
            ph[c,0] = np.sum(q[c,:])/m
            # Store the current variances and means
            variance_old = np.copy(centroids_variances[c*n:(c+1)*n, c*n:(c+1)*n])
            mean_old = np.copy(centroids_means[c,:])
            
            # Update mean
            # Each sample is weighted by the cluster distribution
            # to estimate the mean
            temp_mean = np.zeros((1,n))
            for k in range(m):
                temp_mean +=q[c,k]*data[k,:]
            centroids_means[c, :] = temp_mean/(m*ph[c,0])
            

            # Update variance
            # Look like the normal variance calcul
            # But each term is weighted by the cluster distribution
            temp = np.zeros((n,n))
            for k in range(m):
                delta_x_k = data[k,:] - centroids_means[c,:]
                # delta_x_k = delta_x_k.reshape(3,1)
                temp += q[c,k]*delta_x_k.dot(delta_x_k.T)
            # temp[temp>255*255*m*ph[c,0]] = np.random.randint(200*200,255*255, size=(temp[temp>255*255*m*ph[c,0]].shape))
            temp[temp>255*255*m*ph[c,0]] = variance_old[temp>255*255*m*ph[c,0]] + np.random.randint(10,50, size=(temp[temp>255*255*m*ph[c,0]].shape))
            centroids_variances[c*n:(c+1)*n,c*n:(c+1)*n] = temp/(m*ph[c,0])

            # Stopping criterion
            # print(centroids_variances[:n,:n])
            delta_var = centroids_variances[c*n:(c+1)*n,c*n:(c+1)*n] - variance_old
            delta_mean = centroids_means[c,:]- mean_old
            delta_ += np.abs(np.trace(delta_var) + np.sum(delta_mean))
            print("Delta value: {}".format(delta_))
    return(ph, centroids_means, centroids_variances)

def codesamples(estimator, samples2D, fshape):
    """
    """
    result = np.zeros(samples2D.shape)
    for k in range(samples2D.shape[0]):
        cluster = estimator.predict(samples2D[k,:].reshape(1,3))
        result[k,:] = estimator.means_[cluster,:]
    coded = np.reshape(result, fshape)
    return coded, result




ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
    help="path to the image")

ap.add_argument("-n", "--nClusters", required=True,
    help="path to the image")

args = vars(ap.parse_args())

im2, shape = read_image(args['path'])
# print(shape)
centroids = init_centroids(int(args["nClusters"]))
(mix_coef, model_mean, model_var) = em(im2[:,:], nc=int(args['nClusters']))
print("Mixture coefficients =\n {}\n  Model variances = \n{}\n  Model means=\n{}\n".format(mix_coef, model_var, model_mean))
# estimator = GaussianMixture(int(args["nClusters"]), covariance_type='tied')
# estimator.fit(im2)
# print("Mixture coefficients =\n {}\n  Model means = \n{}\n  Model variances=\n{}\n".format(estimator.weights_, estimator.means_ ,estimator.covariances_))
# coded, result = codesamples(estimator, im2, shape)
# print("Total distorsion: {}".format(compute_distorsion(im2, result)))
# extension = os.path.splitext(args["path"])[1]
# cv2.imwrite(args["path"].replace(extension, "_gmm_tied_out_"+extension), coded)


