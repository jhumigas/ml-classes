# Kmeans and GMM

We consider a simple showcase of clustering on images:

* Kmeans clustering, a vector quantization technique
* Gaussian Mixture Model i.e the EM algorithm applied ot model with latent variables


## Bayesian Theoritical design 

We denote by H and V respectively our hidden and visible variables.  Θ stands for the hyperparameters for all the distributions.

A modelisation with plate notation would look like this : 

```

           ---------  
          |    H    | 
          |    ↓    | 
 Θ ->     |  -----  |
          | |  V  | | 
          | |     | | 
          |  -----  | 
          |         |
           --------- 
```


### Image Mixture Model Design

We consider each pixel characterised by a vector(R,G,B) to be a sample. We want to learn a Gaussian Mixture Model on the set of all those samples.

We make no further assumption other than the one needed to build a GMM : we suppose the visible variables follow a Multivariate Gaussian distributions.
For each cluster, the corresponding distribution is characherized by a `3*1` mean vector and a `3*3`covariance matrice. 

## EM Algorithm

We assume q, a distribution that approximated H|V,Θ. The overall goal of the EM algorithm is to actually maximize a lower bound of the likelihood P(H,V|Θ), called here L*, thus reducing the divergence between q and H|V,Θ
EM involves two main steps : 

* Expectation Step : In this step we maximize L* regarding q (Θ is supposed constant)
* Maximization Step : In this step we maximize L* regarding Θ (q is supposed constant)

In the case of Gaussian Mixture Models, it is possible to show that kmeans clustering is actually a generated 
instance of GMM. 

## Results

Here are a few illustrations on a small portion of an image found around the internet, with 3 clusters.
![Eye Original](eye.png)
![Eye Full](images/eye_gmm_full_out_.png)
![Eye Diag](images/eye_gmm_diag_out_.png)
![Eye Tied](images/eye_gmm_tied_out_.png)
![Eye Tied](images/eye_kmeans.png)

## References

Jeremy Fix, H.F Buet, M. Geist and F. Pennerath, [Machine Learning](http://sirien.metz.supelec.fr/spip.php?article91)