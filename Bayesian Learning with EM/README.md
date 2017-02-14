# Mixture of categorical disttributions and the EM algorithm 

We consider here two cases to apply the EM algorithm. 

* A survey application : Individuals answer a set of questions and we want to group them
* Image application : Simple RGB image on which we apply a Gaussian Mixture Model

## Theoritical design 

For both cases, we denote by H and V respectively our hidden and visible variables.  Θ stands for the hyperparameters for all the distributions.

A modelisation, valid for both examples, with plate notation would look like this.

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

For the survey application, we have the following : 

> n people have to answer a survey containing m questions. 
> To simplify the problem, we assume that each question can be answer by a single answer among k possibilities.

We denote by aij the answer given by person j to question i. aij belongs to the set 1..k .
We want to analyse the questionnaire by clustering people in g groups.

### Image Mixture Model Design

We consider each pixel characterised by a vector(R,G,B) to be a sample. We want to learn a Gaussian Mixture Model on the set of all those samples.

We make no further assumption other than the one needed to build a GMM : we suppose the visible variables follow a Multivariate Gaussian distributions.
For each cluster, the corresponding distribution is characherized by a `3*1` mean vector and a `3*3`covariance matrice. 

### Survey Mixture Model Design 
 
A thorough representation specific to the survey application : 
```
         -------------
        |      H      |
        |      ↓      |
        |  ---------  |
 Θ ->   | |  -----  | |
        | | |  P  | | |
        | | |     | | |
        | |  -----  | |
        | |    ↑    | |
        | |    Q    | |
        |  ---------  |
        |             |
         -------------
```

Where Θ represents a distribution parameter (it is a set here since it contains parameter of distribution H, V). H is the variable relative to the hiddden variable i.e the cluster.
Supposing we have g clusters would mean that H ranges from 1 to g.
Since n persons answer m questions, one question is linked to a set of answers, the strength of a link kind of depend on the conditional probability that an answer is given to the given 
question. We note by H the latent variable i.e the clusters.

We further assume that our visible variables are all independent for the survey case. 

## EM Algorithm

We assume q, a distribution that approximated H|V,Θ. The overall goal of the EM algorithm is to actually maximize a lower bound of the likelihood P(H,V|Θ), called here L*, thus reducing the divergence between q and H|V,Θ
EM involves two main steps : 

* Expectation Step : In this step we maximize L* regarding q (Θ is supposed constant)
* Maximization Step : In this step we maximize L* regarding Θ (q is supposed constant)

In the case of Gaussian Mixture Models, it is possible to show that kmeans clustering is actually a generated 
instance of GMM. 

## Results

Here are a few illustrations on a small portion of an image found around the internet, with 3 clusters.
![Eye Original](https://gitlab.rezometz.org/david.mugisha/ml-classes/blob/a0c4c56d3266db0aeae7b2f647de4003e3ea5492/Bayesian%20Learning%20with%20EM/eye.png)
![Eye Full](https://gitlab.rezometz.org/david.mugisha/ml-classes/blob/a0c4c56d3266db0aeae7b2f647de4003e3ea5492/Bayesian%20Learning%20with%20EM/images/eye_gmm_full_out_.png)
![Eye Diag](https://gitlab.rezometz.org/david.mugisha/ml-classes/blob/a0c4c56d3266db0aeae7b2f647de4003e3ea5492/Bayesian%20Learning%20with%20EM/images/eye_gmm_diag_out_.png)
![Eye Tied](https://gitlab.rezometz.org/david.mugisha/ml-classes/blob/a0c4c56d3266db0aeae7b2f647de4003e3ea5492/Bayesian%20Learning%20with%20EM/images/eye_gmm_tied_out_.png)
![Eye Tied](https://gitlab.rezometz.org/david.mugisha/ml-classes/blob/a0c4c56d3266db0aeae7b2f647de4003e3ea5492/Bayesian%20Learning%20with%20EM/images/eye_kmeans.png)

