# Mixture of categorical disttributions and the EM algorithm 

## Theoritical design 

n people have to answer a questionnaire containing m questions. 
To simplify the problem, we assume that each question can be answer by a single answer among k possibilities.

We denote by aij the answer given by person j to question i. aij belongs to the set 1..k .
We want to analyse the questionnaire by clustering people in g groups.

1.1 Mixture model design 

We can see our problem as a mixture model where : 

* Our visible variables are the answers : Ai or Vi
* Our latent variables are associated to the clusters : H

We consider each variables Vi|H to be independent.

The parameters of our problem are linked to the modelisation of the distribution of each cluster
 