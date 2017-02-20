# Mixture of categorical disttributions and the EM algorithm 

We consider here a simple case to apply the EM algorithm. 

* A survey application : Individuals answer a set of questions and we want to group them

## Theoritical design 

We denote by H and V respectively our hidden and visible variables.  Θ stands for the hyperparameters for all the distributions.

A general modelisation with plate notation would look like this.

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

### Survey Mixture Model Design 
 
A more thorough representation : 
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


## Results

TODO

