import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import norm

# Handle Data

data_df = pd.read_csv('./pima-indians-diabetes.csv')
S = data_df.as_matrix()

# The last column is the label one
# Random_state = 0 so we always get the same split
S_train, S_test = train_test_split(S, test_size = 0.25, random_state=0)

# Summarize Data

## Separate by class
S_train_true = S_train[S_train[:,-1] == 1]
S_train_false = S_train[S_train[:,-1] == 0]

## Mean and variance for each attribute per class 
## We are fitting  Gaussian distributions for each attributes per class i.e
## Evaluate Gaussian paramater for each attribute for each class
mean_true =  np.mean(S_train_true[:,:8], axis = 0)
mean_false = np.mean(S_train_false[:,:8], axis = 0)
variance_true = np.var(S_train_true[:,:8], axis = 0, ddof= 1)
variance_false = np.var(S_train_false[:,:8], axis = 0, ddof= 1)

conditional_probability_true = len(S_train_true)/len(S_train)
conditional_probability_false = len(S_train_false)/len(S_train)

# Make a predictions
probabilities = np.ones((S_test.shape[0], 2))
predictions = np.ones((S_test.shape[0], 1))
for i in range(0, S_test.shape[0]):
    for j in range(0, S_test.shape[1] -1):
        # Compute the probabilities given by parameter for each class
        probabilities[i,0] *= norm(mean_true[j], variance_true[j]).pdf(S_test[i,j]) * conditional_probability_false
        probabilities[i,1] *= norm(mean_false[j], variance_false[j]).pdf(S_test[i,j]) * conditional_probability_true
    if max(probabilities[i,0], probabilities[i,1]) == probabilities[i,0]:
        predictions[i, 0]  = 0

# Risk evaluation
## with L1-loss
print(predictions[:,0].shape)
print(S_test[:,-1].shape)
Risk = np.mean(np.abs(predictions[:,0] - S_test[:,-1]))
print(1-Risk)