# y = sum(w*x) + b 
# Linear Regression = OLS 
# Ridge Regression = OLS + L2 penalization
# LASSO = OLS + L1 penalization for large amount of features and expect only a few to be needed

# For classification 
# Logistic Regression and Linear SVC
# Linear SVC => C up, tolerance down, importance of each individual sample, C down, try to fit the majority

# For big big set, use sag
# Very useful fast to predict, fast to learn, fast to understand
import mglearn
from sklearn.linear_model import LinearRegression

X,y = mglearn.datasets.make_wave()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)