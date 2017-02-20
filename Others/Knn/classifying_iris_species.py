from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

# Loading dataset
iris_dataset = load_iris()

# Printing dataset
print("Keys of iris_dataset: \{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Features names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("Five columns of data:\n{}".format(iris_dataset['data'][:5]))
print("Target:\n{}".format(iris_dataset['target']))

# Splitting dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# Visualizing 
# Create dataframe from data in X_train
# label column using strings in iris_dataset.feature_names
iris_df = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.tools.plotting.scatter_matrix(iris_df, c=y_train, figsize=(15,15), marker='o',
                        hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)

plt.show()

# Designing estimator
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
