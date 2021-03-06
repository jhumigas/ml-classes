import matplotlib.pyplot as plt # for plotting.
import itertools as it          # for smart lazy iterations.
import numpy as np              # for fast array manipulations.
import pickle                   # for dumping python object instances.

# These are scikit modules
import sklearn.datasets 
import sklearn.utils
import sklearn.svm
import sklearn.metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# prepare a grid parameters
params_grid = [
  {'C': [0.001, 0.01, 0.1, 1], 'kernel': ['linear']},
  {'C': [0.001, 0.01, 0.1, 1], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# create and fit a ridge regression model, testing each alpha
svr = sklearn.svm.SVC()
model = Ridge()
grid = GridSearchCV(svr, params_grid)

# Let us load the dataset.
data_size = 2000
digits    = sklearn.datasets.fetch_mldata('MNIST original')
i, o      = sklearn.utils.shuffle(digits.data, digits.target)
data_size = min(data_size,len(i))
inputs    = i[:data_size]
outputs   = o[:data_size]
images    = (data.reshape((28,28))/255.0 for data in inputs)
labels    = np.array([int(i) for i in outputs])

# Let us apply our grid search (see the documentation for parameters)
print()
print('learning with a {}-sized dataset...'.format(data_size))
grid.fit(inputs, labels) # This is learning
print('done')

# Test (on the dataset itself...)
print('######')
print(grid)
print('######')
predicted = grid.predict(inputs)
#print(sklearn.metrics.classification_report(labels, predicted))
#print(sklearn.metrics.confusion_matrix(labels, predicted))

# Cross-validation
print()
print('################')
print('Cross validation')
print('################')
print()

scores = sklearn.model_selection.cross_val_score(grid, inputs, labels, cv=5)
print('scores = ')
for s in scores:
    print('         {:.2%}'.format(1-s))
print('        ------')
print('  risk = {:.2%}'.format(1-np.average(scores)))

# Saving your classifier thanks to pickle
outfile = open('grid-svc.pkl','wb')
pickle.dump(grid,outfile)
print()
print('classifier saved')

# Loading your classifier thanks to pickle
infile = open('grid-svc.pkl','rb')
predictor = pickle.load(infile)
print()
print('predictor loaded')


fig = plt.figure(figsize=(10,10))
data = zip(inputs,images,labels)
for idx, (vec,img,label) in enumerate(it.islice(data,16)):
    plt.subplot(4,4, idx + 1)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('#{} is {}/{}'.format(idx,
                                    predictor.predict([vec])[0],
                                    label))
plt.show()








