import matplotlib.pyplot as plt # for plotting.
import itertools as it          # for smart lazy iterations.
import numpy as np              # for fast array manipulations.
import pickle                   # for dumping python object instances.

# These are scikit modules
import sklearn.datasets 
import sklearn.utils
import sklearn.svm
import sklearn.metrics
import sklearn.cross_validation

# Let us load the dataset.
data_size = 5000
digits    = sklearn.datasets.fetch_mldata('MNIST original')
i, o      = sklearn.utils.shuffle(digits.data, digits.target)
data_size = min(data_size,len(i))
inputs    = i[:data_size]
outputs   = o[:data_size]
images    = (data.reshape((28,28))/255.0 for data in inputs)
labels    = np.array([int(i) for i in outputs])

# Let us apply a linear C-SVM (see the documentation for parameters)
print()
print('learning with a {}-sized dataset...'.format(data_size))
classifier = sklearn.svm.SVC(C=0.1, kernel='linear', tol=1e-3, decision_function_shape='ovo')
classifier.fit(inputs, labels) # This is learning
print('done')

# Test (on the dataset itself...)
print('######')
print(classifier)
print('######')
predicted = classifier.predict(inputs)
#print(sklearn.metrics.classification_report(labels, predicted))
#print(sklearn.metrics.confusion_matrix(labels, predicted))

# Cross-validation
print()
print('################')
print('Cross validation')
print('################')
print()

scores = sklearn.cross_validation.cross_val_score(classifier, inputs, labels, cv=5)
print('scores = ')
for s in scores:
    print('         {:.2%}'.format(1-s))
print('        ------')
print('  risk = {:.2%}'.format(1-np.average(scores)))

# Saving your classifier thanks to pickle
outfile = open('classifier.pkl','wb')
pickle.dump(classifier,outfile)
print()
print('classifier saved')

# Loading your classifier thanks to pickle
infile = open('classifier.pkl','rb')
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








