import matplotlib.pyplot as plt # for plotting.
import itertools as it          # for smart lazy iterations.

# This is scikit modules
import sklearn.datasets 
import sklearn.utils

# Let us load the dataset.
digits = sklearn.datasets.fetch_mldata('MNIST original')
inputs, outputs = sklearn.utils.shuffle(digits.data, digits.target)

# generators (lazy objects) allows for data conditioning.
images        = (input.reshape((28,28))/255.0 for input in inputs)
labels        = (int(i) for i in outputs)
images_labels = zip(images,labels)

# Let us display images (width labels) from #17 to #67 in a
# 5 rows and 10 columns grid.
fig = plt.figure(figsize=(15,10))
start = 17
end   = 67
for idx, (img, label) in enumerate(it.islice(images_labels,start,end)):
    plt.subplot(5, 10, idx + 1)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('#{} is {}'.format(start+idx,label))
plt.show()

