import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from torchvision import datasets, transforms
from sklearn import metrics
from LDAmethod import MyLDA
import os
import timeit

start_time = timeit.default_timer()

# Loading the Fashion MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.FashionMNIST('../data/F_MNIST_data/', download=True, train=True, transform=transform)
test_set = datasets.FashionMNIST('../data/F_MNIST_data/', download=True, train=False, transform=transform)

train_data = train_set.data.numpy()
train_data = train_data.reshape(train_data.shape[0], -1)
train_targets = train_set.targets.numpy()

test_data = test_set.data.numpy()
test_data = test_data.reshape(test_data.shape[0], -1)
test_targets = test_set.targets.numpy()


dim = 11
print("lda", dim)
lda = MyLDA(n_components=dim)
lda.fit(train_data, train_targets)
train_data_lda = lda.transform(train_data)
test_data_lda = lda.transform(test_data)


means = []
covs = []
for i in range(10): # 10 classes in Fashion MNIST
    elements = [x for x, t in zip(train_data_lda, train_targets) if t == i]
    means.append(np.mean(elements, axis=0))
    covs.append(np.cov(np.array(elements).T))

# Predicting on the test dataset with Minimum Mahalanobis Distance Classifier
# Compute the pseudoinverses of the covariance matrices
cov_inv = [np.linalg.pinv(cov) for cov in covs]

# Calculate the differences to the means
diff = np.array([test_data_lda - mean for mean in means])

# Calculate the Mahalanobis distances
distances = np.array([np.sqrt(np.sum((d @ c) * d, axis=-1)) for d, c in zip(diff, cov_inv)])
#  print(distances.shape)  10 * 10000
# Predict the labels
sigma = np.linalg.det(covs).reshape(-1,1)
pred_labels = np.argmin(distances, axis=0)

# Computing accuracy
accuracy = metrics.accuracy_score(test_targets, pred_labels)
print("Accuracy with dimension ", dim, ": ", accuracy)

'''
# Save 10 images with their predicted and true labels
directory = './lda_predictions_400'
if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(10):
    plt.figure()
    #plt.imshow(testset.data[i], cmap='gray')
    plt.imshow(test_set.data[800*i+7])
    plt.title(f'True: {test_set.classes[test_targets[800*i+7]]}, Predicted: {test_set.classes[pred_labels[800*i+7]]}')
    plt.savefig(f'{directory}/image_{i}.jpg')
'''

end_time = timeit.default_timer()
execution_time = end_time - start_time
print(f"The script executed in {execution_time} seconds")



