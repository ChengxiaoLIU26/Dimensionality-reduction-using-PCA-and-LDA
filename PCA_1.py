import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
import numpy as np
import os
import timeit

def get_shape(lst):
    try:
        return [len(lst)] + get_shape(lst[0])
    except TypeError:
        return []

start_time = timeit.default_timer()

# Load Fashion-MNIST Dataset
transform = transforms.ToTensor()
trainset = datasets.FashionMNIST('../data/F_MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('../data/F_MNIST_data/', download=True, train=False, transform=transform)

# Prepare data for PCA
# Flatten the images and convert labels to numpy arrays
train_data = trainset.data.numpy().reshape((len(trainset), -1))
train_labels = trainset.targets.numpy()

test_data = testset.data.numpy().reshape((len(testset), -1))
test_labels = testset.targets.numpy()

# Perform PCA
dim = 60
pca = PCA(n_components=dim)  # Reduce to 50 dimensions
print(dim)
pca.fit(train_data)

train_data_pca = pca.transform(train_data)
test_data_pca = pca.transform(test_data)

# Calculate mean and covariance for each class in training set
means = []
covariances = []  # 10  * 9 * 9

for i in range(10):  # There are 10 classes in Fashion-MNIST
    data_i = train_data_pca[train_labels == i]
    #print(data_i.shape) (6000,9)
    means.append(np.mean(data_i, axis=0))
    covariances.append(np.cov(data_i, rowvar=False))
    #print(i)

#print(get_shape(means))

# Predicting on the test dataset with Minimum Mahalanobis Distance Classifier
# Compute the pseudoinverses of the covariance matrices
cov_inv = [np.linalg.pinv(cov) for cov in covariances]
#print(get_shape(cov_inv)) # 10 * 9*9

# Calculate the differences to the means
diff = np.array([test_data_pca - mean for mean in means])
#print(diff.shape)  # 10  * 10000 * 9

# Calculate the Mahalanobis distances
#distances = np.array([np.sqrt(np.sum((d @ c) * d, axis=-1)) for d, c in zip(diff, cov_inv)])
distances = np.array([np.sum((d @ c) * d, axis=-1) for d, c in zip(diff, cov_inv)])
#print(distances.shape)   # 10  * 10000

sigma = np.linalg.det(covariances).reshape(-1,1)
# Predict the labels
#pred_labels = np.argmin(distances + np.log(sigma), axis=0)
pred_labels = np.argmin(distances, axis=0)

# Computing accuracy
accuracy = metrics.accuracy_score(test_labels, pred_labels)
print("Accuracy with dimension ", dim, ": ", accuracy)


'''
# Save 10 images with their predicted and true labels
directory = './predictions_400'
if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(10):
    plt.figure()
    #plt.imshow(testset.data[i], cmap='gray')
    plt.imshow(testset.data[800*i+7])
    plt.title(f'True: {testset.classes[test_labels[800*i+7]]}, Predicted: {testset.classes[pred_labels[800*i+7]]}')
    plt.savefig(f'{directory}/image_{i}.jpg')
'''

end_time = timeit.default_timer()
execution_time = end_time - start_time
print(f"The script executed in {execution_time} seconds")


