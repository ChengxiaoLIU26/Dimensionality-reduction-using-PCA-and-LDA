import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
import numpy as np
import os
import timeit
import json

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


#dimensions = [2, 4, 6, 9, 15]
dimensions = [2, 4, 6, 9, 15, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600]
accuracies = []
# Perform PCA
for dim in dimensions:
    start_time_1 = timeit.default_timer()
    pca = PCA(n_components=dim)  # Reduce to 50 dimensions
    print('pca',dim)
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
    #print(get_shape(cov_inv))

    # Calculate the differences to the means
    diff = np.array([test_data_pca - mean for mean in means])
    #print(diff.shape)  # 10  * 10000 * 9

    # Calculate the Mahalanobis distances
    distances = np.array([np.sqrt(np.sum((d @ c) * d, axis=-1)) for d, c in zip(diff, cov_inv)])
    #print(distances.shape)   # 10  * 10000
    # Predict the labels
    pred_labels = np.argmin(distances, axis=0)

    # Computing accuracy
    accuracy = metrics.accuracy_score(test_labels, pred_labels)
    accuracies.append(accuracy)
    # print("Accuracy with dimension ", dim, ": ", accuracy)
    end_time_1 = timeit.default_timer()
    execution_time_1 = end_time_1 - start_time_1
    print(f"The script executed in {execution_time_1} seconds")

with open('accuracies_pca.json', 'w') as f:
    json.dump(accuracies, f)

with open('dimensions_pca.json', 'w') as f:
    json.dump(dimensions, f)

# Make the line smoother
xnew = np.linspace(min(dimensions), max(dimensions), 500)
spl = make_interp_spline(dimensions, accuracies, k=3)  # type: BSpline
ynew = spl(xnew)

plt.figure(figsize=(10, 5))  # Adjust as needed
plt.plot(xnew, ynew, 'skyblue', label='PCA')  # You can choose color as per your preference

plt.xlabel('dimension')
plt.ylabel('accuracy')
plt.title('The prediction accuracy as the dimension increases in Fashion-Mnist datasets')

plt.legend(loc='upper right')
plt.savefig('dimension-accuracy_pca_0.png', dpi=300)  # Save the figure



end_time = timeit.default_timer()
execution_time = end_time - start_time
print(f"The script executed in {execution_time} seconds")