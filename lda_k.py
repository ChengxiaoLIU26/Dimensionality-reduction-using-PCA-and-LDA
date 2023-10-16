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
import json


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

#dimensions = [2, 4, 6, 9, 15, 20, 30]
dimensions = [2, 4, 6, 9, 15, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600]
accuracies = []

for dim in dimensions:
    start_time_1 = timeit.default_timer()
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
    pred_labels = np.argmin(distances, axis=0)

    # Computing accuracy
    accuracy = metrics.accuracy_score(test_targets, pred_labels)
    accuracies.append(accuracy)
    #print("Accuracy with dimension ", dim, ": ", accuracy)
    end_time_1 = timeit.default_timer()
    execution_time_1 = end_time_1 - start_time_1
    print(f"The script executed in {execution_time_1} seconds")

with open('accuracies_lda.json', 'w') as f:
    json.dump(accuracies, f)

with open('dimensions_lda.json', 'w') as f:
    json.dump(dimensions, f)

# Make the line smoother
xnew = np.linspace(min(dimensions), max(dimensions), 500)
spl = make_interp_spline(dimensions, accuracies, k=3)  # type: BSpline
ynew = spl(xnew)

plt.figure(figsize=(10, 5))  # Adjust as needed
plt.plot(xnew, ynew, 'green', label='LDA')  # You can choose color as per your preference

plt.xlabel('dimension')
plt.ylabel('accuracy')
plt.title('The prediction accuracy as the dimension increases in Fashion-Mnist datasets')

plt.legend(loc='upper right')
plt.savefig('dimension-accuracy_lda_1.png', dpi=300)  # Save the figure


end_time = timeit.default_timer()
execution_time = end_time - start_time
print(f"The script executed in {execution_time} seconds")