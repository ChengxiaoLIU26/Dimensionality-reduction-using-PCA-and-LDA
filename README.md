# Dimensionality-reduction-using-PCA-and-LDA (NTU assignment in Machine Vision)
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

