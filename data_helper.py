import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np
from math import floor



def preprocess_data(xs, ys, d):
    xs = xs / 255.0
    xs = xs.reshape(xs.shape[0], -1)
    pca = PCA(n_components=d)
    pca.fit(xs)
    pca_data = pca.transform(xs)

    pca_descaler = [[] for _ in range(d)]
    # Data Normalization 
    for i in range(d):
        if pca_data[:,i].min() < 0:
            pca_descaler[i].append(pca_data[:,i].min())
            pca_data[:,i] += np.abs(pca_data[:,i].min())
        else:
            pca_descaler[i].append(pca_data[:,i].min())
            pca_data[:,i] -= pca_data[:,i].min()
        pca_descaler[i].append(pca_data[:,i].max())
        pca_data[:,i] /= pca_data[:,i].max()

    # Remove outliers
    valid_ind = [True for _ in range(len(pca_data))]
    for col in range(pca_data.shape[1]):
        t_data_mean = pca_data[:,col].mean()
        t_data_std = pca_data[:,col].std()
        valid_upper_bound = pca_data[:,col] < t_data_mean+t_data_std*2
        valid_lower_bound = pca_data[:,col] > t_data_mean-t_data_std*2
        valid = np.logical_and(valid_upper_bound,valid_lower_bound)
        valid_ind = np.logical_and(valid_ind, valid)

    pca_data = pca_data[valid_ind]
    pca_data = 2 * np.pi * pca_data
    ys = ys[valid_ind]

    return pca_data, ys


def load_dataset(client_num, q_num):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Filter the training data
    train_mask = (train_labels == 0) | (train_labels == 1)
    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask]

    train_images, train_labels = preprocess_data(train_images, train_labels, q_num)

    # Filter the test data
    test_mask = (test_labels == 0) | (test_labels == 1)
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask]

    test_images, test_labels = preprocess_data(test_images, test_labels, q_num)

    
    # Divide the data into non-overlapping subsets for each client
    n_samples_per_client = floor(len(train_images) / client_num)
    client_data_train_x = []
    client_data_train_y = []
    for i in range(client_num):
        client_data_train_x.append(train_images[i*n_samples_per_client:(i+1)*n_samples_per_client])       
        client_data_train_y.append(train_labels[i*n_samples_per_client:(i+1)*n_samples_per_client])
        
    n_samples_per_client = floor(len(test_images) / client_num)
    client_data_test_x = []
    client_data_test_y = []
    for i in range(client_num):
        client_data_test_x.append(test_images[i*n_samples_per_client:(i+1)*n_samples_per_client])
        client_data_test_y.append(test_labels[i*n_samples_per_client:(i+1)*n_samples_per_client])
                           
        
    return client_data_train_x, client_data_train_y, client_data_test_x, client_data_test_y
    