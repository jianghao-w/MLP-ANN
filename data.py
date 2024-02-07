import os, sys
import numpy as np
from sklearn import datasets

import math

#import libraries as needed

def readDataLabels():
	#read in the data and the labels to feed into the ANN
    data = datasets.load_digits()
    X = data.data
    y = data.target
    images = data.images
    return X, y, images

def to_categorical(y):
	
	#Convert the nominal y values tocategorical

	return y
	
def train_test_split(data,labels,images, n=0.8):
    
    if len(data) != len(labels):
        raise ValueError("Sample size does not match target size")
    
    num_total_sample = len(data)
    
	#split data in training and testing sets
    
    num_train = math.ceil(n * num_total_sample) 
    # print(num_train)
    
    X_train = []
    y_train = []
    
    X_test = []
    y_test = []
    
    shuffled_index =  np.arange(num_total_sample)
    np.random.shuffle(shuffled_index)
    # print(data[:5])
    data = data[shuffled_index]
    labels = labels[shuffled_index]
    images = images[shuffled_index]
    
    X_train = data[:num_train]
    y_train = labels[:num_train]
    images_train = images[:num_train]
    
    X_test = data[num_train:]
    y_test = labels[num_train:]
    images_test = images[num_train:]
        

    # print(num_total_sample, len(X_train),len(y_train), len(X_test), len(y_test))
    return X_train, y_train, X_test, y_test, images_train, images_test



def normalize_data(X):
	# normalize/standardize the data
    
    X_norm = (X - X.min()) / (X.max() - X.min())
    return X_norm
