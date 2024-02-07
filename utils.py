import numpy as np
import math
from scipy.special import xlogy
from matplotlib import pyplot as plt

# import hypertools as hyp

class MSELoss:      # For Reference
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        self.current_prediction = y_pred
        self.current_gt = y_gt

        # MSE = 0.5 x (GT - Prediction)^2
        loss = 0.5 * np.power((y_gt - y_pred), 2)
        return loss

    def grad(self):
        # Derived by calculating dL/dy_pred
        gradient = -1 * (self.current_gt - self.current_prediction)

        # We are creating and emptying buffers to emulate computation graphs in
        # Modern ML frameworks such as Tensorflow and Pytorch. It is not required.
        self.current_prediction = None
        self.current_gt = None

        return gradient


class CrossEntropyLoss:
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_true):
        	
        loss = -np.sum(y_true * np.log(y_pred))/y_pred.shape[0]
        
        return loss

    def grad(self):
        # Calculate Gradients for back propagation
        gradient = None
        return gradient


class SoftmaxActivation:
    def __init__(self):
        pass

    def __call__(self, z):
        
        tmp = z - z.max(axis=1)[:, np.newaxis] 
        np.exp(tmp, out=z)
        z /= z.sum(axis=1)[:, np.newaxis]
        
        return z
    
        

    def gradient(self, A, Y):
        # Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        return A - Y


class SigmoidActivation:    
    def __init__(self):
        pass

    def __call__(self, z):
        # Calculate Activation Function
        return 1 / (1 + np.exp(-z))
       

    def gradient(self, z, delta):
        #  Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        delta *= z
        delta *= (1-z)
        


class ReLUActivation:
    def __init__(self):
        self.z = None
        pass

    def __call__(self, z):
        # y = f(z) = max(z, 0) -> Refer to the computational model of an Artificial Neuron
        self.z = z
        y = np.maximum(z, 0)
        return y

    def gradient(self):
        # dy/dz = 1 if z was > 0 or dy/dz = 0 if z was <= 0
        gradient = np.where(self.z > 0, 1, 0)
        return gradient


def accuracy_score(y_true, y_pred):
    # y_true = np.full((3),1)
    # y_pred = np.full((3),1)
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    
    return accuracy


def plot_digit_images(y, images):
    fig = plt.figure(figsize=(8, 8)) 
    fig.subplots_adjust(left=0, right=0.5, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
    for i in range(36):
        ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])
        ax.imshow(images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
        ax.text(0, 5, str(y[i]))
        
        
def hist_plot(y):
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts)
    plt.xticks(unique)
    plt.xlabel("Label")
    plt.ylabel("Quantity")
    plt.title("Testing dataset plot")
    
def plot_training_process(history):
    plt.plot(history['loss'], label='training loss')
    plt.plot(history['acc'], label='training acc')
    #pyplot.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1.2])
    plt.legend(loc='lower right')
    plt.show()
    

#     hyp.plot(X, ".", hue=y)