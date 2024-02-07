import os, sys
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score, SoftmaxActivation, SigmoidActivation, MSELoss, CrossEntropyLoss, plot_digit_images, hist_plot, plot_training_process

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'      # train/test... Optional mode to avoid training incase you want to load saved model and test only.

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation, loss_function):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs
        

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        
        self.alpha = 0.001

    def generate_weight_and_bias(self, in_units, out_units, activation_type, dtype):
        
        #Normalized Xavier Weight Initialization
        
        f = 2.0 if activation_type == 'sigmoid' else 6.0
        bound  = np.sqrt(f / (in_units + out_units))
            
        np.random.seed(int(time.time()))
        weight = np.random.uniform(-bound, bound, (in_units, out_units))
        bias = np.random.uniform(-bound, bound,  out_units)

        return weight, bias
        
    def initialize_weights(self, dtype):   
        weights = []
        biases = []
        
        #Weight and bias for the input layer
        weight_layer1, bias1 = self.generate_weight_and_bias(self.num_input_features, self.num_hidden_units, 'sigmoid', dtype)      
        weights.append(weight_layer1)
        biases.append(bias1)
        print('weight_layer1 : ' , biases[0].shape)

        
        #Weight and bias for the only hidden layer
        weight_layer2, bias2 = self.generate_weight_and_bias( self.num_hidden_units, self.num_outputs, 'softmax', dtype)
        weights.append(weight_layer2)        
        biases.append(bias2)
        print('weight_layer2 : ' ,  biases[1].shape)
        
        
        return weights, biases

    def forward(self, X_train, y_train):
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        activations = []
        
        activations.append(X_train)        
        
        #Calculate Activation of hidden layer
        
        Z_1 = np.dot(X_train, self.weights[0]) + self.biases[0]
    
        # print(Z_1[0])
        Activation_1 = self.hidden_unit_activation.__call__(Z_1)
        activations.append(Activation_1)

        #Calculate Activation of output layer
        Z_2 = np.dot(Activation_1, self.weights[1]) + self.biases[1]
        Activation_2 = self.output_activation.__call__(Z_2)

        activations.append(Activation_2)
      
        
        return activations

    def compute_weight_gradient(self, layer, activations, deltas , weight_gradients, bias_gradients, data_size):
        d_W = activations[layer].T.dot(deltas[layer]) 
        d_W += self.alpha * self.weights[layer]
        d_W /= data_size
        
        weight_gradients[layer] = d_W        
        bias_gradients[layer] = np.mean(deltas[layer], 0)   
        
        

    def backward(self, activations, y_train, data_size) :  
        weight_gradients  = [None] * 2
        bias_gradients = [None] * 2        
        deltas = [None] * 2
        
        # Output Layer Delta, (Softmax) dL/dz2 = a2 - y
        layer = 1
        # deltas[1] = self.output_activation.gradient(activations[2], y_train)
        deltas[1] = activations[-1] -  y_train

       
        # #dL/dw2 = dL/dz2 * dz2/dw2        
        self.compute_weight_gradient(layer, activations, deltas, weight_gradients, bias_gradients, data_size)

        
        
        # # Hidden Layer Delta (Sigmoid)
        # # dL/dz1 = dL/dz2 * dz2/da1 * da1/dz1
        layer = 0        
        deltas[0] = deltas[1].dot(self.weights[1].T)

        self.hidden_unit_activation.gradient(activations[1], deltas[0])

        
        # #dL/dw1 = dL/dz1 * dz1/dw1
        self.compute_weight_gradient(layer, activations, deltas, weight_gradients, bias_gradients, data_size)
        
        return weight_gradients, bias_gradients

    def update_params(self, weight_gradients, bias_gradients):
        # Take the optimization step.
        for i in range(len(self.weights)):
            self.weights[i] -=  weight_gradients[i]

        for i in range(len(self.biases)):
            self.biases[i] -=  bias_gradients[i]
        
    
    def convert_targets_to_onehot(self, y):
        converted_y = np.zeros((y.size, 10))
        converted_y[np.arange(y.size), y] = 1
        
        return converted_y

    def train(self, train_data, train_target, learning_rate=0.01, num_epochs=300):
        self.learning_rate = learning_rate
        self.weights, self.biases = self.initialize_weights(train_data.dtype)
        
        X_train = np.asarray(train_data)
        y_train = np.asarray(train_target)

        data_size = y_train.shape[0]
        y_train_onehot = self.convert_targets_to_onehot(y_train)

        losses = []
        accs = []
        history = {'loss': losses, 'acc': accs}
        
        
        for epoch in range(num_epochs):           
 
            activations = self.forward(X_train, y_train_onehot)          
            loss = self.loss_function(activations[-1], y_train_onehot)

            
            weight_gradients, bias_gradients = self.backward(activations, y_train_onehot, data_size)

            
            self.update_params(weight_gradients, bias_gradients)
            
            y_pred = np.argmax(activations[-1], axis = 1)
         

            accuracy = accuracy_score(y_train, y_pred)
            
            losses.append(loss)
            accs.append(accuracy)
            
            print('Epoch', epoch, '  Loss: ', loss, end = ' | ')
            print('Accuracy :', accuracy)
        
        plot_training_process(history)

            

    def test(self, test_data, test_target):
        accuracy = 0    # Test accuracy
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        X_test = np.asarray(test_data)
        y_test = np.asarray(test_target)
        
        
        y_test_onehot = self.convert_targets_to_onehot(y_test)       
        activations = self.forward(X_test, y_test_onehot) 
        y_pred = np.argmax(activations[-1], axis = 1)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy


def main(argv):

    # Load dataset
    X, y, images = readDataLabels()     


    # Normalize X, then split data into train and test split. call function in data.py
    X_train, y_train, X_test, y_test, images_train, images_test = train_test_split(normalize_data(X), y, images)
    
    plot_digit_images(y_test, images_test)
    plt.show()
    time.sleep(1)
    # hist_plot( y_test)
    # plt.show()
    
    feature_num = len(X_train[0])
    output_num = 10 # 0 - 9
    
    
    softmax = SoftmaxActivation()
    sigmoid = SigmoidActivation()
    
    crossentropy = CrossEntropyLoss()
    

    
    ann = ANN(feature_num, 16, output_num, sigmoid, softmax, crossentropy)
    
    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        ann.train(X_train, y_train)
        # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.
    test_accuracy = ann.test(X_test, y_test)
    print('Test Accuracy: ', test_accuracy)

if __name__ == "__main__":
    main(sys.argv)
