import os
import pandas as pd
import numpy as np
import data_handler


from numpy.random import randn
from numpy import savetxt
from numpy import loadtxt
from math import sqrt


class FullyConnectedNeuralNetwork:
    def __init__(self):
        self.learnin_rate = 0.00001
        self.layers = []
        self.layers_size = []
        self.layers_without_activation_func = []
        self.weights = []
        self.counter = 0
        self.number_of_layers = 0

    def create_layer(self, num_of_neurons):
        self.layers_size.append(num_of_neurons)
        self.number_of_layers += 1

    def weights_init(self):
        layers_size = self.layers_size
        for i in range(0, self.number_of_layers - 1):
            weight = randn(layers_size[i], layers_size[i+1]) * sqrt(2 / layers_size[i])
            print(weight.shape)
            self.weights.append(weight)

    def change_weights(self, new_weights):
        self.weights = new_weights

    def feed_forward(self, input_layer, target):
        self.layers_without_activation_func = []
        self.layers_without_activation_func.append(input_layer)
        self.layers = []
        self.layers.append(input_layer)
        for i in range(0, self.number_of_layers - 1):
            next_layer = np.dot(self.layers[i], self.weights[i])                         ## multiply matrices
            self.layers_without_activation_func.append(next_layer)                       ## keeping the actual value of the neurons without activating the function over them.
            if i < self.number_of_layers - 2:
                next_layer = np.vectorize(self.RelU)(next_layer)                         ## activating activation function over the neurons.
            self.layers.append(next_layer)
        output_layer = self.layers[self.number_of_layers - 1]
        predicted_result = np.where(output_layer == output_layer.max())[0][0] + 1
        print(f"expected: {target} got: {predicted_result}\n")
        if target == predicted_result:
            self.counter += 1
        return output_layer

    def back_propagation(self, output_layer, target, learning_rate):
        errors_layers = []
        updated_weights =[]
        correct_output = np.zeros(10)
        correct_output[target - 1] = 1
        error_output = correct_output - output_layer
        errors_layers.append(error_output)
        for i in range(self.number_of_layers - 2, -1, -1):
            current_layer_without_activation = self.layers_without_activation_func[i]
            derivative_on_currnet_layer = np.vectorize(self.RelU_derivative)(current_layer_without_activation)
            errors_of_layer_above = errors_layers[self.number_of_layers - 2 - i]
            weights_i = np.transpose(self.weights[i])                                   ## multiply the errors of the above layer with the trandpose of the current weights matrix. 
                                                                                        ## thats how we get the magnitude of the errors.
            magnitude_of_errors = np.dot(errors_of_layer_above, weights_i)              ## the magnitude of the errors in the current layer 
            errors_of_current_layer = derivative_on_currnet_layer * magnitude_of_errors
            errors_layers.append(errors_of_current_layer)
            layer_i = self.layers[i]
            updated_weight = self.weights[i] + (learning_rate * (layer_i.reshape(self.layers_size[i], 1) * errors_of_layer_above))
            
            ## print(self.layers[i]) to check whether the shape of the layers has changed. if it has, it is a problem. 
            updated_weights = [updated_weight] + updated_weights
        return updated_weights

    def train(self, train_data, targets, learning_rate, num_of_epochs=60):
        i = 0
        self.weights_init()
        output_file = open("output.txt", "w")
        while (i <= num_of_epochs):
            self.counter = 0
            i += 1
            for j in range(0, len(train_data) - 1):
                output_layer = self.feed_forward(train_data[j], targets[j])
                new_weights = self.back_propagation(output_layer, targets[j], learning_rate)
                self.change_weights(new_weights)
            epoch_accuracy = (self.counter / len(targets)) * 100
            os.mkdir(f"epoch_{i}")
            output_file.write(f"accuracy of epoch number {i} is: {epoch_accuracy}\n")
            print(f"accuracy of epoch number {i} is: {epoch_accuracy}%\n")
            for k in range(0, len(self.weights)):
                savetxt(f"epoch_{i}/layer_{k}_to_layer_{k+1}_weights.csv", self.weights[k], delimiter=',')
   
    def predict(self, validate_data, targets, epoch_dir):
        weights = []
        self.counter = 0
        for i in range(0, len(self.weights)):
            weights.append(loadtxt(f"{epoch_dir}\\layer_{i}_to_layer_{i+1}_weights.csv", self.weights[i], delimiter=','))
        self.weights = weights
        for j in range(0, len(validate_data)):
            self.feed_forward(validate_data[j], targets[j])
        accuracy = (self.counter / len(targets)) * 100
        print(accuracy)

    def RelU(self, x):
        return np.maximum(0, x)
        
    def RelU_derivative(self, x):
        if x > 0:
            return 1
        else:
            return 0




def main():
    data_extractor = data_handler.ExtractTrainData('data/train.csv')
    train_data = data_extractor.get_train_data()
    train_data = train_data.to_numpy()
    input_layer_size = data_extractor.get_num_of_columns()
    results = data_extractor.get_results_column()
    results = results.to_numpy()

    NN = FullyConnectedNeuralNetwork()
    NN.create_layer(input_layer_size)
    NN.create_layer(2500)
    NN.create_layer(1536)
    NN.create_layer(10)
    # NN.weights_init()
    # NN.feed_forward(train_data[0], results[0])
    NN.train(train_data, results, 0.00001, 9)
    # for k in range(0, len(NN.weights)):
    #     print(NN.weights[k])
    #     savetxt(f"layer_{k}_to_layer_{k+1}_weights.csv", NN.weights[k], delimiter=',')
    #     print("######################################################################################################")
    #     print(loadtxt(f"layer_{k}_to_layer_{k+1}_weights.csv", delimiter=',')) 


main()

# hidden1 = np.dot(input_layer, self.weights[0])
# hidden1 = np.vectorize(self.RelU)(hidden1)
# hidden2 = np.dot(hidden1, self.weights[1])
# hidden2 = np.vectorize(self.RelU)(hidden2)
# output = np.dot(hidden2, self.weights[2])
