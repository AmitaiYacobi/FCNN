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
        self.weights = []
        self.counter = 0

    def create_layer(self, num_of_neurons):
        self.layers_size.append(num_of_neurons)

    def weights_init(self):
        layers_size = self.layers_size
        for i in range(0, len(layers_size) - 1):
            weight = randn(layers_size[i], layers_size[i+1]) * sqrt(2 / layers_size[i])
            self.weights.append(weight)

    def change_weights(self, new_weights):
        self.weights = new_weights

    def feed_forward(self, input_layer, target):
        layers = []
        layers.append(input_layer)
        for i in range(0, len(self.layers_size) - 1):
            next_layer = np.dot(layers[i], self.weights[i])
            if i < len(self.layers_size) - 2:
                next_layer = np.vectorize(self.RelU)(next_layer)
            layers.append(next_layer)
        output_layer = layers[len(self.layers_size) - 1]
        predicted_result = np.where(output_layer == output_layer.max())[0][0] + 1
        print(f"expected: {target} got: {predicted_result}\n")
        if target == predicted_result:
            self.counter += 1
        return output_layer

    def back_propagation(self, output_layer, target, learning_rate):
        # add counter for accuracy and apdate it every iteration in the epoch. After every epoch, calculate the accuracy of the epoch and write it to a file.
        # after every epoch write the final weights of the epoch to a seperated file so it will be possible to use them later (in prediction).
        correct_output = np.zeros(10)
        correct_output[target - 1] = 1
        error_output = correct_output - output_layer
        print(error_output)

    def train(self, train_data, targets, num_of_epochs=60, learning_rate):
        i = 0
        self.weights_init()
        output_file = open("output.txt", "w")
        while (True):
            self.counter = 0
            i += 1
            for j in range(0, len(train_data)):
                output_layer = self.feed_forward(train_data[j], targets[j])
                new_weights = self.back_propagation(output_layer, targets[j], learning_rate)
                self.change_weights(new_weights)
            epoch_accuracy = (self.counter / len(targets)) * 100
            epoch_dir = f"epoch_{i}"
            os.mkdir(epoch_dir)
            output_file.write(f"accuracy of epoch number {i} is: {epoch_accuracy}\n")
            print(f"accuracy of epoch number {i} is: {epoch_accuracy}\n")
            for k in range(0, len(self.weights)):
                savetxt(f"{epoch_dir}\\layer_{k}_to_layer_{k+1}_weights.csv", self.weights[k], delimiter=',')
   
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

    def RelU(self, number):
        if number > 0:
            return number
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
    NN.weights_init()
    for k in range(0, len(NN.weights)):
        print(NN.weights[k])
        savetxt(f"layer_{k}_to_layer_{k+1}_weights.csv", NN.weights[k], delimiter=',')
        print("######################################################################################################")
        print(loadtxt(f"layer_{k}_to_layer_{k+1}_weights.csv", delimiter=',')) 


main()

# hidden1 = np.dot(input_layer, self.weights[0])
# hidden1 = np.vectorize(self.RelU)(hidden1)
# hidden2 = np.dot(hidden1, self.weights[1])
# hidden2 = np.vectorize(self.RelU)(hidden2)
# output = np.dot(hidden2, self.weights[2])

# first_hidden_layer = 2500 neurons
# second_hidden_layer = 1536 neurons
# results_layer = 10
# data extraction:
#   - extract results columns
#   - create new data (new csv file) without the results colunms
#   - clone the data from the file to a pandas table - optional (I will check this out)
# feed forward:
#   - initial weights accoding to some weights initialization method
#   - feed the layers with the row including the calculation of the
#     non-linear function.
#   - print the expected result and the current result.
#   - keep the result (the layer of the result will be
#     represented as 10 neourons)
# back propogation:
#   - execute the back propogation algorithm according to the result
#   - updtate the weights accordingly.
#   - after finishing iterating the whole data, calculate the accuracy of the
#     epoch and write it to a file.
#   -
