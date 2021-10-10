import pandas as pd
import numpy as np
import data_extraction


from numpy.random import randn
from math import sqrt


class FullyConnectedNeuralNetwork:
    def __init__(self):
        self.learnin_rate = 0.00001
        self.layers = []
        self.layers_size  = []
        self.weights = [] 
   
    def create_layer(self, num_of_neurons):
        self.layers_size.append(num_of_neurons)     

    def weights_init(self):
        layers_size = self.layers_size
        for i in range(0, len(layers_size) - 1):
            weight = randn(layers_size[i], layers_size[i+1]) * sqrt(2 / layers_size[i])
            self.weights.append(weight)

    def change_weights(self, new_weights):
        self.weights = new_weights
    
    def feed_forward(self, input_layer, correct_result):
        self.layers.append(input_layer)
        for i in range(0, len(self.layers_size) - 1):
            next_layer = np.dot(self.layers[i], self.weights[i])
            if i <  len(self.layers_size) - 2: 
                next_layer = np.vectorize(self.RelU)(next_layer)
            self.layers.append(next_layer) 
        output_layer = self.layers[len(self.layers_size) - 1]
        predicted_result = np.where(output_layer == output_layer.max())[0][0] + 1
        print(f"expected: {correct_result} got: {predicted_result}\n")
        return output_layer 
   
    def back_propagation(self, output_layer, correct_result):
        # add counter for accuracy and apdate it every iteration in the epoch. After every epoch, calculate the accuracy of the epoch and write it to a file.
        # after every epoch write the final weights of the epoch to a seperated file so it will be possible to use them later (in prediction).
        correct_output = np.zeros(10)
        correct_output[correct_result - 1] = 1
        error_output = correct_output - output_layer 
        pass 

    def train(self, input_layer, correct_results):
        
        pass
        
    def RelU(self, number):
        if number > 0:
            return number
        else:
            return 0



def main():
    data_extractor = data_extraction.ExtractTrainData('data/train.csv')
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
    NN.feed_forward(train_data[0], results[0])

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




