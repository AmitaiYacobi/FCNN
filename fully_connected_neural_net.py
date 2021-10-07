import pandas as pd
import numpy as np
import data_extraction


from numpy.random import randn
from math import sqrt

class FullyConnectedNeuralNet:
    def __init__(self):
        self.learnin_rate = 0.00001
        self.hidden_layers  = []
        self.weights = [] 
   
    def create_layer(self, num_of_neurons):
        self.hidden_layers.append(num_of_neurons)     
    
    def weights_init(self):
        hidden_layers = self.hidden_layers
        input_to_hidden1 = randn(hidden_layers[0], hidden_layers[1]) * sqrt(2 / hidden_layers[0])   
        hidden1_to_hidden2 = randn(hidden_layers[1], hidden_layers[2]) * sqrt(2 / hidden_layers[1])
        hidden2_to_output = randn(hidden_layers[2], hidden_layers[3]) * sqrt(2 / hidden_layers[2])
        self.weights = [input_to_hidden1, hidden1_to_hidden2, hidden2_to_output]

    def change_weights(self, new_weights):
        self.weights = new_weights

    def RelU(self, number):
        if number > 0:
            return number
        else:
            return 0



def main():
    data_extractor = data_extraction.ExtractTrainData('data/train.csv')
    train_data = data_extractor.get_train_data()
    input_layer_size = data_extractor.get_num_of_columns()
    NN = FullyConnectedNeuralNet()
    NN.create_layer(input_layer_size)
    NN.create_layer(2500)
    NN.create_layer(1536)
    NN.create_layer(10)
    NN.weights_init()

main()


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




