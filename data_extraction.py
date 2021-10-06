import pandas as pd
import numpy as np

train_file_path = 'data/train.csv'

class ExtractTrainData:
    def __init__(self, path_to_data_file):
        self.path_to_data_file = path_to_data_file
        self.train_file = None
    
    def get_results_column(self):
        if(self.train_file == None):
            self.train_file  = pd.read_csv(self.path_to_data_file, header=None)
            self.train_file.iloc[:, 0] 

    def get_train_file(self): 
        if(self.train_file == None):
            self.train_file  = pd.read_csv(self.path_to_data_file, header=None)
        self.train_file = self.train_file.iloc[:, 1:]
        return self.train_file      


