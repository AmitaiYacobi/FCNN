import pandas as pd
import numpy as np

train_file_path = 'data/train.csv'

class ExtractTrainData:
    def __init__(self, path_to_data_file):
        self.path_to_data_file = path_to_data_file
        self.train_data = pd.read_csv(self.path_to_data_file, header=None)
    
    def get_results_column(self):
        return self.train_data.iloc[:, 0] 

    def get_train_data(self): 
        return self.train_data.iloc[:, 1:]

    def get_num_of_columns(self):
        return len(self.get_train_data().columns) 
        
