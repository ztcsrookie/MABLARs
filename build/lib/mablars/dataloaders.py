import numpy as np
from weka.core.converters import Loader, load_any_file


class DataLoader:
    def load_csv_data(self, data_path):
        '''
        Load a csv data set.
        :param data_path: a path of a csv file. Each row of the csv file represents a sample and each column represents a variable (or feature). The first row should be the name of the feature. All values should be numeric value.
        :return: the data and a list of the name of each node
        '''
        with open(data_path, 'r') as f:
            node_name_list = f.readline().strip().split(',')
            num_features = len(node_name_list) - 1
        data = np.genfromtxt(data_path, delimiter=',', skip_header=1)

        # x = np.genfromtxt(data_path, delimiter=',', skip_header=1, usecols=range(num_features))
        # original_y = np.genfromtxt(data_path, delimiter=',', skip_header=1, usecols=(num_features), dtype=str)
        return data, node_name_list

    def load_arff_data(self, arff_data_path):
        arff_data = load_any_file(arff_data_path)
        arff_data.class_is_last()
        return arff_data