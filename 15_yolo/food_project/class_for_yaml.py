import os
import pandas as pd

class_file_path = os.path.join(os.getcwd(), './class_counts_result.csv')

data = pd.read_csv(class_file_path, header=None, names=['class', 'count'])
class_data = data['class']
class_data.to_csv('class_for_yaml.csv', sep=':', header=False)
