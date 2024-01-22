"""中位数"""
import numpy as np
import pandas as pd

data = pd.read_json("../data_test/ratings.json")
fracture = data.loc['Fracture']
print(np.median(fracture))  # 4.0
print(fracture.median())  # 4.0
