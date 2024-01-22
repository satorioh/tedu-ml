"""
加权平均值
"""
import numpy as np
import pandas as pd

data = pd.read_json("../data_test/ratings.json")
print(data)
print("==" * 20)
fracture = data.loc['Fracture']
print(fracture)
print("==" * 20)

weights = [1, 10, 1, 1, 1, 10, 1]
weight_average = np.average(fracture, weights=weights)
print(weight_average)
